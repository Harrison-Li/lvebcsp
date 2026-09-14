"""LatentDiT flow-matching training entry point."""

from __future__ import annotations

import argparse
import inspect
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lvebcsp.data.collate import collate_crystal_batch
from lvebcsp.data.representation import build_structure_tensor
from lvebcsp.models.dit import LatentDiT
from lvebcsp.models.interpolants import FlowMatchingInterpolant
from lvebcsp.models.jepa import Lvebm
from lvebcsp.train.ema import init_ema_state, update_ema_state
from lvebcsp.train.train_jepa import (
    batch_to_device,
    build_dataset_from_config,
    build_early_stopper,
    build_jepa_from_config,
    build_lr_scheduler,
    ensure_ratio_in_batch,
    format_learning_rate,
    has_dataset_split,
)
from lvebcsp.common.config import load_config, save_config, select_device
from lvebcsp.common.gpu import maybe_data_parallel, primary_device, resolve_gpu_ids, unwrap_model
from lvebcsp.common.seed import seed_everything
from lvebcsp.common.wandb_logging import (
    finish_wandb,
    init_wandb,
    log_wandb,
    maybe_watch_model,
)


DIRECT_LOG_METRICS = (
    "coord_frac_rmse",
    "coord_velocity_rmse",
    "lattice_len_rel_rmse",
    "lattice_len_angstrom_rmse",
    "lattice_angle_deg_rmse",
    "t_avg",
)


def disable_native_mha_fastpath() -> None:
    """Avoid CUDA native-MHA fastpath crashes seen with DataParallel eval."""

    mha_backend = getattr(torch.backends, "mha", None)
    set_fastpath_enabled = getattr(mha_backend, "set_fastpath_enabled", None)
    if callable(set_fastpath_enabled):
        set_fastpath_enabled(False)


def latent_dit_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the LatentDiT model section."""

    return dict(config.get("latent_dit", {}))


def build_direct_from_config(config: dict[str, Any]) -> LatentDiT:
    """Instantiate LatentDiT from config."""

    model_cfg = latent_dit_config(config)
    model_cfg.pop("residual_output", None)
    model_d_y = int(model_cfg.pop("d_y", config.get("d_y", 8)))
    model_signature = inspect.signature(LatentDiT.__init__)
    valid_model_args = {
        name
        for name, parameter in model_signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in model_signature.parameters.values()):
        valid_model_args.add("order")
    model_cfg = {name: value for name, value in model_cfg.items() if name in valid_model_args}
    return LatentDiT(
        n_max=int(config.get("n_max", 64)),
        d_y=model_d_y,
        d_jepa=int(config.get("d_jepa", 512)),
        **model_cfg,
    )


def build_direct_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Build the LatentDiT AdamW optimizer."""

    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))


def selected_direct_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Return the small LatentDiT metric set exposed in logs."""

    return {name: float(metrics[name]) for name in DIRECT_LOG_METRICS if name in metrics}


def format_direct_epoch_log(
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    train_metrics: dict[str, float],
    learning_rate: float,
    val_metrics: dict[str, float] | None = None,
) -> str:
    """Format the compact LatentDiT epoch log line."""

    fields: list[tuple[str, float]] = [("train_loss", train_loss)]
    if val_loss is not None:
        fields.append(("val_loss", val_loss))
    fields.extend(selected_direct_metrics(train_metrics).items())
    if val_metrics:
        fields.extend((f"val_{name}", value) for name, value in selected_direct_metrics(val_metrics).items())
    metric_text = " ".join(f"{name}={value:.6f}" for name, value in fields)
    metric_text = f"{metric_text} lr={format_learning_rate(learning_rate)}"
    return f"epoch={epoch} {metric_text}"


def direct_wandb_metrics(
    train_loss: float,
    val_loss: float | None,
    train_metrics: dict[str, float],
    learning_rate: float,
    val_metrics: dict[str, float] | None = None,
) -> dict[str, float | None]:
    """Build the compact LatentDiT W&B payload."""

    payload: dict[str, float | None] = {
        "train/loss": train_loss,
        "val/loss": val_loss,
        "trainer/lr": learning_rate,
    }
    payload.update(
        {f"train/{name}": value for name, value in selected_direct_metrics(train_metrics).items()}
    )
    if val_metrics:
        payload.update({f"val/{name}": value for name, value in selected_direct_metrics(val_metrics).items()})
    return payload


def upgrade_legacy_jepa_head_mlp_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: Lvebm,
) -> dict[str, torch.Tensor]:
    """Map old JEPA MLP norm keys to the current MLP module layout."""

    model_state = model.state_dict()
    upgraded = dict(state_dict)
    for key, value in list(state_dict.items()):
        if ".mlp.1." not in key or key in model_state:
            continue
        target_key = key.replace(".mlp.1.", ".mlp.2.")
        if target_key in model_state and tuple(model_state[target_key].shape) == tuple(value.shape):
            upgraded[target_key] = value
            upgraded.pop(key, None)
    return upgraded


def load_jepa_checkpoint(path: str | Path, device: torch.device) -> Lvebm:
    """Load a JEPA checkpoint saved by train_jepa.py."""

    ckpt = torch.load(path, map_location=device)
    model = build_jepa_from_config(ckpt.get("config", {})).to(device)
    model.load_state_dict(upgrade_legacy_jepa_head_mlp_state_dict(ckpt["model"], model))
    return model


def load_direct_model_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load compatible LatentDiT weights while ignoring unrelated stale keys."""

    target = unwrap_model(model)
    current = target.state_dict()
    lattice_key = "lattice_embedder.0.weight"
    if (
        lattice_key in state_dict
        and lattice_key in current
        and tuple(state_dict[lattice_key].shape) != tuple(current[lattice_key].shape)
    ):
        raise RuntimeError(
            "incompatible independent-angle LatentDiT checkpoint: the current "
            "model requires the Gram-Cholesky lattice representation"
        )
    upgraded = {
        name: value
        for name, value in state_dict.items()
        if name in current and tuple(value.shape) == tuple(current[name].shape)
    }
    merged = dict(current)
    merged.update(upgraded)
    target.load_state_dict(merged)


def validate_jepa_latent_dim(config: dict[str, Any], jepa: Lvebm, checkpoint_path: str | Path) -> None:
    """Ensure LatentDiT expects the latent dimension emitted by the JEPA checkpoint."""

    jepa_dim = int(jepa.d_jepa)
    config_dim = config.get("d_jepa")
    if config_dim is None:
        config["d_jepa"] = jepa_dim
        return
    if int(config_dim) != jepa_dim:
        raise ValueError(
            f"LatentDiT config d_jepa={config_dim} does not match JEPA checkpoint "
            f"{checkpoint_path} d_jepa={jepa_dim}. Set d_jepa to {jepa_dim} or use a matching JEPA checkpoint."
        )


def restore_early_stopper(early_stopper: Any, state: dict[str, Any] | None) -> None:
    """Restore EarlyStopping fields when resuming a run."""

    if early_stopper is None or not state:
        return
    for name in ("best", "best_epoch", "wait", "stopped"):
        if name in state:
            setattr(early_stopper, name, state[name])


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    early_stopper: Any,
    device: torch.device,
) -> tuple[int, float]:
    """Load LatentDiT training state and return the next epoch and best monitor value."""

    ckpt = torch.load(path, map_location=device)
    load_direct_model_state_dict(model, ckpt["model"])
    optimizer_loaded = True
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            if "parameter group" not in str(exc):
                raise
            optimizer_loaded = False
    if optimizer_loaded and scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    restore_early_stopper(early_stopper, ckpt.get("early_stopping"))
    next_epoch = int(ckpt.get("epoch", -1)) + 1
    if early_stopper is not None and early_stopper.best is not None:
        best = float(early_stopper.best)
    else:
        best = float(ckpt.get("best", ckpt.get("monitor_loss", float("inf"))))
    return next_epoch, best


def compute_condition_latent(
    jepa: Lvebm,
    batch: dict[str, torch.Tensor],
    source: str = "mixed",
) -> torch.Tensor:
    """Compute z from target, context, or a random mix of both."""

    batch = ensure_ratio_in_batch(batch)
    if source == "mixed":
        source = "target_sim" if random.random() < 0.5 else "context_aug"
    if source == "target_sim":
        return jepa.encode_tgt(batch["peak_d"], batch["peak_i"], batch["peak_mask"])
    if source == "context_aug":
        kwargs = {"atom_types": batch["atom_types"]}
        signature = inspect.signature(jepa.encode_ctx)
        if "ratio" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            kwargs["ratio"] = batch["ratio"]
        return jepa.encode_ctx(batch["aug_peak_d"], batch["aug_peak_i"], batch["aug_peak_mask"], **kwargs)
    raise ValueError(f"Unknown condition_source {source!r}")


def sample_flow_time(batch_size: int, t_min: float, device: torch.device) -> torch.Tensor:
    """Sample flow interpolation times."""

    return FlowMatchingInterpolant(t_min=t_min, device=device)._sample_t(batch_size)


def sample_structure_noise_like(y0: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """Sample Gaussian structure noise while keeping padded atom tokens zero."""

    token_mask = structure_token_mask(atom_mask)
    interpolant = FlowMatchingInterpolant(t_min=0.01, device=y0.device)
    noise = interpolant._centered_gaussian(*y0.shape)
    return noise * token_mask.to(device=y0.device, dtype=y0.dtype).unsqueeze(-1)


def structure_token_mask(atom_mask: torch.Tensor) -> torch.Tensor:
    """Return the valid-token mask for one lattice token plus atom tokens."""

    lattice_mask = torch.ones(atom_mask.shape[0], 1, dtype=torch.bool, device=atom_mask.device)
    return torch.cat([lattice_mask, atom_mask.bool()], dim=1)


def randomly_permute_valid_atoms(
    atom_types: torch.Tensor,
    frac_coords: torch.Tensor,
    atom_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly permute valid atom tokens while keeping all atom fields aligned."""

    if atom_types.ndim != 2:
        raise ValueError(f"atom_types must have shape [B, N], got {tuple(atom_types.shape)}")
    if frac_coords.ndim != 3 or frac_coords.shape[:2] != atom_types.shape:
        raise ValueError(
            f"frac_coords must have shape [B, N, 3] matching atom_types, got {tuple(frac_coords.shape)}"
        )
    if atom_mask.shape != atom_types.shape:
        raise ValueError(f"atom_mask must match atom_types shape, got {tuple(atom_mask.shape)}")

    permuted_types = atom_types.clone()
    permuted_coords = frac_coords.clone()
    permuted_mask = atom_mask.clone()
    for batch_idx in range(atom_types.shape[0]):
        valid_idx = torch.nonzero(atom_mask[batch_idx].bool(), as_tuple=False).flatten()
        if valid_idx.numel() <= 1:
            continue
        source_idx = valid_idx[torch.randperm(valid_idx.numel(), device=atom_types.device)]
        permuted_types[batch_idx, valid_idx] = atom_types[batch_idx, source_idx]
        permuted_coords[batch_idx, valid_idx] = frac_coords[batch_idx, source_idx]
        permuted_mask[batch_idx, valid_idx] = atom_mask[batch_idx, source_idx]
    return permuted_types, permuted_coords, permuted_mask


def direct_criterion_batch(
    y0: torch.Tensor,
    atom_mask: torch.Tensor,
    t: torch.Tensor,
    noisy: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build the dense metadata batch consumed by LatentDiT.criterion."""

    token_mask = structure_token_mask(atom_mask)
    batch = {
        "x_1": y0,
        "t": t,
        "token_mask": token_mask,
    }
    if noisy is not None:
        if "x_t" in noisy:
            batch["x_t"] = noisy["x_t"]
        if "coord_velocity" in noisy:
            batch["coord_velocity"] = noisy["coord_velocity"]
    return batch


def direct_criterion_metrics(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Extract scalar logging metrics from a LatentDiT criterion dictionary."""

    metrics: dict[str, float] = {}
    for name, value in loss_dict.items():
        if name in {"loss", "x_loss"}:
            continue
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            metrics[name] = float(value.detach().cpu())
        else:
            metrics[name] = float(value)
    return metrics


def flow_target_t_min(flow_cfg: dict[str, Any]) -> float:
    """Return the target Meta/AADT-style minimum flow time."""

    return float(flow_cfg.get("t_min", 0.01))


def flow_time_bounds_for_epoch(flow_cfg: dict[str, Any], epoch: int) -> float:
    """Return optional curriculum-adjusted ``t_min`` for one epoch."""

    t_min = flow_target_t_min(flow_cfg)
    curriculum_epochs = int(flow_cfg.get("curriculum_epochs", 0))
    if curriculum_epochs > 0:
        progress = min(max(float(epoch) / float(curriculum_epochs), 0.0), 1.0)
        start_min = float(flow_cfg.get("t_min_start", t_min))
        t_min = start_min + (t_min - start_min) * progress
    if t_min < 0 or t_min >= 0.5:
        raise ValueError(f"Invalid flow t_min={t_min}; expected 0 <= t_min < 0.5")
    return t_min


def build_flow_interpolant(flow_cfg: dict[str, Any], t_min: float, device: torch.device) -> FlowMatchingInterpolant:
    """Build the LatentDiT flow interpolant for a concrete epoch/eval interval."""

    return FlowMatchingInterpolant(
        t_min=t_min,
        corrupt=bool(flow_cfg.get("corrupt", True)),
        device=device,
    )


def corrupt_structure_tensor(
    y0: torch.Tensor,
    atom_mask: torch.Tensor,
    interpolant: FlowMatchingInterpolant,
) -> dict[str, torch.Tensor]:
    """Apply flow-matching corruption to a structure tensor batch."""

    token_mask = structure_token_mask(atom_mask)
    return interpolant.corrupt_batch(
        {
            "x_1": y0,
            "token_mask": token_mask,
            "diffuse_mask": token_mask,
        }
    )


def interpolate_fractional_coords_on_torus(
    coord_noise: torch.Tensor,
    frac_coords: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Interpolate fractional coordinates along the shortest periodic path."""

    t_view = t.reshape(-1, 1, 1)
    delta = torch.remainder(frac_coords - coord_noise + 0.5, 1.0) - 0.5
    return torch.remainder(coord_noise + t_view * delta, 1.0)


def corrupt_fractional_coords(
    frac_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    interpolant: FlowMatchingInterpolant,
) -> dict[str, torch.Tensor]:
    """Corrupt fractional coordinates by adding periodic noise to each target token."""

    interpolant.device = frac_coords.device
    t = interpolant._sample_t(frac_coords.shape[0])
    mask = atom_mask.to(device=frac_coords.device, dtype=torch.bool)
    mask_f = mask.to(dtype=frac_coords.dtype).unsqueeze(-1)
    if interpolant.corrupt:
        sigma = (1.0 - t).reshape(-1, 1, 1)
        coord_noise = torch.rand_like(frac_coords) - 0.5
        x_0 = torch.remainder(frac_coords + coord_noise, 1.0)
        frac_coords_t = torch.remainder(frac_coords + sigma * coord_noise, 1.0)
        coord_velocity = -coord_noise
    else:
        x_0 = torch.zeros_like(frac_coords)
        frac_coords_t = frac_coords
        coord_velocity = torch.zeros_like(frac_coords)
    return {
        "x_t": frac_coords_t * mask_f,
        "x_0": x_0 * mask_f,
        "coord_velocity": coord_velocity * mask_f,
        "t": t,
    }


def corrupt_lattice_and_fractional_coords(
    y0: torch.Tensor,
    frac_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    interpolant: FlowMatchingInterpolant,
) -> dict[str, torch.Tensor]:
    """Corrupt encoded lattice and fractional coordinates with one shared time."""

    interpolant.device = frac_coords.device
    t = interpolant._sample_t(frac_coords.shape[0])
    mask = atom_mask.to(device=frac_coords.device, dtype=torch.bool)
    mask_f = mask.to(dtype=frac_coords.dtype).unsqueeze(-1)
    if interpolant.corrupt:
        t_lattice = t.reshape(-1, 1)
        lattice_noise = torch.randn_like(y0[:, 0, :6])
        lattice_t = (1.0 - t_lattice) * lattice_noise + t_lattice * y0[:, 0, :6]

        sigma = (1.0 - t).reshape(-1, 1, 1)
        coord_noise = torch.rand_like(frac_coords) - 0.5
        coord_x_0 = torch.remainder(frac_coords + coord_noise, 1.0)
        frac_coords_t = torch.remainder(frac_coords + sigma * coord_noise, 1.0)
        coord_velocity = -coord_noise
    else:
        lattice_noise = torch.zeros_like(y0[:, 0, :6])
        coord_x_0 = torch.zeros_like(frac_coords)
        coord_velocity = torch.zeros_like(frac_coords)
        lattice_t = y0[:, 0, :6]
        frac_coords_t = frac_coords
    return {
        "lattice_t": lattice_t,
        "x_t": frac_coords_t * mask_f,
        "lattice_x_0": lattice_noise,
        "x_0": coord_x_0 * mask_f,
        "coord_velocity": coord_velocity * mask_f,
        "t": t,
    }


def train_direct_steps(
    model: LatentDiT,
    jepa: Lvebm,
    batches: list[dict[str, torch.Tensor]],
    steps: int = 10,
    lr: float = 1e-3,
    d_y: int = 8,
) -> list[float]:
    """Tiny-loop helper used by tests and smoke runs."""

    model.train()
    jepa.eval()
    opt = build_direct_optimizer(model, lr=lr, weight_decay=0.0)
    losses: list[float] = []
    for step in range(steps):
        batch = batches[step % len(batches)]
        atom_types, frac_coords, atom_mask = randomly_permute_valid_atoms(
            batch["atom_types"],
            batch["frac_coords"],
            batch["atom_mask"],
        )
        y0 = build_structure_tensor(atom_types, frac_coords, batch["lattice_params"], atom_mask, d_y=d_y)
        condition_batch = dict(batch)
        condition_batch["atom_types"] = atom_types
        with torch.no_grad():
            z = compute_condition_latent(jepa, condition_batch, source="target_sim")
        noisy = corrupt_lattice_and_fractional_coords(
            y0,
            frac_coords,
            atom_mask,
            FlowMatchingInterpolant(t_min=0.01, device=y0.device),
        )
        t = noisy["t"]
        lattice_t = noisy["lattice_t"]
        frac_coords_t = noisy["x_t"]
        opt.zero_grad(set_to_none=True)
        out = model(
            lattice_t,
            frac_coords_t,
            t,
            z,
            atom_types,
            atom_mask,
        )
        loss_dict = model.criterion(
            direct_criterion_batch(y0, atom_mask, t, noisy=noisy),
            out["Y0_pred"],
            pred_coord_velocity=out.get("coord_velocity"),
        )
        loss = loss_dict["loss"]
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def direct_batch_loss(
    model: torch.nn.Module,
    jepa: Lvebm,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    condition_source: str,
    t_min: float,
    loss_cfg: dict[str, Any],
    flow_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the LatentDiT loss for one batch."""

    atom_types, frac_coords, atom_mask = randomly_permute_valid_atoms(
        batch["atom_types"],
        batch["frac_coords"],
        batch["atom_mask"],
    )
    y0 = build_structure_tensor(
        atom_types,
        frac_coords,
        batch["lattice_params"],
        atom_mask,
        d_y=int(config.get("d_y", 8)),
    )
    condition_batch = dict(batch)
    condition_batch["atom_types"] = atom_types
    with torch.no_grad():
        z = compute_condition_latent(
            jepa,
            condition_batch,
            source=condition_source,
        )
    interpolant = build_flow_interpolant(flow_cfg or {}, t_min, y0.device)
    if bool((flow_cfg or {}).get("diffuse_lattice", False)):
        noisy = corrupt_lattice_and_fractional_coords(
            y0,
            frac_coords,
            atom_mask,
            interpolant,
        )
        lattice_t = noisy["lattice_t"]
    else:
        noisy = corrupt_fractional_coords(frac_coords, atom_mask, interpolant)
        lattice_t = torch.randn_like(y0[:, 0, :6])
    t = noisy["t"]
    frac_coords_t = noisy["x_t"]
    out = model(
        lattice_t,
        frac_coords_t,
        t,
        z,
        atom_types,
        atom_mask,
    )
    loss_dict = unwrap_model(model).criterion(
        direct_criterion_batch(y0, atom_mask, t, noisy=noisy),
        out["Y0_pred"],
        loss_cfg=loss_cfg,
        pred_coord_velocity=out.get("coord_velocity"),
    )
    return loss_dict["loss"], direct_criterion_metrics(loss_dict)


def train_one_epoch(
    model: torch.nn.Module,
    jepa: Lvebm,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    params: list[torch.nn.Parameter],
    config: dict[str, Any],
    device: torch.device,
    *,
    epoch: int,
    max_epochs: int,
    flow_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    grad_clip_norm: float = 0.0,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scheduler_interval: str = "epoch",
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float = 0.0,
) -> tuple[float, dict[str, float], float]:
    """Run one standard PyTorch training epoch."""

    model.train()
    jepa.eval()

    epoch_t_min = flow_time_bounds_for_epoch(flow_cfg, epoch)
    running = 0.0
    metric_sums: dict[str, float] = {}
    batches = 0
    progress = tqdm(
        loader,
        desc=f"LatentDiT epoch {epoch + 1}/{max_epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for batch in progress:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = direct_batch_loss(
            model,
            jepa,
            batch,
            config,
            condition_source=str(config.get("condition_source", "mixed")),
            t_min=epoch_t_min,
            loss_cfg=loss_cfg,
            flow_cfg=flow_cfg,
        )
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
        optimizer.step()
        if ema_state is not None:
            update_ema_state(ema_state, unwrap_model(model), ema_decay)
        if scheduler is not None and scheduler_interval == "step":
            scheduler.step()

        batch_loss = float(loss.detach().cpu())
        running += batch_loss
        for name, value in metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
        batches += 1
        running_metrics = {name: total / batches for name, total in metric_sums.items()}
        progress.set_postfix(
            train_loss=f"{running / batches:.4f}",
            lr=format_learning_rate(float(optimizer.param_groups[0]["lr"])),
            **{name: f"{value:.4f}" for name, value in selected_direct_metrics(running_metrics).items()},
        )

    if scheduler is not None and scheduler_interval == "epoch":
        scheduler.step()

    epoch_loss = running / max(batches, 1)
    epoch_metrics = {name: total / max(batches, 1) for name, total in metric_sums.items()}
    return epoch_loss, epoch_metrics, epoch_t_min


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    jepa: Lvebm,
    loader: DataLoader | None,
    config: dict[str, Any],
    device: torch.device,
    *,
    t_min: float,
    loss_cfg: dict[str, Any],
) -> tuple[float | None, dict[str, float]]:
    """Evaluate LatentDiT on a validation loader."""

    if loader is None:
        return None, {}
    model.eval()
    jepa.eval()
    running = 0.0
    metric_sums: dict[str, float] = {}
    batches = 0
    condition_source = str(config.get("validation_condition_source", "context_aug"))
    for batch in loader:
        batch = batch_to_device(batch, device)
        loss, metrics = direct_batch_loss(
            model,
            jepa,
            batch,
            config,
            condition_source=condition_source,
            t_min=t_min,
            loss_cfg=loss_cfg,
            flow_cfg=config.get("flow", {}),
        )
        running += float(loss.detach().cpu())
        for name, value in metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
        batches += 1
    loss = running / max(batches, 1)
    metrics = {name: total / max(batches, 1) for name, total in metric_sums.items()}
    return loss, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train LatentDiT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from, usually outputs/.../last.ckpt")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    disable_native_mha_fastpath()
    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))
    gpu_ids = resolve_gpu_ids(config, device)
    device = primary_device(device, gpu_ids)

    dataset = build_dataset_from_config(config, split="train")
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 4)),
        collate_fn=collate_crystal_batch,
    )
    val_loader = None
    if has_dataset_split(config, "valid"):
        val_dataset = build_dataset_from_config(config, split="valid")
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config.get("eval_batch_size", config.get("batch_size", 64))),
            shuffle=False,
            num_workers=int(config.get("num_workers", 4)),
            collate_fn=collate_crystal_batch,
        )
    jepa = load_jepa_checkpoint(config["jepa_ckpt"], device)
    validate_jepa_latent_dim(config, jepa, config["jepa_ckpt"])
    jepa.eval()
    for param in jepa.parameters():
        param.requires_grad_(False)
    model = build_direct_from_config(config).to(device)
    model = maybe_data_parallel(model, gpu_ids)
    if len(gpu_ids) > 1:
        print(f"using DataParallel on CUDA devices {gpu_ids}")
    params = list(model.parameters())
    opt = build_direct_optimizer(
        model,
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-2)),
    )
    max_epochs = int(config.get("max_epochs", 300))
    optimizer_steps_per_epoch = max(len(loader), 1)
    scheduler, scheduler_interval = build_lr_scheduler(
        opt,
        config,
        max_epochs=max_epochs,
        steps_per_epoch=optimizer_steps_per_epoch,
    )
    ema_decay = float(config.get("ema_decay", 0.0))
    ema_state = init_ema_state(unwrap_model(model)) if ema_decay > 0 else None
    output_dir = Path(config.get("output_dir", "outputs/latent_dit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "training_config.yaml")
    early_stopper = build_early_stopper(config)

    best = float("inf")
    start_epoch = 0
    resume_path = args.resume or config.get("resume_ckpt")
    if resume_path:
        start_epoch, best = load_training_checkpoint(resume_path, model, opt, scheduler, early_stopper, device)
        print(f"resumed LatentDiT from {resume_path} at epoch={start_epoch} best={best:.6f}")
    wandb_run = init_wandb(config, job_type="latent_dit", output_dir=output_dir, default_name=output_dir.name)
    maybe_watch_model(wandb_run, unwrap_model(model), config)
    flow_cfg = config.get("flow", {})
    loss_cfg = config.get("loss", {})
    grad_clip_norm = float(config.get("gradient_clip_norm", 0.0))
    val_t_min = flow_time_bounds_for_epoch(flow_cfg, int(flow_cfg.get("curriculum_epochs", 0)))
    try:
        for epoch in range(start_epoch, max_epochs):
            epoch_loss, epoch_metrics, epoch_t_min = train_one_epoch(
                model,
                jepa,
                loader,
                opt,
                params,
                config,
                device,
                epoch=epoch,
                max_epochs=max_epochs,
                flow_cfg=flow_cfg,
                loss_cfg=loss_cfg,
                grad_clip_norm=grad_clip_norm,
                scheduler=scheduler,
                scheduler_interval=scheduler_interval,
                ema_state=ema_state,
                ema_decay=ema_decay,
            )
            val_loss, val_metrics = evaluate(
                model,
                jepa,
                val_loader,
                config,
                device,
                t_min=val_t_min,
                loss_cfg=loss_cfg,
            )
            monitor_loss = val_loss if val_loss is not None else epoch_loss
            is_best = monitor_loss < best
            if is_best:
                best = monitor_loss
            stop_now = early_stopper.step(monitor_loss, epoch) if early_stopper is not None else False
            ckpt = {
                "model": unwrap_model(model).state_dict(),
                "config": config,
                "jepa_ckpt": config["jepa_ckpt"],
                "epoch": epoch,
                "loss": epoch_loss,
                "metrics": epoch_metrics,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "monitor_loss": monitor_loss,
                "best": best,
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }
            if ema_state is not None:
                ckpt["ema_model"] = ema_state
            if early_stopper is not None:
                ckpt["early_stopping"] = early_stopper.state_dict()
            torch.save(ckpt, output_dir / "last.ckpt")
            if is_best:
                torch.save(ckpt, output_dir / "best.ckpt")
            learning_rate = float(opt.param_groups[0]["lr"])
            print(format_direct_epoch_log(epoch, epoch_loss, val_loss, epoch_metrics, learning_rate, val_metrics))
            epoch_step = (epoch + 1) * optimizer_steps_per_epoch
            log_wandb(
                wandb_run,
                direct_wandb_metrics(epoch_loss, val_loss, epoch_metrics, learning_rate, val_metrics),
                step=epoch_step,
            )
            if stop_now:
                print(
                    "early_stopping "
                    f"epoch={epoch} monitor_loss={monitor_loss:.6f} "
                    f"best={early_stopper.best:.6f} best_epoch={early_stopper.best_epoch} "
                    f"wait={early_stopper.wait}/{early_stopper.patience}"
                )
                break
    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
