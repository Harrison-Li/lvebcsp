"""JEPA training entry point."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from tqdm.auto import tqdm

from lvebcsp.data.cif_dataset import CIFPXRDDataset
from lvebcsp.data.collate import collate_crystal_batch
from lvebcsp.data.mp20_lmdb import MP20LMDBDataset
from lvebcsp.data.organic_lmdb import OrganicLMDBDataset
from lvebcsp.data.crystal_dataset import CrystalDataset, collate_crystal_graphs
from lvebcsp.models.jepa import Lvebm
from lvebcsp.models.layers import AdaLNMLP, GatedMLP, MLP
from lvebcsp.models.module import Predictor
from lvebcsp.common.chemistry import atom_types_to_element_ratios, element_counts_to_ratios
from lvebcsp.common.config import load_config, save_config, select_device
from lvebcsp.common.gpu import primary_device, resolve_gpu_ids, unwrap_model
from lvebcsp.common.distributed import cleanup_distributed, is_distributed, is_main_process, setup_distributed
from lvebcsp.common.seed import seed_everything
from lvebcsp.common.wandb_logging import (
    finish_wandb,
    init_wandb,
    log_wandb,
    maybe_watch_model,
)


JEPA_LOG_METRICS = (
    ("pred", "loss_pred"),
    ("sigreg", "loss_sigreg"),
    ("cos", "sim_diag"),
    ("ctx_std", "context_std"),
    ("tgt_std", "target_std"),
)
JEPA_WANDB_SPLIT_METRICS = (
    ("pred", "loss_pred"),
    ("sigreg", "loss_sigreg"),
)
JEPA_METRIC_NAMES = (
    "loss_pred", "loss_sigreg", "sim_diag", "sim_offdiag", "context_std", "target_std",
)


@dataclass
class EarlyStopping:
    """Track loss plateaus and decide when to stop training."""

    patience: int
    min_delta: float = 0.0
    warmup_epochs: int = 0
    ignore_warmup_metrics: bool = False
    best: float | None = None
    best_epoch: int = -1
    wait: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        self.patience = int(self.patience)
        self.min_delta = float(self.min_delta)
        self.warmup_epochs = int(self.warmup_epochs)
        self.ignore_warmup_metrics = bool(self.ignore_warmup_metrics)
        if self.patience < 1:
            raise ValueError("early_stopping.patience must be >= 1")
        if self.min_delta < 0:
            raise ValueError("early_stopping.min_delta must be >= 0")
        if self.warmup_epochs < 0:
            raise ValueError("early_stopping.warmup_epochs must be >= 0")

    def step(self, value: float, epoch: int) -> bool:
        value = float(value)
        epoch = int(epoch)
        if self.ignore_warmup_metrics and epoch < self.warmup_epochs:
            self.wait = 0
            self.stopped = False
            return False
        # Discard a pre-warmup best restored from a checkpoint written before
        # ``ignore_warmup_metrics`` was introduced.
        if (
            self.ignore_warmup_metrics
            and self.best is not None
            and self.best_epoch < self.warmup_epochs
        ):
            self.best = None
            self.best_epoch = -1
            self.wait = 0
            self.stopped = False
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.wait = 0
            self.stopped = False
            return False
        if epoch < self.warmup_epochs:
            return False
        self.wait += 1
        self.stopped = self.wait >= self.patience
        return self.stopped

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "warmup_epochs": self.warmup_epochs,
            "ignore_warmup_metrics": self.ignore_warmup_metrics,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "wait": self.wait,
            "stopped": self.stopped,
        }


def build_early_stopper(config: dict[str, Any]) -> EarlyStopping | None:
    """Build optional early stopping state from a training config."""

    stopping_cfg = config.get("early_stopping")
    if stopping_cfg in (None, False):
        return None
    if stopping_cfg is True:
        cfg: dict[str, Any] = {}
    elif isinstance(stopping_cfg, dict):
        cfg = dict(stopping_cfg)
    else:
        raise TypeError(f"early_stopping must be a bool, dict, or omitted; got {type(stopping_cfg).__name__}")
    if not bool(cfg.get("enabled", True)):
        return None
    return EarlyStopping(
        patience=int(cfg.get("patience", 20)),
        min_delta=float(cfg.get("min_delta", 0.0)),
        warmup_epochs=int(cfg.get("warmup_epochs", 0)),
        ignore_warmup_metrics=bool(cfg.get("ignore_warmup_metrics", False)),
    )


def selected_jepa_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Return the small JEPA metric set exposed in logs."""

    return selected_jepa_metrics_by_spec(metrics, JEPA_LOG_METRICS)


def selected_jepa_metrics_by_spec(
    metrics: dict[str, float],
    metric_specs: tuple[tuple[str, str], ...],
) -> dict[str, float]:
    """Return compact JEPA metric names from a raw metric mapping."""

    return {
        log_name: float(metrics[raw_name])
        for log_name, raw_name in metric_specs
        if raw_name in metrics
    }


def selected_jepa_metrics_for_split(metrics: dict[str, float], split: str) -> dict[str, float]:
    """Return compact JEPA metric names with a train/valid suffix."""

    return {f"{name}_{split}": value for name, value in selected_jepa_metrics(metrics).items()}


def jepa_log_metric_fields(
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
) -> list[tuple[str, float]]:
    """Return train/valid JEPA metric fields in console-log order."""

    fields: list[tuple[str, float]] = []
    for log_name, raw_name in JEPA_LOG_METRICS:
        if raw_name in train_metrics:
            fields.append((f"{log_name}_train", float(train_metrics[raw_name])))
        if raw_name in valid_metrics:
            fields.append((f"{log_name}_valid", float(valid_metrics[raw_name])))
    return fields


def format_learning_rate(learning_rate: float) -> str:
    """Format a learning rate compactly for console logs."""

    mantissa, exponent = f"{learning_rate:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_jepa_epoch_log(
    epoch: int,
    train_loss: float,
    train_metrics: dict[str, float],
    valid_loss: float | None,
    valid_metrics: dict[str, float],
    learning_rate: float | None = None,
) -> str:
    """Format the compact JEPA epoch log line."""

    fields: list[tuple[str, float]] = [("train_loss", train_loss)]
    if valid_loss is not None:
        fields.append(("valid_loss", valid_loss))
    fields.extend(jepa_log_metric_fields(train_metrics, valid_metrics))
    metric_text = " ".join(f"{name}={value:.6f}" for name, value in fields)
    if learning_rate is not None:
        metric_text = f"{metric_text} lr={format_learning_rate(learning_rate)}"
    return f"epoch={epoch} {metric_text}"


def jepa_wandb_metrics(
    train_loss: float,
    train_metrics: dict[str, float],
    valid_loss: float | None,
    valid_metrics: dict[str, float],
    learning_rate: float | None = None,
) -> dict[str, float | None]:
    """Build the compact JEPA W&B payload."""

    payload: dict[str, float | None] = {
        "train/loss": train_loss,
        "valid/loss": valid_loss,
        "trainer/lr": learning_rate,
    }
    for log_name, raw_name in JEPA_WANDB_SPLIT_METRICS:
        if raw_name in train_metrics:
            payload[f"{log_name}_train"] = float(train_metrics[raw_name])
        if raw_name in valid_metrics:
            payload[f"{log_name}_valid"] = float(valid_metrics[raw_name])
    similarity_metrics = valid_metrics if valid_metrics else train_metrics
    for raw_name in ("sim_diag", "sim_offdiag"):
        if raw_name in similarity_metrics:
            payload[raw_name] = float(similarity_metrics[raw_name])
    return payload


def build_jepa_head(
    config: Any,
    input_dim: int,
    output_dim: int,
    *,
    condition_dim: int | None = None,
) -> nn.Module | None:
    """Build an optional JEPA projection/prediction head from a small local config."""

    if config is None:
        return None
    if isinstance(config, nn.Module):
        return config
    if isinstance(config, str):
        head_type = config
        cfg: dict[str, Any] = {}
    elif isinstance(config, dict):
        cfg = dict(config)
        head_type = str(cfg.pop("type", "mlp"))
    else:
        raise TypeError(f"JEPA head config must be a string, dict, nn.Module, or None; got {type(config).__name__}")

    head_type = head_type.lower().replace("-", "_")
    if head_type in {"identity", "none"}:
        return nn.Identity()

    if head_type == "predictor":
        if condition_dim is None:
            raise ValueError("JEPA head type 'predictor' is only valid for the conditional predictor head")
        if "num_layers" in cfg and "depth" not in cfg:
            cfg["depth"] = cfg.pop("num_layers")
        if "num_heads" in cfg and "heads" not in cfg:
            cfg["heads"] = cfg.pop("num_heads")
        context_dim = int(cfg.pop("context_dim", cfg.get("input_dim", output_dim)))
        cfg.setdefault("input_dim", context_dim)
        cfg.setdefault("condition_dim", condition_dim)
        cfg.setdefault("output_dim", output_dim)
        cfg.setdefault("hidden_dim", output_dim)
        cfg.setdefault("num_frames", 1)
        cfg.setdefault("depth", 2)
        cfg.setdefault("heads", 4)
        cfg["input_dim"] = int(cfg["input_dim"])
        cfg["condition_dim"] = int(cfg["condition_dim"])
        cfg["output_dim"] = int(cfg["output_dim"])
        cfg["hidden_dim"] = int(cfg["hidden_dim"])
        cfg["num_frames"] = int(cfg["num_frames"])
        cfg["depth"] = int(cfg["depth"])
        cfg["heads"] = int(cfg["heads"])
        heads = int(cfg["heads"])
        if heads < 1:
            raise ValueError("predictor.heads must be >= 1")
        cfg.setdefault("dim_head", max(int(cfg["hidden_dim"]) // heads, 1))
        cfg.setdefault("mlp_dim", int(cfg["hidden_dim"]) * 4)
        cfg["dim_head"] = int(cfg["dim_head"])
        cfg["mlp_dim"] = int(cfg["mlp_dim"])
        return Predictor(**cfg)

    if head_type == "mlp":
        cfg.setdefault("input_dim", input_dim)
        cfg.setdefault("output_dim", output_dim)
        return MLP(**cfg)
    if head_type in {"adaln_mlp", "adalnmlp", "ada_ln_mlp"}:
        if condition_dim is None:
            raise ValueError("JEPA head type 'adaln_mlp' is only valid for the conditional predictor head")
        context_dim = int(cfg.pop("context_dim", cfg.pop("input_dim", output_dim)))
        cfg["input_dim"] = context_dim
        cfg.setdefault("cond_dim", condition_dim)
        cfg.setdefault("output_dim", output_dim)
        cfg.setdefault("hidden_dim", output_dim)
        return AdaLNMLP(**cfg)
    if head_type == "gated_mlp":
        cfg.setdefault("input_dim", input_dim)
        cfg.setdefault("output_dim", output_dim)
        cfg.setdefault("hidden_dim", output_dim)
        return GatedMLP(**cfg)
    raise ValueError(f"Unknown JEPA head type {head_type!r}")


def ensure_ratio_in_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Ensure JEPA batches include normalized elemental ratios."""

    if "ratio" in batch:
        return batch
    if "element_counts" not in batch and "atom_types" not in batch:
        return batch
    out = dict(batch)
    if "element_counts" in out:
        out["ratio"] = element_counts_to_ratios(out["element_counts"])
    else:
        out["ratio"] = atom_types_to_element_ratios(out["atom_types"])
    return out


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors and PyG crystal graphs to device."""

    return {key: value.to(device, non_blocking=True) if isinstance(value, (torch.Tensor, Data)) else value for key, value in batch.items()}


def build_jepa_from_config(config: dict[str, Any]) -> Lvebm:
    """Instantiate Lvebm from a config dictionary."""

    loss_cfg = config.get("loss", {})
    d_jepa = int(config.get("d_jepa", 512))
    condition_cfg = config.get("condition_encoder", {})
    condition_hidden_dim = int(condition_cfg.get("hidden_dim", 256))
    predictor_input_dim = d_jepa + condition_hidden_dim
    crystal_cfg = config.get("crystal_encoder") or {}
    encoder_output_dim = crystal_cfg.get("output_dim", d_jepa)
    predictor_cfg = config.get("predictor") or {"type": "mlp"}
    return Lvebm(
        d_jepa=d_jepa,
        crystal_encoder=crystal_cfg,
        condition_encoder=condition_cfg,
        projector=build_jepa_head(config.get("projector"), input_dim=encoder_output_dim, output_dim=d_jepa),
        predictor=build_jepa_head(
            predictor_cfg,
            input_dim=predictor_input_dim,
            output_dim=d_jepa,
            condition_dim=condition_hidden_dim,
        ),
        pred_proj=build_jepa_head(config.get("pred_proj"), input_dim=d_jepa, output_dim=d_jepa),
        stop_gradient=bool(config.get("stop_gradient", config.get("stop_gradient_target", False))),
        sigreg_num_projections=int(loss_cfg.get("sigreg_num_projections", 1024)),
        lambda_sig=float(loss_cfg.get("lambda_sig", 0.1)),
    )


def _peak_encoder_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("peak_encoder", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"peak_encoder must be a mapping, got {type(raw).__name__}")
    return dict(raw)


def resolve_p_max(config: dict[str, Any]) -> int:
    """Return the dataset peak count and validate the encoder can handle it."""

    peak_cfg = _peak_encoder_config(config)
    data_p_max = (
        int(config["p_max"])
        if config.get("p_max") is not None
        else int(peak_cfg.get("p_max", 128))
    )
    encoder_p_max = int(peak_cfg.get("p_max", data_p_max))
    if data_p_max < 1:
        raise ValueError(f"p_max must be >= 1, got {data_p_max}")
    if encoder_p_max < data_p_max:
        raise ValueError(
            f"peak_encoder.p_max={encoder_p_max} is smaller than p_max={data_p_max}; "
            "increase peak_encoder.p_max or lower p_max so JEPA batches fit the encoder"
        )
    return data_p_max


def build_dataset_from_config(config: dict[str, Any], split: str = "train"):
    """Build a manifest-backed CIF dataset or a supported LMDB dataset."""

    split = "valid" if split == "val" else split
    if "graph_manifest_dir" in config:
        return CrystalDataset(
            Path(config["graph_manifest_dir"]) / f"{split}.npz",
            cutoff=config.get("crystal_encoder", {}).get("cutoff", 6.0),
            random_block_geometry=split == "train" and config.get("random_block_geometry", True),
            max_samples=config.get(f"max_{split}_samples"),
        )
    mp20_path = config.get("mp20_lmdb_path") if split == "train" else config.get(f"mp20_{split}_lmdb_path")
    organic_path = config.get("organic_lmdb_path") if split == "train" else config.get(f"organic_{split}_lmdb_path")
    manifest_path = config.get("manifest_path") if split == "train" else config.get(f"{split}_manifest_path")
    p_max = resolve_p_max(config)

    if mp20_path and organic_path:
        raise ValueError(
            f"Configure only one LMDB backend for split={split!r}: "
            "mp20 or organic"
        )
    lmdb_path = organic_path or mp20_path
    if lmdb_path:
        two_theta_range = config.get("two_theta_range", [5.0, 80.0])
        dataset_cls = OrganicLMDBDataset if organic_path else MP20LMDBDataset
        return dataset_cls(
            lmdb_path,
            n_max=int(config.get("n_max", 64)),
            p_max=p_max,
            wavelength=config.get("wavelength", "CuKa"),
            two_theta_range=(float(two_theta_range[0]), float(two_theta_range[1])),
            augmentation=config.get("augmentation", {}) if split == "train" else config.get("validation_augmentation", {"enabled": False}),
            use_conventional=bool(config.get("use_conventional", False)),
            include_pxrd=bool(config.get("include_pxrd", True)),
        )
    if not manifest_path:
        raise KeyError(f"No dataset path configured for split={split!r}")
    return CIFPXRDDataset(
        manifest_path,
        split=split,
        n_max=int(config.get("n_max", 64)),
        p_max=p_max,
        augmentation=config.get("augmentation", {}) if split == "train" else config.get("validation_augmentation", {"enabled": False}),
        include_pxrd=bool(config.get("include_pxrd", True)),
    )


def has_dataset_split(config: dict[str, Any], split: str) -> bool:
    """Return whether the config names a dataset path for a split."""

    split = "valid" if split == "val" else split
    if "graph_manifest_dir" in config:
        return (Path(config["graph_manifest_dir"]) / f"{split}.npz").exists()
    if split == "train":
        return any(
            key in config
            for key in ("mp20_lmdb_path", "organic_lmdb_path", "manifest_path")
        )
    return (
        f"mp20_{split}_lmdb_path" in config
        or f"organic_{split}_lmdb_path" in config
        or f"{split}_manifest_path" in config
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    max_epochs: int,
    steps_per_epoch: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
    """Build a scheduler; the metric interval steps after validation."""

    scheduler_cfg = config.get("scheduler")
    if scheduler_cfg in (None, False):
        return None, "epoch"
    if isinstance(scheduler_cfg, str):
        cfg: dict[str, Any] = {"type": scheduler_cfg}
    elif isinstance(scheduler_cfg, dict):
        cfg = dict(scheduler_cfg)
    else:
        raise TypeError(f"scheduler must be a string, dict, or falsey value; got {type(scheduler_cfg).__name__}")

    scheduler_type = str(cfg.get("type", "")).lower().replace("-", "_")
    if scheduler_type in {"none", "off", "false"}:
        return None, "epoch"
    if scheduler_type in {"reducelronplateau", "reduce_lr_on_plateau", "reduce_on_plateau"}:
        cfg.pop("type", None)
        cfg.pop("interval", None)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg), "metric"
    if scheduler_type not in {"linearwarmupcosineannealinglr", "linear_warmup_cosine_annealing_lr"}:
        raise ValueError(f"Unknown scheduler type {cfg.get('type')!r}")

    interval = str(cfg.get("interval", "epoch")).lower()
    if interval not in {"epoch", "step"}:
        raise ValueError(f"scheduler.interval must be 'epoch' or 'step', got {interval!r}")

    steps_per_epoch = max(int(steps_per_epoch), 1)
    total_steps = max(int(max_epochs), 1)
    if interval == "step":
        total_steps *= steps_per_epoch

    if "warmup_steps" in cfg:
        warmup_steps = int(cfg["warmup_steps"])
    elif "warmup_epochs" in cfg:
        warmup_steps = int(cfg["warmup_epochs"])
        if interval == "step":
            warmup_steps *= steps_per_epoch
    else:
        warmup_steps = int(round(total_steps * float(cfg.get("warmup_fraction", 0.1))))
    warmup_steps = min(max(warmup_steps, 0), total_steps)

    eta_min = float(cfg.get("eta_min", cfg.get("min_lr", 0.0)))
    warmup_start_lr = float(cfg.get("warmup_start_lr", 0.0))

    lr_lambdas = []
    for group in optimizer.param_groups:
        base_lr = float(group["lr"])
        min_factor = eta_min / base_lr if base_lr > 0 else 0.0
        start_factor = warmup_start_lr / base_lr if base_lr > 0 else 0.0
        min_factor = min(max(min_factor, 0.0), 1.0)
        start_factor = min(max(start_factor, 0.0), 1.0)

        def lr_lambda(
            step: int,
            *,
            min_factor: float = min_factor,
            start_factor: float = start_factor,
            warmup_steps: int = warmup_steps,
            total_steps: int = total_steps,
        ) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return start_factor + (1.0 - start_factor) * (step / warmup_steps)
            cosine_steps = max(total_steps - warmup_steps, 1)
            cosine_step = min(max(step - warmup_steps, 0), cosine_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_steps))
            return min_factor + (1.0 - min_factor) * cosine

        lr_lambdas.append(lr_lambda)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas), interval


def normalize_jepa_output(out: dict[str, Any]) -> dict[str, Any]:
    """Reduce DataParallel scalar outputs and convert metrics to plain floats."""

    normalized = dict(out)
    loss = normalized["loss"]
    if isinstance(loss, torch.Tensor) and loss.ndim > 0:
        loss = loss.mean()
    normalized["loss"] = loss

    metrics: dict[str, float] = {}
    for name, value in normalized.get("metrics", {}).items():
        if isinstance(value, torch.Tensor):
            metrics[name] = float(value.detach().mean().cpu())
        else:
            metrics[name] = float(value)
    normalized["metrics"] = metrics
    return normalized


def run_jepa_batch(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Use forward so DDP can synchronize the model's gradients."""
    return normalize_jepa_output(model(batch))


def train_jepa_steps(
    model: Lvebm,
    batches: list[dict[str, torch.Tensor]],
    steps: int = 10,
    lr: float = 1e-4,
) -> list[float]:
    """Tiny-loop helper used by tests and smoke runs."""

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps):
        batch = batches[step % len(batches)]
        opt.zero_grad(set_to_none=True)
        out = run_jepa_batch(model, batch)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].detach().cpu()))
    return losses


def reduce_epoch_statistics(
    total_loss: float,
    metric_totals: dict[str, float],
    samples: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Reuse the DLM trainer's reduction, weighted by crystal count."""
    names = sorted(metric_totals)
    values = torch.tensor(
        [total_loss, float(samples), *(metric_totals[name] for name in names)],
        device=device,
        dtype=torch.float64,
    )
    if is_distributed():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    denominator = values[1].clamp_min(1.0)
    loss = float((values[0] / denominator).cpu())
    metrics = {
        name: float((values[index + 2] / denominator).cpu())
        for index, name in enumerate(names)
    }
    return loss, metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    max_epochs: int,
    grad_clip_norm: float = 0.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_interval: str = "epoch",
    max_batches: int | None = None,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one standard PyTorch training epoch."""

    model.train()
    running = 0.0
    metric_sums = dict.fromkeys(JEPA_METRIC_NAMES, 0.0)
    samples = 0
    progress = tqdm(
        loader,
        desc=f"JEPA epoch {epoch + 1}/{max_epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not is_main_process(),
    )
    for step, batch in enumerate(progress):
        if max_batches is not None and step >= max_batches:
            break
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = run_jepa_batch(model, batch)
        loss = out["loss"]
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler is not None and scheduler_interval == "step":
            scheduler.step()

        batch_loss = float(loss.detach().cpu())
        batch_size = out["pred_emb"].size(0)
        running += batch_loss * batch_size
        for name, value in out["metrics"].items():
            metric_sums[name] += float(value) * batch_size
        samples += batch_size
        running_metrics = {name: total / samples for name, total in metric_sums.items()}
        progress.set_postfix(
            train_loss=f"{running / samples:.4f}",
            lr=format_learning_rate(float(optimizer.param_groups[0]["lr"])),
            **{
                name: f"{value:.4f}"
                for name, value in selected_jepa_metrics_for_split(running_metrics, "train").items()
            },
        )

    if scheduler is not None and scheduler_interval == "epoch":
        scheduler.step()

    return reduce_epoch_statistics(running, metric_sums, samples, device)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader | None, device: torch.device,
    max_batches: int | None = None, amp_dtype: torch.dtype | None = None,
) -> tuple[float | None, dict[str, float]]:
    """Evaluate JEPA loss on a validation loader."""

    if loader is None:
        return None, {}
    model.eval()
    # Validation shards can have different lengths. Avoid DDP forward collectives.
    model = unwrap_model(model)
    running = 0.0
    metric_sums = dict.fromkeys(JEPA_METRIC_NAMES, 0.0)
    samples = 0
    # Use repeatable SIGReg projections for comparable validation losses.
    sigreg = model.sigreg
    previous_seed = sigreg.seed
    sigreg.seed = 0
    try:
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            batch = batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = run_jepa_batch(model, batch)
            batch_size = out["pred_emb"].size(0)
            running += float(out["loss"].detach().cpu()) * batch_size
            for name, value in out["metrics"].items():
                metric_sums[name] += float(value) * batch_size
            samples += batch_size
    finally:
        sigreg.seed = previous_seed
    return reduce_epoch_statistics(running, metric_sums, samples, device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train crystal JEPA from graph manifests")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-dir", dest="graph_manifest_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Resume model, optimizer, scheduler, and epoch from a checkpoint")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"))
    for option in (
        "batch-size", "eval-batch-size", "num-workers", "max-epochs",
        "max-train-batches", "max-eval-batches", "max-train-samples", "max-valid-samples",
    ):
        parser.add_argument(f"--{option}", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.update({key: value for key, value in vars(args).items() if key != "config" and value is not None})
    if "graph_manifest_dir" in config:
        config["graph_manifest_dir"] = str(Path(config["graph_manifest_dir"]).resolve())
    output_dir = Path(config.get("output_dir", "outputs/jepa")).resolve()
    config["output_dir"] = str(output_dir)
    device = select_device(str(config.get("device", "auto")))
    rank, local_rank, world_size = setup_distributed("nccl" if device.type == "cuda" else "gloo")
    wandb_run = None
    try:
        if dist.is_initialized() and device.type == "cuda":
            device = torch.device("cuda", local_rank)
        elif not dist.is_initialized():
            gpu_ids = resolve_gpu_ids(config, device)
            if len(gpu_ids) > 1:
                raise ValueError("Launch one process per GPU with scripts/train_crystal_jepa.sh.")
            device = primary_device(device, gpu_ids)
            if device.type == "cuda":
                torch.cuda.set_device(device)

        precision = config.get("precision", "fp32")
        if precision == "auto":
            precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        amp_dtype = {"fp32": None, "bf16": torch.bfloat16}[precision]
        config["precision"] = precision
        seed = int(config.get("seed", 42))
        seed_everything(seed)
        dataset = build_dataset_from_config(config, split="train")
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True,
        ) if is_distributed() else None
        workers = int(config.get("num_workers", 4))
        loader_options = dict(
            num_workers=workers,
            collate_fn=collate_crystal_graphs,
            pin_memory=device.type == "cuda",
        )
        if workers > 0:
            # NCCL and forked workers are incompatible; retain spawned workers across epochs.
            loader_options.update(multiprocessing_context="spawn", persistent_workers=True)
        batch_size = int(config.get("batch_size", 16))
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=sampler is None,
            sampler=sampler, drop_last=True, **loader_options,
        )
        if not len(loader):
            raise ValueError("Training needs at least batch_size × world_size crystals for a full distributed batch.")
        val_loader = None
        if has_dataset_split(config, "valid"):
            val_dataset = build_dataset_from_config(config, split="valid")
            # Strided validation shards cover every crystal exactly once, without padding.
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(config.get("eval_batch_size", batch_size)),
                sampler=range(rank, len(val_dataset), world_size),
                **loader_options,
            )

        model = build_jepa_from_config(config).to(device)
        if dist.is_initialized():
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
        max_epochs = int(config.get("max_epochs", 100))
        optimizer_steps_per_epoch = min(len(loader), config.get("max_train_batches") or len(loader))
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("learning_rate", 1e-4)),
            weight_decay=float(config.get("weight_decay", 1e-2)),
        )
        scheduler, scheduler_interval = build_lr_scheduler(
            opt, config, max_epochs=max_epochs, steps_per_epoch=optimizer_steps_per_epoch,
        )
        grad_clip_norm = float(config.get("gradient_clip_norm", 0.0))
        early_stopper = build_early_stopper(config)
        best, start_epoch, global_step = float("inf"), 0, 0
        if config.get("resume"):
            checkpoint = torch.load(config["resume"], map_location="cpu", weights_only=False)
            unwrap_model(model).load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint.get("global_step", start_epoch * optimizer_steps_per_epoch)
            best = checkpoint.get("best", checkpoint["monitor_loss"])
            if early_stopper is not None and checkpoint.get("early_stopping"):
                for key in ("best", "best_epoch", "wait", "stopped"):
                    setattr(early_stopper, key, checkpoint["early_stopping"][key])
            del checkpoint

        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, output_dir / "training_config.yaml")
            wandb_run = init_wandb(config, job_type="jepa", output_dir=output_dir, default_name=output_dir.name)
            maybe_watch_model(wandb_run, unwrap_model(model), config)
            print(
                f"device={device} world_size={world_size} precision={precision} "
                f"batch_per_rank={batch_size} global_batch={batch_size * world_size} "
                f"train_samples={len(dataset)} valid_samples={len(val_loader.dataset) if val_loader is not None else 0} "
                f"start_epoch={start_epoch} global_step={global_step}", flush=True,
            )
        for epoch in range(start_epoch, max_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            seed_everything(seed + epoch * world_size + rank)
            epoch_loss, epoch_metrics = train_one_epoch(
                model,
                loader,
                opt,
                device,
                epoch=epoch,
                max_epochs=max_epochs,
                grad_clip_norm=grad_clip_norm,
                scheduler=scheduler,
                scheduler_interval=scheduler_interval,
                max_batches=config.get("max_train_batches"),
                amp_dtype=amp_dtype,
            )
            global_step += optimizer_steps_per_epoch
            val_loss, val_metrics = evaluate(
                model, val_loader, device, config.get("max_eval_batches"), amp_dtype=amp_dtype,
            )
            monitor_loss = val_loss if val_loss is not None else epoch_loss
            if scheduler is not None and scheduler_interval == "metric":
                # All ranks receive the same sample-weighted validation loss.
                scheduler.step(monitor_loss)
            is_best = monitor_loss < best
            if is_best:
                best = monitor_loss
            stop_now = early_stopper.step(monitor_loss, epoch) if early_stopper is not None else False
            if is_main_process():
                ckpt = {
                    "model": unwrap_model(model).state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                    "world_size": world_size,
                    "best": best,
                    "loss": epoch_loss,
                    "train_loss": epoch_loss,
                    "metrics": epoch_metrics,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "monitor_loss": monitor_loss,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "optimizer": opt.state_dict(),
                    "objective": "mse_sigreg_context",
                }
                if early_stopper is not None:
                    ckpt["early_stopping"] = early_stopper.state_dict()
                for name in (("last", "best") if is_best else ("last",)):
                    temporary = output_dir / f"{name}.ckpt.tmp"
                    torch.save(ckpt, temporary)
                    temporary.replace(output_dir / f"{name}.ckpt")
                learning_rate = float(opt.param_groups[0]["lr"])
                print(format_jepa_epoch_log(epoch, epoch_loss, epoch_metrics, val_loss, val_metrics, learning_rate), flush=True)
                log_wandb(
                    wandb_run,
                    jepa_wandb_metrics(epoch_loss, epoch_metrics, val_loss, val_metrics, learning_rate),
                    step=global_step,
                )
                if stop_now:
                    print(
                        f"early_stopping epoch={epoch} monitor_loss={monitor_loss:.6f} "
                        f"best={early_stopper.best:.6f} best_epoch={early_stopper.best_epoch} "
                        f"wait={early_stopper.wait}/{early_stopper.patience}", flush=True,
                    )
            if is_distributed():
                dist.barrier()
            if stop_now:
                break
        if is_main_process() and config.get("packing_eval_samples", 0) and val_loader is not None:
            from lvebcsp.eval.crystal_jepa import evaluate_packing
            best_ckpt = torch.load(output_dir / "best.ckpt", map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(best_ckpt["model"])
            report = evaluate_packing(unwrap_model(model), val_dataset, device,
                                      max_samples=config["packing_eval_samples"])
            (output_dir / "packing_valid.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        if is_distributed():
            dist.barrier()
    finally:
        finish_wandb(wandb_run)
        cleanup_distributed()


if __name__ == "__main__":
    main()
