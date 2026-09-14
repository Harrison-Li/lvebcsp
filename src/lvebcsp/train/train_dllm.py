"""Training entry point for the crystal diffusion language model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import gcd, isfinite
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from lvebcsp.common.config import load_config, save_config, select_device
from lvebcsp.common.distributed import (
    cleanup_distributed,
    is_distributed,
    is_main_process,
    setup_distributed,
)
from lvebcsp.common.gpu import unwrap_model
from lvebcsp.common.seed import seed_everything
from lvebcsp.common.utils import lattice_params_to_matrix_torch
from lvebcsp.common.wandb_logging import (
    finish_wandb,
    init_wandb,
    log_wandb,
    maybe_watch_model,
)
from lvebcsp.data.collate import collate_crystal_batch
from lvebcsp.models.canvas import CrystalCanvas, FormulaConditioner
from lvebcsp.models.discrete_corrupt import UniformDiscreteCorruption
from lvebcsp.models.dlm import CrystalDiffusionLanguageModel
from lvebcsp.models.jepa import Lvebm
from lvebcsp.models.peak_encoder import PeakDetailEncoder
from lvebcsp.train.dllm_checkpoint import (
    TRAIN_SCOPES,
    initialize_from_checkpoint,
    migrate_condition_encoder_state_dict,
    set_dllm_train_scope,
)
from lvebcsp.train.train_jepa import (
    batch_to_device,
    build_dataset_from_config,
    build_early_stopper,
    build_jepa_from_config,
    build_lr_scheduler,
    format_learning_rate,
    has_dataset_split,
)


DLLM_LOG_METRICS = (
    "token_loss",
    "token_accuracy",
    "atom_token_accuracy",
    "nonpad_token_accuracy",
    "exact_count_accuracy",
    "atom_count_mae",
    "coord_frac_rmse",
    "non_atom_coord_velocity_rmse",
    "lattice_len_rel_rmse",
    "lattice_len_angstrom_rmse",
    "lattice_angle_deg_rmse",
    "lattice_velocity_rmse",
    "formula_ratio_loss",
    "formula_ratio_l1",
    "formula_noise_std_avg",
    "self_conditioned_fraction",
    "t_avg",
)

@dataclass
class PreparedDLLMBatch:
    """Noisy model inputs and their clean denoising targets."""

    tokens_t: Tensor
    lattice_t: Tensor
    frac_coords_t: Tensor
    t: Tensor
    clean_tokens: Tensor
    token_corruption_mask: Tensor
    clean_frac_coords: Tensor
    lattice_params: Tensor
    clean_lattice_matrix: Tensor
    lattice_velocity: Tensor
    coord_velocity: Tensor

    def targets(self) -> dict[str, Tensor]:
        return {
            "tokens": self.clean_tokens,
            "token_corruption_mask": self.token_corruption_mask,
            "frac_coords": self.clean_frac_coords,
            "lattice_params": self.lattice_params,
            "lattice_matrix": self.clean_lattice_matrix,
            "lattice_velocity": self.lattice_velocity,
            "coord_velocity": self.coord_velocity,
            "t": self.t,
        }


class CrystalDiffusionCorruption:
    """Create one joint ``t=0 clean -> t=1 prior`` training path.

    Tokens use independent uniform replacement, the full lattice matrix uses a
    Gaussian prior, and every coordinate-canvas position uses a uniform prior
    on the fractional-coordinate torus. Each crystal independently samples a
    discrete timestep. Tokens use ``t ** token_noise_exponent``, while lattice
    and coordinates use ``t ** geometry_noise_exponent``. Both exponents
    default to ``1.0``. Setting the geometry exponent to ``0.5`` keeps geometry
    noisier than tokens at intermediate times, so geometry denoises later.
    """

    def __init__(
        self,
        n_max: int,
        vocab_size: int,
        *,
        num_timesteps: int = 1000,
        token_noise_exponent: float = 1.0,
        geometry_noise_exponent: float = 1.0,
    ) -> None:
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be at least 1")
        if (
            not isfinite(float(token_noise_exponent))
            or float(token_noise_exponent) <= 0.0
        ):
            raise ValueError("token_noise_exponent must be finite and positive")
        if (
            not isfinite(float(geometry_noise_exponent))
            or float(geometry_noise_exponent) <= 0.0
        ):
            raise ValueError(
                "geometry_noise_exponent must be finite and positive"
            )
        self.canvas = CrystalCanvas(n_max)
        self.discrete = UniformDiscreteCorruption(vocab_size)
        self.num_timesteps = int(num_timesteps)
        self.token_noise_exponent = float(token_noise_exponent)
        self.geometry_noise_exponent = float(geometry_noise_exponent)

    def _time(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        t: Tensor | float | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        if t is None:
            timestep = torch.randint(
                1,
                self.num_timesteps + 1,
                (batch_size,),
                device=device,
                generator=generator,
            )
            return timestep.to(dtype=dtype) / self.num_timesteps

        time = torch.as_tensor(t, device=device, dtype=dtype)
        if time.ndim == 0:
            time = time.expand(batch_size)
        else:
            time = time.reshape(-1)
        if time.shape != (batch_size,):
            raise ValueError(f"t must be scalar or have shape [{batch_size}]")
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("t must contain finite values in [0, 1]")
        return time

    def __call__(
        self,
        batch: dict[str, Tensor],
        *,
        t: Tensor | float | None = None,
        generator: torch.Generator | None = None,
    ) -> PreparedDLLMBatch:
        atom_types = batch["atom_types"]
        frac_coords = batch["frac_coords"]
        atom_mask = batch.get("atom_mask")
        lattice_params = batch["lattice_params"]
        canvas = self.canvas.build(atom_types, atom_mask, frac_coords)

        clean_coords = canvas["frac_coords"]
        batch_size = atom_types.shape[0]
        if lattice_params.shape != (batch_size, 6):
            raise ValueError(f"lattice_params must have shape [{batch_size}, 6]")
        if torch.any(canvas["num_atoms"] < 1):
            raise ValueError("each training crystal must contain at least one atom")

        time = self._time(
            batch_size,
            clean_coords.device,
            clean_coords.dtype,
            t,
            generator,
        )
        token_time = time.pow(self.token_noise_exponent)
        geometry_time = time.pow(self.geometry_noise_exponent)
        tokens_t, token_corruption_mask = self.discrete.corrupt(
            canvas["tokens"],
            token_time,
            generator=generator,
            return_mask=True,
        )

        clean_lattice_matrix = lattice_params_to_matrix_torch(
            lattice_params[:, :3],
            lattice_params[:, 3:6],
        ).to(
            device=clean_coords.device,
            dtype=clean_coords.dtype,
        )
        lattice_prior = torch.randn(
            clean_lattice_matrix.shape,
            device=clean_lattice_matrix.device,
            dtype=clean_lattice_matrix.dtype,
            generator=generator,
        )
        lattice_velocity = clean_lattice_matrix - lattice_prior
        lattice_t = (
            clean_lattice_matrix
            - geometry_time[:, None, None] * lattice_velocity
        )

        coord_prior = torch.rand(
            clean_coords.shape,
            device=clean_coords.device,
            dtype=clean_coords.dtype,
            generator=generator,
        )
        coord_velocity = torch.remainder(
            clean_coords - coord_prior + 0.5,
            1.0,
        ) - 0.5
        frac_coords_t = torch.remainder(
            coord_prior
            + (1.0 - geometry_time[:, None, None]) * coord_velocity,
            1.0,
        )

        return PreparedDLLMBatch(
            tokens_t=tokens_t,
            lattice_t=lattice_t,
            frac_coords_t=frac_coords_t,
            t=time,
            clean_tokens=canvas["tokens"],
            token_corruption_mask=token_corruption_mask,
            clean_frac_coords=clean_coords,
            lattice_params=lattice_params,
            clean_lattice_matrix=clean_lattice_matrix,
            lattice_velocity=lattice_velocity,
            coord_velocity=coord_velocity,
        )


def deterministic_dataset_subset(
    dataset: Dataset,
    max_samples: int | None,
    *,
    seed: int,
) -> Dataset:
    """Select a reproducible random subset for bounded experiments."""

    if max_samples is None:
        return dataset
    count = int(max_samples)
    if count < 1:
        raise ValueError("max_samples must be positive")
    if count >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def build_dllm_from_config(config: dict[str, Any]) -> CrystalDiffusionLanguageModel:
    """Instantiate the crystal DLM from the ``dlm`` config section."""

    model_cfg = dict(config.get("dlm", {}))
    model_cfg.pop("max_formula_tokens", None)
    n_max = int(model_cfg.pop("n_max", config.get("n_max", 64)))
    d_jepa = int(model_cfg.pop("d_jepa", config.get("d_jepa", 256)))
    p_max = int(model_cfg.pop("p_max", config.get("p_max", 128)))
    detail_cfg = dict(model_cfg.pop("pxrd_detail", {}))
    pxrd_enabled = bool(model_cfg.pop("pxrd_conditioning", False))
    detail_encoder = None
    if detail_cfg.get("enabled", False):
        detail_hidden_dim = int(detail_cfg.get("hidden_dim", 64))
        detail_encoder = PeakDetailEncoder(
            p_max=p_max,
            hidden_dim=detail_hidden_dim,
            output_dim=int(model_cfg.get("hidden_dim", 256)),
            num_tokens=int(detail_cfg.get("num_tokens", 16)),
            num_heads=int(
                detail_cfg.get(
                    "num_heads",
                    gcd(detail_hidden_dim, int(model_cfg.get("num_heads", 4))),
                )
            ),
            dropout=float(
                detail_cfg.get("dropout", model_cfg.get("dropout", 0.1))
            ),
        )
    return CrystalDiffusionLanguageModel(
        n_max=n_max,
        d_jepa=d_jepa,
        pxrd_detail_encoder=detail_encoder,
        pxrd_conditioning=pxrd_enabled,
        **model_cfg,
    )


def _upgrade_legacy_jepa_state(
    state_dict: dict[str, Tensor],
    model: Lvebm,
) -> dict[str, Tensor]:
    """Map the older JEPA MLP normalization key layout when necessary."""

    model_state = model.state_dict()
    upgraded = dict(state_dict)
    for key, value in list(state_dict.items()):
        if ".mlp.1." not in key or key in model_state:
            continue
        target_key = key.replace(".mlp.1.", ".mlp.2.")
        if target_key in model_state and model_state[target_key].shape == value.shape:
            upgraded[target_key] = value
            upgraded.pop(key, None)
    return upgraded


class FrozenJEPATargetEncoder(nn.Module):
    """The frozen ideal-PXRD branch extracted from a trained JEPA."""

    def __init__(self, jepa: Lvebm) -> None:
        super().__init__()
        self.d_jepa = int(jepa.d_jepa)
        self.peak_encoder = jepa.context_encoder
        self.projector = jepa.projector

    def encode_tgt(
        self,
        peak_d: Tensor,
        peak_i: Tensor,
        peak_mask: Tensor,
    ) -> Tensor:
        return self.projector(self.peak_encoder(peak_d, peak_i, peak_mask))

    def forward(
        self,
        peak_d: Tensor,
        peak_i: Tensor,
        peak_mask: Tensor,
    ) -> Tensor:
        return self.encode_tgt(peak_d, peak_i, peak_mask)


def load_jepa_target_encoder(
    path: str | Path,
    device: torch.device,
) -> FrozenJEPATargetEncoder:
    """Load only the frozen JEPA target branch used for ideal PXRD."""

    checkpoint = torch.load(path, map_location=device)
    jepa = build_jepa_from_config(checkpoint.get("config", {})).to(device)
    jepa.load_state_dict(_upgrade_legacy_jepa_state(checkpoint["model"], jepa))
    model = FrozenJEPATargetEncoder(jepa).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def validate_jepa_latent_dim(
    config: dict[str, Any],
    jepa: nn.Module,
    checkpoint_path: str | Path,
) -> None:
    """Keep the DLM condition width consistent with its JEPA checkpoint."""

    checkpoint_dim = int(jepa.d_jepa)
    configured_dim = config.get("dlm", {}).get("d_jepa", config.get("d_jepa"))
    if configured_dim is None:
        config["d_jepa"] = checkpoint_dim
    elif int(configured_dim) != checkpoint_dim:
        raise ValueError(
            f"DLM d_jepa={configured_dim} does not match {checkpoint_path} "
            f"d_jepa={checkpoint_dim}"
        )


def compute_pxrd_latent(
    jepa_target: nn.Module,
    batch: dict[str, Tensor],
) -> Tensor:
    """Encode the clean, non-augmented PXRD view with the frozen JEPA branch."""

    return jepa_target.encode_tgt(
        batch["peak_d"],
        batch["peak_i"],
        batch["peak_mask"],
    )


def criterion_metrics(values: dict[str, Tensor]) -> dict[str, float]:
    """Detach DLM criterion values for progress and experiment logging."""

    return {
        name: float(value.detach().float().mean().cpu())
        for name, value in values.items()
    }


def sample_formula_noise_stds(
    max_std: float,
    batch_size: int,
    device: torch.device,
    *,
    clean_probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Mix exact formulas with continuously sampled noise levels."""

    if not isfinite(float(max_std)) or max_std < 0:
        raise ValueError("formula max_relative_noise_std must be non-negative")
    if not 0.0 <= float(clean_probability) <= 1.0:
        raise ValueError("formula clean_probability must be in [0, 1]")
    if max_std == 0:
        return torch.zeros(batch_size, device=device)
    noise = float(max_std) * torch.rand(batch_size, device=device, generator=generator)
    clean = torch.rand(batch_size, device=device, generator=generator) < clean_probability
    return noise.masked_fill(clean, 0.0)


def dllm_batch_loss(
    model: nn.Module,
    jepa_target: nn.Module | None,
    batch: dict[str, Tensor],
    corruption: CrystalDiffusionCorruption,
    formula_conditioner: FormulaConditioner,
    *,
    loss_cfg: dict[str, Any] | None = None,
    condition_dropout: float = 0.0,
    formula_max_noise_std: float = 0.0,
    formula_clean_probability: float = 1.0,
    self_condition_probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Corrupt one batch and compute the joint crystal denoising loss."""

    if not 0.0 <= float(condition_dropout) <= 1.0:
        raise ValueError("condition_dropout must be in [0, 1]")
    if not 0.0 <= float(self_condition_probability) <= 1.0:
        raise ValueError("self_condition_probability must be in [0, 1]")
    base_model = unwrap_model(model)
    prepared = corruption(batch, generator=generator)
    formula_noise_std = sample_formula_noise_stds(
        formula_max_noise_std,
        batch["atom_types"].shape[0],
        batch["atom_types"].device,
        clean_probability=formula_clean_probability,
        generator=generator,
    )
    with torch.no_grad():
        if base_model.uses_pxrd_conditioning:
            if jepa_target is None:
                raise ValueError(
                    "jepa_target is required when PXRD conditioning is enabled"
                )
            pxrd_condition = compute_pxrd_latent(jepa_target, batch)
        else:
            pxrd_condition = None
        formula = formula_conditioner.from_atom_types(
            batch["atom_types"],
            batch.get("atom_mask"),
            noise_std=formula_noise_std,
            generator=generator,
        )
    detail_inputs: dict[str, Tensor] = {}
    if base_model.uses_pxrd_detail:
        detail_keys = ("aug_peak_d", "aug_peak_i", "aug_peak_mask")
        missing_detail = [key for key in detail_keys if key not in batch]
        if missing_detail:
            raise KeyError(
                "PXRD detail conditioning requires augmented batch fields: "
                + ", ".join(missing_detail)
            )
        # The local detail path receives the augmented spectrum while JEPA
        # supplies a robust global condition from the clean spectrum view.
        detail_inputs = {
            "peak_d": batch["aug_peak_d"],
            "peak_i": batch["aug_peak_i"],
            "peak_mask": batch["aug_peak_mask"].to(dtype=torch.bool),
        }
    if condition_dropout > 0 and pxrd_condition is not None:
        keep = torch.rand(
            pxrd_condition.shape[0],
            device=pxrd_condition.device,
            generator=generator,
        ) >= float(condition_dropout)
        # Null conditioning is applied per crystal while formula evidence
        # remains fixed. This keeps every DDP worker on the same parameter path.
        pxrd_condition = torch.where(
            keep.unsqueeze(-1),
            pxrd_condition,
            torch.zeros_like(pxrd_condition),
        )
        if detail_inputs:
            detail_inputs["peak_mask"] = (
                detail_inputs["peak_mask"] & keep.unsqueeze(-1)
            )

    c = base_model.encode_condition(
        formula["fractions"],
        formula["uncertainties"],
        z_jepa=pxrd_condition,
        **detail_inputs,
    )
    model_inputs = (
        prepared.lattice_t,
        prepared.frac_coords_t,
        prepared.t,
        c,
        prepared.tokens_t,
    )
    geometry_time_inputs: dict[str, Tensor] = {}
    if corruption.geometry_noise_exponent != 1.0:
        geometry_time_inputs["geometry_noise_level"] = prepared.t.pow(
            corruption.geometry_noise_exponent
        )
    self_conditioned = torch.zeros(
        prepared.tokens_t.shape[0],
        dtype=torch.bool,
        device=prepared.tokens_t.device,
    )
    self_conditioning_logits = None
    if self_condition_probability > 0:
        # Gemma-style self-conditioning: both passes see exactly the same
        # noisy state and timestep.
        with torch.no_grad():
            first_output = model(
                *model_inputs,
                **geometry_time_inputs,
            )
        first_logits = first_output["token_logits"].detach()
        self_conditioned = torch.rand(
            prepared.tokens_t.shape[0],
            device=prepared.tokens_t.device,
            generator=generator,
        ) < float(self_condition_probability)
        self_conditioning_logits = torch.where(
            self_conditioned[:, None, None],
            first_logits,
            torch.zeros_like(first_logits),
        )
    output = model(
        *model_inputs,
        self_conditioning_logits=self_conditioning_logits,
        **geometry_time_inputs,
    )
    values = base_model.criterion(
        prepared.targets(),
        output,
        loss_cfg=loss_cfg,
        formula_fractions=formula["clean_fractions"],
        formula_uncertainties=formula["uncertainties"],
    )
    loss = values["loss"]
    metrics = criterion_metrics(values)
    metrics["formula_noise_std_avg"] = float(
        formula_noise_std.mean().detach().cpu()
    )
    metrics["self_conditioned_fraction"] = float(
        self_conditioned.float().mean().detach().cpu()
    )
    return loss, metrics


def selected_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        name: float(metrics[name])
        for name in DLLM_LOG_METRICS
        if name in metrics
    }


def reduce_epoch_statistics(
    total_loss: float,
    metric_totals: dict[str, float],
    batches: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Average epoch statistics across all DDP workers."""

    names = sorted(metric_totals)
    values = torch.tensor(
        [total_loss, float(batches), *(metric_totals[name] for name in names)],
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


def format_epoch_log(
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    learning_rate: float,
) -> str:
    fields = [f"epoch={epoch}", f"train_loss={train_loss:.6f}"]
    if val_loss is not None:
        fields.append(f"val_loss={val_loss:.6f}")
    fields.extend(
        f"{name}={value:.6f}"
        for name, value in selected_metrics(train_metrics).items()
    )
    fields.extend(
        f"val_{name}={value:.6f}"
        for name, value in selected_metrics(val_metrics).items()
    )
    fields.append(f"lr={format_learning_rate(learning_rate)}")
    return " ".join(fields)


def wandb_metrics(
    train_loss: float,
    val_loss: float | None,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    learning_rate: float,
) -> dict[str, float | None]:
    payload: dict[str, float | None] = {
        "train/loss": train_loss,
        "val/loss": val_loss,
        "trainer/lr": learning_rate,
    }
    payload.update(
        {f"train/{name}": value for name, value in selected_metrics(train_metrics).items()}
    )
    payload.update(
        {f"val/{name}": value for name, value in selected_metrics(val_metrics).items()}
    )
    return payload


def train_one_epoch(
    model: nn.Module,
    jepa_target: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    corruption: CrystalDiffusionCorruption,
    formula_conditioner: FormulaConditioner,
    device: torch.device,
    config: dict[str, Any],
    *,
    epoch: int,
    max_epochs: int,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scheduler_interval: str,
) -> tuple[float, dict[str, float]]:
    """Run one DLM training epoch."""

    model.train()
    if jepa_target is not None:
        jepa_target.eval()
    total_loss = 0.0
    metric_totals: dict[str, float] = {}
    batches = 0
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    progress = tqdm(
        loader,
        desc=f"DLM epoch {epoch + 1}/{max_epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not is_main_process(),
    )
    for batch in progress:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = dllm_batch_loss(
            model,
            jepa_target,
            batch,
            corruption,
            formula_conditioner,
            loss_cfg=config.get("loss", {}),
            condition_dropout=float(config.get("condition_dropout", 0.0)),
            formula_max_noise_std=float(
                config.get("formula", {}).get("max_relative_noise_std", 0.0)
            ),
            formula_clean_probability=float(
                config.get("formula", {}).get("clean_probability", 1.0)
            ),
            self_condition_probability=float(
                config.get("diffusion", {}).get(
                    "self_condition_probability",
                    0.5,
                )
            ),
        )
        loss.backward()
        grad_clip = float(config.get("gradient_clip_norm", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()

        if scheduler is not None and scheduler_interval == "step":
            scheduler.step()

        total_loss += float(loss.detach().cpu())
        for name, value in metrics.items():
            metric_totals[name] = metric_totals.get(name, 0.0) + value
        batches += 1
        averages = {name: value / batches for name, value in metric_totals.items()}
        progress.set_postfix(
            loss=f"{total_loss / batches:.4f}",
            count_mae=f"{averages.get('atom_count_mae', 0.0):.3f}",
            token_acc=f"{averages.get('token_accuracy', 0.0):.3f}",
            frac_rmsd=f"{averages.get('coord_frac_rmse', 0.0):.4f}",
            length_rmse_A=(
                f"{averages.get('lattice_len_angstrom_rmse', 0.0):.3f}"
            ),
            angle_rmse_deg=(
                f"{averages.get('lattice_angle_deg_rmse', 0.0):.2f}"
            ),
            formula_sigma=(
                f"{averages.get('formula_noise_std_avg', 0.0):.3f}"
            ),
            self_cond=(
                f"{averages.get('self_conditioned_fraction', 0.0):.3f}"
            ),
        )

    if scheduler is not None and scheduler_interval == "epoch":
        scheduler.step()
    return reduce_epoch_statistics(total_loss, metric_totals, batches, device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    jepa_target: nn.Module | None,
    loader: DataLoader | None,
    corruption: CrystalDiffusionCorruption,
    formula_conditioner: FormulaConditioner,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[float | None, dict[str, float]]:
    """Evaluate the same joint denoising objective on validation crystals."""

    if loader is None:
        return None, {}
    model.eval()
    if jepa_target is not None:
        jepa_target.eval()
    total_loss = 0.0
    metric_totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        loss, metrics = dllm_batch_loss(
            model,
            jepa_target,
            batch,
            corruption,
            formula_conditioner,
            loss_cfg=config.get("loss", {}),
            formula_max_noise_std=float(
                config.get("formula", {}).get(
                    "validation_max_relative_noise_std", 0.0
                )
            ),
            self_condition_probability=float(
                config.get("diffusion", {}).get(
                    "validation_self_condition_probability",
                    config.get("diffusion", {}).get(
                        "self_condition_probability",
                        0.5,
                    ),
                )
            ),
        )
        total_loss += float(loss.detach().cpu())
        for name, value in metrics.items():
            metric_totals[name] = metric_totals.get(name, 0.0) + value
        batches += 1
    return reduce_epoch_statistics(total_loss, metric_totals, batches, device)


def _restore_early_stopper(stopper: Any, state: dict[str, Any] | None) -> None:
    if stopper is None or not state:
        return
    for name in ("best", "best_epoch", "wait", "stopped"):
        if name in state:
            setattr(stopper, name, state[name])


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    early_stopper: Any,
    device: torch.device,
) -> tuple[int, float]:
    """Restore a DLM run and return its next epoch and best loss."""

    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(
        migrate_condition_encoder_state_dict(checkpoint["model"])
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    _restore_early_stopper(early_stopper, checkpoint.get("early_stopping"))
    best = float(checkpoint.get("best", checkpoint.get("monitor_loss", float("inf"))))
    return int(checkpoint.get("epoch", -1)) + 1, best


def _disable_native_mha_fastpath() -> None:
    """Avoid native-MHA parallel-evaluation crashes on some CUDA builds."""

    mha_backend = getattr(torch.backends, "mha", None)
    set_fastpath_enabled = getattr(mha_backend, "set_fastpath_enabled", None)
    if callable(set_fastpath_enabled):
        set_fastpath_enabled(False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the variable-cardinality crystal DLM"
    )
    parser.add_argument("--config", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        default=None,
        help="Resume model, optimizer, scheduler, and epoch from a compatible checkpoint.",
    )
    checkpoint_group.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights only and start with a fresh optimizer.",
    )
    parser.add_argument(
        "--output-bias-migration",
        default=None,
        choices=("drop", "fold"),
        help=(
            "How --init-checkpoint handles legacy lattice/coordinate Linear "
            "biases: remove them, or fold them into the preceding LayerNorm."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--train-scope", choices=TRAIN_SCOPES, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    diffusion_cfg = config.get("diffusion", {})
    sampling_cfg = config.get("sampling", {})
    for name in ("token_noise_exponent", "geometry_noise_exponent"):
        training_value = float(diffusion_cfg.get(name, 1.0))
        sampling_value = float(sampling_cfg.get(name, 1.0))
        if training_value != sampling_value:
            parser.error(
                f"diffusion.{name}={training_value} must match "
                f"sampling.{name}={sampling_value}"
            )
    if args.resume:
        # An explicit resume continues stage two and must not reapply the
        # stage-one initializer embedded in its original YAML config.
        init_path = None
        resume_path = args.resume
        config.pop("init_checkpoint", None)
    elif args.init_checkpoint:
        init_path = args.init_checkpoint
        resume_path = None
        config.pop("resume_ckpt", None)
    else:
        init_path = config.get("init_checkpoint")
        resume_path = config.get("resume_ckpt")
    if init_path and resume_path:
        parser.error("--init-checkpoint/init_checkpoint cannot be combined with resume")
    output_bias_migration = str(
        args.output_bias_migration
        or config.get("output_bias_migration", "drop")
    )
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            parser.error("--learning-rate must be positive")
        config["learning_rate"] = args.learning_rate
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            parser.error("--max-epochs must be at least 1")
        config["max_epochs"] = args.max_epochs
    train_scope = str(args.train_scope or config.get("train_scope", "all"))
    if train_scope not in TRAIN_SCOPES:
        parser.error(f"train_scope must be one of {TRAIN_SCOPES}")
    config["train_scope"] = train_scope
    if init_path:
        # Weight-only initialization intentionally supersedes a resume path
        # embedded in an older training config.
        config.pop("resume_ckpt", None)
        config["init_checkpoint"] = str(init_path)
        config["initialized_from"] = str(init_path)
        config["output_bias_migration"] = output_bias_migration
    pxrd_enabled = bool(config.get("dlm", {}).get("pxrd_conditioning", False))
    # De novo training can consume structure-only LMDBs/manifests and avoids
    # the cost of simulating or preprocessing spectra that the model ignores.
    config["include_pxrd"] = pxrd_enabled

    rank, local_rank, world_size = setup_distributed(
        config.get("distributed_backend")
    )
    wandb_run = None
    try:
        _disable_native_mha_fastpath()
        seed = int(config.get("seed", 42))
        seed_everything(seed + rank)
        if dist.is_initialized():
            device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = select_device(str(config.get("device", "auto")))

        train_dataset = build_dataset_from_config(config, split="train")
        train_dataset = deterministic_dataset_subset(
            train_dataset,
            config.get("max_train_samples"),
            seed=seed,
        )
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
            if world_size > 1
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config.get("batch_size", 64)),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=int(config.get("num_workers", 4)),
            drop_last=bool(config.get("drop_last", False)),
            collate_fn=collate_crystal_batch,
        )
        val_loader = None
        if has_dataset_split(config, "valid"):
            val_dataset = build_dataset_from_config(config, split="valid")
            val_dataset = deterministic_dataset_subset(
                val_dataset,
                config.get("max_valid_samples"),
                seed=seed + 1,
            )
            val_sampler = (
                DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                )
                if world_size > 1
                else None
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(
                    config.get("eval_batch_size", config.get("batch_size", 64))
                ),
                shuffle=False,
                sampler=val_sampler,
                num_workers=int(config.get("num_workers", 4)),
                collate_fn=collate_crystal_batch,
            )

        jepa_target: nn.Module | None = None
        if pxrd_enabled:
            jepa_checkpoint = config.get("jepa_ckpt")
            if not jepa_checkpoint:
                raise ValueError(
                    "jepa_ckpt is required when PXRD conditioning is enabled"
                )
            jepa_target = load_jepa_target_encoder(jepa_checkpoint, device)
            validate_jepa_latent_dim(config, jepa_target, jepa_checkpoint)
        model: nn.Module = build_dllm_from_config(config).to(device)
        initialization_report: dict[str, Any] | None = None
        if init_path:
            # Keep the unused checkpoint optimizer state off the GPU. Only model
            # tensors are copied to ``model`` by initialize_from_checkpoint.
            initial_checkpoint = torch.load(init_path, map_location="cpu")
            initialization_report = initialize_from_checkpoint(
                model,
                initial_checkpoint,
                strategy=output_bias_migration,
            )
            del initial_checkpoint

        set_dllm_train_scope(model, train_scope)
        if dist.is_initialized():
            find_unused = bool(config.get("ddp_find_unused_parameters", False))
            ddp_kwargs: dict[str, Any] = {
                "broadcast_buffers": False,
                "find_unused_parameters": find_unused,
            }
            if device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[local_rank],
                    output_device=local_rank,
                )
            model = DistributedDataParallel(model, **ddp_kwargs)
        target_model = unwrap_model(model)
        trainable_parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        if is_main_process() and world_size > 1:
            print(f"using DistributedDataParallel with {world_size} processes")
        if is_main_process() and initialization_report is not None:
            removed = ", ".join(initialization_report["removed_keys"]) or "none"
            added_pxrd = len(initialization_report.get("initialized_pxrd_keys", ()))
            print(
                f"initialized DLM from {init_path} "
                f"output_bias_migration={output_bias_migration} "
                f"removed_keys={removed} initialized_pxrd_parameters={added_pxrd}"
            )
        if is_main_process():
            trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
            total_count = sum(parameter.numel() for parameter in model.parameters())
            print(
                f"train_scope={train_scope} trainable_parameters="
                f"{trainable_count}/{total_count}"
            )

        diffusion_cfg = dict(config.get("diffusion", {}))
        corruption = CrystalDiffusionCorruption(
            target_model.n_max,
            target_model.vocab_size,
            num_timesteps=int(diffusion_cfg.get("num_timesteps", 1000)),
            token_noise_exponent=float(
                diffusion_cfg.get("token_noise_exponent", 1.0)
            ),
            geometry_noise_exponent=float(
                diffusion_cfg.get("geometry_noise_exponent", 1.0)
            ),
        )
        formula_conditioner = FormulaConditioner()
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(config.get("learning_rate", 1.0e-4)),
            weight_decay=float(config.get("weight_decay", 1.0e-2)),
        )
        max_epochs = int(config.get("max_epochs", 300))
        steps_per_epoch = max(len(train_loader), 1)
        scheduler, scheduler_interval = build_lr_scheduler(
            optimizer,
            config,
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        early_stopper = build_early_stopper(config)

        output_dir = Path(config.get("output_dir", "outputs/dlm"))
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, output_dir / "training_config.yaml")
        if dist.is_initialized():
            dist.barrier()

        start_epoch = 0
        best = float("inf")
        if resume_path:
            start_epoch, best = load_training_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                early_stopper,
                device,
            )
            if is_main_process():
                print(
                    f"resumed DLM from {resume_path} "
                    f"at epoch={start_epoch} best={best:.6f}"
                )

        if is_main_process():
            wandb_run = init_wandb(
                config,
                job_type="dlm",
                output_dir=output_dir,
                default_name=output_dir.name,
            )
            maybe_watch_model(wandb_run, target_model, config)

        for epoch in range(start_epoch, max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss, train_metrics = train_one_epoch(
                model,
                jepa_target,
                train_loader,
                optimizer,
                corruption,
                formula_conditioner,
                device,
                config,
                epoch=epoch,
                max_epochs=max_epochs,
                scheduler=scheduler,
                scheduler_interval=scheduler_interval,
            )
            val_loss, val_metrics = evaluate(
                model,
                jepa_target,
                val_loader,
                corruption,
                formula_conditioner,
                device,
                config,
            )
            monitor_loss = val_loss if val_loss is not None else train_loss
            is_best = monitor_loss < best
            if is_best:
                best = monitor_loss
            stop_now = (
                early_stopper.step(monitor_loss, epoch)
                if early_stopper is not None
                else False
            )
            if is_main_process():
                checkpoint: dict[str, Any] = {
                    "model": target_model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "loss": train_loss,
                    "metrics": train_metrics,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "monitor_loss": monitor_loss,
                    "best": best,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                }
                if config.get("jepa_ckpt"):
                    checkpoint["jepa_ckpt"] = config["jepa_ckpt"]
                if init_path:
                    checkpoint["initialized_from"] = str(init_path)
                    checkpoint["output_bias_migration"] = output_bias_migration
                if early_stopper is not None:
                    checkpoint["early_stopping"] = early_stopper.state_dict()
                torch.save(checkpoint, output_dir / "last.ckpt")
                if bool(config.get("save_epoch_checkpoints", False)):
                    torch.save(
                        checkpoint,
                        output_dir / f"epoch_{epoch + 1:03d}.ckpt",
                    )
                if is_best:
                    torch.save(checkpoint, output_dir / "best.ckpt")

                learning_rate = float(optimizer.param_groups[0]["lr"])
                print(
                    format_epoch_log(
                        epoch,
                        train_loss,
                        val_loss,
                        train_metrics,
                        val_metrics,
                        learning_rate,
                    )
                )
                log_wandb(
                    wandb_run,
                    wandb_metrics(
                        train_loss,
                        val_loss,
                        train_metrics,
                        val_metrics,
                        learning_rate,
                    ),
                    step=(epoch + 1) * steps_per_epoch,
                )
            if stop_now:
                if is_main_process():
                    print(
                        "early_stopping "
                        f"epoch={epoch} monitor_loss={monitor_loss:.6f} "
                        f"best={early_stopper.best:.6f} "
                        f"best_epoch={early_stopper.best_epoch} "
                        f"wait={early_stopper.wait}/{early_stopper.patience}"
                    )
                break
    finally:
        if is_main_process():
            finish_wandb(wandb_run)
        cleanup_distributed()


if __name__ == "__main__":
    main()
