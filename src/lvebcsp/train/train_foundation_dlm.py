"""Train one shared DLM across organic and inorganic crystal domains.

The denoiser is the repository's existing ``CrystalDiffusionLanguageModel``.
A single E(3)-invariant atomistic-fragment encoder feeds its condition encoder.
Organic molecular templates and inorganic NULL fragments with reduced
composition counts are adapted to the same tensors. Training may use either
domain alone or balance both domains.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from lvebcsp.common.config import load_config, save_config, select_device
from lvebcsp.common.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
)
from lvebcsp.common.gpu import unwrap_model
from lvebcsp.common.seed import seed_everything
from lvebcsp.common.wandb_logging import (
    finish_wandb,
    init_wandb,
    log_wandb,
    maybe_watch_model,
)
from lvebcsp.data.atomistic_fragment import (
    FRAGMENT_TENSOR_KEYS,
    INORGANIC_CONDITION,
    INORGANIC_CONDITION_MODES,
    MULTIPLICITY_TENSOR_KEYS,
    BalancedMaterialsDataset,
    FoundationInorganicDataset,
    FoundationOrganicDataset,
)
from lvebcsp.data.collate import collate_crystal_batch
from lvebcsp.models.canvas import NUM_ELEMENTS
from lvebcsp.models.foundation_dlm import FoundationCrystalDiffusionModel
from lvebcsp.train.dllm_checkpoint import (
    is_pxrd_adapter_key,
    migrate_condition_encoder_state_dict,
)
from lvebcsp.train.train_dllm import (
    CrystalDiffusionCorruption,
    criterion_metrics,
    format_epoch_log,
    load_jepa_target_encoder,
    reduce_epoch_statistics,
    sample_formula_noise_stds,
    wandb_metrics,
)
from lvebcsp.train.train_jepa import build_early_stopper, build_lr_scheduler


FOUNDATION_MODEL_FAMILY = "universal_atomistic_fragment_dlm_v2"
FOUNDATION_MULTIPLICITY_MODEL_FAMILY = (
    "universal_atomistic_fragment_dlm_multiplicity_cfg_v1"
)
FOUNDATION_MODEL_FAMILIES = (
    FOUNDATION_MODEL_FAMILY,
    FOUNDATION_MULTIPLICITY_MODEL_FAMILY,
)
FOUNDATION_TRAINING_DOMAINS = ("organic", "inorganic")
FOUNDATION_CONDITION_SCHEMA: dict[str, Any] = {
    "shared_tensor_fields": (
        *FRAGMENT_TENSOR_KEYS,
        "condition_formula_fractions",
    ),
    "organic_source_fields": (
        "mol_atom_type_2",
        "mol_atom_pos_2",
        "mol2_group_slices",
    ),
    "inorganic_condition": INORGANIC_CONDITION,
    "inorganic_fragment_source": "null_or_explicit_inference_fragment",
    "uses_repeat_dict_2": False,
    "uses_mol_atom_map_2_as_feature": False,
    "uses_mol_atom_degree_2_as_feature": False,
    "uses_target_atom_count": False,
    "composition_is_scale_free": True,
    "uses_domain_label_as_feature": False,
    "uses_explicit_condition_element_counts": True,
    "condition_element_count_sources": {
        "fragment_present": "total_fragment_counts",
        "composition_only": "reduced_composition_counts",
    },
    "uses_explicit_fragment_group_counts": True,
}
FOUNDATION_MULTIPLICITY_CONDITION_SCHEMA: dict[str, Any] = {
    **FOUNDATION_CONDITION_SCHEMA,
    "shared_tensor_fields": (
        *FOUNDATION_CONDITION_SCHEMA["shared_tensor_fields"],
        *MULTIPLICITY_TENSOR_KEYS,
    ),
    "uses_target_atom_count": True,
    "uses_optional_formula_unit_multiplicity": True,
    "multiplicity_classifier_free_guidance": True,
}
FOUNDATION_CONDITION_SCHEMAS: dict[str, dict[str, Any]] = {
    FOUNDATION_MODEL_FAMILY: FOUNDATION_CONDITION_SCHEMA,
    FOUNDATION_MULTIPLICITY_MODEL_FAMILY: (
        FOUNDATION_MULTIPLICITY_CONDITION_SCHEMA
    ),
}


def foundation_uses_multiplicity(config: dict[str, Any]) -> bool:
    """Return and validate the checkpoint-family multiplicity contract."""

    family = config.get("model_family", FOUNDATION_MODEL_FAMILY)
    multiplicity_cfg = dict(config.get("multiplicity_condition", {}))
    enabled = bool(multiplicity_cfg.get("enabled", False))
    expected = family == FOUNDATION_MULTIPLICITY_MODEL_FAMILY
    if family not in FOUNDATION_MODEL_FAMILIES:
        raise ValueError(f"unsupported foundation model family {family!r}")
    if enabled is not expected:
        raise ValueError(
            f"{family} requires multiplicity_condition.enabled={expected}"
        )
    if enabled:
        if int(multiplicity_cfg.get("max_value", config.get("n_max", 160))) < 1:
            raise ValueError("multiplicity_condition.max_value must be positive")
        for key in ("drop_probability", "validation_drop_probability"):
            probability = float(multiplicity_cfg.get(key, 0.0))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"multiplicity_condition.{key} must be in [0, 1]")
    return enabled


def foundation_training_domains(config: dict[str, Any]) -> tuple[str, ...]:
    configured = config.get("training_domains", FOUNDATION_TRAINING_DOMAINS)
    if isinstance(configured, str):
        configured = (configured,)
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError(
            "training_domains must select organic, inorganic, or both"
        )
    domains = tuple(str(domain) for domain in configured)
    unsupported = sorted(set(domains) - set(FOUNDATION_TRAINING_DOMAINS))
    if unsupported:
        raise ValueError(
            "unsupported training_domains: " + ", ".join(unsupported)
        )
    if len(set(domains)) != len(domains):
        raise ValueError("training_domains must not contain duplicates")
    return tuple(domain for domain in FOUNDATION_TRAINING_DOMAINS if domain in domains)


def validate_foundation_checkpoint_contract(checkpoint: dict[str, Any]) -> None:
    family = checkpoint.get("model_family") or checkpoint.get("config", {}).get(
        "model_family"
    )
    if family not in FOUNDATION_MODEL_FAMILIES:
        raise ValueError(
            f"refusing checkpoint family {family!r}; "
            f"expected one of {FOUNDATION_MODEL_FAMILIES!r}"
        )
    expected_schema = FOUNDATION_CONDITION_SCHEMAS[family]
    schema = checkpoint.get("condition_schema")
    if not isinstance(schema, dict):
        raise ValueError("foundation checkpoint is missing condition_schema")
    for key in ("shared_tensor_fields", "organic_source_fields"):
        if tuple(schema.get(key, ())) != expected_schema[key]:
            raise ValueError(f"foundation checkpoint has unexpected {key}")
    for key in (
        "uses_repeat_dict_2",
        "uses_mol_atom_map_2_as_feature",
        "uses_mol_atom_degree_2_as_feature",
        "uses_domain_label_as_feature",
    ):
        if schema.get(key) is not False:
            raise ValueError(f"foundation checkpoint violates {key}=False")
    if schema.get("uses_target_atom_count") is not expected_schema[
        "uses_target_atom_count"
    ]:
        raise ValueError(
            "foundation checkpoint target-count exposure disagrees with its family"
        )
    if schema.get("composition_is_scale_free") is not True:
        raise ValueError("foundation checkpoint composition must be scale-free")
    if schema.get("uses_explicit_condition_element_counts") is not True:
        raise ValueError(
            "foundation checkpoint must use explicit condition element counts"
        )
    if schema.get("condition_element_count_sources") != (
        expected_schema["condition_element_count_sources"]
    ):
        raise ValueError(
            "foundation checkpoint has unexpected condition element-count sources"
        )
    if schema.get("uses_explicit_fragment_group_counts") is not True:
        raise ValueError(
            "foundation checkpoint must use explicit fragment group counts"
        )
    condition = schema.get("inorganic_condition")
    if condition not in INORGANIC_CONDITION_MODES:
        raise ValueError("foundation checkpoint has an invalid inorganic condition")
    configured_condition = checkpoint.get("config", {}).get(
        "inorganic_condition",
        INORGANIC_CONDITION,
    )
    if condition != configured_condition:
        raise ValueError(
            "foundation checkpoint condition schema disagrees with its config"
        )
    if schema.get("inorganic_fragment_source") != (
        "null_or_explicit_inference_fragment"
    ):
        raise ValueError("foundation checkpoint uses an unsafe inorganic fragment")
    config = checkpoint.get("config", {})
    foundation_uses_multiplicity(config)
    if family == FOUNDATION_MULTIPLICITY_MODEL_FAMILY:
        if schema.get("uses_optional_formula_unit_multiplicity") is not True:
            raise ValueError("multiplicity checkpoint is missing its input contract")
        if schema.get("multiplicity_classifier_free_guidance") is not True:
            raise ValueError("multiplicity checkpoint is missing its CFG contract")

def build_foundation_model_from_config(
    config: dict[str, Any],
) -> FoundationCrystalDiffusionModel:
    family = config.get("model_family", FOUNDATION_MODEL_FAMILY)
    if family not in FOUNDATION_MODEL_FAMILIES:
        raise ValueError(f"cannot build foundation DLM from family {family!r}")
    uses_multiplicity = foundation_uses_multiplicity(config)
    dlm_cfg = dict(config.get("dlm", {}))
    forbidden = sorted(
        set(dlm_cfg)
        & {"n_max", "d_jepa", "pxrd_conditioning", "pxrd_detail_encoder", "pxrd_detail"}
    )
    if forbidden:
        raise ValueError(
            "the foundation wrapper owns the external DLM condition; remove: "
            + ", ".join(forbidden)
        )
    fragment_cfg = dict(config.get("fragment_condition", {}))
    unsafe_fragment_keys = sorted(
        set(fragment_cfg)
        & {
            "inorganic_local_environment_atoms",
            "inorganic_periodic_image_radius",
            "fragment_seed",
        }
    )
    if unsafe_fragment_keys:
        raise ValueError(
            "target-derived inorganic fragments are forbidden; remove: "
            + ", ".join(unsafe_fragment_keys)
        )
    encoder_cfg = dict(fragment_cfg.get("encoder", {}))
    inorganic_condition = str(
        config.get("inorganic_condition", INORGANIC_CONDITION)
    )
    multiplicity_cfg = dict(config.get("multiplicity_condition", {}))
    return FoundationCrystalDiffusionModel(
        n_max=int(config.get("n_max", 160)),
        inorganic_condition=inorganic_condition,
        condition_dim=int(encoder_cfg.get("output_dim", 256)),
        pxrd_latent_dim=int(config.get("pxrd_latent_dim", 256)),
        fragment_hidden_dim=int(encoder_cfg.get("hidden_dim", 128)),
        fragment_num_layers=int(encoder_cfg.get("num_layers", 3)),
        fragment_num_heads=int(encoder_cfg.get("num_heads", 4)),
        fragment_num_rbf=int(encoder_cfg.get("num_rbf", 32)),
        fragment_cutoff=float(encoder_cfg.get("cutoff", 8.0)),
        fragment_dropout=float(encoder_cfg.get("dropout", 0.05)),
        multiplicity_conditioning=uses_multiplicity,
        multiplicity_max=int(multiplicity_cfg.get("max_value", config.get("n_max", 160))),
        **dlm_cfg,
    )


def _dataset_path(config: dict[str, Any], domain: str, split: str) -> str:
    split = "valid" if split == "val" else split
    if split == "train":
        candidates = (
            f"{domain}_lmdb_path",
            "mp20_lmdb_path" if domain == "inorganic" else "organic_lmdb_path",
        )
    else:
        candidates = (
            f"{domain}_{split}_lmdb_path",
            f"mp20_{split}_lmdb_path"
            if domain == "inorganic"
            else f"organic_{split}_lmdb_path",
        )
    for key in candidates:
        value = config.get(key)
        if value:
            return str(value)
    raise KeyError(f"missing {domain} LMDB path for split={split!r}")


def build_foundation_domain_datasets(
    config: dict[str, Any],
    split: str,
) -> tuple[FoundationOrganicDataset | None, FoundationInorganicDataset | None]:
    split = "valid" if split == "val" else split
    training_domains = foundation_training_domains(config)
    fragment_cfg = dict(config.get("fragment_condition", {}))
    fragment_n_max = int(fragment_cfg.get("max_atoms", 64))
    inorganic_condition = str(
        config.get("inorganic_condition", INORGANIC_CONDITION)
    )
    if inorganic_condition not in INORGANIC_CONDITION_MODES:
        raise ValueError(f"unsupported inorganic condition {inorganic_condition!r}")
    common: dict[str, Any] = {
        "n_max": int(config.get("n_max", 160)),
        "p_max": int(config.get("p_max", 150)),
        "wavelength": config.get("wavelength", "CuKa"),
        "two_theta_range": tuple(
            float(value)
            for value in config.get("two_theta_range", [5.0, 80.0])
        ),
        "augmentation": {"enabled": False},
        "use_conventional": bool(config.get("use_conventional", False)),
        "fragment_n_max": fragment_n_max,
        "emit_condition_multiplicity": foundation_uses_multiplicity(config),
    }
    organic = (
        FoundationOrganicDataset(
            _dataset_path(config, "organic", split),
            emit_null_pxrd=(inorganic_condition == "pxrd_plus_composition"),
            **common,
        )
        if "organic" in training_domains
        else None
    )
    inorganic = (
        FoundationInorganicDataset(
            _dataset_path(config, "inorganic", split),
            inorganic_condition=inorganic_condition,
            **common,
        )
        if "inorganic" in training_domains
        else None
    )
    return organic, inorganic


def build_balanced_foundation_dataset(
    config: dict[str, Any],
    split: str,
) -> BalancedMaterialsDataset:
    split = "valid" if split == "val" else split
    organic, inorganic = build_foundation_domain_datasets(config, split)
    mixture_cfg = dict(config.get("mixture", {}))
    sample_key = (
        "train_samples_per_domain"
        if split == "train"
        else "validation_samples_per_domain"
    )
    samples = mixture_cfg.get(sample_key)
    return BalancedMaterialsDataset(
        organic,
        inorganic,
        samples_per_domain=None if samples is None else int(samples),
    )


def foundation_formula_condition(
    clean_fractions: Tensor,
    noise_std: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Perturb an already scale-free composition without introducing count."""

    clean = clean_fractions.to(dtype=torch.float32)
    if clean.ndim != 2 or clean.shape[1] != NUM_ELEMENTS + 1:
        raise ValueError("condition_formula_fractions must have shape [B, 119]")
    noise_std = noise_std.to(device=clean.device, dtype=clean.dtype)
    if noise_std.shape != clean.shape[:1]:
        raise ValueError("formula noise_std must have shape [B]")
    if not torch.isfinite(noise_std).all() or torch.any(noise_std < 0):
        raise ValueError("formula noise_std must be finite and non-negative")
    if not torch.isfinite(clean).all() or torch.any(clean < 0):
        raise ValueError("condition formula fractions must be finite and non-negative")
    clean = clean.clone()
    clean[:, 0] = 0
    totals = clean.sum(dim=-1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError("each condition formula must contain an element")
    clean = clean / totals
    observed = clean * torch.exp(
        noise_std[:, None]
        * torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    )
    observed[:, 0] = 0
    observed = observed / observed.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return {
        "fractions": observed,
        "uncertainties": noise_std[:, None] * (clean > 0),
        "clean_fractions": clean,
    }


def _fragment_inputs(
    batch: dict[str, Any],
    *,
    drop_probability: float,
    generator: torch.Generator | None,
) -> tuple[dict[str, Tensor], Tensor]:
    missing = [name for name in FRAGMENT_TENSOR_KEYS if name not in batch]
    if missing:
        raise KeyError("foundation batch is missing: " + ", ".join(missing))
    if not 0.0 <= float(drop_probability) <= 1.0:
        raise ValueError("fragment_condition.drop_probability must be in [0, 1]")
    mask = batch["fragment_mask"].bool().clone()
    dropped = torch.zeros(mask.shape[0], device=mask.device, dtype=torch.bool)
    if drop_probability > 0:
        dropped = torch.rand(
            mask.shape[0], device=mask.device, generator=generator
        ) < float(drop_probability)
        # Remove atomistic geometry while retaining the explicitly allowed
        # element-count/composition condition.
        mask[dropped] = False
    return {
        "fragment_atom_types": batch["fragment_atom_types"],
        "fragment_coords": batch["fragment_coords"],
        "fragment_mask": mask,
        "fragment_group_indices": batch["fragment_group_indices"],
        "condition_element_counts": batch["condition_element_counts"],
    }, dropped


def _multiplicity_inputs(
    model: FoundationCrystalDiffusionModel,
    batch: dict[str, Any],
    *,
    drop_probability: float,
    generator: torch.Generator | None,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Prepare multiplicity-only classifier-free dropout for one batch."""

    batch_size = int(batch["atom_types"].shape[0])
    device = batch["atom_types"].device
    empty = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if not model.uses_multiplicity_conditioning:
        if float(drop_probability) != 0.0:
            raise ValueError(
                "multiplicity dropout requires a multiplicity-conditioned model"
            )
        return {}, empty, empty
    if not 0.0 <= float(drop_probability) <= 1.0:
        raise ValueError("multiplicity_condition.drop_probability must be in [0, 1]")
    missing = [name for name in MULTIPLICITY_TENSOR_KEYS if name not in batch]
    if missing:
        raise KeyError("foundation batch is missing: " + ", ".join(missing))
    multiplicity = batch["condition_multiplicity"]
    available = batch["condition_multiplicity_mask"].to(
        device=device,
        dtype=torch.bool,
    )
    if multiplicity.shape != (batch_size,) or available.shape != (batch_size,):
        raise ValueError("multiplicity batch fields must have shape [B]")
    dropped = empty
    if drop_probability > 0:
        dropped = available & (
            torch.rand(batch_size, device=device, generator=generator)
            < float(drop_probability)
        )
    present = available & ~dropped
    return {
        "condition_multiplicity": multiplicity,
        "condition_multiplicity_mask": present,
    }, available, dropped


def _pxrd_inputs(
    model: FoundationCrystalDiffusionModel,
    jepa_target: nn.Module | None,
    batch: dict[str, Any],
) -> dict[str, Tensor]:
    if not model.uses_pxrd_conditioning:
        return {}
    if jepa_target is None:
        raise ValueError(
            "jepa_target is required for pxrd_plus_composition training"
        )
    required = ("peak_d", "peak_i", "peak_mask", "condition_pxrd_mask")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError("PXRD foundation batch is missing: " + ", ".join(missing))
    present = batch["condition_pxrd_mask"].to(dtype=torch.bool)
    if present.shape != (batch["fragment_mask"].shape[0],):
        raise ValueError("condition_pxrd_mask must have shape [B]")
    batch_size = int(present.shape[0])

    parameter = next(model.parameters())
    latent = torch.zeros(
        batch_size,
        model.pxrd_latent_dim,
        device=present.device,
        dtype=parameter.dtype,
    )
    if present.any():
        encoded = jepa_target.encode_tgt(
            batch["peak_d"][present],
            batch["peak_i"][present],
            batch["peak_mask"][present],
        )
        present_count = int(present.sum().item())
        if encoded.shape != (present_count, model.pxrd_latent_dim):
            raise ValueError(
                "frozen JEPA output does not match foundation pxrd_latent_dim"
            )
        latent[present] = encoded.to(latent)
    return {"pxrd_latent": latent, "pxrd_mask": present}


def foundation_dlm_batch_loss(
    model: nn.Module,
    batch: dict[str, Any],
    corruption: CrystalDiffusionCorruption,
    *,
    jepa_target: nn.Module | None = None,
    loss_cfg: dict[str, Any] | None = None,
    formula_max_noise_std: float = 0.0,
    formula_clean_probability: float = 1.0,
    self_condition_probability: float = 0.5,
    fragment_drop_probability: float = 0.0,
    multiplicity_drop_probability: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute one mixed-domain loss with optional multiplicity conditioning."""

    if not 0.0 <= float(self_condition_probability) <= 1.0:
        raise ValueError("self_condition_probability must be in [0, 1]")
    base_model = unwrap_model(model)
    if not isinstance(base_model, FoundationCrystalDiffusionModel):
        raise TypeError("foundation trainer requires FoundationCrystalDiffusionModel")

    prepared = corruption(batch, generator=generator)
    batch_size = int(batch["atom_types"].shape[0])
    formula_noise_std = sample_formula_noise_stds(
        formula_max_noise_std,
        batch_size,
        batch["atom_types"].device,
        clean_probability=formula_clean_probability,
        generator=generator,
    )
    formula = foundation_formula_condition(
        batch["condition_formula_fractions"],
        formula_noise_std,
        generator=generator,
    )
    fragment_inputs, dropped = _fragment_inputs(
        batch,
        drop_probability=fragment_drop_probability,
        generator=generator,
    )
    multiplicity_inputs, multiplicity_available, multiplicity_dropped = (
        _multiplicity_inputs(
            base_model,
            batch,
            drop_probability=multiplicity_drop_probability,
            generator=generator,
        )
    )
    pxrd_inputs = _pxrd_inputs(base_model, jepa_target, batch)
    geometry_inputs: dict[str, Tensor] = {}
    if corruption.geometry_noise_exponent != 1.0:
        geometry_inputs["geometry_noise_level"] = prepared.t.pow(
            corruption.geometry_noise_exponent
        )
    common_inputs = (
        prepared.lattice_t,
        prepared.frac_coords_t,
        prepared.t,
        prepared.tokens_t,
        formula["fractions"],
        formula["uncertainties"],
    )

    self_conditioned = torch.zeros(
        batch_size, device=prepared.tokens_t.device, dtype=torch.bool
    )
    self_conditioning_logits = None
    if self_condition_probability > 0:
        with torch.no_grad():
            first_output = model(
                *common_inputs,
                **fragment_inputs,
                **multiplicity_inputs,
                **pxrd_inputs,
                **geometry_inputs,
            )
        self_conditioned = torch.rand(
            batch_size,
            device=prepared.tokens_t.device,
            generator=generator,
        ) < float(self_condition_probability)
        self_conditioning_logits = torch.where(
            self_conditioned[:, None, None],
            first_output["token_logits"].detach(),
            torch.zeros_like(first_output["token_logits"]),
        )
    output = model(
        *common_inputs,
        self_conditioning_logits=self_conditioning_logits,
        **fragment_inputs,
        **multiplicity_inputs,
        **pxrd_inputs,
        **geometry_inputs,
    )
    values = base_model.criterion(
        prepared.targets(),
        output,
        loss_cfg=loss_cfg,
        formula_fractions=formula["clean_fractions"],
        formula_uncertainties=formula["uncertainties"],
    )
    metrics = criterion_metrics(values)
    metrics["fragment_atoms_mean"] = float(
        batch["fragment_mask"].sum(dim=-1).float().mean().detach().cpu()
    )
    metrics["condition_atoms_mean"] = float(
        batch["condition_element_counts"]
        .sum(dim=-1)
        .float()
        .mean()
        .detach()
        .cpu()
    )
    metrics["fragment_condition_dropped_fraction"] = float(
        dropped.float().mean().detach().cpu()
    )
    available_count = multiplicity_available.sum().clamp_min(1)
    metrics["multiplicity_available_fraction"] = float(
        multiplicity_available.float().mean().detach().cpu()
    )
    metrics["multiplicity_condition_dropped_fraction"] = float(
        (multiplicity_dropped.sum() / available_count).detach().cpu()
    )
    if multiplicity_available.any():
        metrics["multiplicity_mean"] = float(
            batch["condition_multiplicity"][multiplicity_available]
            .float()
            .mean()
            .detach()
            .cpu()
        )
    else:
        metrics["multiplicity_mean"] = 0.0
    metrics["formula_noise_std_avg"] = float(
        formula_noise_std.mean().detach().cpu()
    )
    metrics["self_conditioned_fraction"] = float(
        self_conditioned.float().mean().detach().cpu()
    )
    domains = batch.get("material_domain")
    if isinstance(domains, list) and domains:
        metrics["organic_fraction"] = sum(
            str(domain) == "organic" for domain in domains
        ) / len(domains)
    return values["loss"], metrics


def _run_epoch(
    model: nn.Module,
    jepa_target: nn.Module | None,
    loader: DataLoader,
    corruption: CrystalDiffusionCorruption,
    device: torch.device,
    config: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scheduler_interval: str = "epoch",
    epoch: int = 0,
    max_epochs: int = 1,
    generator: torch.Generator | None = None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    metric_totals: dict[str, float] = {}
    examples = 0
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    iterator = tqdm(
        loader,
        desc=(
            f"Foundation DLM epoch {epoch + 1}/{max_epochs}"
            if training
            else "Foundation DLM validation"
        ),
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not is_main_process(),
    )
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in iterator:
            batch = {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            formula_cfg = dict(config.get("formula", {}))
            fragment_cfg = dict(config.get("fragment_condition", {}))
            multiplicity_cfg = dict(config.get("multiplicity_condition", {}))
            loss, metrics = foundation_dlm_batch_loss(
                model,
                batch,
                corruption,
                jepa_target=jepa_target,
                loss_cfg=config.get("loss", {}),
                formula_max_noise_std=float(
                    formula_cfg.get(
                        "max_relative_noise_std"
                        if training
                        else "validation_max_relative_noise_std",
                        0.0,
                    )
                ),
                formula_clean_probability=(
                    float(formula_cfg.get("clean_probability", 1.0))
                    if training
                    else 1.0
                ),
                self_condition_probability=float(
                    config.get("diffusion", {}).get(
                        "self_condition_probability"
                        if training
                        else "validation_self_condition_probability",
                        config.get("diffusion", {}).get(
                            "self_condition_probability", 0.5
                        ),
                    )
                ),
                fragment_drop_probability=(
                    float(fragment_cfg.get("drop_probability", 0.0))
                    if training
                    else float(
                        fragment_cfg.get("validation_drop_probability", 0.0)
                    )
                ),
                multiplicity_drop_probability=(
                    float(multiplicity_cfg.get("drop_probability", 0.0))
                    if training
                    else float(
                        multiplicity_cfg.get(
                            "validation_drop_probability",
                            0.0,
                        )
                    )
                ),
                generator=generator,
            )
            if optimizer is not None:
                loss.backward()
                grad_clip = float(config.get("gradient_clip_norm", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
                optimizer.step()
                if scheduler is not None and scheduler_interval == "step":
                    scheduler.step()
            batch_size = int(batch["atom_types"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            for name, value in metrics.items():
                metric_totals[name] = (
                    metric_totals.get(name, 0.0) + float(value) * batch_size
                )
            examples += batch_size
            iterator.set_postfix(
                loss=f"{total_loss / examples:.4f}",
                count_mae=(
                    f"{metric_totals.get('atom_count_mae', 0.0) / examples:.3f}"
                ),
                organic=(
                    f"{metric_totals.get('organic_fraction', 0.0) / examples:.2f}"
                ),
            )
    if training and scheduler is not None and scheduler_interval == "epoch":
        scheduler.step()
    return reduce_epoch_statistics(total_loss, metric_totals, examples, device)


def make_validation_generator(
    device: torch.device,
    seed: int,
    *,
    rank: int = 0,
) -> torch.Generator:
    """Create the same rank-local validation corruption stream every epoch."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(rank))
    return generator


def initialize_backbone_from_checkpoint(
    model: FoundationCrystalDiffusionModel,
    checkpoint_path: str | Path,
) -> tuple[int, int]:
    """Copy shape-compatible DLM weights, never the old condition projection."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    if not isinstance(source, dict):
        raise TypeError("initial backbone checkpoint has no model state dictionary")
    source = migrate_condition_encoder_state_dict({
        (key.removeprefix("module.")): value for key, value in source.items()
    })
    target = model.backbone.state_dict()
    loaded = 0
    for key, target_value in list(target.items()):
        if is_pxrd_adapter_key(key):
            continue
        candidate = source.get(key)
        if candidate is None:
            candidate = source.get(f"backbone.{key}")
        if isinstance(candidate, Tensor) and candidate.shape == target_value.shape:
            target[key] = candidate
            loaded += 1
    if loaded < 1:
        raise ValueError(
            f"no shape-compatible DLM backbone tensors found in {checkpoint_path}"
        )
    model.backbone.load_state_dict(target)
    return loaded, len(target)


def initialize_foundation_from_checkpoint(
    model: FoundationCrystalDiffusionModel,
    checkpoint_path: str | Path,
) -> tuple[int, int]:
    """Warm-start a multiplicity model from every compatible v2 tensor."""

    if not model.uses_multiplicity_conditioning:
        raise ValueError(
            "full foundation initialization is reserved for the multiplicity model"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_foundation_checkpoint_contract(checkpoint)
    source_family = checkpoint.get("model_family") or checkpoint.get(
        "config", {}
    ).get("model_family")
    if source_family != FOUNDATION_MODEL_FAMILY:
        raise ValueError("multiplicity conditioning must initialize from foundation v2")
    source = checkpoint.get("model")
    if not isinstance(source, dict):
        raise TypeError("initial foundation checkpoint has no model state dictionary")
    source = migrate_condition_encoder_state_dict(
        {key.removeprefix("module."): value for key, value in source.items()}
    )
    target = model.state_dict()
    loaded = 0
    missing_required: list[str] = []
    new_prefix = "condition_encoder.multiplicity_conditioner."
    for key, target_value in list(target.items()):
        if key.startswith(new_prefix):
            continue
        candidate = source.get(key)
        if isinstance(candidate, Tensor) and candidate.shape == target_value.shape:
            target[key] = candidate
            loaded += 1
        else:
            missing_required.append(key)
    if missing_required:
        preview = ", ".join(missing_required[:5])
        raise ValueError("foundation v2 initializer is missing tensors: " + preview)
    model.load_state_dict(target)
    return loaded, len(target)


def _restore_early_stopper(stopper: Any, state: dict[str, Any] | None) -> None:
    if stopper is None or not state:
        return
    for name in ("best", "best_epoch", "wait", "stopped"):
        if name in state:
            setattr(stopper, name, state[name])


def _load_resume_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    early_stopper: Any,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    validate_foundation_checkpoint_contract(checkpoint)
    unwrap_model(model).load_state_dict(
        migrate_condition_encoder_state_dict(checkpoint["model"])
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    _restore_early_stopper(early_stopper, checkpoint.get("early_stopping"))
    best = float(checkpoint.get("best", checkpoint.get("monitor_loss", float("inf"))))
    return int(checkpoint.get("epoch", -1)) + 1, best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-backbone", default=None)
    parser.add_argument("--init-foundation", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    selected_initializers = sum(
        value is not None
        for value in (args.resume, args.init_backbone, args.init_foundation)
    )
    if selected_initializers > 1:
        raise ValueError(
            "--resume, --init-backbone, and --init-foundation are mutually exclusive"
        )
    config = load_config(args.config)
    configured_family = config.get("model_family", FOUNDATION_MODEL_FAMILY)
    if configured_family not in FOUNDATION_MODEL_FAMILIES:
        raise ValueError(
            f"config model_family={configured_family!r} is not foundation DLM"
        )
    model_family = str(configured_family)
    config["model_family"] = model_family
    uses_multiplicity = foundation_uses_multiplicity(config)
    for argument, key in (
        (args.output_dir, "output_dir"),
        (args.batch_size, "batch_size"),
        (args.learning_rate, "learning_rate"),
        (args.max_epochs, "max_epochs"),
    ):
        if argument is not None:
            config[key] = argument
    if int(config.get("batch_size", 1)) < 1:
        raise ValueError("batch_size must be positive")
    if float(config.get("learning_rate", 0.0)) <= 0:
        raise ValueError("learning_rate must be positive")
    if int(config.get("max_epochs", 1)) < 1:
        raise ValueError("max_epochs must be positive")
    training_domains = foundation_training_domains(config)
    config["training_domains"] = list(training_domains)
    inorganic_condition = str(
        config.get("inorganic_condition", INORGANIC_CONDITION)
    )
    if inorganic_condition not in INORGANIC_CONDITION_MODES:
        raise ValueError(f"unsupported inorganic condition {inorganic_condition!r}")
    config["inorganic_condition"] = inorganic_condition
    if inorganic_condition == "composition_only" and config.get("jepa_ckpt"):
        raise ValueError(
            "composition_only must not configure a JEPA/PXRD checkpoint"
        )

    diffusion_cfg = dict(config.get("diffusion", {}))
    sampling_cfg = dict(config.get("sampling", {}))
    for name in ("token_noise_exponent", "geometry_noise_exponent"):
        train_value = float(diffusion_cfg.get(name, 1.0))
        sample_value = float(sampling_cfg.get(name, 1.0))
        if train_value != sample_value:
            raise ValueError(
                f"diffusion.{name}={train_value} must match "
                f"sampling.{name}={sample_value}"
            )

    output_dir = Path(
        config.get(
            "output_dir",
            (
                "weights/foundation_dlm_organic_multiplicity_cfg_v1"
                if uses_multiplicity
                else "weights/foundation_dlm_organic_inorganic_v2"
            ),
        )
    )
    if args.resume is not None:
        resume_path = Path(args.resume)
        if output_dir.resolve() != resume_path.resolve().parent:
            raise ValueError("a foundation resume checkpoint must be in output_dir")
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        validate_foundation_checkpoint_contract(resume_checkpoint)
        resume_family = resume_checkpoint.get("model_family") or (
            resume_checkpoint.get("config", {}).get("model_family")
        )
        if resume_family != model_family:
            raise ValueError("resume checkpoint family does not match the config")
    else:
        existing = sorted(output_dir.glob("*.ckpt"))
        if existing:
            raise FileExistsError(
                "refusing to overwrite foundation checkpoints: "
                + ", ".join(str(path) for path in existing)
            )

    rank, local_rank, world_size = setup_distributed(
        config.get("distributed_backend")
    )
    wandb_run = None
    try:
        seed = int(config.get("seed", 42))
        validation_seed = int(config.get("validation_seed", seed + 1_000_003))
        seed_everything(seed + rank)
        if dist.is_initialized():
            device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = select_device(str(config.get("device", "auto")))

        jepa_target = None
        if inorganic_condition == "pxrd_plus_composition":
            jepa_checkpoint = config.get("jepa_ckpt")
            if not jepa_checkpoint:
                raise ValueError(
                    "pxrd_plus_composition requires a frozen jepa_ckpt"
                )
            jepa_target = load_jepa_target_encoder(jepa_checkpoint, device)
            checkpoint_dim = int(jepa_target.d_jepa)
            configured_dim = config.get("pxrd_latent_dim")
            if configured_dim is not None and int(configured_dim) != checkpoint_dim:
                raise ValueError(
                    f"pxrd_latent_dim={configured_dim} does not match "
                    f"{jepa_checkpoint} d_jepa={checkpoint_dim}"
                )
            config["pxrd_latent_dim"] = checkpoint_dim

        train_dataset = build_balanced_foundation_dataset(config, "train")
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
            batch_size=int(config.get("batch_size", 128)),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=int(config.get("num_workers", 4)),
            drop_last=bool(config.get("drop_last", False)),
            collate_fn=collate_crystal_batch,
        )

        val_loader = None
        val_dataset: BalancedMaterialsDataset | None = None
        try:
            val_dataset = build_balanced_foundation_dataset(config, "valid")
        except KeyError:
            val_dataset = None
        if val_dataset is not None:
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
                    config.get("eval_batch_size", config.get("batch_size", 128))
                ),
                shuffle=False,
                sampler=val_sampler,
                num_workers=int(config.get("num_workers", 4)),
                collate_fn=collate_crystal_batch,
            )

        target_model = build_foundation_model_from_config(config)
        initial_backbone_checkpoint = args.init_backbone or config.get(
            "initial_backbone_checkpoint"
        )
        initial_foundation_checkpoint = args.init_foundation or config.get(
            "initial_foundation_checkpoint"
        )
        if (
            not args.resume
            and initial_backbone_checkpoint
            and initial_foundation_checkpoint
        ):
            raise ValueError(
                "initial_backbone_checkpoint and initial_foundation_checkpoint "
                "are mutually exclusive"
            )
        if uses_multiplicity and not args.resume and not initial_foundation_checkpoint:
            raise ValueError(
                "multiplicity conditioning requires initial_foundation_checkpoint"
            )
        initialized = None
        initialization_kind = None
        initial_checkpoint = None
        if initial_foundation_checkpoint and not args.resume:
            initial_checkpoint = initial_foundation_checkpoint
            initialization_kind = "foundation"
            initialized = initialize_foundation_from_checkpoint(
                target_model,
                initial_foundation_checkpoint,
            )
        elif initial_backbone_checkpoint and not args.resume:
            initial_checkpoint = initial_backbone_checkpoint
            initialization_kind = "backbone"
            initialized = initialize_backbone_from_checkpoint(
                target_model,
                initial_backbone_checkpoint,
            )
        model: nn.Module = target_model.to(device)
        if dist.is_initialized():
            ddp_kwargs: dict[str, Any] = {
                "broadcast_buffers": False,
                "find_unused_parameters": bool(
                    config.get("ddp_find_unused_parameters", False)
                ),
            }
            if device.type == "cuda":
                ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
            model = DistributedDataParallel(model, **ddp_kwargs)
        target_model = unwrap_model(model)
        parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config.get("learning_rate", 1.0e-4)),
            weight_decay=float(config.get("weight_decay", 1.0e-2)),
        )
        max_epochs = int(config.get("max_epochs", 150))
        scheduler, scheduler_interval = build_lr_scheduler(
            optimizer,
            config,
            max_epochs=max_epochs,
            steps_per_epoch=max(len(train_loader), 1),
        )
        early_stopper = build_early_stopper(config)
        monitor_metric = str(config.get("monitor_metric", "loss"))
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

        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, output_dir / "training_config.yaml")
            print(
                f"model_family={model_family} "
                f"trainable_parameters={sum(p.numel() for p in parameters)} "
                f"output_dir={output_dir} domains={'+'.join(training_domains)} "
                "shared_backbone=CrystalDiffusionLanguageModel "
                f"monitor_metric={monitor_metric} "
                f"multiplicity_conditioned={str(uses_multiplicity).lower()} "
                "condition_element_counts=fragment_or_reduced_composition "
                f"inorganic_condition={inorganic_condition}"
            )
            if initialized is not None:
                print(
                    f"initialized_{initialization_kind}={initial_checkpoint} "
                    f"loaded_tensors={initialized[0]}/{initialized[1]} "
                    f"multiplicity_adapter_initialized={str(uses_multiplicity).lower()}"
                )
        if dist.is_initialized():
            dist.barrier()

        start_epoch = 0
        best = float("inf")
        if args.resume:
            start_epoch, best = _load_resume_checkpoint(
                args.resume,
                model,
                optimizer,
                scheduler,
                early_stopper,
                device,
            )
            if is_main_process():
                print(
                    f"resumed foundation DLM from {args.resume} "
                    f"at epoch={start_epoch} best={best:.6f}"
                )

        if is_main_process():
            wandb_run = init_wandb(
                config,
                job_type="universal-materials-fragment-dlm",
                output_dir=output_dir,
                default_name=output_dir.name,
            )
            maybe_watch_model(wandb_run, target_model, config)

        for epoch in range(start_epoch, max_epochs):
            train_dataset.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss, train_metrics = _run_epoch(
                model,
                jepa_target,
                train_loader,
                corruption,
                device,
                config,
                optimizer=optimizer,
                scheduler=scheduler,
                scheduler_interval=scheduler_interval,
                epoch=epoch,
                max_epochs=max_epochs,
            )
            if val_loader is None:
                val_loss, val_metrics = None, {}
            else:
                if val_dataset is not None:
                    val_dataset.set_epoch(0)
                val_loss, val_metrics = _run_epoch(
                    model,
                    jepa_target,
                    val_loader,
                    corruption,
                    device,
                    config,
                    optimizer=None,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    generator=make_validation_generator(
                        device,
                        validation_seed,
                        rank=rank,
                    ),
                )
            if monitor_metric == "loss":
                monitor_loss = val_loss if val_loss is not None else train_loss
            else:
                monitor_values = val_metrics if val_loss is not None else train_metrics
                if monitor_metric not in monitor_values:
                    raise KeyError(
                        f"monitor_metric={monitor_metric!r} was not emitted by "
                        "the training criterion"
                    )
                monitor_loss = float(monitor_values[monitor_metric])
            stop_now = (
                early_stopper.step(monitor_loss, epoch)
                if early_stopper is not None
                else False
            )
            if early_stopper is None:
                is_best = monitor_loss < best
                if is_best:
                    best = monitor_loss
            else:
                is_best = early_stopper.best_epoch == epoch
                best = (
                    float(early_stopper.best)
                    if early_stopper.best is not None
                    else float("inf")
                )
            if is_main_process():
                checkpoint: dict[str, Any] = {
                    "model_family": model_family,
                    "condition_schema": {
                        **FOUNDATION_CONDITION_SCHEMAS[model_family],
                        "inorganic_condition": inorganic_condition,
                    },
                    "model": target_model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "loss": train_loss,
                    "metrics": train_metrics,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "monitor_loss": monitor_loss,
                    "monitor_metric": monitor_metric,
                    "best": best,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                }
                if early_stopper is not None:
                    checkpoint["early_stopping"] = early_stopper.state_dict()
                torch.save(checkpoint, output_dir / "last.ckpt")
                if is_best:
                    torch.save(checkpoint, output_dir / "best.ckpt")
                learning_rate = float(optimizer.param_groups[0]["lr"])
                extras = (
                    f" organic_fraction={train_metrics.get('organic_fraction', 0.0):.6f}"
                    f" fragment_atoms={train_metrics.get('fragment_atoms_mean', 0.0):.6f}"
                    f" condition_atoms={train_metrics.get('condition_atoms_mean', 0.0):.6f}"
                    " multiplicity_available="
                    f"{train_metrics.get('multiplicity_available_fraction', 0.0):.6f}"
                    " multiplicity_dropped="
                    f"{train_metrics.get('multiplicity_condition_dropped_fraction', 0.0):.6f}"
                )
                print(
                    format_epoch_log(
                        epoch,
                        train_loss,
                        val_loss,
                        train_metrics,
                        val_metrics,
                        learning_rate,
                    )
                    + extras
                )
                payload = wandb_metrics(
                    train_loss,
                    val_loss,
                    train_metrics,
                    val_metrics,
                    learning_rate,
                )
                payload.update(
                    {
                        "train/organic_fraction": train_metrics.get(
                            "organic_fraction", 0.0
                        ),
                        "train/fragment_atoms_mean": train_metrics.get(
                            "fragment_atoms_mean", 0.0
                        ),
                        "train/condition_atoms_mean": train_metrics.get(
                            "condition_atoms_mean", 0.0
                        ),
                        "val/condition_atoms_mean": val_metrics.get(
                            "condition_atoms_mean", 0.0
                        ),
                        "train/multiplicity_available_fraction": train_metrics.get(
                            "multiplicity_available_fraction", 0.0
                        ),
                        "train/multiplicity_condition_dropped_fraction": (
                            train_metrics.get(
                                "multiplicity_condition_dropped_fraction", 0.0
                            )
                        ),
                        "train/multiplicity_mean": train_metrics.get(
                            "multiplicity_mean", 0.0
                        ),
                        "val/multiplicity_available_fraction": val_metrics.get(
                            "multiplicity_available_fraction", 0.0
                        ),
                    }
                )
                log_wandb(
                    wandb_run,
                    payload,
                    step=(epoch + 1) * max(len(train_loader), 1),
                )
            if dist.is_initialized():
                stop_tensor = torch.tensor(
                    int(stop_now), device=device, dtype=torch.int32
                )
                dist.broadcast(stop_tensor, src=0)
                stop_now = bool(stop_tensor.item())
            if stop_now:
                if is_main_process():
                    print(f"early stopping at epoch={epoch}")
                break
    finally:
        if is_main_process():
            finish_wandb(wandb_run)
        cleanup_distributed()


if __name__ == "__main__":
    main()
