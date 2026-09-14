"""Train the existing DLM with an organic molecular-template condition.

This entry point is deliberately separate from :mod:`train_dllm`, but it uses
the exact same :class:`CrystalDiffusionLanguageModel`. It deterministically
vectorizes only ``mol_atom_type_2`` and its template-group boundaries, then
mixes that vector through the DLM condition encoder.
Unit-cell repeats and the target atom count are excluded, so the DLM must
predict EOS. There is no new denoising architecture or PXRD/JEPA encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
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
from lvebcsp.data.collate import collate_crystal_batch
from lvebcsp.data.organic_lmdb import OrganicLMDBDataset
from lvebcsp.models.canvas import NUM_ELEMENTS
from lvebcsp.models.dlm import CrystalDiffusionLanguageModel
from lvebcsp.train.dllm_checkpoint import migrate_condition_encoder_state_dict
from lvebcsp.train.train_dllm import (
    CrystalDiffusionCorruption,
    criterion_metrics,
    deterministic_dataset_subset,
    format_epoch_log,
    reduce_epoch_statistics,
    sample_formula_noise_stds,
    wandb_metrics,
)
from lvebcsp.train.train_jepa import build_early_stopper, build_lr_scheduler


ORGANIC_TEMPLATE_MODEL_FAMILY = "organic_mol_template_dlm_v2"
ORGANIC_TEMPLATE_CONDITION_SCHEMA = {
    "source_fields": ("mol_atom_type_2", "mol2_group_slices"),
    "uses_repeat_dict_2": False,
    "uses_target_atom_count": False,
}


MOLECULE_TEMPLATE_KEYS = (
    "molecule_template_atom_types",
    "molecule_template_mask",
    "molecule_template_group_indices",
)


def validate_organic_checkpoint_contract(checkpoint: dict[str, Any]) -> None:
    """Reject checkpoints that do not prove the template-only contract."""

    family = checkpoint.get("model_family") or checkpoint.get("config", {}).get(
        "model_family"
    )
    if family != ORGANIC_TEMPLATE_MODEL_FAMILY:
        raise ValueError(
            f"refusing organic checkpoint family {family!r}; "
            f"expected {ORGANIC_TEMPLATE_MODEL_FAMILY!r}"
        )
    schema = checkpoint.get("condition_schema")
    if not isinstance(schema, dict):
        raise ValueError("organic checkpoint is missing condition_schema")
    source_fields = tuple(schema.get("source_fields", ()))
    if source_fields != ORGANIC_TEMPLATE_CONDITION_SCHEMA["source_fields"]:
        raise ValueError(
            f"organic checkpoint has unexpected condition fields {source_fields!r}"
        )
    if schema.get("uses_repeat_dict_2") is not False:
        raise ValueError("organic checkpoint condition must exclude repeat_dict_2")
    if schema.get("uses_target_atom_count") is not False:
        raise ValueError("organic checkpoint condition must exclude target atom count")


def molecule_template_inputs(batch: dict[str, Any]) -> dict[str, Tensor]:
    missing = [name for name in MOLECULE_TEMPLATE_KEYS if name not in batch]
    if missing:
        raise KeyError(
            "organic molecular-template batch is missing: " + ", ".join(missing)
        )
    return {name: batch[name] for name in MOLECULE_TEMPLATE_KEYS}


def molecular_condition_dim(max_groups: int) -> int:
    """Return the deterministic molecular-condition vector width.

    Each group contributes only its one-copy element counts and a presence
    flag. The suffix contains aggregate template counts, template size, and
    number of unique template groups. None is a unit-cell atom count.
    """

    max_groups = int(max_groups)
    if max_groups < 1:
        raise ValueError("molecule_condition.max_groups must be positive")
    element_width = NUM_ELEMENTS + 1
    return max_groups * (element_width + 1) + element_width + 2


def molecular_condition_from_batch(
    batch: dict[str, Any],
    *,
    max_groups: int,
) -> tuple[Tensor, Tensor]:
    """Vectorize ``mol_atom_type_2`` without a second learned model.

    Returns ``(condition, template_element_counts)``. The second value is also
    used to form the DLM's elemental-fraction condition, so neither condition
    is derived from the target crystal atom list or unit-cell multiplicities.
    """

    fields = molecule_template_inputs(batch)
    atom_types = fields["molecule_template_atom_types"].long()
    mask = fields["molecule_template_mask"].bool()
    group_indices = fields["molecule_template_group_indices"].long()
    if atom_types.ndim != 2:
        raise ValueError("molecule_template_atom_types must have shape [B, M]")
    expected = atom_types.shape
    for name, value in (
        ("molecule_template_mask", mask),
        ("molecule_template_group_indices", group_indices),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    batch_size = atom_types.shape[0]
    max_groups = int(max_groups)
    molecular_condition_dim(max_groups)
    template_atoms = mask.sum(dim=-1)
    if torch.any(template_atoms < 1):
        raise ValueError("each molecular template must contain at least one atom")
    active_types = atom_types[mask]
    if torch.any((active_types < 1) | (active_types > NUM_ELEMENTS)):
        raise ValueError("active molecular-template atoms must be in [1, 118]")
    if torch.any(group_indices[mask] < 1) or torch.any(
        group_indices[mask] > max_groups
    ):
        raise ValueError("active molecular-template group index is out of range")

    dtype = torch.float32
    active_atom_types = torch.where(mask, atom_types, torch.zeros_like(atom_types))
    atom_one_hot = F.one_hot(
        active_atom_types,
        num_classes=NUM_ELEMENTS + 1,
    ).to(dtype)
    atom_one_hot[..., 0] = 0
    group_slots = torch.where(
        mask,
        group_indices - 1,
        torch.zeros_like(group_indices),
    )
    scatter_index = group_slots.unsqueeze(-1).expand_as(atom_one_hot)
    group_counts = torch.zeros(
        batch_size,
        max_groups,
        NUM_ELEMENTS + 1,
        device=atom_types.device,
        dtype=dtype,
    )
    group_counts.scatter_add_(1, scatter_index, atom_one_hot)
    group_atom_totals = group_counts.sum(dim=-1)
    group_mask = group_atom_totals > 0
    num_groups = group_mask.sum(dim=-1)
    template_counts = group_counts.sum(dim=1)

    group_features = torch.cat(
        (
            torch.log1p(group_counts),
            group_mask.to(dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    global_features = torch.cat(
        (
            torch.log1p(template_counts),
            torch.log1p(template_atoms.to(dtype)).unsqueeze(-1),
            torch.log1p(num_groups.to(dtype)).unsqueeze(-1),
        ),
        dim=-1,
    )
    condition = torch.cat(
        (group_features.flatten(start_dim=1), global_features),
        dim=-1,
    )
    expected_dim = molecular_condition_dim(max_groups)
    if condition.shape != (batch_size, expected_dim):
        raise AssertionError(
            f"molecular condition has shape {tuple(condition.shape)}, "
            f"expected {(batch_size, expected_dim)}"
        )
    return condition, template_counts


def molecular_formula_condition(
    element_counts: Tensor,
    noise_std: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Create DLM formula tensors from molecular metadata, not the target."""

    if element_counts.ndim != 2 or element_counts.shape[1] != NUM_ELEMENTS + 1:
        raise ValueError("molecular element counts must have shape [B, 119]")
    noise_std = noise_std.to(device=element_counts.device, dtype=torch.float32)
    if noise_std.shape != element_counts.shape[:1]:
        raise ValueError("formula noise_std must have shape [B]")
    clean = element_counts.to(torch.float32).clone()
    observed = clean * torch.exp(
        noise_std[:, None]
        * torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    )
    clean[:, 0] = 0
    observed[:, 0] = 0
    clean_totals = clean.sum(dim=-1, keepdim=True)
    observed_totals = observed.sum(dim=-1, keepdim=True)
    if torch.any(clean_totals <= 0) or torch.any(observed_totals <= 0):
        raise ValueError("each molecular condition must contain at least one element")
    present = clean > 0
    return {
        "fractions": observed / observed_totals,
        "uncertainties": noise_std[:, None] * present,
        "clean_fractions": clean / clean_totals,
    }


def build_organic_dllm_from_config(
    config: dict[str, Any],
) -> CrystalDiffusionLanguageModel:
    """Build the repository's existing DLM with its vector condition enabled."""

    model_cfg = dict(config.get("dlm", {}))
    model_cfg.pop("max_formula_tokens", None)
    forbidden = sorted(
        key
        for key in ("pxrd_conditioning", "pxrd_detail", "jepa_ckpt", "d_jepa")
        if key in model_cfg
    )
    if forbidden:
        raise ValueError(
            "the organic trainer owns the external condition input; remove: "
            + ", ".join(forbidden)
        )
    n_max = int(model_cfg.pop("n_max", config.get("n_max", 160)))
    condition_cfg = dict(config.get("molecule_condition", {}))
    max_groups = int(condition_cfg.get("max_groups", 8))
    model = CrystalDiffusionLanguageModel(
        n_max=n_max,
        d_jepa=molecular_condition_dim(max_groups),
        pxrd_conditioning=True,
        **model_cfg,
    )
    # The base class zero-initializes this attachment for stage-two PXRD
    # finetuning. This is a from-scratch molecular-conditioned run, so allow
    # the condition to influence the first update while keeping its scale low.
    if not bool(condition_cfg.get("zero_init_projection", False)):
        if model.condition_encoder.z_proj is None:
            raise AssertionError("DLM generic condition projection is missing")
        nn.init.xavier_uniform_(
            model.condition_encoder.z_proj[-1].weight,
            gain=0.1,
        )
        nn.init.zeros_(model.condition_encoder.z_proj[-1].bias)
    return model


def build_organic_dataset(
    config: dict[str, Any],
    split: str,
) -> OrganicLMDBDataset:
    split = "valid" if split == "val" else split
    key = "organic_lmdb_path" if split == "train" else f"organic_{split}_lmdb_path"
    path = config.get(key)
    if not path:
        raise KeyError(f"missing required organic dataset path: {key}")
    if config.get("mp20_lmdb_path") or config.get(f"mp20_{split}_lmdb_path"):
        raise ValueError("the organic trainer refuses MP-20/inorganic dataset paths")
    condition_cfg = dict(config.get("molecule_condition", {}))
    template_n_max = int(
        condition_cfg.get("max_template_atoms", config.get("n_max", 160))
    )
    two_theta_range = config.get("two_theta_range", [5.0, 80.0])
    return OrganicLMDBDataset(
        path,
        n_max=int(config.get("n_max", 160)),
        p_max=int(config.get("p_max", 150)),
        wavelength=config.get("wavelength", "CuKa"),
        two_theta_range=(float(two_theta_range[0]), float(two_theta_range[1])),
        augmentation={"enabled": False},
        use_conventional=bool(config.get("use_conventional", False)),
        include_pxrd=False,
        include_molecule_template=True,
        molecule_template_n_max=template_n_max,
    )


def organic_dllm_batch_loss(
    model: nn.Module,
    batch: dict[str, Tensor],
    corruption: CrystalDiffusionCorruption,
    *,
    molecule_condition_max_groups: int = 8,
    loss_cfg: dict[str, Any] | None = None,
    formula_max_noise_std: float = 0.0,
    formula_clean_probability: float = 1.0,
    self_condition_probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute one organic molecular-template-conditioned denoising loss."""

    if not 0.0 <= float(self_condition_probability) <= 1.0:
        raise ValueError("self_condition_probability must be in [0, 1]")
    base_model = unwrap_model(model)
    if not isinstance(base_model, CrystalDiffusionLanguageModel):
        raise TypeError("organic trainer requires CrystalDiffusionLanguageModel")
    expected_condition_dim = molecular_condition_dim(molecule_condition_max_groups)
    if not base_model.uses_pxrd_conditioning or base_model.d_jepa != expected_condition_dim:
        raise ValueError(
            "DLM condition encoder does not match molecule_condition.max_groups"
        )
    prepared = corruption(batch, generator=generator)
    formula_noise_std = sample_formula_noise_stds(
        formula_max_noise_std,
        batch["atom_types"].shape[0],
        batch["atom_types"].device,
        clean_probability=formula_clean_probability,
        generator=generator,
    )
    molecular_condition, molecular_counts = molecular_condition_from_batch(
        batch,
        max_groups=molecule_condition_max_groups,
    )
    formula = molecular_formula_condition(
        molecular_counts,
        formula_noise_std,
        generator=generator,
    )
    c = base_model.encode_condition(
        formula["fractions"],
        formula["uncertainties"],
        z_jepa=molecular_condition,
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
        with torch.no_grad():
            first_output = model(
                *model_inputs,
                **geometry_time_inputs,
            )
        self_conditioned = torch.rand(
            prepared.tokens_t.shape[0],
            device=prepared.tokens_t.device,
            generator=generator,
        ) < float(self_condition_probability)
        self_conditioning_logits = torch.where(
            self_conditioned[:, None, None],
            first_output["token_logits"].detach(),
            torch.zeros_like(first_output["token_logits"]),
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

    metrics["molecule_template_atoms_mean"] = float(
        batch["molecule_template_mask"]
        .sum(dim=-1)
        .float()
        .mean()
        .detach()
        .cpu()
    )
    metrics["formula_noise_std_avg"] = float(
        formula_noise_std.mean().detach().cpu()
    )
    metrics["self_conditioned_fraction"] = float(
        self_conditioned.float().mean().detach().cpu()
    )
    return loss, metrics


def _run_epoch(
    model: nn.Module,
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
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    metric_totals: dict[str, float] = {}
    batches = 0
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    iterator = tqdm(
        loader,
        desc=(f"Organic DLM epoch {epoch + 1}/{max_epochs}" if training else "Organic DLM validation"),
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
            condition_cfg = dict(config.get("molecule_condition", {}))
            loss, metrics = organic_dllm_batch_loss(
                model,
                batch,
                corruption,
                molecule_condition_max_groups=int(
                    condition_cfg.get("max_groups", 8)
                ),
                loss_cfg=config.get("loss", {}),
                formula_max_noise_std=float(
                    formula_cfg.get(
                        "max_relative_noise_std" if training else "validation_max_relative_noise_std",
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
                        "self_condition_probability" if training else "validation_self_condition_probability",
                        config.get("diffusion", {}).get("self_condition_probability", 0.5),
                    )
                ),
            )
            if optimizer is not None:
                loss.backward()
                grad_clip = float(config.get("gradient_clip_norm", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
                optimizer.step()
                if scheduler is not None and scheduler_interval == "step":
                    scheduler.step()
            total_loss += float(loss.detach().cpu())
            for name, value in metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + float(value)
            batches += 1
            iterator.set_postfix(
                loss=f"{total_loss / batches:.4f}",
                count_mae=f"{metric_totals.get('atom_count_mae', 0.0) / batches:.3f}",
                template_atoms=(
                    f"{metric_totals.get('molecule_template_atoms_mean', 0.0) / batches:.2f}"
                ),
            )
    if training and scheduler is not None and scheduler_interval == "epoch":
        scheduler.step()
    return reduce_epoch_statistics(total_loss, metric_totals, batches, device)


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
    validate_organic_checkpoint_contract(checkpoint)
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    configured_family = config.get("model_family")
    if configured_family not in (None, ORGANIC_TEMPLATE_MODEL_FAMILY):
        raise ValueError(
            f"config model_family={configured_family!r} is not organic-template"
        )
    config["model_family"] = ORGANIC_TEMPLATE_MODEL_FAMILY
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        config["learning_rate"] = args.learning_rate
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError("--max-epochs must be positive")
        config["max_epochs"] = args.max_epochs

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
        config.get("output_dir", "weights/dlm_ccdc_organic_mol_template_n160_v2")
    )
    if args.resume is None:
        existing = [
            path
            for name in ("best.ckpt", "last.ckpt")
            if (path := output_dir / name).exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing checkpoints: "
                + ", ".join(str(path) for path in existing)
            )

    rank, local_rank, world_size = setup_distributed(
        config.get("distributed_backend")
    )
    wandb_run = None
    try:
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

        train_dataset: Dataset = build_organic_dataset(config, "train")
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
            batch_size=int(config.get("batch_size", 128)),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=int(config.get("num_workers", 4)),
            drop_last=bool(config.get("drop_last", False)),
            collate_fn=collate_crystal_batch,
        )
        val_loader = None
        if config.get("organic_valid_lmdb_path"):
            val_dataset: Dataset = build_organic_dataset(config, "valid")
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
                    config.get("eval_batch_size", config.get("batch_size", 128))
                ),
                shuffle=False,
                sampler=val_sampler,
                num_workers=int(config.get("num_workers", 4)),
                collate_fn=collate_crystal_batch,
            )

        model: nn.Module = build_organic_dllm_from_config(config).to(device)
        if dist.is_initialized():
            ddp_kwargs: dict[str, Any] = {
                "broadcast_buffers": False,
                "find_unused_parameters": bool(
                    config.get("ddp_find_unused_parameters", False)
                ),
            }
            if device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[local_rank],
                    output_device=local_rank,
                )
            model = DistributedDataParallel(model, **ddp_kwargs)
        target_model = unwrap_model(model)
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
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
                f"model_family={ORGANIC_TEMPLATE_MODEL_FAMILY} "
                f"trainable_parameters={sum(p.numel() for p in parameters)} "
                f"output_dir={output_dir} "
                "condition=mol_atom_type_2+mol2_group_slices "
                "target_atom_count_conditioned=false"
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
                    f"resumed organic-template DLM from {args.resume} "
                    f"at epoch={start_epoch} best={best:.6f}"
                )

        if is_main_process():
            wandb_run = init_wandb(
                config,
                job_type="organic-mol-template-dlm",
                output_dir=output_dir,
                default_name=output_dir.name,
            )
            maybe_watch_model(wandb_run, target_model, config)

        for epoch in range(start_epoch, max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss, train_metrics = _run_epoch(
                model,
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
                val_loss, val_metrics = _run_epoch(
                    model,
                    val_loader,
                    corruption,
                    device,
                    config,
                    optimizer=None,
                    epoch=epoch,
                    max_epochs=max_epochs,
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
                    "model_family": ORGANIC_TEMPLATE_MODEL_FAMILY,
                    "condition_schema": dict(
                        ORGANIC_TEMPLATE_CONDITION_SCHEMA
                    ),
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
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                }
                if early_stopper is not None:
                    checkpoint["early_stopping"] = early_stopper.state_dict()
                torch.save(checkpoint, output_dir / "last.ckpt")
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
                    step=(epoch + 1) * max(len(train_loader), 1),
                )
            if dist.is_initialized():
                stop_tensor = torch.tensor(
                    int(stop_now),
                    device=device,
                    dtype=torch.int32,
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
