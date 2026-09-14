"""Checkpoint migration helpers for DLM heads and optional PXRD adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


OUTPUT_HEAD_NAMES = ("lattice_head", "coord_head")
PXRD_ADAPTER_NAMES = (
    "z_proj",
    "pxrd_detail_encoder",
    "pxrd_detail_norm",
    "pxrd_detail_condition_proj",
)
FORMULA_ENCODER_NAMES = (
    "formula_uncertainty_proj",
    "formula_feature_embedder",
    "formula_pool",
    "formula_token_gate",
)
CONDITION_ENCODER_NAMES = (
    *PXRD_ADAPTER_NAMES,
    "formula_condition_norm",
    *FORMULA_ENCODER_NAMES,
)
TRAIN_SCOPES = ("all", "output-heads", "pxrd-adapter")


def is_pxrd_adapter_key(key: str) -> bool:
    """Return whether a state-dict key belongs only to the PXRD attachment."""

    parts = key.split(".")
    return any(name in parts for name in PXRD_ADAPTER_NAMES)


def is_formula_encoder_key(key: str) -> bool:
    return any(name in key.split(".") for name in FORMULA_ENCODER_NAMES)


def migrate_condition_encoder_state_dict(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Move legacy root-level condition weights under ``condition_encoder``."""

    migrated: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if "condition_encoder" not in parts:
            for index, part in enumerate(parts):
                if part in CONDITION_ENCODER_NAMES:
                    parts.insert(index, "condition_encoder")
                    break
        migrated[".".join(parts)] = value
    return migrated


def _matching_key(state_dict: Mapping[str, Tensor], suffix: str) -> str | None:
    """Return the unique state-dict key ending in ``suffix``."""

    matches = [key for key in state_dict if key == suffix or key.endswith(f".{suffix}")]
    if len(matches) > 1:
        raise ValueError(
            f"checkpoint contains multiple keys ending in {suffix!r}: {matches}"
        )
    return matches[0] if matches else None


def migrate_biasless_output_heads(
    state_dict: Mapping[str, Tensor],
    *,
    strategy: str = "drop",
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Remove legacy output biases, optionally folding them into LayerNorm.

    ``drop`` starts fine-tuning from the genuinely bias-removed function.
    ``fold`` adjusts the preceding LayerNorm beta so that each head initially
    reproduces the old affine output as closely as numerical precision allows.
    """

    if strategy not in {"drop", "fold"}:
        raise ValueError("strategy must be 'drop' or 'fold'")

    migrated = dict(state_dict)
    removed_keys: list[str] = []
    fold_residuals: dict[str, float] = {}

    for head_name in OUTPUT_HEAD_NAMES:
        bias_suffix = f"{head_name}.1.bias"
        bias_key = _matching_key(migrated, bias_suffix)
        if bias_key is None:
            continue

        bias = migrated[bias_key]
        if strategy == "fold":
            prefix = bias_key[: -len(bias_suffix)]
            weight_key = f"{prefix}{head_name}.1.weight"
            norm_bias_key = f"{prefix}{head_name}.0.bias"
            if weight_key not in migrated or norm_bias_key not in migrated:
                raise KeyError(
                    f"cannot fold {bias_key!r}: expected {weight_key!r} and "
                    f"{norm_bias_key!r}"
                )

            weight = migrated[weight_key]
            norm_bias = migrated[norm_bias_key]
            if weight.ndim != 2 or bias.shape != weight.shape[:1]:
                raise ValueError(
                    f"incompatible shapes for {head_name}: "
                    f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
                )
            if norm_bias.shape != weight.shape[1:]:
                raise ValueError(
                    f"incompatible LayerNorm bias for {head_name}: "
                    f"expected {tuple(weight.shape[1:])}, got {tuple(norm_bias.shape)}"
                )

            # The heads are very wide (hidden_dim >> output_dim), so a
            # full-row-rank W normally gives an exact solution to W @ delta=b.
            solve_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
            solve_weight = weight.to(dtype=solve_dtype)
            solve_bias = bias.to(device=weight.device, dtype=solve_dtype)
            delta = torch.linalg.pinv(solve_weight) @ solve_bias
            residual = solve_weight @ delta - solve_bias
            migrated[norm_bias_key] = norm_bias + delta.to(
                device=norm_bias.device,
                dtype=norm_bias.dtype,
            )
            fold_residuals[head_name] = float(residual.abs().max().cpu())

        migrated.pop(bias_key)
        removed_keys.append(bias_key)

    report: dict[str, Any] = {
        "strategy": strategy,
        "removed_keys": tuple(removed_keys),
        "fold_residuals": fold_residuals,
    }
    return migrated, report


def checkpoint_model_state(checkpoint: Any) -> Mapping[str, Tensor]:
    """Extract a model state dict from a training checkpoint or raw state dict."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    model_state = checkpoint.get("model")
    if isinstance(model_state, Mapping):
        return model_state
    if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("checkpoint does not contain a 'model' state dict")


def initialize_from_checkpoint(
    model: nn.Module,
    checkpoint: Any,
    *,
    strategy: str = "drop",
) -> dict[str, Any]:
    """Initialize shared DLM weights, allowing new conditioning modules.

    Missing PXRD or formula-conditioner parameters retain their initialization.
    """

    checkpoint_state = migrate_condition_encoder_state_dict(
        checkpoint_model_state(checkpoint)
    )
    state_dict, report = migrate_biasless_output_heads(
        checkpoint_state,
        strategy=strategy,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    allowed_missing = tuple(
        key
        for key in missing
        if is_pxrd_adapter_key(key) or is_formula_encoder_key(key)
    )
    disallowed_missing = tuple(key for key in missing if key not in allowed_missing)
    if disallowed_missing or unexpected:
        fields = []
        if disallowed_missing:
            fields.append(f"missing keys: {list(disallowed_missing)}")
        if unexpected:
            fields.append(f"unexpected keys: {list(unexpected)}")
        raise RuntimeError("incompatible DLM checkpoint; " + "; ".join(fields))
    report["initialized_pxrd_keys"] = tuple(
        key for key in missing if is_pxrd_adapter_key(key)
    )
    report["initialized_formula_keys"] = tuple(
        key for key in missing if is_formula_encoder_key(key)
    )
    return report


def set_dllm_train_scope(model: nn.Module, scope: str) -> tuple[nn.Parameter, ...]:
    """Select all, output-head, or newly attached PXRD parameters."""

    if scope not in TRAIN_SCOPES:
        raise ValueError(f"scope must be one of {TRAIN_SCOPES}, got {scope!r}")

    for parameter in model.parameters():
        parameter.requires_grad_(scope == "all")

    if scope == "output-heads":
        for head_name in OUTPUT_HEAD_NAMES:
            head = getattr(model, head_name, None)
            if not isinstance(head, nn.Module):
                raise AttributeError(f"model has no module named {head_name!r}")
            for parameter in head.parameters():
                parameter.requires_grad_(True)
    elif scope == "pxrd-adapter":
        if not bool(getattr(model, "uses_pxrd_conditioning", False)):
            raise ValueError(
                "train scope 'pxrd-adapter' requires PXRD conditioning"
            )
        selected_names: list[str] = []
        for name, parameter in model.named_parameters():
            if is_pxrd_adapter_key(name):
                parameter.requires_grad_(True)
                selected_names.append(name)
        if not selected_names:
            raise ValueError("PXRD conditioning model has no adapter parameters")

    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError(f"train scope {scope!r} selected no parameters")
    return parameters
