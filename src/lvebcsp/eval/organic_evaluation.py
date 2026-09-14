"""Evaluate formula- or molecular-template-conditioned DLMs on CCDC LMDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lvebcsp.data.atomistic_fragment import (
    FRAGMENT_TENSOR_KEYS,
    MULTIPLICITY_TENSOR_KEYS,
    FoundationOrganicDataset,
)
from lvebcsp.data.organic_lmdb import OrganicLMDBDataset
from lvebcsp.eval.mp20_realpxrd_solver import evaluate_mp20_dllm
from lvebcsp.train.train_foundation_dlm import (
    FOUNDATION_MODEL_FAMILIES,
    FOUNDATION_MODEL_FAMILY,
    FOUNDATION_MULTIPLICITY_MODEL_FAMILY,
    build_foundation_model_from_config,
    foundation_formula_condition,
    validate_foundation_checkpoint_contract,
)
from lvebcsp.train.train_organic_dllm import (
    MOLECULE_TEMPLATE_KEYS,
    ORGANIC_TEMPLATE_MODEL_FAMILY,
    build_organic_dllm_from_config,
    molecular_condition_from_batch,
    molecular_formula_condition,
    validate_organic_checkpoint_contract,
)


def _organic_molecular_condition(
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    formula_noise_std: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    condition_cfg = dict(config.get("molecule_condition", {}))
    condition, element_counts = molecular_condition_from_batch(
        batch,
        max_groups=int(condition_cfg.get("max_groups", 8)),
    )
    noise_std = torch.full(
        (element_counts.shape[0],),
        float(formula_noise_std),
        device=element_counts.device,
        dtype=torch.float32,
    )
    formula = molecular_formula_condition(
        element_counts,
        noise_std,
        generator=generator,
    )
    return condition, formula


def _foundation_organic_condition(
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    formula_noise_std: float,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    fractions = batch["condition_formula_fractions"]
    noise_std = torch.full(
        (fractions.shape[0],),
        float(formula_noise_std),
        device=fractions.device,
        dtype=torch.float32,
    )
    formula = foundation_formula_condition(
        fractions,
        noise_std,
        generator=generator,
    )
    condition = {name: batch[name] for name in FRAGMENT_TENSOR_KEYS}
    if config.get("model_family") == FOUNDATION_MULTIPLICITY_MODEL_FAMILY:
        for name in MULTIPLICITY_TENSOR_KEYS:
            if name in batch:
                condition[name] = batch[name]
    return condition, formula


def evaluate_organic_dllm(
    checkpoint_path: str | Path,
    lmdb_path: str | Path,
    out_dir: str | Path,
    *,
    max_cases: int = 50,
    offset: int = 0,
    num_evals: int = 1,
    num_steps: int = 100,
    top_k: int | None = None,
    seed: int = 42,
    formula_noise_std: float = 0.0,
    coord_update_mode: str | None = None,
    coord_mask_start_fraction: float | None = None,
    token_noise_exponent: float | None = None,
    geometry_noise_exponent: float | None = None,
    multiplicity: int | None = None,
    use_target_multiplicity: bool = False,
    guidance_scale: float | None = None,
) -> dict[str, Any]:
    """Evaluate organic candidates and select conditioning by checkpoint family."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    family = checkpoint.get("model_family") or checkpoint.get("config", {}).get(
        "model_family"
    )
    if multiplicity is not None and int(multiplicity) < 1:
        raise ValueError("multiplicity must be a positive integer")
    if multiplicity is not None and use_target_multiplicity:
        raise ValueError(
            "multiplicity and use_target_multiplicity are mutually exclusive"
        )
    uses_supplied_multiplicity = multiplicity is not None or use_target_multiplicity
    if (
        uses_supplied_multiplicity
        and family != FOUNDATION_MULTIPLICITY_MODEL_FAMILY
    ):
        raise ValueError(
            "the selected checkpoint was not trained with multiplicity"
        )
    if (
        guidance_scale is not None
        and not uses_supplied_multiplicity
        and float(guidance_scale) != 1.0
    ):
        raise ValueError("multiplicity guidance requires a multiplicity input")
    shared_kwargs: dict[str, Any] = {
        "max_cases": max_cases,
        "offset": offset,
        "num_evals": num_evals,
        "num_steps": num_steps,
        "top_k": top_k,
        "seed": seed,
        "formula_noise_std": formula_noise_std,
        "coord_update_mode": coord_update_mode,
        "coord_mask_start_fraction": coord_mask_start_fraction,
        "token_noise_exponent": token_noise_exponent,
        "geometry_noise_exponent": geometry_noise_exponent,
        "guidance_scale": (
            guidance_scale if uses_supplied_multiplicity else 1.0
        ),
        "dataset_cls": OrganicLMDBDataset,
    }
    if family == ORGANIC_TEMPLATE_MODEL_FAMILY:
        validate_organic_checkpoint_contract(checkpoint)
        config = checkpoint["config"]
        condition_cfg = dict(config.get("molecule_condition", {}))
        shared_kwargs.update(
            {
                "model_builder": build_organic_dllm_from_config,
                "external_condition_builder": _organic_molecular_condition,
                "condition_tensor_keys": MOLECULE_TEMPLATE_KEYS,
                "dataset_kwargs": {
                    "include_pxrd": False,
                    "include_molecule_template": True,
                    "molecule_template_n_max": int(
                        condition_cfg.get(
                            "max_template_atoms",
                            config.get("n_max", 160),
                        )
                    ),
                },
            }
        )
    elif family in FOUNDATION_MODEL_FAMILIES:
        validate_foundation_checkpoint_contract(checkpoint)
        config = checkpoint["config"]
        fragment_cfg = dict(config.get("fragment_condition", {}))
        condition_tensor_keys = (
            *FRAGMENT_TENSOR_KEYS,
            "condition_formula_fractions",
        )
        if uses_supplied_multiplicity:
            condition_tensor_keys = (
                *condition_tensor_keys,
                *MULTIPLICITY_TENSOR_KEYS,
            )
        shared_kwargs.update(
            {
                "dataset_cls": FoundationOrganicDataset,
                "model_builder": build_foundation_model_from_config,
                "external_condition_builder": _foundation_organic_condition,
                "condition_tensor_keys": condition_tensor_keys,
                "dataset_kwargs": {
                    "include_pxrd": False,
                    "fragment_n_max": int(fragment_cfg.get("max_atoms", 64)),
                    "emit_condition_multiplicity": uses_supplied_multiplicity,
                    "condition_multiplicity_override": multiplicity,
                },
            }
        )
    elif isinstance(family, str) and family.startswith(
        "universal_atomistic_fragment_dlm_"
    ):
        # Produce the checkpoint-contract error here instead of silently
        # evaluating an older foundation family as a generic de novo DLM.
        validate_foundation_checkpoint_contract(checkpoint)
    del checkpoint

    return evaluate_mp20_dllm(
        checkpoint_path,
        lmdb_path,
        out_dir,
        **shared_kwargs,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a de novo DLM checkpoint on CCDC organic-crystal LMDB "
            "rows using the RealPXRD-Solver RMSD standard"
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Accepted for command compatibility. DLM architecture and "
            "sampling defaults are read from --dllm_ckpt."
        ),
    )
    parser.add_argument("--lmdb", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dllm_ckpt", required=True)
    parser.add_argument("--max_cases", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num_evals", type=int, default=None)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Alias for --num_evals",
    )
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--formula_noise_std",
        type=float,
        default=0.0,
        help="Relative standard deviation for continuous formula noise",
    )
    parser.add_argument(
        "--coord_update_mode",
        choices=("endpoint", "euler", "heun"),
        default=None,
        help="Override the checkpoint continuous reverse solver",
    )
    parser.add_argument(
        "--coord_mask_start_fraction",
        type=float,
        default=None,
        help="Override when EOS masking feeds back into coordinate states",
    )
    parser.add_argument(
        "--token_noise_exponent",
        type=float,
        default=None,
        help="Override the checkpoint discrete-token noise exponent",
    )
    parser.add_argument(
        "--geometry_noise_exponent",
        type=float,
        default=None,
        help="Override the checkpoint lattice/coordinate noise exponent",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=None,
        help="Use one explicit formula-unit multiplicity for every selected case",
    )
    parser.add_argument(
        "--use_target_multiplicity",
        action="store_true",
        help="Use each benchmark target's formula-unit multiplicity (oracle input)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Multiplicity CFG strength; checkpoint default is used when omitted",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    num_evals = args.num_evals
    if num_evals is None:
        num_evals = args.num_samples if args.num_samples is not None else 1
    result = evaluate_organic_dllm(
        args.dllm_ckpt,
        args.lmdb,
        args.out_dir,
        max_cases=args.max_cases,
        offset=args.offset,
        num_evals=num_evals,
        num_steps=args.num_steps,
        top_k=args.top_k,
        seed=args.seed,
        formula_noise_std=args.formula_noise_std,
        coord_update_mode=args.coord_update_mode,
        coord_mask_start_fraction=args.coord_mask_start_fraction,
        token_noise_exponent=args.token_noise_exponent,
        geometry_noise_exponent=args.geometry_noise_exponent,
        multiplicity=args.multiplicity,
        use_target_multiplicity=args.use_target_multiplicity,
        guidance_scale=args.guidance_scale,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
