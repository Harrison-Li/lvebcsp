"""MP-20 evaluation with the RealPXRD-Solver RMSD success rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from lvebcsp.data.mp20_lmdb import MP20LMDBDataset
from lvebcsp.data.representation import tensor_to_structure
from lvebcsp.inference.export_cif import export_structure_to_cif
from lvebcsp.losses.validity import has_usable_lattice
from lvebcsp.common.config import load_config, select_device
from lvebcsp.common.seed import seed_everything

REALPXRD_SOLVER_MATCHER = {"stol": 0.5, "angle_tol": 10.0, "ltol": 0.3}
RMSD_MATCH_THRESHOLD = 0.5
DEFAULT_TOP_K = 20


def reference_structure_from_item(item: dict[str, Any]) -> Structure:
    """Build the MP-20 reference structure represented by one dataset item."""

    atom_types = item["atom_types"].long()
    atom_mask = item["atom_mask"].bool()
    species = [int(z) for z, keep in zip(atom_types.tolist(), atom_mask.tolist()) if keep and int(z) > 0]
    coords = item["frac_coords"][atom_mask].numpy()
    lattice_params = [float(value) for value in item["lattice_params"].tolist()]
    return Structure(Lattice.from_parameters(*lattice_params), species, coords, coords_are_cartesian=False)


def finite_rmsd_values(rmsd_values: list[float | None]) -> list[float]:
    """Return finite RMSD values from optional values."""

    return [float(value) for value in rmsd_values if value is not None and math.isfinite(float(value))]


def rmsd_is_match(rmsd: float | None, threshold: float = RMSD_MATCH_THRESHOLD) -> bool:
    """Return whether one candidate satisfies the pasted solver's RMSD threshold."""

    return rmsd is not None and math.isfinite(float(rmsd)) and float(rmsd) < float(threshold)


def best_matched_rmsd(rmsd_values: list[float | None], threshold: float = RMSD_MATCH_THRESHOLD) -> float | None:
    """Best RMSD among candidates that count as matched."""

    matched = [float(value) for value in rmsd_values if rmsd_is_match(value, threshold)]
    return min(matched) if matched else None


def min_finite_rmsd(rmsd_values: list[float | None]) -> float | None:
    """Best finite RMSD, even when it does not pass the match threshold."""

    finite = finite_rmsd_values(rmsd_values)
    return min(finite) if finite else None


def rms_dist_or_none(matcher: StructureMatcher, candidate: Structure, gt_primitive: Structure) -> float | None:
    """Call StructureMatcher.get_rms_dist(candidate, gt_primitive) and return rms_dist[0]."""

    # Pymatgen's supercell search can become effectively unbounded for a
    # singular lattice. Treat those samples as invalid before entering the
    # matcher; they cannot represent a physical crystal or a structural match.
    if not has_usable_lattice(candidate):
        return None
    rms_dist = matcher.get_rms_dist(candidate, gt_primitive)
    if rms_dist is None:
        return None
    value = rms_dist[0] if isinstance(rms_dist, tuple) else rms_dist
    value = float(value)
    return value if math.isfinite(value) else None


def reduced_composition_counts(atom_types: torch.Tensor) -> torch.Tensor:
    """Return integer element counts divided by their greatest common divisor."""

    counts = torch.bincount(atom_types.long().cpu(), minlength=119)
    divisor = 0
    for count in counts[counts > 0].tolist():
        divisor = math.gcd(divisor, int(count))
    return counts // max(divisor, 1)


def lattices_are_compatible(
    matcher: StructureMatcher,
    candidate: Structure,
    reference: Structure,
) -> bool:
    """Test the RealPXRD lattice tolerances independently of atomic coordinates."""

    if not has_usable_lattice(candidate) or not has_usable_lattice(reference):
        return False
    origin = [[0.0, 0.0, 0.0]]
    candidate_probe = Structure(candidate.lattice, [1], origin)
    reference_probe = Structure(reference.lattice, [1], origin)
    return rms_dist_or_none(matcher, candidate_probe, reference_probe) is not None


def case_metrics_from_rmsd(
    rmsd_values: list[float | None],
    *,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = RMSD_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """Compute one-case metrics from RMSD values in generation order."""

    top_k_values = rmsd_values[: int(top_k)]
    best_rmsd = best_matched_rmsd(rmsd_values, threshold)
    min_rmsd = min_finite_rmsd(rmsd_values)
    best_index = None
    if min_rmsd is not None:
        best_index = next(
            index
            for index, value in enumerate(rmsd_values)
            if value is not None and math.isfinite(float(value)) and float(value) == min_rmsd
        )
    return {
        "matched": best_rmsd is not None,
        "top_1_match": bool(rmsd_values) and rmsd_is_match(rmsd_values[0], threshold),
        "top_k_match": any(rmsd_is_match(value, threshold) for value in top_k_values),
        "best_rmsd": best_rmsd,
        "min_rmsd": min_rmsd,
        "best_candidate_index": best_index,
    }


def mean_or_none(values: list[float | None]) -> float | None:
    """Mean over finite optional values."""

    finite = finite_rmsd_values(values)
    return sum(finite) / len(finite) if finite else None


def select_saved_candidate_index(
    rmsd_values: list[float | None],
    valid_indices: list[int],
    threshold: float = RMSD_MATCH_THRESHOLD,
) -> int | None:
    """Pick the candidate to export as atom{atoms}_pre_*.cif."""

    matched = [
        (float(value), index)
        for index, value in enumerate(rmsd_values)
        if rmsd_is_match(value, threshold)
    ]
    if matched:
        return min(matched, key=lambda item: (item[0], item[1]))[1]
    finite = [
        (float(value), index)
        for index, value in enumerate(rmsd_values)
        if value is not None and math.isfinite(float(value))
    ]
    if finite:
        return min(finite, key=lambda item: (item[0], item[1]))[1]
    return valid_indices[0] if valid_indices else None


def save_per_case_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write per-case metrics to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "material_id",
        "atoms",
        "num_evals",
        "valid_candidates",
        "finite_rmsd_candidates",
        "matched",
        "top_1_match",
        "top_k_match",
        "best_rmsd",
        "min_rmsd",
        "best_candidate_index",
        "saved_candidate_index",
        "best_candidate_cif",
        "gt_primitive_cif",
        "case_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def evaluate_mp20_realpxrd_solver(
    config: dict[str, Any],
    lmdb_path: str | Path,
    out_dir: str | Path,
    *,
    max_cases: int | None = None,
    offset: int = 0,
    num_evals: int | None = None,
    num_steps: int | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Generate MP-20 candidates and evaluate them with RMSD < 0.5 success."""

    from lvebcsp.inference.sample import integrate_flow, load_models

    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))
    jepa, direct = load_models(config, device)
    p_max = int(config.get("p_max", 128))
    n_max = int(config.get("n_max", 20))
    count = int(num_evals or config.get("num_evals", config.get("num_samples", 20)))
    steps = int(num_steps or config.get("num_steps", 100))
    selected_top_k = int(top_k or config.get("top_k", DEFAULT_TOP_K))
    two_theta_range = config.get("two_theta_range", [5.0, 80.0])
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MP20LMDBDataset(
        lmdb_path,
        n_max=n_max,
        p_max=p_max,
        wavelength=config.get("wavelength", "CuKa"),
        two_theta_range=(float(two_theta_range[0]), float(two_theta_range[1])),
        augmentation={"enabled": False},
        use_conventional=bool(config.get("use_conventional", False)),
    )
    end = len(dataset) if max_cases is None else min(len(dataset), offset + int(max_cases))
    total_cases = max(end - offset, 0)
    matcher = StructureMatcher(**REALPXRD_SOLVER_MATCHER)
    started_at = time.monotonic()
    per_case: list[dict[str, Any]] = []

    print(
        "evaluating MP-20 with RealPXRD-Solver RMSD standard "
        "offset={offset} total={total} num_evals={num_evals} num_steps={steps} top_k={top_k}".format(
            offset=offset,
            total=total_cases,
            num_evals=count,
            steps=steps,
            top_k=selected_top_k,
        ),
        flush=True,
    )

    for index in range(offset, end):
        item = dataset[index]
        gt_primitive = reference_structure_from_item(item).get_primitive_structure()
        atoms = len(gt_primitive)
        case_dir = output_dir / f"case_{index:06d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        gt_primitive_cif = export_structure_to_cif(gt_primitive, case_dir / "gt_primitive.cif")

        atom_types_one = item["atom_types"].unsqueeze(0).to(device)
        atom_mask_one = item["atom_mask"].unsqueeze(0).to(device)
        ratio_one = item["ratio"].unsqueeze(0).to(device)
        with torch.inference_mode():
            z = jepa.encode_ctx(
                item["peak_d"].unsqueeze(0).to(device),
                item["peak_i"].unsqueeze(0).to(device),
                item["peak_mask"].unsqueeze(0).to(device),
                atom_types=atom_types_one,
                ratio=ratio_one,
            )
            atom_types = atom_types_one.repeat(count, 1)
            atom_mask = atom_mask_one.repeat(count, 1)
            frac_coords = torch.rand(count, n_max, 3, device=device)
            frac_coords = frac_coords * atom_mask.float().unsqueeze(-1)
            y_final = integrate_flow(
                direct,
                frac_coords,
                z.repeat(count, 1),
                atom_types,
                atom_mask,
                num_steps=steps,
                guidance_scale=float(config.get("guidance_scale", 1.5)),
                sampler=str(config.get("sampler", "euler")),
                t_end=float(config.get("sample_t_max", config.get("t_end", 0.95))),
                final_denoise=bool(config.get("final_denoise", True)),
                diffuse_lattice=bool(config.get("diffuse_lattice", config.get("flow", {}).get("diffuse_lattice", False))),
            )

        candidate_structures: list[Structure | None] = [None] * count
        rmsd_values: list[float | None] = [None] * count
        valid_indices: list[int] = []
        for sample_idx in range(count):
            try:
                candidate = tensor_to_structure(y_final[sample_idx].cpu(), item["atom_types"], item["atom_mask"])
                candidate_structures[sample_idx] = candidate
                if has_usable_lattice(candidate):
                    valid_indices.append(sample_idx)
                    rmsd_values[sample_idx] = rms_dist_or_none(matcher, candidate, gt_primitive)
            except Exception:
                rmsd_values[sample_idx] = None

        metrics = case_metrics_from_rmsd(rmsd_values, top_k=selected_top_k, threshold=RMSD_MATCH_THRESHOLD)
        saved_index = select_saved_candidate_index(rmsd_values, valid_indices, threshold=RMSD_MATCH_THRESHOLD)
        best_candidate_cif = None
        if saved_index is not None and candidate_structures[saved_index] is not None:
            suffix = metrics["best_rmsd"] if metrics["matched"] else None
            best_candidate_cif = export_structure_to_cif(
                candidate_structures[saved_index],
                case_dir / f"atom{atoms}_pre_{suffix}.cif",
            )

        case_record = {
            "index": index,
            "material_id": item.get("material_id", f"mp20_{index}"),
            "atoms": atoms,
            "num_evals": count,
            "valid_candidates": len(valid_indices),
            "finite_rmsd_candidates": len(finite_rmsd_values(rmsd_values)),
            "saved_candidate_index": saved_index,
            "best_candidate_cif": str(best_candidate_cif) if best_candidate_cif else None,
            "gt_primitive_cif": str(gt_primitive_cif),
            "case_dir": str(case_dir),
            **metrics,
        }
        per_case.append(case_record)

        elapsed = time.monotonic() - started_at
        done = len(per_case)
        top_k_rate = sum(int(row["top_k_match"]) for row in per_case) / max(done, 1)
        print(
            "case={done}/{total} index={index} matched={matched} top_k_rate={top_k_rate:.4f} "
            "valid={valid}/{count} elapsed={elapsed:.1f}s".format(
                done=done,
                total=total_cases,
                index=index,
                matched=bool(metrics["matched"]),
                top_k_rate=top_k_rate,
                valid=len(valid_indices),
                count=count,
                elapsed=elapsed,
            ),
            flush=True,
        )

    per_case_csv = save_per_case_csv(per_case, output_dir / "per_case_metrics.csv")
    total = len(per_case)
    matched_best = [row["best_rmsd"] for row in per_case]
    top_1_matches = sum(int(row["top_1_match"]) for row in per_case)
    top_k_matches = sum(int(row["top_k_match"]) for row in per_case)
    matched_cases = sum(int(row["matched"]) for row in per_case)
    total_candidates = sum(int(row["num_evals"]) for row in per_case)
    valid_candidates = sum(int(row["valid_candidates"]) for row in per_case)
    finite_rmsd_candidates = sum(int(row["finite_rmsd_candidates"]) for row in per_case)
    summary: dict[str, Any] = {
        "lmdb_path": str(lmdb_path),
        "out_dir": str(output_dir),
        "offset": offset,
        "num_cases": total,
        "num_evals": count,
        "num_steps": steps,
        "top_k": selected_top_k,
        "sampler": str(config.get("sampler", "euler")),
        "sample_t_max": float(config.get("sample_t_max", config.get("t_end", 0.95))),
        "final_denoise": bool(config.get("final_denoise", True)),
        "rmsd_threshold": RMSD_MATCH_THRESHOLD,
        "matcher": dict(REALPXRD_SOLVER_MATCHER),
        "structure_matcher_rms_dist_units": "normalized_by_average_free_length_per_atom",
        "top_1_match_rate": top_1_matches / total if total else 0.0,
        "top_k_match_rate": top_k_matches / total if total else 0.0,
        f"top_{selected_top_k}_match_rate": top_k_matches / total if total else 0.0,
        "match_rate": matched_cases / total if total else 0.0,
        "matched_cases": matched_cases,
        "best_rmsd": min_finite_rmsd(matched_best),
        "mean_rmsd": mean_or_none(matched_best),
        "candidate_validity_rate": valid_candidates / total_candidates if total_candidates else 0.0,
        "candidate_rmsd_rate": finite_rmsd_candidates / total_candidates if total_candidates else 0.0,
        "per_case_metrics_csv": str(per_case_csv),
        "metrics_json": str(output_dir / "metrics.json"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def evaluate_mp20_dllm(
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
    guidance_scale: float | None = None,
    dataset_cls: type[MP20LMDBDataset] = MP20LMDBDataset,
    model_builder: Callable[[dict[str, Any]], torch.nn.Module] | None = None,
    external_condition_builder: Callable[
        [dict[str, torch.Tensor], dict[str, Any], float, torch.Generator],
        tuple[
            torch.Tensor | dict[str, torch.Tensor],
            dict[str, torch.Tensor],
        ],
    ]
    | None = None,
    condition_tensor_keys: tuple[str, ...] = (),
    dataset_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate DLM candidates per case and report top-1/top-k matches.

    ``dataset_cls`` and the optional builder hooks allow schema-specific LMDB
    adapters to reuse the same sampling and RealPXRD-Solver metric
    implementation. Defaults retain the original MP-20/PXRD behavior.
    """

    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if num_evals < 1:
        raise ValueError("num_evals must be positive")
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if (
        not math.isfinite(float(formula_noise_std))
        or float(formula_noise_std) < 0.0
    ):
        raise ValueError("formula_noise_std must be finite and non-negative")
    if coord_update_mode is not None and coord_update_mode not in {
        "endpoint",
        "euler",
        "heun",
    }:
        raise ValueError(
            "coord_update_mode must be 'endpoint', 'euler', or 'heun'"
        )
    if (
        coord_mask_start_fraction is not None
        and (
            not math.isfinite(float(coord_mask_start_fraction))
            or not 0.0 <= float(coord_mask_start_fraction) <= 1.0
        )
    ):
        raise ValueError(
            "coord_mask_start_fraction must be finite and in [0, 1]"
        )
    if (
        token_noise_exponent is not None
        and (
            not math.isfinite(float(token_noise_exponent))
            or float(token_noise_exponent) <= 0.0
        )
    ):
        raise ValueError("token_noise_exponent must be finite and positive")
    if (
        geometry_noise_exponent is not None
        and (
            not math.isfinite(float(geometry_noise_exponent))
            or float(geometry_noise_exponent) <= 0.0
        )
    ):
        raise ValueError(
            "geometry_noise_exponent must be finite and positive"
        )
    if guidance_scale is not None and (
        not math.isfinite(float(guidance_scale)) or float(guidance_scale) < 0.0
    ):
        raise ValueError("guidance_scale must be finite and non-negative")
    selected_top_k = min(DEFAULT_TOP_K, num_evals) if top_k is None else top_k
    if selected_top_k < 1:
        raise ValueError("top_k must be positive")
    if selected_top_k > num_evals:
        raise ValueError("top_k cannot exceed num_evals")
    # The matching helpers also serve crystal reconstruction, which does not
    # depend on the legacy DLM sampling model.
    from lvebcsp.models.canvas import FormulaConditioner
    from lvebcsp.train.dllm_checkpoint import initialize_from_checkpoint
    from lvebcsp.train.train_dllm import (
        build_dllm_from_config, compute_pxrd_latent, load_jepa_target_encoder,
    )

    seed_everything(seed)
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    sampling_cfg = dict(config.get("sampling", {}))
    diffusion_cfg = dict(config.get("diffusion", {}))
    if coord_update_mode is None:
        coord_update_mode = str(
            sampling_cfg.get("coord_update_mode", "endpoint")
        )
    if coord_mask_start_fraction is None:
        coord_mask_start_fraction = float(
            sampling_cfg.get("coord_mask_start_fraction", 0.0)
        )
    if token_noise_exponent is None:
        token_noise_exponent = float(
            sampling_cfg.get(
                "token_noise_exponent",
                diffusion_cfg.get("token_noise_exponent", 1.0),
            )
        )
    if geometry_noise_exponent is None:
        geometry_noise_exponent = float(
            sampling_cfg.get(
                "geometry_noise_exponent",
                diffusion_cfg.get("geometry_noise_exponent", 1.0),
            )
        )
    if guidance_scale is None:
        guidance_scale = float(sampling_cfg.get("guidance_scale", 1.0))
    if coord_update_mode not in {"endpoint", "euler", "heun"}:
        raise ValueError(
            "checkpoint sampling.coord_update_mode must be 'endpoint', "
            "'euler', or 'heun'"
        )
    if (
        not math.isfinite(float(coord_mask_start_fraction))
        or not 0.0 <= float(coord_mask_start_fraction) <= 1.0
    ):
        raise ValueError(
            "checkpoint sampling.coord_mask_start_fraction must be finite "
            "and in [0, 1]"
        )
    if (
        not math.isfinite(float(token_noise_exponent))
        or float(token_noise_exponent) <= 0.0
    ):
        raise ValueError(
            "checkpoint token_noise_exponent must be finite and positive"
        )
    if (
        not math.isfinite(float(geometry_noise_exponent))
        or float(geometry_noise_exponent) <= 0.0
    ):
        raise ValueError(
            "checkpoint geometry_noise_exponent must be finite and positive"
        )
    if not math.isfinite(float(guidance_scale)) or float(guidance_scale) < 0.0:
        raise ValueError(
            "checkpoint sampling.guidance_scale must be finite and non-negative"
        )
    device = select_device(str(config.get("device", "auto")))
    selected_model_builder = model_builder or build_dllm_from_config
    model = selected_model_builder(config).to(device).eval()
    initialize_from_checkpoint(model, checkpoint)
    if external_condition_builder is not None and model.uses_pxrd_detail:
        raise ValueError(
            "an external vector condition cannot replace required PXRD detail inputs"
        )
    jepa = None
    if model.uses_pxrd_conditioning and external_condition_builder is None:
        jepa_checkpoint = config.get("jepa_ckpt")
        if not jepa_checkpoint:
            raise ValueError(
                "checkpoint config requires jepa_ckpt for PXRD conditioning"
            )
        jepa = load_jepa_target_encoder(jepa_checkpoint, device)
    two_theta_range = config.get("two_theta_range", [5.0, 80.0])
    dataset_init: dict[str, Any] = {
        "n_max": model.n_max,
        "p_max": int(config.get("p_max", 128)),
        "wavelength": config.get("wavelength", "CuKa"),
        "two_theta_range": (
            float(two_theta_range[0]),
            float(two_theta_range[1]),
        ),
        "augmentation": {"enabled": False},
        "include_pxrd": (
            model.uses_pxrd_conditioning
            and external_condition_builder is None
        ),
    }
    dataset_init.update(dataset_kwargs or {})
    dataset = dataset_cls(lmdb_path, **dataset_init)
    end = min(len(dataset), offset + max_cases)
    items = [dataset[index] for index in range(offset, end)]
    if not items:
        raise ValueError("the requested evaluation range is empty")
    tensor_keys = [
        "atom_types",
        "atom_mask",
    ]
    if model.uses_pxrd_conditioning and external_condition_builder is None:
        tensor_keys.extend(("peak_d", "peak_i", "peak_mask"))
    for key in condition_tensor_keys:
        if key not in tensor_keys:
            tensor_keys.append(key)
    batch = {
        key: torch.stack([item[key] for item in items]).to(device)
        for key in tensor_keys
    }
    formula_noise_seed = seed + 1_000_003
    formula_generator = torch.Generator(device=device).manual_seed(formula_noise_seed)
    z_jepa = None
    if external_condition_builder is None:
        formula = FormulaConditioner().from_atom_types(
            batch["atom_types"],
            batch["atom_mask"],
            noise_std=float(formula_noise_std),
            generator=formula_generator,
        )
    else:
        with torch.inference_mode():
            z_jepa, formula = external_condition_builder(
                batch,
                config,
                float(formula_noise_std),
                formula_generator,
            )
        required_formula_keys = {
            "fractions",
            "uncertainties",
            "clean_fractions",
        }
        missing_formula_keys = required_formula_keys.difference(formula)
        if missing_formula_keys:
            raise KeyError(
                "external condition builder is missing formula fields: "
                + ", ".join(sorted(missing_formula_keys))
            )
        if isinstance(z_jepa, dict):
            if not z_jepa or any(
                value.shape[0] != len(items) for value in z_jepa.values()
            ):
                raise ValueError("external condition batch size does not match cases")
        elif z_jepa.shape[0] != len(items):
            raise ValueError("external condition batch size does not match cases")
    generator = torch.Generator(device=device).manual_seed(seed)
    started_at = time.monotonic()
    if jepa is not None:
        with torch.inference_mode():
            z_jepa = compute_pxrd_latent(jepa, batch)
    output_dir = Path(out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    matcher = StructureMatcher(**REALPXRD_SOLVER_MATCHER)
    rows: list[dict[str, Any]] = []
    matched = top_1_matches = top_k_matches = 0
    valid = count_exact = reduced_formula_exact = full_composition_exact = 0
    invalid_lattice_candidates = 0
    top_k_count_exact = top_k_formula_exact = top_k_composition_exact = 0
    lattice_compatible = top_k_lattice_compatible = 0
    finite_rmsd: list[float] = []
    best_matched_rmsd: list[float] = []
    for batch_index, (dataset_index, item) in enumerate(
        zip(range(offset, end), items)
    ):
        def repeat_case(tensor: torch.Tensor) -> torch.Tensor:
            repeats = (num_evals,) + (1,) * (tensor.ndim - 1)
            return tensor[batch_index : batch_index + 1].repeat(repeats)

        detail_inputs: dict[str, torch.Tensor] = {}
        if model.uses_pxrd_detail:
            detail_inputs = {
                "peak_d": repeat_case(batch["peak_d"]),
                "peak_i": repeat_case(batch["peak_i"]),
                "peak_mask": repeat_case(batch["peak_mask"]),
            }
        with torch.inference_mode():
            sampling_inputs = {
                "num_steps": num_steps,
                "coord_update_mode": coord_update_mode,
                "coord_mask_start_fraction": coord_mask_start_fraction,
                "token_noise_exponent": token_noise_exponent,
                "geometry_noise_exponent": geometry_noise_exponent,
                "guidance_scale": guidance_scale,
                "generator": generator,
                **detail_inputs,
            }
            if isinstance(z_jepa, dict):
                sampled = model.sample(
                    formula_fractions=repeat_case(formula["fractions"]),
                    formula_uncertainties=repeat_case(
                        formula["uncertainties"]
                    ),
                    **{
                        name: repeat_case(value)
                        for name, value in z_jepa.items()
                    },
                    **sampling_inputs,
                )
            else:
                sampled = model.sample(
                    None if z_jepa is None else repeat_case(z_jepa),
                    repeat_case(formula["fractions"]),
                    repeat_case(formula["uncertainties"]),
                    **sampling_inputs,
                )

        case_dir = output_dir / f"case_{dataset_index:06d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        gt = reference_structure_from_item(item).get_primitive_structure()
        gt_path = export_structure_to_cif(gt, case_dir / "gt_primitive.cif")
        gt_types = item["atom_types"][item["atom_mask"]]
        candidate_structures: list[Structure | None] = [None] * num_evals
        candidate_records: list[dict[str, Any]] = []
        rmsd_values: list[float | None] = [None] * num_evals
        valid_indices: list[int] = []
        for candidate_index in range(num_evals):
            pred_atoms = int(sampled["num_atoms"][candidate_index].item())
            pred_types = sampled["atom_types"][candidate_index][
                sampled["atom_mask"][candidate_index]
            ].cpu()
            exact_count = pred_atoms == len(gt_types)
            exact_formula = torch.equal(
                reduced_composition_counts(pred_types),
                reduced_composition_counts(gt_types),
            )
            exact_composition = torch.equal(
                torch.bincount(pred_types, minlength=119),
                torch.bincount(gt_types, minlength=119),
            )
            candidate_valid = False
            candidate_lattice_compatible = False
            try:
                pred = Structure(
                    Lattice.from_parameters(
                        *sampled["lattice_params"][candidate_index]
                        .cpu()
                        .tolist()
                    ),
                    pred_types.tolist(),
                    sampled["frac_coords"][candidate_index][
                        sampled["atom_mask"][candidate_index]
                    ]
                    .cpu()
                    .numpy(),
                    coords_are_cartesian=False,
                )
                candidate_structures[candidate_index] = pred
                if has_usable_lattice(pred):
                    valid_indices.append(candidate_index)
                    candidate_valid = True
                    rmsd_values[candidate_index] = rms_dist_or_none(
                        matcher,
                        pred,
                        gt,
                    )
                    candidate_lattice_compatible = (
                        rmsd_values[candidate_index] is not None
                        or (
                            exact_composition
                            and lattices_are_compatible(matcher, pred, gt)
                        )
                    )
                else:
                    invalid_lattice_candidates += 1
            except Exception:
                pass
            rmsd = rmsd_values[candidate_index]
            if rmsd is not None:
                finite_rmsd.append(rmsd)
            candidate_records.append(
                {
                    "candidate_index": candidate_index,
                    "pred_atoms": pred_atoms,
                    "count_exact": exact_count,
                    "reduced_formula_exact": exact_formula,
                    "full_composition_exact": exact_composition,
                    "valid": candidate_valid,
                    "lattice_compatible": candidate_lattice_compatible,
                    "rmsd": rmsd,
                    "matched": rmsd_is_match(rmsd),
                }
            )

        metrics = case_metrics_from_rmsd(
            rmsd_values,
            top_k=selected_top_k,
            threshold=RMSD_MATCH_THRESHOLD,
        )
        saved_index = select_saved_candidate_index(
            rmsd_values,
            valid_indices,
            threshold=RMSD_MATCH_THRESHOLD,
        )
        pred_path = None
        if saved_index is not None and candidate_structures[saved_index] is not None:
            suffix = metrics["best_rmsd"] if metrics["matched"] else None
            pred_path = export_structure_to_cif(
                candidate_structures[saved_index],
                case_dir / f"atom{len(gt)}_pre_{suffix}.cif",
            )

        top_1 = candidate_records[0]
        top_k_records = candidate_records[:selected_top_k]
        top_k_has_count = any(row["count_exact"] for row in top_k_records)
        top_k_has_formula = any(
            row["reduced_formula_exact"] for row in top_k_records
        )
        top_k_has_composition = any(
            row["full_composition_exact"] for row in top_k_records
        )
        top_k_has_lattice = any(
            row["lattice_compatible"] for row in top_k_records
        )
        count_exact += int(top_1["count_exact"])
        reduced_formula_exact += int(top_1["reduced_formula_exact"])
        full_composition_exact += int(top_1["full_composition_exact"])
        lattice_compatible += int(top_1["lattice_compatible"])
        top_k_count_exact += int(top_k_has_count)
        top_k_formula_exact += int(top_k_has_formula)
        top_k_composition_exact += int(top_k_has_composition)
        top_k_lattice_compatible += int(top_k_has_lattice)
        top_1_matches += int(metrics["top_1_match"])
        top_k_matches += int(metrics["top_k_match"])
        matched += int(metrics["matched"])
        valid += len(valid_indices)
        if metrics["best_rmsd"] is not None:
            best_matched_rmsd.append(float(metrics["best_rmsd"]))
        row = {
            "index": dataset_index,
            "material_id": item["material_id"],
            "gt_atoms": len(gt_types),
            "num_evals": num_evals,
            "top_k": selected_top_k,
            "formula_noise_std": float(formula_noise_std),
            "clean_formula_fractions": {
                str(atomic_number): float(fraction)
                for atomic_number, fraction in enumerate(
                    formula["clean_fractions"][batch_index].cpu().tolist()
                )
                if atomic_number > 0 and fraction > 0
            },
            "observed_formula_fractions": {
                str(atomic_number): float(fraction)
                for atomic_number, fraction in enumerate(
                    formula["fractions"][batch_index].cpu().tolist()
                )
                if atomic_number > 0 and fraction > 0
            },
            "valid_candidates": len(valid_indices),
            "finite_rmsd_candidates": len(finite_rmsd_values(rmsd_values)),
            "pred_atoms": top_1["pred_atoms"],
            "count_exact": top_1["count_exact"],
            "reduced_formula_exact": top_1["reduced_formula_exact"],
            "full_composition_exact": top_1["full_composition_exact"],
            "top_k_any_count_exact": top_k_has_count,
            "top_k_any_reduced_formula_exact": top_k_has_formula,
            "top_k_any_composition_exact": top_k_has_composition,
            "lattice_compatible": top_1["lattice_compatible"],
            "top_k_any_lattice_compatible": top_k_has_lattice,
            "valid": top_1["valid"],
            "rmsd": top_1["rmsd"],
            "matched": metrics["matched"],
            "top_1_match": metrics["top_1_match"],
            "top_k_match": metrics["top_k_match"],
            "best_rmsd": metrics["best_rmsd"],
            "min_rmsd": metrics["min_rmsd"],
            "best_candidate_index": metrics["best_candidate_index"],
            "saved_candidate_index": saved_index,
            "gt_cif": str(gt_path),
            "predicted_cif": str(pred_path) if pred_path else None,
            "candidates": candidate_records,
        }
        rows.append(row)
        with (case_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)

        done = len(rows)
        top_k_rate = top_k_matches / done
        print(
            "case={done}/{total} index={index} top_1={top_1} "
            "top_{top_k}={top_k_match} top_{top_k}_rate={rate:.4f} "
            "valid={valid}/{num_evals}".format(
                done=done,
                total=len(items),
                index=dataset_index,
                top_1=metrics["top_1_match"],
                top_k=selected_top_k,
                top_k_match=metrics["top_k_match"],
                rate=top_k_rate,
                valid=len(valid_indices),
                num_evals=num_evals,
            ),
            flush=True,
        )

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [name for name in rows[0] if name != "candidates"]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    total = len(rows)
    total_candidates = total * num_evals
    top_k_failures = total - top_k_matches
    formula_limited_failures = sum(
        int(not row["matched"] and not row["top_k_any_reduced_formula_exact"])
        for row in rows
    )
    atom_count_limited_failures = sum(
        int(
            not row["matched"]
            and row["top_k_any_reduced_formula_exact"]
            and not row["top_k_any_composition_exact"]
        )
        for row in rows
    )
    lattice_limited_failures = sum(
        int(
            not row["matched"]
            and row["top_k_any_composition_exact"]
            and not row["top_k_any_lattice_compatible"]
        )
        for row in rows
    )
    coordinate_limited_failures = sum(
        int(
            not row["matched"]
            and row["top_k_any_lattice_compatible"]
        )
        for row in rows
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_class": dataset_cls.__name__,
        "lmdb_path": str(Path(lmdb_path).resolve()),
        "offset": offset,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "num_samples": total,
        "num_cases": total,
        "num_evals": num_evals,
        "total_candidates": total_candidates,
        "num_denoise_steps": num_steps,
        "candidates_per_case": num_evals,
        "top_k": selected_top_k,
        "seed": seed,
        "formula_noise_std": float(formula_noise_std),
        "coord_update_mode": coord_update_mode,
        "coord_mask_start_fraction": float(coord_mask_start_fraction),
        "token_noise_exponent": float(token_noise_exponent),
        "geometry_noise_exponent": float(geometry_noise_exponent),
        "guidance_scale": float(guidance_scale),
        "external_condition": external_condition_builder is not None,
        "formula_noise_seed": formula_noise_seed,
        "valid_structures": valid,
        "invalid_lattice_candidates": invalid_lattice_candidates,
        "invalid_lattice_candidate_rate": invalid_lattice_candidates / total_candidates,
        "candidate_validity_rate": valid / total_candidates,
        "top_1_matches": top_1_matches,
        "top_1_match_rate": top_1_matches / total,
        "top_k_matches": top_k_matches,
        "top_k_match_rate": top_k_matches / total,
        f"top_{selected_top_k}_matches": top_k_matches,
        f"top_{selected_top_k}_match_rate": top_k_matches / total,
        "matched": matched,
        "match_rate": matched / total,
        "count_exact": count_exact,
        "count_exact_rate": count_exact / total,
        "top_1_count_exact": count_exact,
        "top_1_count_exact_rate": count_exact / total,
        f"top_{selected_top_k}_any_count_exact": top_k_count_exact,
        f"top_{selected_top_k}_any_count_exact_rate": top_k_count_exact / total,
        "reduced_formula_exact": reduced_formula_exact,
        "reduced_formula_exact_rate": reduced_formula_exact / total,
        "top_1_reduced_formula_exact": reduced_formula_exact,
        "top_1_reduced_formula_exact_rate": reduced_formula_exact / total,
        f"top_{selected_top_k}_any_reduced_formula_exact": top_k_formula_exact,
        f"top_{selected_top_k}_any_reduced_formula_exact_rate": (
            top_k_formula_exact / total
        ),
        "full_composition_exact": full_composition_exact,
        "full_composition_exact_rate": full_composition_exact / total,
        "top_1_composition_exact": full_composition_exact,
        "top_1_composition_exact_rate": full_composition_exact / total,
        f"top_{selected_top_k}_any_composition_exact": top_k_composition_exact,
        f"top_{selected_top_k}_any_composition_exact_rate": (
            top_k_composition_exact / total
        ),
        "lattice_compatible": lattice_compatible,
        "lattice_compatible_rate": lattice_compatible / total,
        f"top_{selected_top_k}_any_lattice_compatible": top_k_lattice_compatible,
        f"top_{selected_top_k}_any_lattice_compatible_rate": (
            top_k_lattice_compatible / total
        ),
        "top_k_failures": top_k_failures,
        "failure_attribution": {
            "formula": formula_limited_failures,
            "atom_count_or_stoichiometric_multiplicity": atom_count_limited_failures,
            "lattice": lattice_limited_failures,
            "coordinates": coordinate_limited_failures,
        },
        "finite_rmsd_count": len(finite_rmsd),
        "rmsd_threshold": RMSD_MATCH_THRESHOLD,
        "matcher": dict(REALPXRD_SOLVER_MATCHER),
        "minimum_finite_rmsd": min(finite_rmsd) if finite_rmsd else None,
        "mean_finite_rmsd": mean_or_none(finite_rmsd),
        "minimum_best_matched_rmsd": (
            min(best_matched_rmsd) if best_matched_rmsd else None
        ),
        "mean_best_matched_rmsd": mean_or_none(best_matched_rmsd),
        "sampling_seconds": time.monotonic() - started_at,
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MP-20 with the RealPXRD-Solver RMSD standard")
    parser.add_argument("--config", required=True)
    parser.add_argument("--lmdb", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num_evals", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None, help="Alias for --num_evals")
    parser.add_argument("--num_steps", type=int, default=None)
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
        help=(
            "Continuous reverse solver override. By default use the checkpoint "
            "sampling profile (legacy checkpoints fall back to endpoint)."
        ),
    )
    parser.add_argument(
        "--coord_mask_start_fraction",
        type=float,
        default=None,
        help=(
            "Override the checkpoint fraction at which EOS masking may feed "
            "back into later coordinate states. New Euler profiles use 1.0; "
            "legacy endpoint profiles use 0.0."
        ),
    )
    parser.add_argument(
        "--token_noise_exponent",
        type=float,
        default=None,
        help=(
            "Override the discrete-token noise schedule exponent. Use 0.5 "
            "for token noise sqrt(t), while geometry remains on t."
        ),
    )
    parser.add_argument(
        "--geometry_noise_exponent",
        type=float,
        default=None,
        help=(
            "Override the lattice/coordinate noise schedule exponent. Use "
            "0.5 to make geometry denoise later than tokens on schedule t."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Override classifier-free guidance strength (1 disables guidance)",
    )
    parser.add_argument("--dllm_ckpt", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dllm_ckpt:
        num_evals = args.num_evals if args.num_evals is not None else args.num_samples
        result = evaluate_mp20_dllm(
            args.dllm_ckpt,
            args.lmdb,
            args.out_dir,
            max_cases=args.max_cases or 50,
            offset=args.offset,
            num_evals=num_evals if num_evals is not None else 1,
            num_steps=args.num_steps or 100,
            top_k=args.top_k,
            seed=args.seed,
            formula_noise_std=args.formula_noise_std,
            coord_update_mode=args.coord_update_mode,
            coord_mask_start_fraction=args.coord_mask_start_fraction,
            token_noise_exponent=args.token_noise_exponent,
            geometry_noise_exponent=args.geometry_noise_exponent,
            guidance_scale=args.guidance_scale,
        )
        print(json.dumps(result, indent=2))
        return
    num_evals = args.num_evals if args.num_evals is not None else args.num_samples
    result = evaluate_mp20_realpxrd_solver(
        load_config(args.config),
        args.lmdb,
        args.out_dir,
        max_cases=args.max_cases,
        offset=args.offset,
        num_evals=num_evals,
        num_steps=args.num_steps,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
