"""Evaluation metrics for ranked structure candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure

from lvebcsp.eval.structure_match import build_matcher, structure_rms, structures_match
from lvebcsp.losses.validity import structure_validity_penalty
from lvebcsp.common.config import load_config

LEGACY_REALPXRD_MATCHER_CONFIG = {"stol": 0.5, "angle_tol": 10.0, "ltol": 0.3}
REFERENCE_SINGLE_MATCHER_CONFIG = {"stol": 0.7, "angle_tol": 5.0, "ltol": 0.2}
REFERENCE_BATCH_MATCHER_CONFIG = {"stol": 0.5, "angle_tol": 10.0, "ltol": 0.3}
REFERENCE_TOP_K = [1, 5, 10,20]

# Keep the original public constant for callers that import the previous evaluator.
REALPXRD_MATCHER_CONFIG = LEGACY_REALPXRD_MATCHER_CONFIG


def load_ranked_paths(pred_dir: str | Path, ranked_csv: str | Path | None = None) -> list[Path]:
    """Load candidate paths from CSV if provided, otherwise sorted CIF files."""

    if ranked_csv and Path(ranked_csv).exists():
        with Path(ranked_csv).open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [Path(row["candidate_path"]) for row in sorted(rows, key=lambda row: int(row.get("rank", 999999)))]
    return sorted(Path(pred_dir).glob("*.cif"))


def lattice_rmse(pred: Structure, reference: Structure) -> float:
    """RMSE over six lattice parameters."""

    pred_values = np.asarray([pred.lattice.a, pred.lattice.b, pred.lattice.c, pred.lattice.alpha, pred.lattice.beta, pred.lattice.gamma])
    ref_values = np.asarray([reference.lattice.a, reference.lattice.b, reference.lattice.c, reference.lattice.alpha, reference.lattice.beta, reference.lattice.gamma])
    return float(np.sqrt(np.mean((pred_values - ref_values) ** 2)))


def first_true_rank(values: list[bool]) -> int | None:
    """Return the 1-indexed rank of the first True value."""

    for index, value in enumerate(values, start=1):
        if value:
            return index
    return None


def finite_values(values: list[float | None]) -> list[float]:
    """Return finite floating-point values from optional metric lists."""

    return [float(value) for value in values if value is not None and np.isfinite(value)]


def mean_or_none(values: list[float | None]) -> float | None:
    """Return a mean when at least one finite value is available."""

    finite = finite_values(values)
    return float(np.mean(finite)) if finite else None


def best_or_none(values: list[float | None]) -> float | None:
    """Return the lowest finite value when available."""

    finite = finite_values(values)
    return min(finite) if finite else None


def normalize_top_k(top_k: list[int] | int | None, default: list[int] | None = None) -> list[int]:
    """Normalize top-k config values from YAML/CLI-style inputs."""

    if top_k is None:
        return list(default or REFERENCE_TOP_K)
    if isinstance(top_k, int):
        return [top_k]
    return [int(value) for value in top_k]


def reference_case_metrics(rms_values: list[float | None], top_k: list[int] | int | None = None) -> dict[str, Any]:
    """Metrics used by the reference evaluation script for one ranked case."""

    top_k_values = normalize_top_k(top_k, REFERENCE_TOP_K)
    best_rms = best_or_none(rms_values)
    metrics: dict[str, Any] = {
        "match_rate": 1.0 if best_rms is not None else 0.0,
        "rms_dist": best_rms,
    }
    for k in top_k_values:
        best_top_k = best_or_none(rms_values[:k])
        metrics[f"match_rate_top_{k}"] = 1.0 if best_top_k is not None else 0.0
        metrics[f"rms_dist_top_{k}"] = best_top_k
    return metrics


def aggregate_reference_metrics(
    per_case_rms_values: list[list[float | None]],
    top_k: list[int] | int | None = None,
) -> dict[str, Any]:
    """Aggregate the reference top-k evaluator over cases.

    This mirrors the pasted evaluator: for each case, take the minimum finite
    StructureMatcher RMS distance in the selected candidates, count cases with
    any finite RMS as matched, and average only over matched cases.
    """

    top_k_values = normalize_top_k(top_k, REFERENCE_TOP_K)
    total = len(per_case_rms_values)
    best_all = [best_or_none(values) for values in per_case_rms_values]
    metrics: dict[str, Any] = {
        "match_rate": len(finite_values(best_all)) / total if total else 0.0,
        "rms_dist": mean_or_none(best_all),
    }
    for k in top_k_values:
        best_top_k = [best_or_none(values[:k]) for values in per_case_rms_values]
        metrics[f"match_rate_top_{k}"] = len(finite_values(best_top_k)) / total if total else 0.0
        metrics[f"rms_dist_top_{k}"] = mean_or_none(best_top_k)
    return metrics


def evaluate_candidates(
    pred_dir: str | Path,
    reference_cif: str | Path,
    ranked_csv: str | Path | None = None,
    matcher_config: dict[str, float] | None = None,
    top_k: list[int] | None = None,
) -> dict[str, Any]:
    """Compute ranked candidate metrics, including RealPXRD-Solver benchmark fields."""

    if top_k is None:
        top_k = [1, 5, 20]
    elif isinstance(top_k, int):
        top_k = [top_k]
    matcher_config = {**REALPXRD_MATCHER_CONFIG, **(matcher_config or {})}
    matcher = build_matcher(
        stol=float(matcher_config.get("stol", 0.7)),
        angle_tol=float(matcher_config.get("angle_tol", 10.0)),
        ltol=float(matcher_config.get("ltol", 0.3)),
    )
    reference = Structure.from_file(reference_cif)
    paths = load_ranked_paths(pred_dir, ranked_csv)
    structures: list[Structure | None] = []
    for path in paths:
        try:
            structures.append(Structure.from_file(path))
        except Exception:
            structures.append(None)

    matches: list[bool] = []
    validity: list[bool] = []
    lattice_errors_by_rank: list[float | None] = []
    rms_values: list[float | None] = []
    for structure in structures:
        if structure is None:
            matches.append(False)
            validity.append(False)
            lattice_errors_by_rank.append(None)
            rms_values.append(None)
            continue
        matches.append(structures_match(structure, reference, matcher))
        validity.append(structure_validity_penalty(structure)[0] == 0)
        lattice_errors_by_rank.append(lattice_rmse(structure, reference))
        rms_values.append(structure_rms(structure, reference, matcher))

    finite_lattice_errors = finite_values(lattice_errors_by_rank)
    top_1_structural_rmse = rms_values[0] if rms_values else None
    best_structural_rmse = best_or_none(rms_values)
    metrics: dict[str, Any] = {
        "num_candidates": len(structures),
        "validity_rate": float(np.mean(validity)) if validity else 0.0,
        "lattice_rmse_best_ranked": lattice_errors_by_rank[0] if lattice_errors_by_rank else None,
        "coordinate_rms_best_match": best_structural_rmse,
        "candidate_diversity_lattice_std": float(np.std(finite_lattice_errors)) if finite_lattice_errors else 0.0,
        "latent_energy_ranking_accuracy": bool(matches[0]) if matches else False,
        "realpxrd_matcher_stol": float(matcher_config["stol"]),
        "realpxrd_matcher_angle_tol": float(matcher_config["angle_tol"]),
        "realpxrd_matcher_ltol": float(matcher_config["ltol"]),
        "structure_matcher_rms_dist_units": "normalized_by_average_free_length_per_atom",
        "realpxrd_first_match_rank": first_true_rank(matches),
        "realpxrd_structural_rmse_top_1": top_1_structural_rmse,
        "realpxrd_structural_rmse_best": best_structural_rmse,
        "realpxrd_rms_dist_top_1": top_1_structural_rmse,
        "realpxrd_rms_dist_best": best_structural_rmse,
    }
    for k in top_k:
        top_k_match = bool(any(matches[:k]))
        best_top_k_rms = best_or_none(rms_values[:k])
        metrics[f"top_{k}_structure_match"] = top_k_match
        metrics[f"realpxrd_top_{k}_match_rate"] = 1.0 if top_k_match else 0.0
        metrics[f"realpxrd_top_{k}_match_rate_percent"] = 100.0 if top_k_match else 0.0
        metrics[f"realpxrd_structural_rmse_best_top_{k}"] = best_top_k_rms
        metrics[f"realpxrd_rms_dist_best_top_{k}"] = best_top_k_rms
    return metrics


def evaluate_candidates_reference_method(
    pred_dir: str | Path,
    reference_cif: str | Path,
    ranked_csv: str | Path | None = None,
    matcher_config: dict[str, float] | None = None,
    top_k: list[int] | int | None = None,
) -> dict[str, Any]:
    """Evaluate ranked candidates with the reference RMSE/match-rate method."""

    top_k_values = normalize_top_k(top_k, REFERENCE_TOP_K)
    matcher_config = {**REFERENCE_BATCH_MATCHER_CONFIG, **(matcher_config or {})}
    matcher = build_matcher(
        stol=float(matcher_config.get("stol", REFERENCE_BATCH_MATCHER_CONFIG["stol"])),
        angle_tol=float(matcher_config.get("angle_tol", REFERENCE_BATCH_MATCHER_CONFIG["angle_tol"])),
        ltol=float(matcher_config.get("ltol", REFERENCE_BATCH_MATCHER_CONFIG["ltol"])),
    )
    reference = Structure.from_file(reference_cif)
    paths = load_ranked_paths(pred_dir, ranked_csv)
    rms_values: list[float | None] = []
    for path in paths:
        try:
            structure = Structure.from_file(path)
            rms_values.append(structure_rms(structure, reference, matcher))
        except Exception:
            rms_values.append(None)

    metrics = reference_case_metrics(rms_values, top_k_values)
    metrics.update(
        {
            "evaluation_method": "theirs",
            "num_candidates": len(paths),
            "reference_matcher_stol": float(matcher_config["stol"]),
            "reference_matcher_angle_tol": float(matcher_config["angle_tol"]),
            "reference_matcher_ltol": float(matcher_config["ltol"]),
            "structure_matcher_rms_dist_units": "normalized_by_average_free_length_per_atom",
        }
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked CIF candidates")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--reference_cif", default=None)
    parser.add_argument("--ranked_csv", default=None)
    parser.add_argument("--eval_method", choices=["theirs", "reference", "legacy"], default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else {}
    pred_dir = args.pred_dir or config.get("pred_dir")
    reference_cif = args.reference_cif or config.get("reference_cif")
    ranked_csv = args.ranked_csv or config.get("ranked_csv")
    if not pred_dir or not reference_cif:
        raise SystemExit("--pred_dir and --reference_cif are required")
    eval_method = str(args.eval_method or config.get("eval_method") or "theirs").lower()
    if eval_method == "reference":
        eval_method = "theirs"
    if eval_method == "legacy":
        metrics = evaluate_candidates(
            pred_dir,
            reference_cif,
            ranked_csv,
            matcher_config=config.get("structure_matcher", {}),
            top_k=config.get("metrics", {}).get("top_k", [1, 5, 20]),
        )
    else:
        metrics = evaluate_candidates_reference_method(
            pred_dir,
            reference_cif,
            ranked_csv,
            matcher_config=config.get("reference_structure_matcher", {}),
            top_k=config.get("reference_metrics", {}).get("top_k", REFERENCE_TOP_K),
        )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
