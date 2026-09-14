"""MP-20 RealPXRD-Solver evaluation after JEPA screening.

This keeps the RMSD success rule from ``mp20_realpxrd_solver`` but changes the
candidate order: generate a larger pool, rank the valid pool with JEPA PXRD
alignment, then evaluate only the screened structures in that ranked order.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from lvebcsp.data.mp20_lmdb import MP20LMDBDataset
from lvebcsp.data.representation import tensor_to_structure
from lvebcsp.data.xrd import DIPeakList
from lvebcsp.eval.mp20_realpxrd_solver import (
    DEFAULT_TOP_K,
    REALPXRD_SOLVER_MATCHER,
    RMSD_MATCH_THRESHOLD,
    best_matched_rmsd,
    case_metrics_from_rmsd,
    finite_rmsd_values,
    mean_or_none,
    min_finite_rmsd,
    reference_structure_from_item,
    rms_dist_or_none,
    select_saved_candidate_index,
)
from lvebcsp.inference.export_cif import export_structure_to_cif
from lvebcsp.inference.rank_pxrd import rank_structures
from lvebcsp.inference.sample import integrate_flow, load_models
from lvebcsp.common.config import load_config, select_device
from lvebcsp.common.seed import seed_everything

DEFAULT_NUM_GENERATED = 64


def item_peak_list(item: dict[str, Any]) -> DIPeakList:
    """Return the MP-20 row's target PXRD peaks as a DIPeakList."""

    return DIPeakList(
        peak_d=item["peak_d"],
        peak_i=item["peak_i"],
        peak_mask=item["peak_mask"],
    )


def jepa_screening_weights(config: dict[str, Any]) -> dict[str, float]:
    """Resolve ranking weights for JEPA screening.

    The default is pure JEPA latent energy. Add either of these to the config to
    include peak distance or validity penalties:

    ``jepa_screening: {ranking: {w_latent: 1.0, w_peak: 0.5, w_valid: 10.0}}``
    ``screening: {ranking_weights: {...}}``
    """

    for key in ("jepa_screening", "screening"):
        section = config.get(key)
        if isinstance(section, dict):
            weights = section.get("ranking") or section.get("ranking_weights")
            if isinstance(weights, dict):
                return {str(name): float(value) for name, value in weights.items()}
    return {"w_latent": 1.0, "w_peak": 0.0, "w_valid": 0.0}


def jepa_screening_validity(config: dict[str, Any]) -> dict[str, float]:
    """Resolve validity thresholds used during JEPA ranking."""

    for key in ("jepa_screening", "screening"):
        section = config.get(key)
        if isinstance(section, dict) and isinstance(section.get("validity"), dict):
            return {str(name): float(value) for name, value in section["validity"].items()}
    validity = config.get("validity", {})
    return {str(name): float(value) for name, value in validity.items()} if isinstance(validity, dict) else {}


def save_screened_ranked_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Save JEPA-ranked screened candidates, including RMSD diagnostics."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "screen_rank",
        "generated_index",
        "candidate_path",
        "score",
        "latent_energy",
        "peak_distance",
        "validity_penalty",
        "min_pair_distance",
        "oracle_rms_dist",
        "formula",
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "alpha",
        "beta",
        "gamma",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_jepa_per_case_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write per-case JEPA-screened RealPXRD metrics."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "material_id",
        "atoms",
        "num_generated",
        "generation_batch_size",
        "num_evals",
        "screen_top_k",
        "valid_candidates",
        "rankable_candidates",
        "screened_candidates",
        "finite_rmsd_candidates",
        "matched",
        "top_1_match",
        "top_k_match",
        "best_rmsd",
        "min_rmsd",
        "best_candidate_index",
        "best_generated_index",
        "saved_candidate_index",
        "saved_generated_index",
        "top_1_generated_index",
        "best_candidate_cif",
        "gt_primitive_cif",
        "screened_ranked_csv",
        "full_pool_ranked_csv",
        "pool_matched",
        "pool_best_rmsd",
        "pool_min_rmsd",
        "pool_first_match_jepa_rank",
        "pool_best_match_jepa_rank",
        "pool_min_rmsd_jepa_rank",
        "pool_finite_rmsd_candidates",
        "case_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


@torch.inference_mode()
def generate_structures_for_item(
    item: dict[str, Any],
    jepa,
    direct,
    device: torch.device,
    config: dict[str, Any],
    *,
    num_generated: int,
    generation_batch_size: int,
    num_steps: int,
    n_max: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Generate a pool of structures for one MP-20 target."""

    atom_types_one = item["atom_types"].unsqueeze(0).to(device)
    atom_mask_one = item["atom_mask"].unsqueeze(0).to(device)
    ratio_one = item["ratio"].unsqueeze(0).to(device)
    z_query = jepa.encode_ctx(
        item["peak_d"].unsqueeze(0).to(device),
        item["peak_i"].unsqueeze(0).to(device),
        item["peak_mask"].unsqueeze(0).to(device),
        atom_types=atom_types_one,
        ratio=ratio_one,
    )

    records: list[dict[str, Any]] = []
    generated = 0
    while generated < int(num_generated):
        chunk = min(int(generation_batch_size), int(num_generated) - generated)
        atom_types = atom_types_one.repeat(chunk, 1)
        atom_mask = atom_mask_one.repeat(chunk, 1)
        frac_coords = torch.rand(chunk, n_max, 3, device=device)
        frac_coords = frac_coords * atom_mask.float().unsqueeze(-1)
        y_final = integrate_flow(
            direct,
            frac_coords,
            z_query.repeat(chunk, 1),
            atom_types,
            atom_mask,
            num_steps=num_steps,
            guidance_scale=float(config.get("guidance_scale", 1.5)),
            sampler=str(config.get("sampler", "euler")),
            t_end=float(config.get("sample_t_max", config.get("t_end", 0.95))),
            final_denoise=bool(config.get("final_denoise", True)),
            diffuse_lattice=bool(config.get("diffuse_lattice", config.get("flow", {}).get("diffuse_lattice", False))),
        )
        for local_idx in range(chunk):
            generated_index = generated + local_idx
            try:
                structure = tensor_to_structure(
                    y_final[local_idx].cpu(),
                    item["atom_types"],
                    item["atom_mask"],
                )
            except Exception:
                continue
            records.append({"generated_index": generated_index, "structure": structure})
        generated += chunk
    return z_query.squeeze(0), records


def rank_generated_structures(
    records: list[dict[str, Any]],
    exp_peaks: DIPeakList,
    pred_emb: torch.Tensor,
    jepa,
    case_dir: Path,
    *,
    p_max: int,
    wavelength: str | float,
    ranking_weights: dict[str, float],
    validity: dict[str, float],
    save_all_candidates: bool = False,
) -> list[dict[str, Any]]:
    """Rank generated structures by JEPA score and attach original indices."""

    candidates_dir = case_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    path_to_record: dict[str, dict[str, Any]] = {}
    structures: list[tuple[Path, Structure]] = []
    for record in records:
        generated_index = int(record["generated_index"])
        candidate_path = candidates_dir / f"candidate_{generated_index:04d}.cif"
        if save_all_candidates:
            export_structure_to_cif(record["structure"], candidate_path)
        record["candidate_path"] = candidate_path
        path_to_record[str(candidate_path)] = record
        structures.append((candidate_path, record["structure"]))

    ranked_rows = rank_structures(
        structures,
        exp_peaks,
        pred_emb,
        jepa,
        p_max=p_max,
        wavelength=wavelength,
        ranking_weights=ranking_weights,
        validity=validity,
    )
    for row in ranked_rows:
        record = path_to_record[row["candidate_path"]]
        row["generated_index"] = int(record["generated_index"])
        row["_structure"] = record["structure"]
    return ranked_rows


def export_screened_rows(screened_rows: list[dict[str, Any]], case_dir: Path) -> None:
    """Export screened candidates and update their CSV paths."""

    screened_dir = case_dir / "screened"
    screened_dir.mkdir(parents=True, exist_ok=True)
    for screen_idx, row in enumerate(screened_rows, start=1):
        row["screen_rank"] = screen_idx
        output_path = screened_dir / f"rank_{screen_idx:04d}_gen_{int(row['generated_index']):04d}.cif"
        export_structure_to_cif(row["_structure"], output_path)
        row["candidate_path"] = str(output_path)


def evaluate_screened_rows(
    screened_rows: list[dict[str, Any]],
    matcher: StructureMatcher,
    gt_primitive: Structure,
) -> list[float | None]:
    """Compute solver RMSD values for JEPA-screened rows in rank order."""

    rmsd_values: list[float | None] = []
    for row in screened_rows:
        if "oracle_rms_dist" in row:
            rmsd = row["oracle_rms_dist"]
        else:
            try:
                rmsd = rms_dist_or_none(matcher, row["_structure"], gt_primitive)
            except Exception:
                rmsd = None
        row["oracle_rms_dist"] = rmsd
        rmsd_values.append(rmsd)
    return rmsd_values


def diagnose_full_pool_rows(
    ranked_rows: list[dict[str, Any]],
    matcher: StructureMatcher,
    gt_primitive: Structure,
    *,
    threshold: float = RMSD_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """Compute oracle RMSD diagnostics for all generated rows in JEPA rank order."""

    rmsd_values: list[float | None] = []
    for row in ranked_rows:
        try:
            rmsd = rms_dist_or_none(matcher, row["_structure"], gt_primitive)
        except Exception:
            rmsd = None
        row["oracle_rms_dist"] = rmsd
        rmsd_values.append(rmsd)

    first_match_rank = None
    best_match_rank = None
    best_match_rmsd = None
    for rank, rmsd in enumerate(rmsd_values, start=1):
        if rmsd is not None and float(rmsd) < float(threshold):
            if first_match_rank is None:
                first_match_rank = rank
            if best_match_rmsd is None or float(rmsd) < float(best_match_rmsd):
                best_match_rmsd = float(rmsd)
                best_match_rank = rank
    min_rmsd = min_finite_rmsd(rmsd_values)
    min_rank = None
    if min_rmsd is not None:
        min_rank = next(
            rank
            for rank, value in enumerate(rmsd_values, start=1)
            if value is not None and float(value) == float(min_rmsd)
        )
    return {
        "pool_matched": best_match_rmsd is not None,
        "pool_best_rmsd": best_match_rmsd,
        "pool_min_rmsd": min_rmsd,
        "pool_first_match_jepa_rank": first_match_rank,
        "pool_best_match_jepa_rank": best_match_rank,
        "pool_min_rmsd_jepa_rank": min_rank,
        "pool_finite_rmsd_candidates": len(finite_rmsd_values(rmsd_values)),
    }


def generated_index_at(rows: list[dict[str, Any]], index: int | None) -> int | None:
    """Return the original generation index for a screened row index."""

    if index is None or index < 0 or index >= len(rows):
        return None
    return int(rows[index]["generated_index"])


def evaluate_mp20_realpxrd_jepa_screen(
    config: dict[str, Any],
    lmdb_path: str | Path,
    out_dir: str | Path,
    *,
    max_cases: int | None = None,
    offset: int = 0,
    num_generated: int | None = None,
    screen_top_k: int | None = None,
    num_steps: int | None = None,
    generation_batch_size: int | None = None,
    top_k: int | None = None,
    save_all_candidates: bool = False,
    diagnose_full_pool: bool = False,
) -> dict[str, Any]:
    """Generate, JEPA-screen, then evaluate MP-20 candidates with RMSD < 0.5."""

    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))
    jepa, direct = load_models(config, device)
    p_max = int(config.get("p_max", 128))
    n_max = int(config.get("n_max", 20))
    generated_count = int(num_generated or config.get("num_generated", config.get("num_samples", DEFAULT_NUM_GENERATED)))
    selected_screen_top_k = int(screen_top_k or config.get("screen_top_k", config.get("top_k", DEFAULT_TOP_K)))
    steps = int(num_steps or config.get("num_steps", 100))
    gen_batch = int(generation_batch_size or config.get("generation_batch_size", generated_count))
    selected_top_k = int(top_k or config.get("top_k", min(DEFAULT_TOP_K, selected_screen_top_k)))
    if generated_count <= 0:
        raise ValueError("num_generated must be positive")
    if selected_screen_top_k <= 0:
        raise ValueError("screen_top_k must be positive")
    if gen_batch <= 0:
        raise ValueError("generation_batch_size must be positive")
    two_theta_range = config.get("two_theta_range", [5.0, 80.0])
    ranking_weights = jepa_screening_weights(config)
    validity = jepa_screening_validity(config)
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
        "evaluating MP-20 with JEPA screening + RealPXRD-Solver RMSD "
        "offset={offset} total={total} num_generated={num_generated} screen_top_k={screen_top_k} "
        "num_steps={steps} generation_batch_size={gen_batch} top_k={top_k}".format(
            offset=offset,
            total=total_cases,
            num_generated=generated_count,
            screen_top_k=selected_screen_top_k,
            steps=steps,
            gen_batch=gen_batch,
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

        z_query, generated_records = generate_structures_for_item(
            item,
            jepa,
            direct,
            device,
            config,
            num_generated=generated_count,
            generation_batch_size=gen_batch,
            num_steps=steps,
            n_max=n_max,
        )
        ranked_rows = rank_generated_structures(
            generated_records,
            item_peak_list(item),
            z_query,
            jepa,
            case_dir,
            p_max=p_max,
            wavelength=config.get("wavelength", "CuKa"),
            ranking_weights=ranking_weights,
            validity=validity,
            save_all_candidates=save_all_candidates,
        )
        pool_diagnostics: dict[str, Any] = {}
        full_pool_ranked_csv = None
        if diagnose_full_pool:
            pool_diagnostics = diagnose_full_pool_rows(ranked_rows, matcher, gt_primitive)
            full_pool_ranked_csv = save_screened_ranked_csv(ranked_rows, case_dir / "full_pool_ranked_candidates.csv")

        screened_rows = ranked_rows[:selected_screen_top_k]
        export_screened_rows(screened_rows, case_dir)
        rmsd_values = evaluate_screened_rows(screened_rows, matcher, gt_primitive)
        screened_ranked_csv = save_screened_ranked_csv(screened_rows, case_dir / "screened_ranked_candidates.csv")

        metrics = case_metrics_from_rmsd(rmsd_values, top_k=selected_top_k, threshold=RMSD_MATCH_THRESHOLD)
        saved_index = select_saved_candidate_index(
            rmsd_values,
            list(range(len(screened_rows))),
            threshold=RMSD_MATCH_THRESHOLD,
        )
        best_candidate_cif = None
        if saved_index is not None:
            suffix = best_matched_rmsd(rmsd_values, RMSD_MATCH_THRESHOLD)
            best_candidate_cif = export_structure_to_cif(
                screened_rows[saved_index]["_structure"],
                case_dir / f"atom{atoms}_pre_{suffix}.cif",
            )

        case_record = {
            "index": index,
            "material_id": item.get("material_id", f"mp20_{index}"),
            "atoms": atoms,
            "num_generated": generated_count,
            "generation_batch_size": gen_batch,
            "num_evals": len(screened_rows),
            "screen_top_k": selected_screen_top_k,
            "valid_candidates": len(generated_records),
            "rankable_candidates": len(ranked_rows),
            "screened_candidates": len(screened_rows),
            "finite_rmsd_candidates": len(finite_rmsd_values(rmsd_values)),
            "best_generated_index": generated_index_at(screened_rows, metrics["best_candidate_index"]),
            "saved_candidate_index": saved_index,
            "saved_generated_index": generated_index_at(screened_rows, saved_index),
            "top_1_generated_index": generated_index_at(screened_rows, 0),
            "best_candidate_cif": str(best_candidate_cif) if best_candidate_cif else None,
            "gt_primitive_cif": str(gt_primitive_cif),
            "screened_ranked_csv": str(screened_ranked_csv),
            "full_pool_ranked_csv": str(full_pool_ranked_csv) if full_pool_ranked_csv else None,
            "case_dir": str(case_dir),
            **pool_diagnostics,
            **metrics,
        }
        per_case.append(case_record)

        elapsed = time.monotonic() - started_at
        done = len(per_case)
        top_k_rate = sum(int(row["top_k_match"]) for row in per_case) / max(done, 1)
        print(
            "case={done}/{total} index={index} matched={matched} top_k_rate={top_k_rate:.4f} "
            "generated_valid={valid}/{generated} rankable={rankable} screened={screened} elapsed={elapsed:.1f}s".format(
                done=done,
                total=total_cases,
                index=index,
                matched=bool(metrics["matched"]),
                top_k_rate=top_k_rate,
                valid=len(generated_records),
                generated=generated_count,
                rankable=len(ranked_rows),
                screened=len(screened_rows),
                elapsed=elapsed,
            ),
            flush=True,
        )

    per_case_csv = save_jepa_per_case_csv(per_case, output_dir / "per_case_metrics.csv")
    total = len(per_case)
    matched_best = [row["best_rmsd"] for row in per_case]
    top_1_matches = sum(int(row["top_1_match"]) for row in per_case)
    top_k_matches = sum(int(row["top_k_match"]) for row in per_case)
    matched_cases = sum(int(row["matched"]) for row in per_case)
    total_generated = sum(int(row["num_generated"]) for row in per_case)
    total_valid = sum(int(row["valid_candidates"]) for row in per_case)
    total_rankable = sum(int(row["rankable_candidates"]) for row in per_case)
    total_screened = sum(int(row["screened_candidates"]) for row in per_case)
    total_finite_rmsd = sum(int(row["finite_rmsd_candidates"]) for row in per_case)
    pool_matched_cases = sum(int(row.get("pool_matched", False)) for row in per_case)
    pool_matches_missed = sum(
        int(bool(row.get("pool_matched", False)) and not bool(row["matched"]))
        for row in per_case
    )
    first_match_ranks = [row.get("pool_first_match_jepa_rank") for row in per_case]
    summary: dict[str, Any] = {
        "eval_mode": "jepa_screened_realpxrd_solver",
        "lmdb_path": str(lmdb_path),
        "out_dir": str(output_dir),
        "offset": offset,
        "num_cases": total,
        "num_generated": generated_count,
        "generation_batch_size": gen_batch,
        "screen_top_k": selected_screen_top_k,
        "num_evals": selected_screen_top_k,
        "num_steps": steps,
        "top_k": selected_top_k,
        "sampler": str(config.get("sampler", "euler")),
        "sample_t_max": float(config.get("sample_t_max", config.get("t_end", 0.95))),
        "final_denoise": bool(config.get("final_denoise", True)),
        "screening_weights": ranking_weights,
        "screening_validity": validity,
        "diagnose_full_pool": bool(diagnose_full_pool),
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
        "candidate_validity_rate": total_valid / total_generated if total_generated else 0.0,
        "candidate_rankable_rate": total_rankable / total_valid if total_valid else 0.0,
        "screened_rmsd_rate": total_finite_rmsd / total_screened if total_screened else 0.0,
        "full_pool_match_rate": pool_matched_cases / total if diagnose_full_pool and total else None,
        "jepa_recall_of_pool_matches": matched_cases / pool_matched_cases
        if diagnose_full_pool and pool_matched_cases
        else None,
        "pool_matches_missed_by_screening": pool_matches_missed if diagnose_full_pool else None,
        "mean_pool_first_match_jepa_rank": mean_or_none(first_match_ranks) if diagnose_full_pool else None,
        "per_case_metrics_csv": str(per_case_csv),
        "metrics_json": str(output_dir / "metrics.json"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MP-20 after JEPA screening with RealPXRD-Solver RMSD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--lmdb", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num_generated", type=int, default=None, help="Number of structures to generate before screening")
    parser.add_argument("--num_samples", type=int, default=None, help="Alias for --num_generated")
    parser.add_argument("--screen_top_k", type=int, default=None, help="Number of JEPA-ranked structures to evaluate")
    parser.add_argument("--num_evals", type=int, default=None, help="Alias for --screen_top_k")
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--generation_batch_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None, help="Top-k metric over the screened ranked structures")
    parser.add_argument("--save_all_candidates", action="store_true", help="Also export every valid generated candidate CIF")
    parser.add_argument(
        "--diagnose_full_pool",
        action="store_true",
        help="Compute oracle RMSD for every generated/ranked candidate to diagnose JEPA screening recall",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    num_generated = args.num_generated if args.num_generated is not None else args.num_samples
    screen_top_k = args.screen_top_k if args.screen_top_k is not None else args.num_evals
    result = evaluate_mp20_realpxrd_jepa_screen(
        load_config(args.config),
        args.lmdb,
        args.out_dir,
        max_cases=args.max_cases,
        offset=args.offset,
        num_generated=num_generated,
        screen_top_k=screen_top_k,
        num_steps=args.num_steps,
        generation_batch_size=args.generation_batch_size,
        top_k=args.top_k,
        save_all_candidates=args.save_all_candidates,
        diagnose_full_pool=args.diagnose_full_pool,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
