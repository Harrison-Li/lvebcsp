"""Rank candidate structures against experimental PXRD."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable

import torch
from pymatgen.core import Structure

from lvebcsp.data.xrd import DIPeakList, pattern_to_di, simulate_pxrd
from lvebcsp.losses.alignment import latent_energy
from lvebcsp.losses.validity import has_usable_lattice, structure_validity_penalty
from lvebcsp.models.jepa import Lvebm


def peak_list_distance(exp: DIPeakList, sim: DIPeakList) -> float:
    """Nearest-neighbor d-spacing mismatch weighted by experimental intensities."""

    exp_d = exp.peak_d[exp.peak_mask]
    exp_i = exp.peak_i[exp.peak_mask]
    sim_d = sim.peak_d[sim.peak_mask]
    sim_i = sim.peak_i[sim.peak_mask]
    if exp_d.numel() == 0 or sim_d.numel() == 0:
        return float("inf")
    distances = torch.cdist(exp_d.view(-1, 1), sim_d.view(-1, 1)).min(dim=1).values
    intensity_gap = torch.cdist(exp_i.view(-1, 1), sim_i.view(-1, 1)).min(dim=1).values
    weights = exp_i / exp_i.sum().clamp_min(1e-6)
    return float(torch.sum(weights * (distances + 0.25 * intensity_gap)).cpu())


def latent_energy_from_vectors(query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for normalized latent energy."""

    return latent_energy(query, candidate)


def rank_structures(
    structures: Iterable[tuple[Path, Structure]],
    exp_peaks: DIPeakList,
    pred_emb: torch.Tensor,
    jepa: Lvebm,
    p_max: int = 128,
    wavelength: str | float = "CuKa",
    ranking_weights: dict[str, float] | None = None,
    validity: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Score and sort candidate structures."""

    ranking_weights = ranking_weights or {"w_latent": 1.0, "w_peak": 0.5, "w_valid": 10.0}
    validity = validity or {}
    rows: list[dict[str, Any]] = []
    device = pred_emb.device
    jepa.eval()
    with torch.no_grad():
        for candidate_path, structure in structures:
            if not has_usable_lattice(
                structure,
                min_angle=float(validity.get("min_angle", 30.0)),
                max_angle=float(validity.get("max_angle", 150.0)),
            ):
                continue
            try:
                penalty, valid_metrics = structure_validity_penalty(
                    structure,
                    min_distance=float(validity.get("min_distance", 0.5)),
                    min_angle=float(validity.get("min_angle", 30.0)),
                    max_angle=float(validity.get("max_angle", 150.0)),
                )
            except Exception:
                penalty = 1.0e3
                valid_metrics = {"min_pair_distance": float("inf")}
            try:
                pattern = simulate_pxrd(structure, wavelength=wavelength)
                sim_peaks = pattern_to_di(pattern, p_max=p_max, wavelength=wavelength)
                if not bool(sim_peaks.peak_mask.any()):
                    raise ValueError("candidate PXRD simulation produced no peaks")
                z_candidate = jepa.encode_tgt(
                    sim_peaks.peak_d.unsqueeze(0).to(device),
                    sim_peaks.peak_i.unsqueeze(0).to(device),
                    sim_peaks.peak_mask.unsqueeze(0).to(device),
                )
                e_latent = float(latent_energy(pred_emb.view(1, -1), z_candidate).squeeze(0).cpu())
                d_peak = peak_list_distance(exp_peaks, sim_peaks)
            except Exception:
                e_latent = 1.0e6
                d_peak = 1.0e6
                penalty += 1.0e3
            lattice = structure.lattice
            score = (
                float(ranking_weights.get("w_latent", 1.0)) * e_latent
                + float(ranking_weights.get("w_peak", 0.5)) * d_peak
                + float(ranking_weights.get("w_valid", 10.0)) * penalty
            )
            rows.append(
                {
                    "candidate_path": str(candidate_path),
                    "score": score,
                    "latent_energy": e_latent,
                    "peak_distance": d_peak,
                    "validity_penalty": penalty,
                    "min_pair_distance": valid_metrics["min_pair_distance"],
                    "formula": structure.composition.reduced_formula,
                    "lattice_a": lattice.a,
                    "lattice_b": lattice.b,
                    "lattice_c": lattice.c,
                    "alpha": lattice.alpha,
                    "beta": lattice.beta,
                    "gamma": lattice.gamma,
                }
            )
    rows.sort(key=lambda row: row["score"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def save_ranked_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Save ranking rows to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No candidate rows to save")
    fieldnames = [
        "rank",
        "candidate_path",
        "score",
        "latent_energy",
        "peak_distance",
        "validity_penalty",
        "min_pair_distance",
        "formula",
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "alpha",
        "beta",
        "gamma",
    ]
    if any("oracle_rms_dist" in row for row in rows):
        fieldnames.append("oracle_rms_dist")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rank CIF candidates against PXRD")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args(argv)
    raise SystemExit("Use lvebcsp.inference.sample for checkpoint-backed ranking.")


if __name__ == "__main__":
    main()
