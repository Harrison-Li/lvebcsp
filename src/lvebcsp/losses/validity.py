"""Simple geometric validity checks and penalties."""

from __future__ import annotations

import math

import numpy as np
import torch
from pymatgen.core import Structure

from lvebcsp.data.representation import decode_lattice_params


def lattice_validity_penalty(
    encoded_lattice: torch.Tensor,
    min_angle: float = 30.0,
    max_angle: float = 150.0,
) -> torch.Tensor:
    """Soft penalty for non-positive lengths or unreasonable angles."""

    params = decode_lattice_params(encoded_lattice)
    lengths = params[..., :3]
    angles = params[..., 3:6]
    length_penalty = torch.relu(1e-4 - lengths).mean()
    angle_penalty = torch.relu(min_angle - angles).mean() + torch.relu(angles - max_angle).mean()
    return length_penalty + angle_penalty / 180.0


def min_pair_distance(structure: Structure) -> float:
    """Return the smallest pairwise distance in angstrom, or inf for one atom."""

    if len(structure) < 2:
        return math.inf
    distance_matrix = structure.distance_matrix
    mask = ~torch.eye(len(structure), dtype=torch.bool).numpy()
    return float(distance_matrix[mask].min())


def structure_validity_penalty(
    structure: Structure,
    min_distance: float = 0.5,
    min_angle: float = 30.0,
    max_angle: float = 150.0,
) -> tuple[float, dict[str, float]]:
    """Check basic structural validity and return a non-negative penalty."""

    lattice = structure.lattice
    lengths = [lattice.a, lattice.b, lattice.c]
    angles = [lattice.alpha, lattice.beta, lattice.gamma]
    penalty = 0.0
    if not all(math.isfinite(value) and value > 0 for value in lengths):
        penalty += 10.0
    if not all(math.isfinite(value) and min_angle <= value <= max_angle for value in angles):
        penalty += 5.0
    min_dist = min_pair_distance(structure)
    if min_dist < min_distance:
        penalty += float((min_distance - min_dist) / max(min_distance, 1e-6))
    return penalty, {"min_pair_distance": min_dist}


def has_usable_lattice(
    structure: Structure,
    min_angle: float = 30.0,
    max_angle: float = 150.0,
    min_length: float = 1.0e-3,
    min_volume: float = 1.0e-6,
) -> bool:
    """Return whether a structure's lattice is safe for pymatgen matching/simulation."""

    try:
        lattice = structure.lattice
        matrix = np.asarray(lattice.matrix, dtype=float)
        lengths = np.asarray([lattice.a, lattice.b, lattice.c], dtype=float)
        angles = np.asarray([lattice.alpha, lattice.beta, lattice.gamma], dtype=float)
        volume = float(lattice.volume)
    except Exception:
        return False
    return bool(
        np.isfinite(matrix).all()
        and np.isfinite(lengths).all()
        and np.isfinite(angles).all()
        and math.isfinite(volume)
        and np.all(lengths >= min_length)
        and np.all((angles >= min_angle) & (angles <= max_angle))
        and abs(volume) >= min_volume
    )
