"""PXRD simulation and d-spacing/intensity peak-list conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

CU_KA_WAVELENGTH = 1.5406
WAVELENGTHS = {
    "CuKa": CU_KA_WAVELENGTH,
    "CuKalpha": CU_KA_WAVELENGTH,
    "CuKa1": 1.54056,
    "MoKa": 0.71073,
}


@dataclass(frozen=True)
class DIPeakList:
    """Padded d-spacing/intensity representation."""

    peak_d: torch.Tensor
    peak_i: torch.Tensor
    peak_mask: torch.Tensor


def wavelength_to_angstrom(wavelength: str | float) -> float:
    """Resolve a named or numeric X-ray wavelength in angstrom."""

    if isinstance(wavelength, (int, float, np.integer, np.floating)):
        resolved = float(wavelength)
    else:
        if wavelength in WAVELENGTHS:
            resolved = WAVELENGTHS[wavelength]
        else:
            try:
                resolved = float(XRDCalculator(wavelength=wavelength).wavelength)
            except Exception as exc:
                raise ValueError(
                    f"Unknown wavelength {wavelength!r}; local names are {sorted(WAVELENGTHS)}"
                ) from exc
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"Wavelength must be positive and finite, got {wavelength!r}")
    return resolved


def structure_to_primitive(structure: Structure) -> Structure:
    """Convert a structure to a primitive cell when pymatgen can determine one."""

    try:
        return SpacegroupAnalyzer(structure, symprec=0.1).find_primitive() or structure.get_primitive_structure()
    except Exception:
        return structure.get_primitive_structure()


def simulate_pxrd(
    structure: Structure,
    wavelength: str | float = "CuKa",
    two_theta_range: tuple[float, float] = (5.0, 80.0),
):
    """Simulate a powder diffraction pattern with pymatgen."""

    # Resolve aliases ourselves so simulation and subsequent two-theta-to-d
    # conversion use exactly the same wavelength.  Pymatgen's ``CuKa`` alias
    # is 1.54184 A, whereas this project defines ``CuKa`` as 1.5406 A.
    calc = XRDCalculator(wavelength=wavelength_to_angstrom(wavelength))
    return calc.get_pattern(structure, two_theta_range=two_theta_range)


def two_theta_to_d(two_theta_deg: np.ndarray | torch.Tensor | float, wavelength_angstrom: float = CU_KA_WAVELENGTH):
    """Convert diffraction angle 2theta in degrees to d-spacing using Bragg's law."""

    if isinstance(two_theta_deg, torch.Tensor):
        theta = torch.deg2rad(two_theta_deg.to(torch.float32) / 2.0)
        return wavelength_angstrom / (2.0 * torch.sin(theta).clamp_min(1e-12))
    array = np.asarray(two_theta_deg, dtype=np.float64)
    theta = np.deg2rad(array / 2.0)
    return wavelength_angstrom / (2.0 * np.clip(np.sin(theta), 1e-12, None))


def normalize_intensity(intensities: Iterable[float] | np.ndarray | torch.Tensor):
    """Scale intensities so that the maximum positive value is 1."""

    if isinstance(intensities, torch.Tensor):
        values = intensities.to(torch.float32).clamp_min(0.0)
        max_value = values.max() if values.numel() else torch.tensor(0.0, device=values.device)
        return values / max_value.clamp_min(1e-12)
    values = np.asarray(intensities, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    max_value = float(values.max()) if values.size else 0.0
    if max_value <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values / max_value


def pad_peak_list(peak_d: Iterable[float], peak_i: Iterable[float], p_max: int) -> DIPeakList:
    """Pad or truncate a peak list to fixed tensors."""

    d = torch.as_tensor(np.asarray(list(peak_d), dtype=np.float32), dtype=torch.float32)
    i = torch.as_tensor(np.asarray(list(peak_i), dtype=np.float32), dtype=torch.float32)
    if d.numel() != i.numel():
        raise ValueError(f"peak_d and peak_i must have the same length, got {d.numel()} and {i.numel()}")
    keep = min(int(p_max), d.numel())
    out_d = torch.zeros(p_max, dtype=torch.float32)
    out_i = torch.zeros(p_max, dtype=torch.float32)
    mask = torch.zeros(p_max, dtype=torch.bool)
    if keep:
        out_d[:keep] = d[:keep]
        out_i[:keep] = i[:keep]
        mask[:keep] = True
    return DIPeakList(out_d, out_i, mask)


def _pattern_arrays(pattern) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(pattern, "x") and hasattr(pattern, "y"):
        return np.asarray(pattern.x, dtype=np.float64), np.asarray(pattern.y, dtype=np.float64)
    if isinstance(pattern, dict):
        return np.asarray(pattern["two_theta"], dtype=np.float64), np.asarray(pattern["intensity"], dtype=np.float64)
    if isinstance(pattern, (tuple, list)) and len(pattern) >= 2:
        return np.asarray(pattern[0], dtype=np.float64), np.asarray(pattern[1], dtype=np.float64)
    raise TypeError("pattern must be a pymatgen pattern, dict, or (two_theta, intensity) pair")


def pattern_to_di(
    pattern,
    p_max: int,
    min_intensity: float = 1e-4,
    wavelength: str | float = "CuKa",
) -> DIPeakList:
    """Convert an XRD pattern to a normalized, sorted, padded d-I peak list."""

    two_theta, intensity = _pattern_arrays(pattern)
    intensity = normalize_intensity(intensity)
    valid = np.isfinite(two_theta) & np.isfinite(intensity) & (intensity >= min_intensity)
    two_theta = two_theta[valid]
    intensity = intensity[valid]
    if two_theta.size:
        order = np.argsort(two_theta)  # increasing angle equals decreasing d.
        two_theta = two_theta[order]
        intensity = intensity[order]
    d_spacing = two_theta_to_d(two_theta, wavelength_to_angstrom(wavelength))
    return pad_peak_list(d_spacing[:p_max], intensity[:p_max], p_max)
