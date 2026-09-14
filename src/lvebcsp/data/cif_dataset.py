"""Manifest-backed CIF dataset for PXRD JEPA and direct CSP training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset

from lvebcsp.common.chemistry import element_counts_to_ratios, formula_to_element_counts

from lvebcsp.data.augment import PXRDDataAugmenter
from lvebcsp.data.manifest import ManifestEntry, read_manifest
from lvebcsp.data.xrd import simulate_pxrd, structure_to_primitive


class CIFPXRDDataset(Dataset):
    """Load CIFs and return padded structure and d-I PXRD tensors."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        n_max: int = 64,
        p_max: int = 128,
        wavelength: str | float = "CuKa",
        two_theta_range: tuple[float, float] = (5.0, 80.0),
        augmentation: dict[str, Any] | None = None,
        include_pxrd: bool = True,
    ) -> None:
        self.entries: list[ManifestEntry] = read_manifest(manifest_path, split=split)
        self.n_max = int(n_max)
        self.p_max = int(p_max)
        self.wavelength = wavelength
        self.two_theta_range = two_theta_range
        self.augmenter = PXRDDataAugmenter.from_config(augmentation)
        self.include_pxrd = bool(include_pxrd)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        if not entry.cif_path.exists():
            raise FileNotFoundError(
                f"CIF file listed in manifest does not exist: {entry.cif_path}"
            )
        structure = structure_to_primitive(Structure.from_file(entry.cif_path))
        if len(structure) > self.n_max:
            raise ValueError(
                f"{entry.cif_path} has {len(structure)} atoms after primitive "
                f"conversion, above n_max={self.n_max}"
            )

        atom_types = torch.zeros(self.n_max, dtype=torch.long)
        frac_coords = torch.zeros(self.n_max, 3, dtype=torch.float32)
        atom_mask = torch.zeros(self.n_max, dtype=torch.bool)
        for i, site in enumerate(structure.sites):
            atom_types[i] = int(site.specie.Z)
            frac_coords[i] = torch.as_tensor(site.frac_coords % 1.0, dtype=torch.float32)
            atom_mask[i] = True

        lattice = structure.lattice
        lattice_params = torch.tensor(
            [lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma],
            dtype=torch.float32,
        )

        formula = entry.formula or structure.composition.reduced_formula
        element_counts = formula_to_element_counts(formula)
        item = {
            "material_id": entry.material_id,
            "formula": formula,
            "element_counts": element_counts,
            "ratio": element_counts_to_ratios(element_counts),
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "atom_mask": atom_mask,
            "lattice_params": lattice_params,
        }
        if not self.include_pxrd:
            return item

        pattern = simulate_pxrd(
            structure,
            wavelength=self.wavelength,
            two_theta_range=self.two_theta_range,
        )
        peaks = self.augmenter.to_di(
            pattern.x,
            pattern.y,
            p_max=self.p_max,
            wavelength=self.wavelength,
            two_theta_range=self.two_theta_range,
            augment=False,
        )
        aug_peaks = self.augmenter.to_di(
            pattern.x,
            pattern.y,
            p_max=self.p_max,
            wavelength=self.wavelength,
            two_theta_range=self.two_theta_range,
            augment=True,
        )
        item.update(
            {
                "peak_d": peaks.peak_d,
                "peak_i": peaks.peak_i,
                "peak_mask": peaks.peak_mask,
                "aug_peak_d": aug_peaks.peak_d,
                "aug_peak_i": aug_peaks.peak_i,
                "aug_peak_mask": aug_peaks.peak_mask,
            }
        )
        return item
