"""CCDC organic-crystal LMDB dataset adapter."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import torch

from lvebcsp.common.chemistry import (
    atom_types_to_element_counts,
    element_counts_to_ratios,
)
from lvebcsp.data.mp20_lmdb import MP20LMDBDataset, _atomic_number


class OrganicLMDBDataset(MP20LMDBDataset):
    """Read zero-padded CCDC rows into the shared crystal batch schema."""

    def __init__(
        self,
        lmdb_path: str | Path,
        *args: Any,
        include_molecule_template: bool = False,
        molecule_template_n_max: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(lmdb_path, *args, **kwargs)
        self.include_molecule_template = bool(include_molecule_template)
        self.molecule_template_n_max = int(
            self.n_max
            if molecule_template_n_max is None
            else molecule_template_n_max
        )
        if self.molecule_template_n_max < 1:
            raise ValueError("molecule_template_n_max must be positive")

    def _load_row(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self._length:
            raise IndexError(
                f"Organic LMDB index {index} out of range for length {self._length}"
            )

        key = f"{index:08d}".encode("utf-8")
        env = self._open_env()
        with env.begin() as txn:
            payload = txn.get(key)
        if payload is None:
            raise IndexError(f"No organic LMDB row for index {index} (key={key!r})")

        row = pickle.loads(gzip.decompress(payload))
        if not isinstance(row, dict):
            raise TypeError(f"Expected organic LMDB row dict, got {type(row).__name__}")
        row = dict(row)
        required = ("crystal_atom_type", "crystal_atom_pos", "lattice_matrix")
        missing = [name for name in required if name not in row]
        if missing:
            raise KeyError(
                "Organic LMDB row is missing required fields: " + ", ".join(missing)
            )

        row["atom_type"] = row["crystal_atom_type"]
        row["atom_pos"] = row["crystal_atom_pos"]
        row.setdefault(
            "material_id",
            str(row.get("cif_filename") or f"organic_{key.decode('utf-8')}"),
        )
        row.setdefault("_dataset_index", index)
        row.setdefault("_lmdb_key", key.decode("utf-8"))
        return row

    def _molecule_template_tensors(
        self,
        row: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Encode only the deduplicated molecular templates.

        ``mol_atom_type_2`` concatenates one copy of each unique molecular
        building block and ``mol2_group_slices`` separates those blocks.
        Unit-cell multiplicities, target atom counts, and target coordinates
        are deliberately excluded so the DLM must infer its EOS position.
        """

        required = ("mol_atom_type_2", "mol2_group_slices")
        missing = [name for name in required if name not in row]
        if missing:
            raise KeyError(
                "Organic LMDB row is missing molecular-template fields: "
                + ", ".join(missing)
            )

        symbols = [str(symbol) for symbol in list(row["mol_atom_type_2"])]
        template_atoms = len(symbols)
        if template_atoms < 1:
            raise ValueError("mol_atom_type_2 must contain at least one atom")
        if template_atoms > self.molecule_template_n_max:
            raise ValueError(
                f"molecular template has {template_atoms} atoms, above "
                f"molecule_template_n_max={self.molecule_template_n_max}"
            )

        raw_slices = list(row["mol2_group_slices"])
        if not raw_slices:
            raise ValueError("mol2_group_slices must contain at least one group")
        atom_types = torch.zeros(self.molecule_template_n_max, dtype=torch.long)
        mask = torch.zeros(self.molecule_template_n_max, dtype=torch.bool)
        group_indices = torch.zeros(self.molecule_template_n_max, dtype=torch.long)
        group_starts = torch.zeros(self.molecule_template_n_max, dtype=torch.bool)
        group_ends = torch.zeros(self.molecule_template_n_max, dtype=torch.bool)
        covered = torch.zeros(template_atoms, dtype=torch.bool)

        for group_index, raw_slice in enumerate(raw_slices, start=1):
            if len(raw_slice) != 2:
                raise ValueError(
                    f"molecular-template slice must be a pair, got {raw_slice!r}"
                )
            start, end = (int(raw_slice[0]), int(raw_slice[1]))
            if not 0 <= start < end <= template_atoms:
                raise ValueError(
                    f"invalid molecular-template slice {(start, end)} for "
                    f"{template_atoms} atoms"
                )
            if covered[start:end].any():
                raise ValueError("mol2_group_slices contains overlapping groups")
            covered[start:end] = True
            group_indices[start:end] = group_index
            group_starts[start] = True
            group_ends[end - 1] = True

        if not covered.all():
            raise ValueError(
                "mol2_group_slices must cover every mol_atom_type_2 atom exactly once"
            )
        atom_types[:template_atoms] = torch.tensor(
            [_atomic_number(symbol) for symbol in symbols],
            dtype=torch.long,
        )
        mask[:template_atoms] = True
        return {
            "molecule_template_atom_types": atom_types,
            "molecule_template_mask": mask,
            "molecule_template_group_indices": group_indices,
            "molecule_template_group_starts": group_starts,
            "molecule_template_group_ends": group_ends,
            "molecule_template_num_groups": torch.tensor(
                len(raw_slices),
                dtype=torch.long,
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self.include_molecule_template:
            return super().__getitem__(index)

        # Load the row once. Calling the parent implementation and then loading
        # template metadata would decompress every 1.2M-row training sample
        # twice.
        row = self._load_row(index)
        atom_types, frac_coords, atom_mask, lattice_params = self._structure_tensors(
            row
        )
        material_id = str(
            row.get("material_id") or f"organic_{row.get('_lmdb_key', index)}"
        )
        element_counts = atom_types_to_element_counts(atom_types)
        item: dict[str, Any] = {
            "material_id": material_id,
            "formula": "",
            "element_counts": element_counts,
            "ratio": element_counts_to_ratios(element_counts),
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "atom_mask": atom_mask,
            "lattice_params": lattice_params,
            **self._molecule_template_tensors(row),
        }
        if not self.include_pxrd:
            return item

        two_theta, intensity = self._spectrum(row)
        wavelength = self._spectrum_wavelength(row)
        peaks = self.augmenter.to_di(
            two_theta,
            intensity,
            p_max=self.p_max,
            wavelength=wavelength,
            two_theta_range=self.two_theta_range,
            augment=False,
        )
        aug_peaks = self.augmenter.to_di(
            two_theta,
            intensity,
            p_max=self.p_max,
            wavelength=wavelength,
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
