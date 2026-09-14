"""MP-20 LMDB dataset adapter."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from pymatgen.core import Element, Lattice
from torch.utils.data import Dataset

from lvebcsp.common.chemistry import atom_types_to_element_counts, element_counts_to_ratios

from lvebcsp.data.augment import PXRDDataAugmenter


def _atomic_number(symbol: str) -> int:
    return int(Element(symbol).Z)


class MP20LMDBDataset(Dataset):
    """Read MP-20 gzip-pickled LMDB rows into the CIFPXRDDataset batch schema."""

    def __init__(
        self,
        lmdb_path: str | Path,
        n_max: int = 20,
        p_max: int = 128,
        wavelength: str | float = "CuKa",
        two_theta_range: tuple[float, float] = (5.0, 80.0),
        augmentation: dict[str, Any] | None = None,
        use_conventional: bool = False,
        include_pxrd: bool = True,
    ) -> None:
        self.lmdb_path = Path(lmdb_path)
        self.n_max = int(n_max)
        self.p_max = int(p_max)
        self.wavelength = wavelength
        self.two_theta_range = two_theta_range
        self.augmenter = PXRDDataAugmenter.from_config(augmentation)
        self.use_conventional = bool(use_conventional)
        self.include_pxrd = bool(include_pxrd)
        self._env: lmdb.Environment | None = None
        self._keys: tuple[bytes, ...] | None = None
        env = self._open_env()
        with env.begin() as txn:
            self._length = int(txn.stat()["entries"])
        self._close_env()

    def _open_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                subdir=False,
            )
        return self._env

    def _close_env(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    def __len__(self) -> int:
        return self._length

    def _ordinal_keys(self) -> tuple[bytes, ...]:
        if self._keys is None:
            env = self._open_env()
            with env.begin() as txn:
                self._keys = tuple(txn.cursor().iternext(keys=True, values=False))
        return self._keys

    def _load_row(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self._length:
            raise IndexError(
                f"MP-20 LMDB index {index} out of range for length {self._length}"
            )

        key = str(index).encode("utf-8")
        env = self._open_env()
        with env.begin() as txn:
            payload = txn.get(key)
        if payload is None:
            key = self._ordinal_keys()[index]
            with env.begin() as txn:
                payload = txn.get(key)
        if payload is None:
            raise IndexError(f"No MP-20 LMDB row for index {index}")
        row = pickle.loads(gzip.decompress(payload))
        if not isinstance(row, dict):
            raise TypeError(f"Expected MP-20 row dict, got {type(row).__name__}")
        row = dict(row)
        row.setdefault("_dataset_index", index)
        row.setdefault("_lmdb_key", key.decode("utf-8", errors="replace"))
        return row

    @staticmethod
    def _first_present(row: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in row:
                return row[key]
        raise KeyError(f"MP-20 LMDB row is missing all fields: {', '.join(keys)}")

    @staticmethod
    def _as_frac_coords(values: Any) -> np.ndarray:
        coords = np.asarray(values, dtype=np.float32)
        if coords.ndim == 3:
            coords = coords[0]
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Expected fractional coordinates with shape (N, 3), got {coords.shape}"
            )
        return coords

    def _structure_tensors(
        self,
        row: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        has_conventional = all(
            key in row
            for key in (
                "conventional_atom_type",
                "conventional_atom_pos",
                "conventional_lattice_matrix",
            )
        )
        if self.use_conventional and has_conventional:
            species = list(row["conventional_atom_type"])
            frac_coords = self._as_frac_coords(row["conventional_atom_pos"])
            lattice_matrix = np.asarray(row["conventional_lattice_matrix"], dtype=np.float64)
        else:
            species = list(self._first_present(row, "atom_type", "p_atom_type"))
            frac_coords = self._as_frac_coords(self._first_present(row, "atom_pos", "p_atom_pos"))
            lattice_matrix = np.asarray(
                self._first_present(row, "lattice_matrix", "p_lattice_matrix"),
                dtype=np.float64,
            )

        if len(species) > self.n_max:
            raise ValueError(f"MP-20 row has {len(species)} atoms, above n_max={self.n_max}")
        if len(species) != len(frac_coords):
            raise ValueError(
                f"MP-20 row has {len(species)} species but {len(frac_coords)} coordinate rows"
            )
        atomic_numbers = [_atomic_number(symbol) for symbol in species]
        frac_coords = np.remainder(frac_coords, 1.0)
        atom_types = torch.zeros(self.n_max, dtype=torch.long)
        coords = torch.zeros(self.n_max, 3, dtype=torch.float32)
        atom_mask = torch.zeros(self.n_max, dtype=torch.bool)
        for idx, atomic_number in enumerate(atomic_numbers):
            atom_types[idx] = atomic_number
            coords[idx] = torch.as_tensor(frac_coords[idx], dtype=torch.float32)
            atom_mask[idx] = True

        lattice = Lattice(lattice_matrix)
        lattice_params = torch.tensor([*lattice.abc, *lattice.angles], dtype=torch.float32)
        return atom_types, coords, atom_mask, lattice_params

    def _spectrum(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        two_theta = np.asarray(row["pxrd_x"], dtype=np.float64)
        intensity = np.asarray(row["pxrd_y"], dtype=np.float64)
        return two_theta, intensity

    def _spectrum_wavelength(self, row: dict[str, Any]) -> str | float:
        """Use row-level experimental metadata with the dataset value as fallback."""

        wavelength = row.get("wavelength")
        if wavelength is None or (isinstance(wavelength, str) and not wavelength.strip()):
            return self.wavelength
        if isinstance(wavelength, np.generic):
            return float(wavelength)
        return wavelength

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._load_row(index)
        atom_types, frac_coords, atom_mask, lattice_params = self._structure_tensors(row)
        material_id = str(row.get("material_id") or f"mp20_{row.get('_lmdb_key', index)}")
        element_counts = atom_types_to_element_counts(atom_types)
        item = {
            "material_id": material_id,
            "formula": "",
            "element_counts": element_counts,
            "ratio": element_counts_to_ratios(element_counts),
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "atom_mask": atom_mask,
            "lattice_params": lattice_params,
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
