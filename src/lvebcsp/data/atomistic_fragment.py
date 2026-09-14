"""Leakage-safe fragment conditions for the shared foundation DLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from lvebcsp.common.chemistry import (
    atom_types_to_element_counts,
    element_counts_to_formula_multiplicity,
    element_counts_to_reduced_counts,
    element_counts_to_ratios,
)
from lvebcsp.data.mp20_lmdb import MP20LMDBDataset, _atomic_number
from lvebcsp.data.organic_lmdb import OrganicLMDBDataset
from lvebcsp.models.canvas import NUM_ELEMENTS


# Change this to ``pxrd_plus_composition`` only when PXRD is available to the
# deployed inorganic model. Both modes keep the target structure out of the
# fragment condition.
INORGANIC_CONDITION = "composition_only"
INORGANIC_CONDITION_MODES = (
    "composition_only",
    "pxrd_plus_composition",
)

FRAGMENT_TENSOR_KEYS = (
    "fragment_atom_types",
    "fragment_coords",
    "fragment_mask",
    "fragment_group_indices",
    "condition_element_counts",
)
MULTIPLICITY_TENSOR_KEYS = (
    "condition_multiplicity",
    "condition_multiplicity_mask",
)


def normalized_element_fractions(atom_types: Tensor) -> Tensor:
    """Return scale-free elemental fractions with shape ``[119]``."""

    atom_types = torch.as_tensor(atom_types, dtype=torch.long).reshape(-1)
    if torch.any((atom_types != 0) & ((atom_types < 1) | (atom_types > NUM_ELEMENTS))):
        raise ValueError("atomic numbers must be zero padding or lie in [1, 118]")
    active = atom_types[(atom_types >= 1) & (atom_types <= NUM_ELEMENTS)]
    if active.numel() < 1:
        raise ValueError("a condition must contain at least one chemical element")
    counts = torch.bincount(active, minlength=NUM_ELEMENTS + 1).float()
    counts[0] = 0
    return counts / counts.sum()


def null_fragment(fragment_n_max: int) -> dict[str, Tensor]:
    """Return the padded NULL_FRAGMENT used by composition-only inputs."""

    fragment_n_max = int(fragment_n_max)
    if fragment_n_max < 1:
        raise ValueError("fragment_n_max must be positive")
    return {
        "fragment_atom_types": torch.zeros(fragment_n_max, dtype=torch.long),
        "fragment_coords": torch.zeros(fragment_n_max, 3, dtype=torch.float32),
        "fragment_mask": torch.zeros(fragment_n_max, dtype=torch.bool),
        "fragment_group_indices": torch.zeros(fragment_n_max, dtype=torch.long),
        "condition_element_counts": torch.zeros(
            NUM_ELEMENTS + 1,
            dtype=torch.float32,
        ),
    }


def pack_fragment(
    atom_types: Tensor,
    coords: Tensor,
    group_indices: Tensor,
    *,
    fragment_n_max: int,
) -> dict[str, Tensor]:
    """Validate and pad a genuinely supplied inference fragment."""

    atom_types = torch.as_tensor(atom_types, dtype=torch.long).reshape(-1)
    coords = torch.as_tensor(coords, dtype=torch.float32)
    group_indices = torch.as_tensor(group_indices, dtype=torch.long).reshape(-1)
    count = int(atom_types.numel())
    if count < 1:
        raise ValueError("an atomistic fragment must contain at least one atom")
    if count > int(fragment_n_max):
        raise ValueError(
            f"fragment contains {count} atoms, above fragment_n_max={fragment_n_max}"
        )
    if coords.shape != (count, 3):
        raise ValueError(f"fragment coordinates must have shape [{count}, 3]")
    if group_indices.shape != (count,):
        raise ValueError(f"fragment group indices must have shape [{count}]")
    if torch.any((atom_types < 1) | (atom_types > NUM_ELEMENTS)):
        raise ValueError("fragment atomic numbers must be in [1, 118]")
    if torch.any(group_indices < 1):
        raise ValueError("active fragment group indices must be positive")
    if not torch.isfinite(coords).all():
        raise ValueError("fragment coordinates must be finite")

    result = null_fragment(fragment_n_max)
    result["fragment_atom_types"][:count] = atom_types
    result["fragment_coords"][:count] = coords
    result["fragment_mask"][:count] = True
    result["fragment_group_indices"][:count] = group_indices
    result["condition_element_counts"] = atom_types_to_element_counts(atom_types)
    return result


def organic_template_fragment(
    row: dict[str, Any],
    *,
    fragment_n_max: int,
) -> dict[str, Tensor]:
    """Adapt one-copy molecular metadata to the universal condition schema."""

    required = ("mol_atom_type_2", "mol_atom_pos_2", "mol2_group_slices")
    missing = [name for name in required if name not in row]
    if missing:
        raise KeyError(
            "organic row is missing atomistic-template fields: "
            + ", ".join(missing)
        )

    symbols = [str(symbol) for symbol in list(row["mol_atom_type_2"])]
    if not symbols:
        raise ValueError("mol_atom_type_2 must contain at least one atom")
    atom_types = torch.tensor(
        [_atomic_number(symbol) for symbol in symbols],
        dtype=torch.long,
    )

    raw_coords = torch.as_tensor(row["mol_atom_pos_2"], dtype=torch.float32)
    if raw_coords.ndim == 3:
        if raw_coords.shape[0] < 1:
            raise ValueError("mol_atom_pos_2 contains no coordinate view")
        coords = raw_coords[0].clone()
    elif raw_coords.ndim == 2:
        coords = raw_coords.clone()
    else:
        raise ValueError(
            "mol_atom_pos_2 must have shape [M, 3] or [V, M, 3]"
        )
    if coords.shape != (len(symbols), 3):
        raise ValueError(
            "one mol_atom_pos_2 view must align with mol_atom_type_2; got "
            f"{tuple(coords.shape)} for {len(symbols)} atoms"
        )

    slices = list(row["mol2_group_slices"])
    if not slices:
        raise ValueError("mol2_group_slices must contain at least one group")
    group_indices = torch.zeros(len(symbols), dtype=torch.long)
    covered = torch.zeros(len(symbols), dtype=torch.bool)
    for group_index, raw_slice in enumerate(slices, start=1):
        if len(raw_slice) != 2:
            raise ValueError(f"fragment slice must be a pair, got {raw_slice!r}")
        start, end = int(raw_slice[0]), int(raw_slice[1])
        if not 0 <= start < end <= len(symbols):
            raise ValueError(
                f"invalid fragment slice {(start, end)} for {len(symbols)} atoms"
            )
        if covered[start:end].any():
            raise ValueError("mol2_group_slices contains overlapping groups")
        covered[start:end] = True
        group_indices[start:end] = group_index
        coords[start:end] -= coords[start:end].mean(dim=0, keepdim=True)
    if not covered.all():
        raise ValueError("mol2_group_slices must cover every template atom")

    result = pack_fragment(
        atom_types,
        coords,
        group_indices,
        fragment_n_max=fragment_n_max,
    )
    result["condition_formula_fractions"] = element_counts_to_ratios(
        result["condition_element_counts"]
    )
    return result


def _target_item(
    dataset: MP20LMDBDataset,
    row: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    atom_types, frac_coords, atom_mask, lattice_params = dataset._structure_tensors(
        row
    )
    material_id = str(
        row.get("material_id") or f"{prefix}_{row.get('_lmdb_key', 'unknown')}"
    )
    element_counts = atom_types_to_element_counts(atom_types)
    return {
        "material_id": material_id,
        "formula": "",
        "element_counts": element_counts,
        "ratio": element_counts_to_ratios(element_counts),
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "atom_mask": atom_mask,
        "lattice_params": lattice_params,
    }


class FoundationOrganicDataset(OrganicLMDBDataset):
    """Organic targets conditioned on one copy of each molecular template."""

    def __init__(
        self,
        lmdb_path: str | Path,
        *args: Any,
        fragment_n_max: int = 64,
        emit_null_pxrd: bool = False,
        emit_condition_multiplicity: bool = False,
        condition_multiplicity_override: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("include_molecule_template", None)
        kwargs.pop("molecule_template_n_max", None)
        kwargs["include_pxrd"] = False
        super().__init__(
            lmdb_path,
            *args,
            include_molecule_template=False,
            **kwargs,
        )
        self.fragment_n_max = int(fragment_n_max)
        self.emit_null_pxrd = bool(emit_null_pxrd)
        self.emit_condition_multiplicity = bool(emit_condition_multiplicity)
        self.condition_multiplicity_override = (
            None
            if condition_multiplicity_override is None
            else int(condition_multiplicity_override)
        )
        if (
            self.condition_multiplicity_override is not None
            and self.condition_multiplicity_override < 1
        ):
            raise ValueError("condition_multiplicity_override must be positive")
        if (
            self.condition_multiplicity_override is not None
            and not self.emit_condition_multiplicity
        ):
            raise ValueError(
                "condition_multiplicity_override requires "
                "emit_condition_multiplicity=True"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._load_row(index)
        item = _target_item(self, row, prefix="organic")
        item.update(
            organic_template_fragment(row, fragment_n_max=self.fragment_n_max)
        )
        if self.emit_condition_multiplicity:
            if self.condition_multiplicity_override is None:
                multiplicity, available = element_counts_to_formula_multiplicity(
                    item["element_counts"],
                    item["condition_element_counts"],
                )
            else:
                multiplicity = torch.tensor(
                    self.condition_multiplicity_override,
                    dtype=torch.long,
                )
                available = torch.tensor(True)
            item["condition_multiplicity"] = multiplicity
            item["condition_multiplicity_mask"] = available
        if self.emit_null_pxrd:
            item.update(
                {
                    "peak_d": torch.zeros(self.p_max, dtype=torch.float32),
                    "peak_i": torch.zeros(self.p_max, dtype=torch.float32),
                    "peak_mask": torch.zeros(self.p_max, dtype=torch.bool),
                    "condition_pxrd_mask": torch.tensor(False),
                }
            )
        item["material_domain"] = "organic"
        return item


class FoundationInorganicDataset(MP20LMDBDataset):
    """Inorganic targets with composition and a NULL_FRAGMENT condition."""

    def __init__(
        self,
        lmdb_path: str | Path,
        *args: Any,
        fragment_n_max: int = 64,
        inorganic_condition: str = INORGANIC_CONDITION,
        emit_condition_multiplicity: bool = False,
        condition_multiplicity_override: int | None = None,
        **kwargs: Any,
    ) -> None:
        if inorganic_condition not in INORGANIC_CONDITION_MODES:
            raise ValueError(
                "inorganic_condition must be 'composition_only' or "
                "'pxrd_plus_composition'"
            )
        include_pxrd = inorganic_condition == "pxrd_plus_composition"
        requested_pxrd = kwargs.pop("include_pxrd", include_pxrd)
        if bool(requested_pxrd) != include_pxrd:
            raise ValueError(
                "include_pxrd must match the selected inorganic_condition"
            )
        super().__init__(
            lmdb_path,
            *args,
            include_pxrd=include_pxrd,
            **kwargs,
        )
        self.fragment_n_max = int(fragment_n_max)
        self.inorganic_condition = inorganic_condition
        self.emit_condition_multiplicity = bool(emit_condition_multiplicity)
        self.condition_multiplicity_override = (
            None
            if condition_multiplicity_override is None
            else int(condition_multiplicity_override)
        )
        if (
            self.condition_multiplicity_override is not None
            and self.condition_multiplicity_override < 1
        ):
            raise ValueError("condition_multiplicity_override must be positive")
        if (
            self.condition_multiplicity_override is not None
            and not self.emit_condition_multiplicity
        ):
            raise ValueError(
                "condition_multiplicity_override requires "
                "emit_condition_multiplicity=True"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        target_counts = atom_types_to_element_counts(
            item["atom_types"][item["atom_mask"]]
        )
        reduced_counts = element_counts_to_reduced_counts(target_counts)
        item["condition_formula_fractions"] = element_counts_to_ratios(
            reduced_counts
        )
        item.update(null_fragment(self.fragment_n_max))
        # Reduced composition is invariant to unit-cell repetition and is
        # therefore safe to expose even when no atomistic fragment is present.
        item["condition_element_counts"] = reduced_counts
        if self.emit_condition_multiplicity:
            if self.condition_multiplicity_override is None:
                multiplicity, available = element_counts_to_formula_multiplicity(
                    target_counts,
                    reduced_counts,
                )
            else:
                multiplicity = torch.tensor(
                    self.condition_multiplicity_override,
                    dtype=torch.long,
                )
                available = torch.tensor(True)
            item["condition_multiplicity"] = multiplicity
            item["condition_multiplicity_mask"] = available
        if self.inorganic_condition == "pxrd_plus_composition":
            item.pop("aug_peak_d", None)
            item.pop("aug_peak_i", None)
            item.pop("aug_peak_mask", None)
            item["condition_pxrd_mask"] = torch.tensor(True)
        item["material_domain"] = "inorganic"
        return item


class BalancedMaterialsDataset(Dataset[dict[str, Any]]):
    """Present each selected material domain equally often in every epoch."""

    def __init__(
        self,
        organic: Dataset[dict[str, Any]] | None = None,
        inorganic: Dataset[dict[str, Any]] | None = None,
        *,
        samples_per_domain: int | None = None,
    ) -> None:
        selected = tuple(
            (name, dataset)
            for name, dataset in (
                ("organic", organic),
                ("inorganic", inorganic),
            )
            if dataset is not None
        )
        if not selected:
            raise ValueError("at least one material domain must be selected")
        if any(len(dataset) < 1 for _, dataset in selected):
            raise ValueError("selected material domains must contain data")
        self.domain_names = tuple(name for name, _ in selected)
        self.datasets = tuple(dataset for _, dataset in selected)
        self.samples_per_domain = int(
            max(len(dataset) for dataset in self.datasets)
            if samples_per_domain is None
            else samples_per_domain
        )
        if self.samples_per_domain < 1:
            raise ValueError("samples_per_domain must be positive")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.datasets) * self.samples_per_domain

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        domain = int(index) % len(self.datasets)
        position = int(index) // len(self.datasets)
        dataset = self.datasets[domain]
        offset = self.epoch * self.samples_per_domain
        source_index = (position + offset) % len(dataset)
        item = dict(dataset[source_index])
        item["material_domain"] = self.domain_names[domain]
        return item
