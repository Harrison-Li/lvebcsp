"""CIF export helpers."""

from __future__ import annotations

from pathlib import Path

from pymatgen.core import Structure


def export_structure_to_cif(structure: Structure, path: str | Path) -> Path:
    """Write one pymatgen Structure to CIF."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(output_path), fmt="cif")
    return output_path
