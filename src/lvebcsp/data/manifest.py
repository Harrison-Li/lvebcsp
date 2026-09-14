"""Manifest CSV parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ManifestEntry:
    """One row in a CIF manifest."""

    cif_path: Path
    material_id: str
    split: str
    formula: str


def read_manifest(path: str | Path, split: str | None = None) -> list[ManifestEntry]:
    """Read a manifest with required cif_path and optional metadata columns."""

    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path)
    if "cif_path" not in frame.columns:
        raise ValueError("Manifest CSV must contain a cif_path column")
    if split is not None and "split" in frame.columns:
        frame = frame[frame["split"] == split]
    entries: list[ManifestEntry] = []
    for idx, row in frame.iterrows():
        cif_path = Path(str(row["cif_path"]))
        if not cif_path.is_absolute():
            cif_path = manifest_path.parent / cif_path
        entries.append(
            ManifestEntry(
                cif_path=cif_path,
                material_id=str(row.get("material_id", f"row_{idx}")),
                split=str(row.get("split", "")),
                formula=str(row.get("formula", "")),
            )
        )
    return entries
