"""Batch collation for manifest datasets."""

from __future__ import annotations

from typing import Any

import torch

from lvebcsp.common.chemistry import element_counts_to_ratios


def collate_crystal_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor fields and keep string fields as lists."""

    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values
    if "ratio" not in batch and "element_counts" in batch:
        batch["ratio"] = element_counts_to_ratios(batch["element_counts"])
    return batch
