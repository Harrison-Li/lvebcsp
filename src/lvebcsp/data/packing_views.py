"""Packing-preserving observations for controlled crystal JEPA experiments."""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch_geometric.data import Data


def mask_atom_geometry(graph: Data, fraction: float = .25, *,
                       generator: torch.Generator | None = None,
                       visible: Tensor | None = None) -> Data:
    """Hide atomic positions and all incident edges without moving visible atoms.

    Atomic identities, atom count and cell remain observed. Hidden positions are
    replaced by zeros; fractional coordinates, cached distances and unspecified
    target-derived attributes are deliberately not copied. The encoder already
    supports ``geometry_known`` and isolated atoms, so no model changes are
    necessary. This constructs a partial observation, not another crystal.
    """
    if graph.pos.device.type != "cpu":
        raise ValueError("Construct masked observations on CPU before batching")
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("Mask fraction must be finite and in [0, 1)")
    n = int(graph.num_nodes)
    if n < 2:
        raise ValueError("Masking requires at least two atoms")
    if visible is None:
        hidden_count = min(n - 1, max(1, int(round(fraction * n)))) if fraction else 0
        visible = torch.ones(n, dtype=torch.bool)
        visible[torch.randperm(n, generator=generator)[:hidden_count]] = False
    if visible.device.type != "cpu" or visible.dtype != torch.bool or visible.shape != (n,) or not visible.any():
        raise ValueError("Visible must be a boolean atom mask with at least one observed atom")
    src, dst = graph.edge_index
    keep = visible[src] & visible[dst]
    context = Data(z=graph.z.clone(), pos=graph.pos.clone().masked_fill(~visible[:, None], 0.),
                   edge_index=graph.edge_index[:, keep].clone(),
                   geometry_known=visible.clone(), num_nodes=n)
    if "cell" in graph:
        context.cell = graph.cell.clone()
    if "edge_shifts" in graph:
        context.edge_shifts = graph.edge_shifts[keep].clone()
    for key in ("periodic", "num_atoms", "block_instance_id", "atom_map"):
        if key in graph:
            context[key] = graph[key].clone()
    return context
