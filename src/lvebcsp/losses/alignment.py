"""Latent alignment losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_energy(z_query: torch.Tensor, z_candidate: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance in normalized latent space."""

    query = F.normalize(z_query, dim=-1)
    candidate = F.normalize(z_candidate, dim=-1)
    return torch.sum((query - candidate) ** 2, dim=-1)
