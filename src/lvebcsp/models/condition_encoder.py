"""Building-block multiplicity conditioning for crystal JEPA."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_add_pool

from lvebcsp.models.layers import MLP


class ConditionEncoder(nn.Module):
    """Bind each representative block embedding to its absolute copy count.

    block_embeddings: [M, D], one embedding per representative graph.
    multiplicity: [M], number of copies of each block in the target cell.
    block_batch: [M], crystal assignment for each block, numbered from zero.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_encoder = MLP(input_dim + 1, hidden_dim, hidden_dim)

    def forward(
        self, block_embeddings: Tensor, multiplicity: Tensor, block_batch: Tensor,
    ) -> Tensor:
        counts = multiplicity.to(block_embeddings).log1p().unsqueeze(-1)
        tokens = self.block_encoder(torch.cat([block_embeddings, counts], dim=-1))
        return global_add_pool(tokens, block_batch)
