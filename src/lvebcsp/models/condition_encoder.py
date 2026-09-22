"""Building-block multiplicity conditioning for crystal JEPA."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from lvebcsp.models.layers import MLP


class ConditionEncoder(nn.Module):
    """Bind each representative block embedding to its absolute copy count.

    block_embeddings: [M, T, D], T tokens per representative graph.
    multiplicity: [M], number of copies of each block in the target cell.
    block_batch: [M], crystal assignment for each block, numbered from zero.
    M = representative graphs, B = crystals, D = input_dim, H = hidden_dim.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_encoder = MLP(input_dim + 1, hidden_dim, hidden_dim)

    def forward(
        self, block_embeddings: Tensor, multiplicity: Tensor, block_batch: Tensor,
        size: int | None = None,
    ) -> Tensor:
        counts = multiplicity.to(block_embeddings).log1p()[:, None, None]  # [M, 1, 1]
        counts = counts.expand(-1, block_embeddings.size(1), -1)  # [M, T, 1]
        tokens = self.block_encoder(torch.cat([block_embeddings, counts], dim=-1))  # [M, T, D + 1] -> [M, T, H]
        return scatter(tokens, block_batch, dim=0, dim_size=size, reduce="sum")  # [B, T, H]; retain token slots
