"""Decomposition-free crystal encoder using local cutoff-attention and latent Transformer.

Architecture:
Atomic species + coordinates + lattice
                    │
       Periodic cutoff-neighbor construction
                    │
       Local atom Transformer blocks
       Attention only over cutoff neighbors
                    │
       32 learned queries attend to all atoms
                    │
           32 latent tokens
                    │
         Latent Transformer blocks
                    │
          Global crystal embedding

Accepts pre-computed crystal graphs as PyTorch Geometric Data or Batch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax, to_dense_batch

from lvebcsp.models.module import AtomEmbedding
from lvebcsp.models.layers import MLP


@dataclass(frozen=True)
class EncoderConfig:
    atom_dim: int = 64
    dim: int = 256
    output_dim: int = 512
    num_latents: int = 32
    local_layers: int = 3
    latent_layers: int = 4
    heads: int = 8
    radial_basis: int = 32
    cutoff: float = 6.0
    atom_feature_dim: int = 0
    dropout: float = 0.0


class LocalAtomTransformerBlock(nn.Module):
    """Local atom Transformer block with multi-head self-attention strictly over cutoff neighbors."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        a = cfg.atom_dim
        self.heads = cfg.heads
        self.head_dim = a // self.heads

        self.q_proj = nn.Linear(a, a)
        self.k_proj = nn.Linear(a, a)
        self.v_proj = nn.Linear(a, a)
        self.edge_bias_proj = nn.Linear(a, self.heads, bias=False)
        self.edge_val_proj = nn.Linear(a, a, bias=False)
        self.out_proj = nn.Linear(a, a)

        self.self_edge_emb = nn.Parameter(torch.zeros(a))

        self.norm1 = nn.LayerNorm(a)
        self.norm2 = nn.LayerNorm(a)
        self.ffn = MLP(a, 4 * a, a)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        h: Tensor,
        edges: Tensor,
        edge_features: Tensor,
        weight: Tensor,
    ) -> Tensor:
        """Run cutoff neighbor multi-head attention.

        Args:
            h: Atom embeddings [N, a].
            edges: [2, E] where edges[0] is source (src) and edges[1] is receiver (dst).
            edge_features: [E, a] radial edge representations.
            weight: [E] smooth cosine cutoff weights in [0, 1].

        Returns:
            Updated atom embeddings [N, a].
        """
        norm_h = self.norm1(h)
        N = h.shape[0]
        device = h.device
        dtype = h.dtype

        src, dst = edges

        # Include self-loops so each atom attends to itself with weight 1.0
        self_nodes = torch.arange(N, device=device, dtype=torch.long)
        all_src = torch.cat([src, self_nodes], dim=0)
        all_dst = torch.cat([dst, self_nodes], dim=0)
        all_weight = torch.cat([weight, torch.ones(N, device=device, dtype=weight.dtype)], dim=0)
        all_edge_feat = torch.cat([edge_features, self.self_edge_emb.expand(N, -1)], dim=0)

        # Multi-head Q, K, V projections
        q = self.q_proj(norm_h).view(N, self.heads, self.head_dim)
        k = self.k_proj(norm_h).view(N, self.heads, self.head_dim)
        v = self.v_proj(norm_h).view(N, self.heads, self.head_dim)

        # Q at receiver (dst), K at source image (src)
        q_dst = q[all_dst]  # [E_total, heads, head_dim]
        k_src = k[all_src]  # [E_total, heads, head_dim]
        dot = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.head_dim)  # [E_total, heads]

        edge_bias = self.edge_bias_proj(all_edge_feat)  # [E_total, heads]
        scores = dot + edge_bias

        attn_weights = softmax(scores, all_dst, num_nodes=N)

        # Modulate attention weights by the smooth cutoff envelope
        attn_weights = attn_weights * all_weight.unsqueeze(-1)

        # Values incorporate edge feature representation
        v_src = v[all_src]
        edge_v = self.edge_val_proj(all_edge_feat).view(-1, self.heads, self.head_dim)
        msg = attn_weights.unsqueeze(-1) * (v_src + edge_v)  # [E_total, heads, head_dim]

        attn_out = torch.zeros((N, self.heads, self.head_dim), device=device, dtype=dtype)
        attn_out.index_add_(0, all_dst, msg)
        attn_out = attn_out.view(N, self.heads * self.head_dim)
        attn_out = self.out_proj(attn_out)

        h = h + self.dropout(attn_out)
        h = h + self.ffn(self.norm2(h))
        return h


class Attention(nn.Module):
    """Noncausal Scaled Dot-Product Attention. True in valid indicates a real key."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_dim, self.dropout = heads, dim // heads, dropout
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, query: Tensor, memory: Tensor, valid: Tensor | None = None) -> Tensor:
        b, qn, d = query.shape

        def split(x: Tensor) -> Tensor:
            return x.reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)

        q = split(self.query(query))
        k, v = (split(x) for x in self.key_value(memory).chunk(2, -1))
        mask = None if valid is None else valid[:, None, None, :]
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(result.transpose(1, 2).reshape(b, qn, d))


class CrossBlock(nn.Module):
    """Cross-attention block from queries to memory."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        d = cfg.dim
        self.q_norm = nn.LayerNorm(d)
        self.kv_norm = nn.LayerNorm(d)
        self.ff_norm = nn.LayerNorm(d)
        self.attention = Attention(d, cfg.heads, cfg.dropout)
        self.ffn = MLP(d, 4 * d, d)

    def forward(self, query: Tensor, memory: Tensor, valid: Tensor | None = None) -> Tensor:
        query = query + self.attention(self.q_norm(query), self.kv_norm(memory), valid)
        return query + self.ffn(self.ff_norm(query))


class LatentBlock(nn.Module):
    """Self-attention block for latent tokens."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        d = cfg.dim
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attention = Attention(d, cfg.heads, cfg.dropout)
        self.ffn = MLP(d, 4 * d, d)

    def forward(self, tokens: Tensor) -> Tensor:
        h = self.norm1(tokens)
        tokens = tokens + self.attention(h, h)
        return tokens + self.ffn(self.norm2(tokens))


class CrystalEncoder(nn.Module):
    """Local atom attention followed by latent-token attention and global pooling.

    Input is a PyG Data or Batch with z [N] (atomic numbers), pos [N, 3],
    and precomputed edge_index [2, E] (source image -> receiver). Optional
    fields: cell [1, 3, 3] per graph, edge_shifts [E, 3] (integer source-image
    translations), periodic [1], geometry_known [N], and
    atom_features [N, F]. Batch.from_data_list or a PyG DataLoader handles
    graph collation. Missing cells/shifts mean isolated, unshifted geometry;
    missing geometry_known means all atoms are observed.

    Inputs and config are assumed valid; neighbor graphs must cover config.cutoff.
    forward returns [B, output_dim]; forward_features also returns latent
    tokens, padded atom tokens, and their validity mask.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or EncoderConfig()
        a, d = cfg.atom_dim, cfg.dim

        self.atom_embedding = AtomEmbedding(a, max_num_elements=119)
        self.geometry_embedding = nn.Embedding(2, a)
        self.extra_features = (
            nn.Linear(cfg.atom_feature_dim, a, bias=False)
            if cfg.atom_feature_dim
            else None
        )

        self.register_buffer("radial_centers", torch.linspace(0, cfg.cutoff, cfg.radial_basis))
        self.radial_width = cfg.cutoff / (cfg.radial_basis - 1)
        self.edge_embedding = MLP(cfg.radial_basis, a, a)

        # Local atom Transformer blocks with attention strictly over cutoff neighbors
        self.local_layers = nn.ModuleList(
            [LocalAtomTransformerBlock(cfg) for _ in range(cfg.local_layers)]
        )
        self.atom_to_memory = nn.Linear(a, d)

        # 32 learned latent queries cross-attending to all atoms
        self.latent_queries = nn.Parameter(torch.randn(1, cfg.num_latents, d) * 0.02)
        self.metadata_embedding = MLP(10, d, d)
        self.tokenizer = CrossBlock(cfg)

        # Latent Transformer blocks
        self.latent_transformer = nn.ModuleList(
            [LatentBlock(cfg) for _ in range(cfg.latent_layers)]
        )

        # Readout query pooling for global crystal embedding
        self.readout_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.readout = CrossBlock(cfg)
        self.output = nn.Linear(d, cfg.output_dim)

    def _metadata(
        self, cells: Tensor, periodic: Tensor, observed: Tensor, sizes: Tensor,
    ) -> Tensor:
        identity = torch.eye(3, dtype=cells.dtype, device=cells.device)
        cell = torch.where(periodic[:, None, None], cells, identity)
        gram = cell @ cell.transpose(-1, -2)
        lengths = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
        cosines = torch.stack(
            [
                gram[:, 0, 1] / (lengths[:, 0] * lengths[:, 1]),
                gram[:, 0, 2] / (lengths[:, 0] * lengths[:, 2]),
                gram[:, 1, 2] / (lengths[:, 1] * lengths[:, 2]),
            ],
            dim=-1,
        )
        cell_features = torch.cat(
            [
                lengths.log(),
                cosines.clamp(-1, 1),
                cell.det().abs().clamp_min(1e-12).log()[:, None],
            ],
            dim=-1,
        ) * periodic[:, None]
        return torch.cat(
            [
                cell_features,
                periodic[:, None].to(cell_features.dtype),
                observed,
                sizes.to(cell_features.dtype).log1p()[:, None],
            ],
            dim=-1,
        )

    def forward_features(self, batch: Data | Batch) -> dict[str, Tensor]:
        if not isinstance(batch, Batch):
            batch = Batch.from_data_list([batch])

        z, pos = batch.z, batch.pos
        sizes = batch.ptr[1:] - batch.ptr[:-1]
        cells = batch.get("cell", pos.new_zeros(batch.num_graphs, 3, 3)).reshape(-1, 3, 3)
        periodic = batch.get("periodic", cells.abs().sum(dim=(-1, -2)) > 1e-6).view(-1)
        known = batch.get("geometry_known", torch.ones_like(z, dtype=torch.bool))

        atoms = self.atom_embedding(z) + self.geometry_embedding(known.long())
        if self.extra_features is not None:
            atoms = atoms + self.extra_features(batch.atom_features)

        src, dst = batch.edge_index
        shifts = batch.get("edge_shifts", pos.new_zeros(src.numel(), 3))
        translation = torch.einsum("ei,eij->ej", shifts.to(pos.dtype), cells[batch.batch[dst]])
        distance = (pos[src] + translation - pos[dst]).norm(dim=-1)
        radial = torch.exp(
            -0.5 * ((distance[:, None] - self.radial_centers) / self.radial_width).square()
        )
        edge_features = self.edge_embedding(radial)
        weight = 0.5 * (1 + torch.cos(math.pi * (distance / self.config.cutoff).clamp(0, 1)))
        for layer in self.local_layers:
            atoms = layer(atoms, batch.edge_index, edge_features, weight)

        memory, valid = to_dense_batch(
            self.atom_to_memory(atoms), batch.batch, batch_size=batch.num_graphs,
        )
        observed = pos.new_zeros(batch.num_graphs).index_add(0, batch.batch, known.to(pos.dtype))
        metadata = self._metadata(cells, periodic, (observed / sizes)[:, None], sizes)
        queries = self.latent_queries.expand(batch.num_graphs, -1, -1)
        queries = queries + self.metadata_embedding(metadata)[:, None, :]
        tokens = self.tokenizer(queries, memory, valid)

        # 3. Latent Transformer blocks
        for layer in self.latent_transformer:
            tokens = layer(tokens)

        # 4. Readout pooling -> Global crystal embedding
        pooled = self.readout(self.readout_query.expand(batch.num_graphs, -1, -1), tokens)
        embedding = self.output(pooled[:, 0])

        return {
            "embedding": embedding,
            "latent_tokens": tokens,
            "atom_tokens": memory,
            "valid_mask": valid,
        }

    def forward(self, batch: Data | Batch) -> Tensor:
        return self.forward_features(batch)["embedding"]


class CrystalPatchEncoder(CrystalEncoder):
    """Crystal Patch Encoder returning 32 latent patch tokens [B, 32, D]."""

    def __init__(self, dim: int = 256, **kwargs: Any) -> None:
        if isinstance(dim, EncoderConfig):
            cfg = dim
        else:
            cfg = EncoderConfig(dim=dim, **kwargs)
        super().__init__(cfg)

    def forward(self, batch: Data | Batch) -> Tensor:
        """Forward pass returning patch tokens [B, 32, D]."""
        return self.forward_features(batch)["latent_tokens"]


class PeakEncoder(nn.Module):
    """PeakEncoder for backward compatibility with JEPA imports."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        d_jepa: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        config = config or {}
        self.d_jepa = int(d_jepa)
        hidden_dim = int(config.get("hidden_dim", 256))
        self.d_proj = nn.Linear(1, hidden_dim)
        self.i_proj = nn.Linear(1, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.d_jepa),
            nn.GELU(),
            nn.Linear(self.d_jepa, self.d_jepa),
        )
        self.norm = nn.LayerNorm(self.d_jepa)

    def forward(
        self,
        peak_d: Tensor,
        peak_i: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Encode d-spacing and intensity peaks to a latent representation."""
        if peak_i is None and isinstance(peak_d, (tuple, list)):
            peak_d, peak_i = peak_d[0], peak_d[1]
            if len(peak_d) > 2 and mask is None:
                mask = peak_d[2]

        if peak_i is None:
            peak_i = torch.ones_like(peak_d)

        if peak_d.ndim == 2:
            peak_d = peak_d.unsqueeze(-1)
        if peak_i.ndim == 2:
            peak_i = peak_i.unsqueeze(-1)

        feat = self.d_proj(peak_d.float()) + self.i_proj(peak_i.float())
        if mask is not None:
            feat = feat * mask.unsqueeze(-1).float()
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0).float()
            pooled = feat.sum(dim=1) / denom
        else:
            pooled = feat.mean(dim=1)

        return self.norm(self.out_proj(pooled))