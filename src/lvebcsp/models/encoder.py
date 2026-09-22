"""Decomposition-free crystal encoder using local cutoff-attention and latent Transformer.

Architecture:
Atomic species + coordinates + lattice
                    │
       Periodic cutoff-neighbor construction
                    │
       Canonical lattice + presence condition atoms and edges
                    │
       LocalAtomTransformer
       Radial features + cutoff-weighted neighbor attention
                    │
       UniversalEncoder.compress
       32 learned queries attend to all atoms
                    │
           32 latent tokens
                    │
         Latent Transformer blocks
                    │
        32 crystal latent tokens

Accepts pre-computed crystal graphs as PyTorch Geometric Data or Batch.

Shape notation: B = graphs, N = total atoms, N_max = padded atoms per graph,
E = edges, E_total = E + N (including self-loops), T = learned latent slots,
A = atom_dim, D = dim, D_out = output_dim, R = radial_basis.
The default token path is [B, 32, 256] -> [B, 32, 512]; T is not time.
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

from lvebcsp.common.utils import lattice_params_to_matrix_torch
from lvebcsp.data.representation import lattice_matrix_to_params
from lvebcsp.models.module import AtomEmbedding, GaussianBasis, PolynomialCutoff
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
            edge_features: [E, a] radial and lattice edge representations.
            weight: [E] smooth polynomial cutoff weights in [0, 1].

        Returns:
            Updated atom embeddings [N, a].
        """
        norm_h = self.norm1(h)  # [N, A]
        N = h.shape[0]
        device = h.device
        dtype = h.dtype

        src, dst = edges  # [E], [E]

        # Include self-loops so each atom attends to itself with weight 1.0
        self_nodes = torch.arange(N, device=device, dtype=torch.long)  # [N]
        all_src = torch.cat([src, self_nodes], dim=0)  # [E_total]
        all_dst = torch.cat([dst, self_nodes], dim=0)  # [E_total]
        all_weight = torch.cat([weight, torch.ones(N, device=device, dtype=weight.dtype)], dim=0)  # [E_total]
        all_edge_feat = torch.cat([edge_features, self.self_edge_emb.expand(N, -1)], dim=0)  # [E_total, A]

        # Multi-head Q, K, V projections
        q = self.q_proj(norm_h).view(N, self.heads, self.head_dim)  # [N, heads, head_dim]
        k = self.k_proj(norm_h).view(N, self.heads, self.head_dim)  # [N, heads, head_dim]
        v = self.v_proj(norm_h).view(N, self.heads, self.head_dim)  # [N, heads, head_dim]

        # Q at receiver (dst), K at source image (src)
        q_dst = q[all_dst]  # [E_total, heads, head_dim]
        k_src = k[all_src]  # [E_total, heads, head_dim]
        dot = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.head_dim)  # [E_total, heads]

        edge_bias = self.edge_bias_proj(all_edge_feat)  # [E_total, heads]
        scores = dot + edge_bias  # [E_total, heads]

        # Include the cutoff envelope in the softmax normalization; self weights are 1.
        scores = scores + all_weight.clamp_min(torch.finfo(all_weight.dtype).tiny).log().unsqueeze(-1)
        scores = scores.masked_fill(all_weight.unsqueeze(-1) == 0, -torch.inf)
        attn_weights = softmax(scores, all_dst, num_nodes=N)  # [E_total, heads]

        # Values incorporate edge feature representation
        v_src = v[all_src]  # [E_total, heads, head_dim]
        edge_v = self.edge_val_proj(all_edge_feat).view(-1, self.heads, self.head_dim)  # [E_total, heads, head_dim]
        msg = attn_weights.unsqueeze(-1) * (v_src + edge_v)  # [E_total, heads, head_dim]

        attn_out = torch.zeros((N, self.heads, self.head_dim), device=device, dtype=dtype)  # [N, heads, head_dim]
        attn_out.index_add_(0, all_dst, msg)  # [N, heads, head_dim]
        attn_out = attn_out.view(N, self.heads * self.head_dim)  # [N, A]
        attn_out = self.out_proj(attn_out)  # [N, A]

        h = h + self.dropout(attn_out)  # [N, A]
        h = h + self.ffn(self.norm2(h))  # [N, A]
        return h  # [N, A]


class LocalAtomTransformer(nn.Module):
    """Build geometry-aware per-atom features with local attention message passing.
    
    The canonical lattice matrix reconstructed from cell lengths and angles,
    plus a learned absent/present state, conditions initial atoms and every
    local layer's edge messages before atom-only token compression. The
    original cell is retained for periodic distances so rotating positions and
    cell together leaves both geometry paths invariant.

    Inputs and config are assumed valid; neighbor graphs must cover config.cutoff.
    Owns atom/lattice embeddings, periodic distances, the radial basis and smooth
    cutoff, and the attention stack. The cutoff gates attention once per layer.
    forward returns aggregated atom features [N, atom_dim].
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or EncoderConfig()
        a = cfg.atom_dim

        self.atom_embedding = AtomEmbedding(a, max_num_elements=119)
        self.geometry_embedding = nn.Embedding(2, a)
        self.extra_features = (
            nn.Linear(cfg.atom_feature_dim, a, bias=False)
            if cfg.atom_feature_dim
            else None
        )

        self.radial_basis = GaussianBasis(r_max=cfg.cutoff, num_basis=cfg.radial_basis)
        self.envelope = PolynomialCutoff(p=5, r_c=cfg.cutoff)
        self.edge_embedding = MLP(cfg.radial_basis, a, a)

        # Local atom Transformer blocks with attention strictly over cutoff neighbors
        self.layers = nn.ModuleList(
            [LocalAtomTransformerBlock(cfg) for _ in range(cfg.local_layers)]
        )

        self.lattice_embedding = MLP(9, a, a, norm_fn=None)
        self.lattice_presence_embedding = nn.Embedding(2, a)

    def forward(self, batch: Data | Batch) -> Tensor:
        if not isinstance(batch, Batch):
            batch = Batch.from_data_list([batch])

        z, pos = batch.z, batch.pos  # [N], [N, 3]
        cells = batch.get("cell", pos.new_zeros(batch.num_graphs, 3, 3)).reshape(-1, 3, 3)  # [B, 3, 3]
        periodic = batch.get("periodic", torch.ones_like(cells[:, 0, 0], dtype=torch.bool))  # [B]
        cells = cells.masked_fill(~periodic.reshape(-1, 1, 1), 0)
        known = batch.get("geometry_known", torch.ones_like(z, dtype=torch.bool))  # [N]

        provided = cells.ne(0).any(dim=(-2, -1))  # [B]; 0 = absent, 1 = provided
        with torch.autocast(device_type=cells.device.type, enabled=False):
            # A safe temporary cell prevents undefined angles for absent cells.
            safe_cells = torch.where(
                provided[:, None, None], cells.float(),
                torch.eye(3, device=cells.device, dtype=torch.float32),
            )
            params = lattice_matrix_to_params(safe_cells)  # [B, 6], Angstrom/degrees
            canonical = lattice_params_to_matrix_torch(params[:, :3], params[:, 3:])
            features = canonical.flatten(1) / self.config.cutoff  # [B, 9]; fixed length scale
        geometry = self.lattice_embedding(features) * provided[:, None]  # [B, A]
        lattice = geometry + self.lattice_presence_embedding(provided.long())  # [B, A]

        atoms = self.atom_embedding(z) + self.geometry_embedding(known.long())  # [N, A]
        if self.extra_features is not None:
            atoms = atoms + self.extra_features(batch.atom_features)  # [N, A]

        # Initial conditioning also reaches atoms with no neighbor edges.
        atoms = atoms + lattice[batch.batch].to(atoms.dtype)  # [N, A]

        src, dst = batch.edge_index  # [E], [E]
        shifts = batch.get("edge_shifts", pos.new_zeros(src.numel(), 3))  # [E, 3]
        # Preserve periodic geometry precision when neural layers use BF16 autocast.
        with torch.autocast(device_type=pos.device.type, enabled=False):
            translation = torch.einsum("ei,eij->ej", shifts.to(pos.dtype), cells[batch.batch[dst]])  # [E, 3]
        distance = (pos[src] + translation - pos[dst]).norm(dim=-1)  # [E]

        # Apply the envelope once, in each local layer's attention normalization.
        fc = self.envelope(distance)  # [E]
        radial = self.radial_basis(distance)  # [E, R]
        edge_features = self.edge_embedding(radial)  # [E, A]
        edge_features = edge_features + lattice[batch.batch[dst]].to(edge_features.dtype)
        for layer in self.layers:
            atoms = layer(atoms, batch.edge_index, edge_features, fc)  # [N, A]
        return atoms  # [N, A]


class Attention(nn.Module):
    """Noncausal multi-head attention. True in valid indicates a real key."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, query: Tensor, memory: Tensor, valid: Tensor | None = None) -> Tensor:
        # query: [B, T, D]; memory: [B, S, D]; optional valid: [B, S].
        # S = N_max for atom memory, or T for latent self-attention.
        result, _ = self.attention(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=None if valid is None else ~valid,
            need_weights=False,
        )
        return result  # [B, T, D]


class CrossBlock(nn.Module):
    """Cross-attention block from queries to memory. Compress per atom token features into a token list
    reduce the attn computation cost.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        d = cfg.dim
        self.q_norm = nn.LayerNorm(d)
        self.kv_norm = nn.LayerNorm(d)
        self.ff_norm = nn.LayerNorm(d)
        self.attention = Attention(d, cfg.heads, cfg.dropout)
        self.ffn = MLP(d, 4 * d, d)

    def forward(self, query: Tensor, memory: Tensor, valid: Tensor | None = None) -> Tensor:
        query = query + self.attention(self.q_norm(query), self.kv_norm(memory), valid)  # [B, T, D]
        return query + self.ffn(self.ff_norm(query))  # [B, T, D]


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
        h = self.norm1(tokens)  # [B, T, D]
        tokens = tokens + self.attention(h, h)  # [B, T, D]
        return tokens + self.ffn(self.norm2(tokens))  # [B, T, D]


class UniversalEncoder(nn.Module):
    """Compress aggregated per-atom features into fixed-length crystal tokens.

    ``local_encoder`` owns all atom, lattice, radial and local-attention work.
    ``compress`` accepts its [N, atom_dim] features and graph membership only;
    learned queries and latent self-attention produce [B, num_latents, output_dim].

    ``forward`` retains the graph-based API by delegating to LocalAtomTransformer
    before compression. ``return_atoms=True`` also returns its atom features.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or EncoderConfig()
        d = cfg.dim
        self.local_encoder = LocalAtomTransformer(cfg)
        self.atom_to_memory = nn.Linear(cfg.atom_dim, d)

        # Learned latent queries cross-attending to lattice-conditioned atoms
        self.latent_queries = nn.Parameter(torch.randn(1, cfg.num_latents, d) * 0.02)  # [1, T, D]
        self.tokenizer = CrossBlock(cfg)

        # Latent Transformer blocks
        self.latent_transformer = nn.ModuleList(
            [LatentBlock(cfg) for _ in range(cfg.latent_layers)]
        )

        # Project each token independently, preserving the latent sequence.
        self.output = nn.Linear(d, cfg.output_dim)

    def forward(self, batch: Data | Batch, *, return_atoms: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if not isinstance(batch, Batch):
            batch = Batch.from_data_list([batch])
        atoms = self.local_encoder(batch)  # [N, A]
        tokens = self.compress(
            atoms, batch.batch, num_graphs=batch.num_graphs,
            max_num_nodes=batch.get("max_num_nodes"),
        )
        return (tokens, atoms) if return_atoms else tokens

    def compress(
        self,
        atoms: Tensor,
        graph_index: Tensor,
        *,
        num_graphs: int | None = None,
        max_num_nodes: int | None = None,
    ) -> Tensor:
        """Compress [N, A] atom features grouped by graph_index [N] to [B, T, D_out]."""
        memory, valid = to_dense_batch(
            self.atom_to_memory(atoms), graph_index, batch_size=num_graphs,
            max_num_nodes=max_num_nodes,
        )  # memory: [B, N_max, D]; valid: [B, N_max]

        queries = self.latent_queries.expand(memory.shape[0], -1, -1)  # [B, T, D]
        tokens = self.tokenizer(queries, memory, valid)  # [B, T, D]

        # Refine the compressed tokens.
        for layer in self.latent_transformer:
            tokens = layer(tokens)  # [B, T, D]

        tokens = self.output(tokens)  # [B, T, D_out]; defaults: [B, 32, 512]
        return tokens


class TransformerPeakEncoder(nn.Module):
    """Interact over d-I peaks before reducing to a single CLS condition.

    Peaks are canonically ordered by decreasing d (then decreasing intensity),
    so learned rank embeddings describe physical peak order, not input order.
    ``hidden_dim`` controls token width; ``p_max`` is a separate peak capacity.
    The output width is set by the predictor's conditioning interface.
    """

    version = "transformer_peaks_v1"
    defaults = dict(p_max=80, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1)

    def __init__(
        self,
        p_max: int | dict[str, Any] = 80,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        d_jepa: int = 512,
    ) -> None:
        super().__init__()
        if isinstance(p_max, dict):
            cfg = dict(p_max)
            if cfg.pop("version", self.version) != self.version:
                raise ValueError(f"TransformerPeakEncoder requires version={self.version!r}")
            legacy_dim = cfg.pop("n_dim", None)
            configured_dim = cfg.pop("d_jepa", d_jepa)
            if int(configured_dim) != int(d_jepa) or (legacy_dim is not None and int(legacy_dim) != int(d_jepa)):
                raise ValueError("peak_encoder d_jepa/n_dim must match the requested condition output width")
            p_max = int(cfg.pop("p_max", self.defaults["p_max"]))
            hidden_dim = int(cfg.pop("hidden_dim", hidden_dim))
            num_layers = int(cfg.pop("num_layers", num_layers))
            num_heads = int(cfg.pop("num_heads", num_heads))
            dropout = float(cfg.pop("dropout", dropout))
            if cfg:
                raise ValueError(f"Unknown TransformerPeakEncoder config keys: {', '.join(sorted(cfg))}")
        self.p_max, self.hidden_dim, self.d_jepa = int(p_max), int(hidden_dim), int(d_jepa)
        num_layers, num_heads, dropout = int(num_layers), int(num_heads), float(dropout)
        if min(self.p_max, self.hidden_dim, self.d_jepa, num_layers, num_heads) < 1:
            raise ValueError("Peak capacity, dimensions, layers and heads must be positive")
        if self.hidden_dim % num_heads:
            raise ValueError("peak_encoder.hidden_dim must be divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("peak_encoder.dropout must be in [0, 1)")

        self.d_mlp = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.GELU(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim))
        self.i_mlp = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.GELU(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim))
        self.rank_embedding = nn.Embedding(self.p_max, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=4 * self.hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        # TransformerEncoder clones one layer; give each layer its own random weights.
        for block in self.transformer.layers:
            for name, parameter in block.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif name.endswith("bias"):
                    nn.init.zeros_(parameter)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.proj = nn.Linear(self.hidden_dim, self.d_jepa)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.rank_embedding.weight, std=0.02)

    def forward(self, peak_d: Tensor, peak_i: Tensor, peak_mask: Tensor) -> Tensor:
        """Return [B, d_jepa]; masked values have no effect or input gradient."""
        if peak_d.ndim != 2 or peak_i.shape != peak_d.shape:
            raise ValueError("peak_d and peak_i must have matching [B, P] shapes")
        if peak_mask.dtype != torch.bool or peak_mask.shape != peak_d.shape:
            raise ValueError("Peak mask must be boolean with shape [B, P]")
        if peak_d.device != peak_i.device or peak_d.device != peak_mask.device:
            raise ValueError("Peak values and mask must be on the same device")
        batch, p_count = peak_d.shape
        if p_count > self.p_max:
            raise ValueError(f"Peak count {p_count} exceeds encoder p_max={self.p_max}")
        # Sanitize before either MLP: NaN * zero is still NaN, including in backward.
        d = peak_d.float().masked_fill(~peak_mask, 1.)
        intensity = peak_i.float().masked_fill(~peak_mask, 0.)
        if not (torch.isfinite(d).all() and (d > 0).all() and
                torch.isfinite(intensity).all() and (intensity >= 0).all()):
            raise ValueError("Valid peaks require finite positive d and finite nonnegative intensity")

        # Stable secondary sorting handles coincident peak positions consistently.
        by_intensity = intensity.argsort(dim=1, descending=True, stable=True)
        ordered_d = d.masked_fill(~peak_mask, -torch.inf).gather(1, by_intensity)
        order = by_intensity.gather(1, ordered_d.argsort(dim=1, descending=True, stable=True))
        count = int(peak_mask.sum(1).max()) if batch else 0
        # Remove trailing padding for the whole batch without discarding any valid peak.
        order = order[:, :count]
        valid = peak_mask.gather(1, order)
        d, intensity = d.gather(1, order), intensity.gather(1, order)
        ranks = torch.arange(count, device=peak_d.device)
        # Match the existing encoder's reciprocal-spacing convention.
        h = (self.d_mlp(d.clamp_min(1e-6).reciprocal().unsqueeze(-1))
             + self.i_mlp(intensity.unsqueeze(-1)) + self.rank_embedding(ranks)[None])
        h = h.masked_fill(~valid[..., None], 0.)
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1).to(h.dtype), h), dim=1)
        key_padding_mask = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=peak_d.device), ~valid), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        return self.proj(self.norm(encoded[:, 0]))


def build_peak_encoder(config: dict[str, Any] | None, d_jepa: int) -> nn.Module:
    """Select the PXRD encoder without changing historical checkpoint key names."""
    config = config or {}
    version = config.get("version", PeakEncoder.version)
    if version == TransformerPeakEncoder.version:
        return TransformerPeakEncoder(config, d_jepa=d_jepa)
    raise ValueError(f"Unknown PeakEncoder version {version!r}")
