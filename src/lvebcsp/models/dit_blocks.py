"""Transformer blocks for LatentDiT."""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor
from lvebcsp.models.layers import MLP


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.frequency_embedding_dim = hidden_dim
        self.mlp = MLP(
            input_dim=self.frequency_embedding_dim,
            hidden_dim=hidden_dim * 4,
            output_dim=hidden_dim,
        )

    @staticmethod
    def timestep_embedding(t, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Conditional Adaptive LayerNorm Transformer denoiser block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=0, 
            bias=True,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # Original context modules:
        # self.cross_norm_q = nn.LayerNorm(
        #     hidden_dim, elementwise_affine=False, eps=1e-6
        # )
        # self.cross_norm_kv = nn.LayerNorm(hidden_dim, eps=1e-6)
        # self.cross_attn = nn.MultiheadAttention(
        #     embed_dim=hidden_dim,
        #     num_heads=num_heads,
        #     dropout=0,
        #     bias=True,
        #     batch_first=True,
        # )
        # self.cross_gate = nn.Sequential(
        #     nn.SiLU(), nn.Linear(hidden_dim, hidden_dim, bias=True)
        # )
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * 4,
            hidden_dim,
            act_fn=lambda: nn.GELU(approximate="tanh"),
            norm_fn=nn.LayerNorm,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        mask: Tensor,
        # context: Tensor | None = None,
        # context_mask: Tensor | None = None,
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=-1)

        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(_x, _x, _x, key_padding_mask=~mask, need_weights=False)[0]
        )
        # Original token-context path:
        # if context is not None:
        #     key_padding_mask = (
        #         None
        #         if context_mask is None
        #         else ~context_mask.to(dtype=torch.bool)
        #     )
        #     cross = self.cross_attn(
        #         self.cross_norm_q(x),
        #         self.cross_norm_kv(context),
        #         self.cross_norm_kv(context),
        #         key_padding_mask=key_padding_mask,
        #         need_weights=False,
        #     )[0]
        #     x = x + self.cross_gate(c).unsqueeze(1) * cross
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
