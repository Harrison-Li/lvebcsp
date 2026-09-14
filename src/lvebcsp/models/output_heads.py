"""Output heads for direct structure tokens."""

from __future__ import annotations

import torch
from torch import nn, Tensor

from lvebcsp.data.representation import decode_lattice_params


class FinalHead(nn.Module):
    """Separate lattice and coordinate heads."""

    def __init__(self, hidden_dim: int, d_y: int = 8, zero_init: bool = False) -> None:
        super().__init__()
        self.d_y = int(d_y)
        self.lattice_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 6))
        self.coord_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        if zero_init:
            nn.init.zeros_(self.lattice_head[-1].weight)
            nn.init.zeros_(self.lattice_head[-1].bias)
            nn.init.zeros_(self.coord_head[-1].weight)
            nn.init.zeros_(self.coord_head[-1].bias)

    @staticmethod
    def atom_pooling(atom_tokens: Tensor, atom_mask: Tensor):
        mask_f = atom_mask.unsqueeze(-1)
        denom = mask_f.sum(dim=1)
        return (atom_tokens * mask_f).sum(dim=1) / denom

    def forward(self, tokens: Tensor, atom_mask: Tensor, frac_coords_t: Tensor, t: Tensor) -> dict[str, Tensor]:
        """Project hidden tokens into lattice fields and coordinate velocities."""

        batch, total_tokens, _ = tokens.shape
        y0 = torch.zeros(batch, total_tokens, self.d_y, device=tokens.device, dtype=tokens.dtype)

        lattice_token = tokens[:, 0]
        atom_tokens = tokens[:, 1:]

        pooled_atoms = self.atom_pooling(atom_tokens, atom_mask)
        lattice_context = lattice_token + pooled_atoms
        lattice_encoded = self.lattice_head(lattice_context)

        atom_mask_f = atom_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        coord_velocity = self.coord_head(tokens[:, 1:]) * atom_mask_f

        t = t.reshape(-1).to(device=tokens.device, dtype=tokens.dtype)
        if t.shape[0] != batch:
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
        sigma = (1.0 - t).reshape(batch, 1, 1)
        frac_coords_t = frac_coords_t.to(device=tokens.device, dtype=tokens.dtype)
        frac_coords = torch.remainder(frac_coords_t + sigma * coord_velocity, 1.0) * atom_mask_f

        y0[:, 0, :6] = lattice_encoded
        y0[:, 1:, :3] = frac_coords

        return {
            "Y0_pred": y0,
            "lattice_pred": decode_lattice_params(lattice_encoded),
            "coord_pred": frac_coords,
            "coord_velocity": coord_velocity,
        }
