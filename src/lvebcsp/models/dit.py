"""Direct structure diffusion/flow Transformer conditioned on JEPA latents."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lvebcsp.data.representation import (
    decode_lattice_params,
    scale_normalized_lattice_gram,
)
from lvebcsp.models.dit_blocks import DiTBlock, TimestepEmbedder
from lvebcsp.models.module import AtomEmbedding
from lvebcsp.models.output_heads import FinalHead


def pos_embedding(indices: torch.Tensor, emb_dim: int, max_len: int = 2048) -> torch.Tensor:
    """Create sine/cosine positional embeddings for integer token indices."""

    half_dim = emb_dim // 2
    frequencies = torch.arange(half_dim, device=indices.device, dtype=torch.float32)
    scales = max_len ** (2 * frequencies / max(emb_dim, 1))
    angles = indices.to(torch.float32)[..., None] * math.pi / scales
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding


class DiffsuionTransformer(nn.Module):
    """DiT-style denoiser that predicts lattice parameters and fractional coordinates."""

    def __init__(
        self,
        n_max: int = 64,
        d_y: int | None = None,
        d_jepa: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        zero_init_output: bool = False,
        lattice_representation: str = "gram_cholesky",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.n_max = int(n_max)
        self.d_y = int(d_y if d_y is not None else 8)
        self.d_jepa = int(d_jepa)
        self.hidden_dim = int(hidden_dim)
        # self.num_condition_tokens = int(kwargs.get("num_condition_tokens", 4))
        self.lattice_representation = str(lattice_representation).lower()
        if self.lattice_representation != "gram_cholesky":
            raise ValueError(
                "lattice_representation must be 'gram_cholesky'"
            )

        self.lattice_embedder = nn.Sequential(
            nn.Linear(6 + 9, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.atom_embedder = AtomEmbedding(fea_dim=self.hidden_dim, max_num_elements=119)
        self.t_embedder = TimestepEmbedder(self.hidden_dim)
        self.z_proj = nn.Sequential(
            nn.Linear(self.d_jepa, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # self.jepa_context_proj = nn.Linear(
        #     self.d_jepa, self.num_condition_tokens * self.hidden_dim
        # )
        # self.jepa_context_norm = nn.LayerNorm(self.hidden_dim)
        self.null_z = nn.Parameter(torch.zeros(self.hidden_dim))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.hidden_dim,
                    num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_heads = FinalHead(
            self.hidden_dim,
            d_y=self.d_y,
            zero_init=zero_init_output,
        )
        self.initialize_weights(zero_init_output=zero_init_output)

    def initialize_weights(self, zero_init_output: bool = False) -> None:
        """Initialize direct-path glue while preserving DiTBlock internals."""

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

        self.apply(_basic_init)
        nn.init.normal_(self.atom_embedder.embedding.weight, std=0.02)
        nn.init.zeros_(self.atom_embedder.embedding.weight[0])
        nn.init.zeros_(self.null_z)

        for module in self.modules():
            if hasattr(module, "adaLN_modulation"):
                modulation = getattr(module, "adaLN_modulation")
                if isinstance(modulation, nn.Sequential) and isinstance(modulation[-1], nn.Linear):
                    nn.init.zeros_(modulation[-1].weight)
                    nn.init.zeros_(modulation[-1].bias)
            # Original context gate initialization:
            # if hasattr(module, "cross_gate"):
            #     cross_gate = getattr(module, "cross_gate")
            #     if isinstance(cross_gate, nn.Sequential):
            #         nn.init.zeros_(cross_gate[-1].weight)
            #         nn.init.zeros_(cross_gate[-1].bias)

        if zero_init_output:
            nn.init.zeros_(self.output_heads.lattice_head[-1].weight)
            nn.init.zeros_(self.output_heads.lattice_head[-1].bias)
            nn.init.zeros_(self.output_heads.coord_head[-1].weight)
            nn.init.zeros_(self.output_heads.coord_head[-1].bias)

    def forward(
        self,
        lattice_t: Tensor,
        frac_coords_t: Tensor,
        t: Tensor,
        z_jepa: Tensor,
        atom_types: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor]:
        """Predict structure flow fields from current fractional coordinates."""

        batch_size = frac_coords_t.shape[0]
        device = frac_coords_t.device
        t = t.expand(batch_size) if t.ndim == 0 else t.reshape(-1)
        if t.shape[0] != batch_size:
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
        if z_jepa is not None and tuple(z_jepa.shape) != (batch_size, self.d_jepa):
            raise ValueError(f"z_jepa must have shape [B, {self.d_jepa}], got {tuple(z_jepa.shape)}")
        atom_mask = mask.to(dtype=torch.bool)
        structure_mask = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=device), atom_mask],
            dim=1,
        )

        gram_features = scale_normalized_lattice_gram(lattice_t).flatten(
            start_dim=1
        ).to(lattice_t)
        lattice_input = torch.cat((lattice_t, gram_features), dim=-1)

        lattice_embs = self.lattice_embedder(lattice_input).unsqueeze(1)
        

        atom_embs = self.atom_embedder(atom_types.long())
        atom_embs += self.frac_coords_embedder(frac_coords_t)
        x = torch.cat([lattice_embs, atom_embs], dim=1) # [B, 1 + n_atoms, h]
        # Absolute token-index embeddings break atom-token permutation equivariance.
        # token_index = torch.cumsum(structure_mask, dim=-1, dtype=torch.int64) - 1
        # x += pos_embedding(token_index, self.hidden_dim)
        x = x * structure_mask.to(dtype=x.dtype).unsqueeze(-1)

        t_emb = self.t_embedder(t)
        if z_jepa is None:
            z_emb = self.null_z.unsqueeze(0).expand(batch_size, -1)
        else:
            z_emb = self.z_proj(z_jepa.to(dtype=t_emb.dtype))

        c = z_emb + t_emb
        # Original JEPA token-context path:
        # jepa_context = None
        # if z_jepa is not None and self.num_condition_tokens > 0:
        #     jepa_context = self.jepa_context_proj(
        #         z_jepa.to(device=device, dtype=x.dtype)
        #     )
        #     jepa_context = jepa_context.reshape(
        #         batch_size, self.num_condition_tokens, self.hidden_dim
        #     )
        #     jepa_context = self.jepa_context_norm(jepa_context)
        for block in self.blocks:
            x = block(
                x,
                c,
                structure_mask,
                # context=jepa_context,
            )
            x = x * structure_mask.to(dtype=x.dtype).unsqueeze(-1)
        tokens = self.final_norm(x)
        return self.output_heads(tokens, atom_mask, frac_coords_t, t)

    def criterion(self, noisy_dense_encoded_batch, pred_x, loss_cfg=None, pred_coord_velocity: Tensor | None = None):
        loss_cfg = loss_cfg or {}

        target = noisy_dense_encoded_batch["x_1"]
        t = noisy_dense_encoded_batch["t"].reshape(-1)
        token_mask = noisy_dense_encoded_batch["token_mask"]

        atom_w = token_mask[:, 1:].to(dtype=pred_x.dtype).unsqueeze(-1)
        lattice_w = token_mask[:, 0].to(dtype=pred_x.dtype)

        norm = (1.0 - torch.minimum(t, t.new_tensor(0.9))).clamp_min(1.0e-6)

        # coordinate loss
        pred_frac = torch.remainder(pred_x[:, 1:, :3].to(torch.float32), 1.0)
        target_frac = torch.remainder(target[:, 1:, :3].to(torch.float32), 1.0)

        coord_diff = torch.remainder(pred_frac - target_frac + 0.5, 1.0) - 0.5
        coord_denom = atom_w.sum(dim=(1, 2)).clamp_min(1.0) * 3.0
        coord_mse = torch.sum(coord_diff.square() * atom_w, dim=(1, 2)) / coord_denom
        coord_velocity_rmse = None
        if "coord_velocity" in noisy_dense_encoded_batch and "x_t" in noisy_dense_encoded_batch:
            noisy_frac = torch.remainder(noisy_dense_encoded_batch["x_t"].to(torch.float32), 1.0)
            target_velocity = noisy_dense_encoded_batch["coord_velocity"].to(torch.float32)
            sigma = (1.0 - t).clamp_min(1.0e-6)
            if pred_coord_velocity is None:
                pred_velocity = (torch.remainder(pred_frac - noisy_frac + 0.5, 1.0) - 0.5) / sigma[:, None, None]
            else:
                pred_velocity = pred_coord_velocity.to(torch.float32)
            velocity_diff = pred_velocity - target_velocity
            coord_velocity_mse = torch.sum(velocity_diff.square() * atom_w, dim=(1, 2)) / coord_denom
            coord_loss = coord_velocity_mse.mean()
            coord_velocity_rmse = torch.sqrt(coord_velocity_mse.mean().clamp_min(0.0)).detach()
        else:
            coord_ps = torch.sum((coord_diff / norm[:, None, None]).square() * atom_w, dim=(1, 2)) / coord_denom
            coord_loss = coord_ps.mean()

        # lattice loss
        pred_lattice = decode_lattice_params(pred_x[:, 0, :6])
        target_lattice = decode_lattice_params(target[:, 0, :6])

        length_err = (pred_lattice[:, :3] - target_lattice[:, :3]) / target_lattice[:, :3].clamp_min(1.0e-6)
        length_rel_mse = length_err.square().mean(dim=-1)
        length_abs_mse = (pred_lattice[:, :3] - target_lattice[:, :3]).square().mean(dim=-1)
        length_ps = (length_err / norm[:, None]).square().mean(dim=-1)
        length_loss = torch.sum(length_ps * lattice_w) / lattice_w.sum().clamp_min(1.0)

        pred_angle = torch.deg2rad(pred_lattice[:, 3:6])
        target_angle = torch.deg2rad(target_lattice[:, 3:6])

        angle_err = torch.stack(
            [
                torch.sin(pred_angle) - torch.sin(target_angle),
                torch.cos(pred_angle) - torch.cos(target_angle),
            ],
            dim=-1,
        )

        angle_ps = (angle_err / norm[:, None, None]).square().mean(dim=(-1, -2))
        angle_loss = torch.sum(angle_ps * lattice_w) / lattice_w.sum().clamp_min(1.0)
        angle_deg_mse = (pred_lattice[:, 3:6] - target_lattice[:, 3:6]).square().mean(dim=-1)

        w_coord = float(loss_cfg.get("w_coord", 1.0))
        w_length = float(loss_cfg.get("w_lattice_length_rel", 1.0))
        w_angle = float(loss_cfg.get("w_lattice_angle", 1.0))

        loss = w_coord * coord_loss + w_length * length_loss + w_angle * angle_loss

        metrics = {
            "loss": loss,
            "coord_frac_rmse": torch.sqrt(coord_mse.mean().clamp_min(0.0)).detach(),
            "lattice_len_rel_rmse": torch.sqrt(
                (torch.sum(length_rel_mse * lattice_w) / lattice_w.sum().clamp_min(1.0)).clamp_min(0.0)
            ).detach(),
            "lattice_len_angstrom_rmse": torch.sqrt(
                (torch.sum(length_abs_mse * lattice_w) / lattice_w.sum().clamp_min(1.0)).clamp_min(0.0)
            ).detach(),
            "lattice_angle_deg_rmse": torch.sqrt(
                (torch.sum(angle_deg_mse * lattice_w) / lattice_w.sum().clamp_min(1.0)).clamp_min(0.0)
            ).detach(),
            "t_avg": t.detach().mean(),
        }
        if coord_velocity_rmse is not None:
            metrics["coord_velocity_rmse"] = coord_velocity_rmse
        return metrics

    def guided_forward(
        self,
        lattice_t: Tensor,
        frac_coords_t: Tensor,
        t: Tensor,
        z_jepa: Tensor,
        atom_types: Tensor,
        mask: Tensor,
        guidance_scale: float = 1.0,
    ) -> dict[str, Tensor]:
        """Classifier-free guidance over the JEPA latent condition."""

        cond = self.forward(lattice_t, frac_coords_t, t, z_jepa, atom_types, mask)
        if float(guidance_scale) == 1.0:
            return cond
        uncond = self.forward(lattice_t, frac_coords_t, t, None, atom_types, mask)
        scale = float(guidance_scale)
        y0 = torch.zeros_like(cond["Y0_pred"])
        lattice_encoded = uncond["Y0_pred"][:, 0, :6] + scale * (
            cond["Y0_pred"][:, 0, :6] - uncond["Y0_pred"][:, 0, :6]
        )
        coord_velocity = uncond["coord_velocity"] + scale * (cond["coord_velocity"] - uncond["coord_velocity"])
        lattice_params = decode_lattice_params(lattice_encoded)
        atom_mask_f = mask.to(device=y0.device, dtype=y0.dtype).unsqueeze(-1)
        t_vec = t.expand(frac_coords_t.shape[0]) if t.ndim == 0 else t.reshape(-1)
        sigma = (1.0 - t_vec).to(device=y0.device, dtype=y0.dtype).reshape(-1, 1, 1)
        frac_coords = torch.remainder(frac_coords_t.to(device=y0.device, dtype=y0.dtype) + sigma * coord_velocity, 1.0)
        frac_coords = frac_coords * atom_mask_f
        y0[:, 0, :6] = lattice_encoded
        y0[:, 1:, :3] = frac_coords
        y0[:, 1:, 3:] = y0[:, 1:, 3:] * atom_mask_f
        return {
            "Y0_pred": y0,
            "lattice_pred": lattice_params,
            "coord_pred": frac_coords,
            "coord_velocity": coord_velocity,
        }


# Public name used throughout the training and inference entry points.
LatentDiT = DiffsuionTransformer
