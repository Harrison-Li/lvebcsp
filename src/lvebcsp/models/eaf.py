"""Flow matching on frozen JEPA tokens, with a final crystal decoder.

Following https://github.com/lillian039/ELF/tree/pytorch_elf, the shared
backbone stays in embedding space until decoding. The crystal objective is
velocity MSE plus JEPA(generated crystal) versus JEPA(true crystal) MSE.
Species and counts are supplied by the building-block condition.
"""

from __future__ import annotations

import math

import numpy as np
from pymatgen.optimization.neighbors import find_points_in_spheres
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from lvebcsp.data.representation import encoded_lattice_to_matrix
from lvebcsp.models.condition_encoder import ConditionEncoder
from lvebcsp.models.dit_blocks import DiTBlock, TimestepEmbedder
from lvebcsp.models.encoder import UniversalEncoder
from lvebcsp.models.interpolants.flow_matching import FlowMatchingInterpolant
from lvebcsp.models.layers import MLP


class LocalAtomDecoder(nn.Module):
    """Expand each latent into queries with supplied species, without atom self-attention."""

    def __init__(self, latent_dim=256, max_atoms=32, num_elements=119, heads=8):
        super().__init__()
        self.max_atoms = max_atoms
        self.query = nn.Parameter(torch.randn(1, max_atoms, latent_dim))
        self.atom_embedding = nn.Embedding(num_elements, latent_dim, padding_idx=0)
        self.cross_attn = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2), nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.coord_head = nn.Linear(latent_dim, 3)

    def forward(self, patch_latent: Tensor, atom_types: Tensor) -> Tensor:
        b, k, d = patch_latent.shape
        z = patch_latent.reshape(b * k, 1, d)
        species = self.atom_embedding(atom_types.reshape(b * k, self.max_atoms))
        q = self.query.expand(b * k, -1, -1) + species
        h, _ = self.cross_attn(q, z, z, need_weights=False)
        # With one key, attention alone repeats the same value for every atom.
        h = q + h
        return self.coord_head(h + self.ffn(h)).reshape(b, k * self.max_atoms, 3)


class EAF(nn.Module):
    """Conditional flow on [B, num_latents, token_dim] frozen JEPA embeddings."""

    def __init__(
        self,
        encoder: UniversalEncoder,
        projector: nn.Module | None = None,
        condition_encoder: ConditionEncoder | None = None,
        token_dim: int = 512,
        hidden_dim: int = 256,
        layers: int = 6,
        heads: int = 8,
        max_atoms: int = 32,
        dropout: float = 0.0,
        t_min: float = 0.01,
        num_steps: int = 100,
        loss_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        if not 0 <= t_min < 0.5 or num_steps < 1:
            raise ValueError("Flow matching requires 0 <= t_min < 0.5 and num_steps >= 1.")
        if encoder.config.atom_feature_dim:
            raise ValueError("Crystal decoding requires a JEPA encoder without extra atom features.")
        self.encoder = encoder
        self.projector = projector if projector is not None else nn.Identity()
        self.condition_encoder = condition_encoder or ConditionEncoder(token_dim, hidden_dim)
        for module in (self.encoder, self.projector, self.condition_encoder):
            module.requires_grad_(False).eval()
        self.num_latents = encoder.config.num_latents
        self.token_dim = token_dim
        self.loss_weights = {"flow": 1.0, "jepa": 1.0, **(loss_weights or {})}
        if self.loss_weights.keys() - {"flow", "jepa"}:
            raise ValueError("EAF supports only 'flow' and 'jepa' loss weights.")
        self.interpolant = FlowMatchingInterpolant(
            t_min=t_min, num_timesteps=num_steps, center_noise=False,
        )

        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.condition_proj = nn.Linear(self.condition_encoder.hidden_dim, hidden_dim)
        self.slot_embedding = nn.Parameter(torch.randn(1, self.num_latents, hidden_dim) * 0.02)
        self.time_embedding = TimestepEmbedder(hidden_dim)
        self.mode_embedding = nn.Embedding(2, hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, token_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)
        self.decoder = LocalAtomDecoder(hidden_dim, max_atoms, heads=heads)
        self.lattice_head = MLP(hidden_dim, hidden_dim, 6)
        # Start from a nondegenerate cell on the encoder's physical length scale.
        nn.init.zeros_(self.lattice_head.mlp[-1].weight)
        nn.init.zeros_(self.lattice_head.mlp[-1].bias)
        with torch.no_grad():
            self.lattice_head.mlp[-1].bias[:3].fill_(math.log(2 * encoder.config.cutoff))

    def train(self, mode: bool = True):
        super().train(mode)
        for module in (self.encoder, self.projector, self.condition_encoder):
            module.eval()
        return self

    def encode(self, graph: Data | Batch) -> Tensor:
        # Frozen weights still allow gradients with respect to generated geometry.
        return self.projector(self.encoder(graph))

    @torch.no_grad()
    def encode_condition(self, batch) -> Tensor:
        blocks = self.encode(batch["context"])
        return self.condition_encoder(
            blocks, batch["multiplicity"], batch["block_batch"], size=batch["atom_types"].size(0),
        )

    def backbone(self, z: Tensor, t: Tensor, condition: Tensor, *, decode: bool = False) -> Tensor:
        h = self.input_proj(z) + self.condition_proj(condition) + self.slot_embedding
        c = self.time_embedding(t * 1000) + self.mode_embedding.weight[int(decode)]
        mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        for block in self.blocks:
            h = block(h, c, mask)
        return self.output_norm(h)

    def velocity(self, z: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        return self.velocity_head(self.backbone(z, t, condition))

    def decode_atoms(self, hidden: Tensor, atom_types: Tensor) -> dict[str, Tensor]:
        capacity = self.num_latents * self.decoder.max_atoms
        if atom_types.size(1) > capacity:
            raise ValueError(f"Condition has {atom_types.size(1)} atoms; decoder capacity is {capacity}.")
        atom_types = F.pad(atom_types, (0, capacity - atom_types.size(1)))
        mask = atom_types != 0
        coords = self.decoder(hidden, atom_types).float().remainder(1)
        lattice = self.lattice_head(hidden.mean(dim=1)).float()
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            cell = encoded_lattice_to_matrix(lattice)
        return {"frac_coords": coords * mask.unsqueeze(-1), "atom_types": atom_types,
                "atom_mask": mask, "lattice": lattice, "cell": cell}

    def decode(self, tokens: Tensor, condition: Tensor, atom_types: Tensor) -> dict[str, Tensor]:
        t = tokens.new_ones(tokens.size(0))
        return self.decode_atoms(self.backbone(tokens, t, condition, decode=True), atom_types)

    def generated_graph(self, prediction: dict[str, Tensor]) -> Batch:
        """Rebuild periodic neighbors; detach topology only, retaining geometry gradients."""
        graphs = []
        for frac, cell, species, mask in zip(
            prediction["frac_coords"], prediction["cell"],
            prediction["atom_types"], prediction["atom_mask"],
        ):
            with torch.autocast(device_type=frac.device.type, enabled=False):
                pos = frac[mask].float() @ cell.float()
            xyz = pos.detach().cpu().double().numpy()
            lattice = cell.detach().cpu().double().numpy()
            dst, src, shifts, _ = find_points_in_spheres(
                xyz, xyz, self.encoder.config.cutoff, np.ones(3, dtype=np.int64), lattice,
            )
            keep = (src != dst) | (shifts != 0).any(axis=1)
            graphs.append(Data(
                z=species[mask], pos=pos, cell=cell.unsqueeze(0),
                edge_index=torch.as_tensor(np.stack([src[keep], dst[keep]]),
                                           dtype=torch.long, device=pos.device),
                edge_shifts=torch.as_tensor(shifts[keep], dtype=pos.dtype, device=pos.device),
            ))
        return Batch.from_data_list(graphs)

    def forward(self, batch) -> dict[str, Tensor | dict[str, Tensor]]:
        condition = self.encode_condition(batch)
        with torch.no_grad():
            clean = self.encode(batch["target"]).float()
        valid = torch.ones(clean.shape[:2], dtype=torch.bool, device=clean.device)
        noisy = self.interpolant.corrupt_batch(
            {"x_1": clean, "token_mask": valid, "diffuse_mask": valid},
        )
        t, z_t = noisy["t"], noisy["x_t"]
        v_target = self.interpolant._x_vector_field(t[..., None], clean, z_t)
        v_pred = self.velocity(z_t, t[:, 0], condition).float()
        loss_flow = F.mse_loss(v_pred, v_target)

        # Estimate the clean endpoint, then decode once at t=1. No training rollout.
        endpoint = z_t + (1 - t[..., None]) * v_pred
        prediction = self.decode(endpoint, condition, batch["atom_types"])
        generated = self.encode(self.generated_graph(prediction)).float()
        loss_jepa = F.mse_loss(generated, clean)
        loss = self.loss_weights["flow"] * loss_flow + self.loss_weights["jepa"] * loss_jepa
        return {"loss": loss, "metrics": {"flow": loss_flow, "jepa": loss_jepa,
                "target_std": clean.std(dim=0, unbiased=False).mean()}}

    @torch.no_grad()
    def sample(self, batch, steps: int | None = None, initial_noise: Tensor | None = None):
        steps = self.interpolant.num_timesteps if steps is None else steps
        if steps < 1:
            raise ValueError("Sampling requires at least one flow step.")
        condition = self.encode_condition(batch)
        shape = (condition.size(0), self.num_latents, self.token_dim)
        z = torch.randn(shape, device=condition.device) if initial_noise is None else initial_noise.clone()
        if z.shape != shape:
            raise ValueError(f"Expected initial_noise shape {shape}, got {tuple(z.shape)}.")
        for step in range(steps):
            t = z.new_full((z.size(0),), step / steps)
            z = z + self.velocity(z, t, condition) / steps
        return {"tokens": z, **self.decode(z, condition, batch["atom_types"])}


# Existing training and inference entry points import this name.
CrystalLatentDiffusion = EAF
