"""PXRD-conditioned prediction of a fixed geometric packing descriptor.

This optional terminal readout complements the learned JEPA crystal tokens. Its
target is a periodic pair-distance histogram, so joint encoder optimization
cannot remove its packing dependence. It learns descriptor prediction and
candidate ranking, not atom coordinates or the original JEPA token correction.
"""
from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from pymatgen.core import Structure
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from lvebcsp.data.crystal_pxrd import pxrd_settings


def periodic_pair_histogram(graph: Data, *, cutoff: float = 6., bins: int = 24,
                            rounding_decimals: int = 4) -> Tensor:
    """Count periodic neighbors per atom, including molecular internal pairs.

    Molecules remain rigid in the packing task, so internal pair counts are
    constant across candidate packings. Both directions of a pair are counted,
    as are neighbors in distinct periodic images; zero-shift self pairs are not.
    The returned descriptor is deliberately fixed and non-differentiable.
    """
    if cutoff <= 0 or bins < 1 or len(graph.z) == 0:
        raise ValueError("A nonempty crystal, positive cutoff and bins are required")
    cell = graph.cell.detach().cpu().numpy().reshape(3, 3)
    pos = graph.pos.detach().cpu().numpy()
    numbers = graph.z.detach().cpu().numpy()
    structure = Structure(cell, numbers, pos, coords_are_cartesian=True)
    distance = structure.get_neighbor_list(cutoff + 10. ** -rounding_decimals)[3]
    distance = np.round(distance, rounding_decimals)
    histogram = np.histogram(distance[distance <= cutoff], bins=np.linspace(0., cutoff, bins + 1))[0]
    return torch.from_numpy(histogram.astype(np.float32) / len(numbers))


def reciprocal_profile(d: Tensor, intensity: Tensor, mask: Tensor, *, bins: int = 256,
                       wavelength: float = 1.5406, angles: tuple[float, float] = (5., 80.)) -> Tensor:
    """Area-normalized intensity-weighted Gaussian peaks on a 1/d grid.

    Gaussian sigma is one grid interval. Peak positions and intensities stay
    paired, and padding is removed before reciprocal and intensity operations.
    """
    if bins < 2 or wavelength <= 0 or not 0 < angles[0] < angles[1] < 180:
        raise ValueError("Invalid reciprocal profile settings")
    if mask.dtype != torch.bool or d.shape != intensity.shape or d.shape != mask.shape:
        raise ValueError("Expected matching d/intensity/boolean-mask arrays")
    if not mask.any(-1).all():
        raise ValueError("Every observation must contain a valid peak")
    if not torch.isfinite(d[mask]).all() or not torch.isfinite(intensity[mask]).all():
        raise ValueError("Valid peaks must be finite")
    if (d[mask] <= 0).any() or (intensity[mask] < 0).any():
        raise ValueError("Valid spacings must be positive and intensities nonnegative")
    limits = [2 * math.sin(math.radians(a / 2)) / wavelength for a in angles]
    grid = torch.linspace(*limits, bins, device=d.device, dtype=torch.float32)
    width = (limits[1] - limits[0]) / (bins - 1)
    q = d.float().masked_fill(~mask, 1.).clamp_min(1e-6).reciprocal()
    weight = intensity.float().masked_fill(~mask, 0.)
    values = (torch.exp(-.5 * ((q[..., None] - grid) / width).square()) * weight[..., None]).sum(-2)
    mass = values.sum(-1, keepdim=True)
    if not torch.isfinite(values).all() or (mass <= 0).any():
        raise ValueError("Nonfinite or empty diffraction profile")
    return values / mass


class PackingReadout(nn.Module):
    """Predict physical descriptor from context geometry and observed PXRD only.

    Features have shape [..., bins] and [..., profile_bins]. All normalization
    statistics and the residual scale are fitted on training families and saved
    in the checkpoint. Candidate structures enter only the subsequent scoring.
    """
    def __init__(self, *, bins: int = 24, profile_bins: int = 256, hidden: int = 256,
                 use_condition: bool = True, cutoff: float = 6., rounding_decimals: int = 4,
                 wavelength: float = 1.5406, angles: tuple[float, float] = (5., 80.)) -> None:
        super().__init__()
        self.bins, self.profile_bins = bins, profile_bins
        self.use_condition = use_condition
        self.cutoff, self.rounding_decimals = cutoff, rounding_decimals
        self.wavelength, self.angles = wavelength, tuple(angles)
        self.register_buffer("feature_mean", torch.zeros(profile_bins))
        self.register_buffer("feature_scale", torch.ones(profile_bins))
        self.register_buffer("residual_scale", torch.ones(()))
        self.context_norm = nn.LayerNorm(bins)
        self.net = nn.Sequential(nn.Linear(bins + profile_bins, hidden), nn.GELU(),
                                 nn.LayerNorm(hidden), nn.Linear(hidden, bins))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context_histogram: Tensor, profile: Tensor) -> Tensor:
        if context_histogram.shape[:-1] != profile.shape[:-1]:
            raise ValueError("Context and observation batch shapes must match")
        condition = (profile - self.feature_mean) / self.feature_scale
        if not self.use_condition:
            condition = torch.zeros_like(condition)
        features = torch.cat((self.context_norm(context_histogram), condition), dim=-1)
        return context_histogram + self.residual_scale * self.net(features)

    def describe(self, graphs: Data | Sequence[Data]) -> Tensor:
        if isinstance(graphs, Batch):
            graphs = graphs.to_data_list()
        elif isinstance(graphs, Data):
            graphs = [graphs]
        return torch.stack([periodic_pair_histogram(g, cutoff=self.cutoff, bins=self.bins,
                            rounding_decimals=self.rounding_decimals) for g in graphs]).to(self.feature_mean.device)

    def predict_graphs(self, context_graphs: Data | Sequence[Data], peak_d: Tensor,
                       peak_i: Tensor, peak_mask: Tensor) -> Tensor:
        context = self.describe(context_graphs)
        device = self.feature_mean.device
        profile = reciprocal_profile(peak_d.to(device), peak_i.to(device), peak_mask.to(device),
                   bins=self.profile_bins, wavelength=self.wavelength, angles=self.angles)
        if profile.ndim == 1:
            profile = profile[None]
        return self(context, profile)

    def candidate_energies(self, predicted_histogram: Tensor, candidates: Data | Sequence[Data]) -> Tensor:
        """Return [queries, candidates] mean squared distances (lower is better)."""
        descriptors = self.describe(candidates)
        return (predicted_histogram.reshape(-1, self.bins)[:, None] - descriptors[None]).square().mean(-1)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "PackingReadout":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("target") != "periodic_pair_histogram" or checkpoint.get("arm") not in {"profile_pxrd", "context_only"}:
            raise ValueError("Expected a physical-descriptor profile or context-only checkpoint")
        settings = pxrd_settings(checkpoint["pxrd"])
        state = checkpoint["state_dict"]
        model = cls(bins=checkpoint["bins"], profile_bins=state["feature_mean"].numel(),
                    hidden=checkpoint["hidden"], use_condition=checkpoint["use_condition"],
                    cutoff=checkpoint["cutoff"], rounding_decimals=checkpoint["rounding_decimals"],
                    wavelength=settings["wavelength"], angles=settings["two_theta_range"])
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()
