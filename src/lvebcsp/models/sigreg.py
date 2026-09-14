"""Adapted from https://github.com/lucas-maes/le-wm/blob/main/module.py"""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer
    SIGReg(Z) = 1/M \sum_{m=1}^M T^{m}
    T^{m} = \int \omega(t) |\phi(t;h_m)-\phi_0|^2 dt
    h_m = Z \cdot u_m
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024, seed: int | None = None):
        super().__init__()
        self.num_proj = num_proj
        self.seed = seed
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: [..., D]
        """
        if proj.ndim < 2:
            raise ValueError(f"SIGReg expects at least 2 dimensions [..., D], got {tuple(proj.shape)}")
        samples = proj.reshape(-1, proj.size(-1)).to(torch.float32)
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=samples.device)
            generator.manual_seed(int(self.seed))
        # Build Gaussian basis from hypersphere directions.
        basis = torch.randn(samples.size(-1), self.num_proj, device=samples.device, generator=generator)
        basis = basis.div_(basis.norm(p=2, dim=0).clamp_min(1e-12))
        # compute the epps-pulley statistic
        x_t = (samples @ basis).unsqueeze(-1) * self.t.to(device=samples.device)
        # Euler's transformation of real part and imaginary part
        err = (x_t.cos().mean(dim=0) - self.phi.to(device=samples.device)).square() + x_t.sin().mean(dim=0).square()
        statistic = (err @ self.weights.to(device=samples.device)) * samples.size(0)
        return statistic.mean()  # average over projections and time
