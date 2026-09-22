"""Copied from https://github.com/lucas-maes/le-wm/blob/main/module.py"""

import torch
import math
from torch import nn


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer
    SIGReg(Z) = 1/M \sum_{m=1}^M T^{m}
    T^{m} = \int \omega(t) |\phi(t;h_m)-\phi_0|^2 dt
    h_m = Z \cdot u_m
    """
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
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
        proj: (groups, samples, features), or (samples, features).
        Crystal slots and slot differences use (T, B, D), with crystals as samples.
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time


class VISReg(nn.Module):
    """
    VISReg regularizer.

    Drop-in usage:
        reg = VISReg(num_proj=1024)
        loss = reg(z)

    Input:
        z: (..., N, D)

        N = number of samples
        D = embedding dimension

        Examples:
            (B, D)
            (T, B, D)

    The loss contains:
        1. variance / scale regularization
        2. sliced-Wasserstein Gaussian shape regularization
        3. centering regularization
    """

    def __init__(
        self,
        num_proj=1024,
        scale_weight=1.0,
        shape_weight=1.0,
        center_weight=1.0,
        eps=1e-4,
    ):
        super().__init__()

        self.num_proj = num_proj
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight
        self.eps = eps

    def forward(self, z):
        """
        Args:
            z: (..., N, D)

        Returns:
            scalar VISReg loss
        """

        if z.ndim < 2:
            raise ValueError(
                f"VISReg expects (..., N, D), got {tuple(z.shape)}"
            )

        N = z.size(-2)
        D = z.size(-1)

        if N < 2:
            raise ValueError(
                "VISReg requires at least 2 samples."
            )

        # ---------------------------------------------------------
        # Center
        # ---------------------------------------------------------
        mu = z.mean(dim=-2, keepdim=True)
        zc = z - mu

        # ---------------------------------------------------------
        # Per-feature scale
        #
        # Use population std because we are matching N(0, I).
        # ---------------------------------------------------------
        std = torch.sqrt(
            zc.square().mean(dim=-2, keepdim=True) + self.eps
        )

        # Explicit anti-collapse / variance term
        scale_loss = (std - 1.0).square().mean()

        # ---------------------------------------------------------
        # Shape normalization
        #
        # Important: stop gradient through std.
        # The shape loss therefore does not try to control scale.
        # ---------------------------------------------------------
        zn = zc / std.detach().clamp_min(self.eps)

        # ---------------------------------------------------------
        # Random directions on unit sphere
        #
        # A: (D, K)
        # ---------------------------------------------------------
        A = torch.randn(
            D,
            self.num_proj,
            device=z.device,
            dtype=z.dtype,
        )

        A = A / A.norm(
            p=2,
            dim=0,
            keepdim=True,
        ).clamp_min(self.eps)

        # ---------------------------------------------------------
        # Random 1-D projections
        #
        # (..., N, D) @ (D, K)
        #       ->
        # (..., N, K)
        # ---------------------------------------------------------
        projected = zn @ A

        # Empirical quantiles
        projected = projected.sort(dim=-2).values

        # ---------------------------------------------------------
        # Standard Gaussian target quantiles
        #
        # q_i = Phi^{-1}((i + 0.5) / N)
        # ---------------------------------------------------------
        p = (
            torch.arange(
                N,
                device=z.device,
                dtype=z.dtype,
            )
            + 0.5
        ) / N

        q = math.sqrt(2.0) * torch.erfinv(
            2.0 * p - 1.0
        )

        # Make q broadcast against (..., N, K)
        q_shape = [1] * projected.ndim
        q_shape[-2] = N
        q_shape[-1] = 1

        q = q.view(q_shape)

        # ---------------------------------------------------------
        # Sliced Wasserstein shape loss
        # ---------------------------------------------------------
        shape_loss = (
            projected - q
        ).square().mean()

        # ---------------------------------------------------------
        # Center regularization
        # ---------------------------------------------------------
        center_loss = mu.square().mean()

        # ---------------------------------------------------------
        # Total VISReg
        # ---------------------------------------------------------
        loss = (
            self.scale_weight * scale_loss
            + self.shape_weight * shape_loss
            + self.center_weight * center_loss
        )

        return loss