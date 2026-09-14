"""Flow-matching corruption utilities for structure tensors."""

from __future__ import annotations

import copy

import torch


class FlowMatchingInterpolant:
    """Interpolant for simple Gaussian flow matching.

    This is intentionally kept close to the AADT/protein-frame-flow style
    implementation: one ``t_min`` parameter, centered Gaussian noise, optional
    ``diffuse_mask``, and Euler helpers for clean-endpoint denoisers.
    """

    def __init__(
        self,
        t_min: float = 1.0e-2,
        corrupt: bool = True,
        num_timesteps: int = 100,
        self_condition: bool = False,
        self_condition_prob: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        self.t_min = t_min
        self.corrupt = corrupt
        self.num_timesteps = num_timesteps
        self.self_condition = self_condition
        self.self_condition_prob = self_condition_prob
        self.device = device

    def _sample_t(self, batch_size: int) -> torch.Tensor:
        t = torch.rand(batch_size, device=self.device)
        return t * (1.0 - 2.0 * self.t_min) + self.t_min

    def _centered_gaussian(self, batch_size: int, num_tokens: int, emb_dim: int = 3) -> torch.Tensor:
        noise = torch.randn(batch_size, num_tokens, emb_dim, device=self.device)
        return noise - torch.mean(noise, dim=-2, keepdims=True)

    def _corrupt_x(
        self,
        x_1: torch.Tensor,
        t: torch.Tensor,
        token_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_0 = self._centered_gaussian(*x_1.shape)
        x_t = (1.0 - t[..., None]) * x_0 + t[..., None] * x_1
        x_t = x_t * diffuse_mask[..., None] + x_1 * (~diffuse_mask[..., None])
        return x_t * token_mask[..., None]

    def corrupt_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Corrupts a batch by sampling ``t`` and interpolating to noisy samples.

        Required keys:
        - ``x_1``: clean data tensor, shape ``[B, N, D]``
        - ``token_mask``: true for valid tokens, false for padding
        - ``diffuse_mask``: true for tokens to diffuse, false for fixed tokens
        """

        noisy_batch = copy.deepcopy(batch)
        x_1 = batch["x_1"]
        token_mask = batch["token_mask"]
        diffuse_mask = batch["diffuse_mask"]
        batch_size, _ = diffuse_mask.shape

        self.device = x_1.device
        t = self._sample_t(batch_size)[:, None]
        noisy_batch["t"] = t

        if self.corrupt:
            x_t = self._corrupt_x(x_1, t, token_mask, diffuse_mask)
        else:
            x_t = x_1
        if torch.any(torch.isnan(x_t)):
            raise ValueError("NaN in x_t during corruption")
        noisy_batch["x_t"] = x_t

        return noisy_batch

    def _x_vector_field(self, t: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return (x_1 - x_t) / (1.0 - t)

    def _x_euler_step(self, d_t: torch.Tensor, t: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        assert d_t > 0
        x_vf = self._x_vector_field(t, x_1, x_t)
        return x_t + x_vf * d_t

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_timesteps={self.num_timesteps}, self_condition={self.self_condition})"
