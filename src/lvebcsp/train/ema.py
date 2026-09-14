"""Exponential moving average helpers for training checkpoints."""

from __future__ import annotations

import torch
from torch import nn


def init_ema_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module state dict for EMA tracking."""

    return {name: value.detach().clone() for name, value in module.state_dict().items()}


@torch.no_grad()
def update_ema_state(ema_state: dict[str, torch.Tensor], module: nn.Module, decay: float) -> None:
    """Update an EMA state dict from the current module state."""

    decay = float(decay)
    for name, value in module.state_dict().items():
        target = ema_state[name]
        source = value.detach()
        if torch.is_floating_point(target) or torch.is_complex(target):
            target.mul_(decay).add_(source.to(dtype=target.dtype), alpha=1.0 - decay)
        else:
            target.copy_(source)
