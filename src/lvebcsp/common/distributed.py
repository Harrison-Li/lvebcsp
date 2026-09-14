"""Distributed training helpers for PyTorch DDP."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed(backend: str | None = None) -> tuple[int, int, int]:
    """Initialize the distributed process group if launched via torchrun.

    Returns:
        (rank, local_rank, world_size).  When running without torchrun the
        values are (0, 0, 1) and no process group is created.
    """

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(
            backend=selected_backend,
            rank=rank,
            world_size=world_size,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    # Single-GPU fallback
    return 0, 0, 1


def cleanup_distributed() -> None:
    """Destroy the distributed process group if one exists."""

    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """Return whether distributed training is active."""

    return dist.is_initialized() and dist.get_world_size() > 1


def is_main_process() -> bool:
    """Return True on the rank-0 process (or always True in single-GPU mode)."""

    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank() -> int:
    """Return the global rank (0 when not distributed)."""

    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Return world size (1 when not distributed)."""

    return dist.get_world_size() if dist.is_initialized() else 1


@torch.no_grad()
def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor and divide by world size.

    Returns the input unchanged when not running distributed.
    """

    if not is_distributed():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t.div_(dist.get_world_size())
    return t
