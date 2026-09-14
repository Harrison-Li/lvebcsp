"""GPU selection helpers for training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def _parse_gpu_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def resolve_gpu_ids(config: dict[str, Any], device: torch.device) -> list[int]:
    """Resolve CUDA device ids from ``num_gpus`` or optional explicit ``gpu_ids``."""

    if "gpu_ids" in config:
        gpu_ids = _parse_gpu_ids(config.get("gpu_ids"))
    else:
        num_gpus = int(config.get("num_gpus", 1))
        if num_gpus < 0:
            raise ValueError("num_gpus must be >= 0")
        start = int(device.index or 0)
        gpu_ids = list(range(start, start + num_gpus))

    if device.type != "cuda":
        if "gpu_ids" in config or len(gpu_ids) > 1:
            raise RuntimeError("GPU training requires device: cuda or device: auto with CUDA available")
        return []

    if not gpu_ids:
        raise ValueError("num_gpus must be >= 1 when training on CUDA; set device: cpu for CPU training")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Duplicate GPU ids are not allowed: {gpu_ids}")

    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("CUDA device was selected, but torch.cuda.device_count() is 0")
    max_id = max(gpu_ids)
    if max_id >= available:
        raise RuntimeError(f"Requested GPU ids {gpu_ids}, but only {available} CUDA device(s) are visible")
    return gpu_ids


def primary_device(device: torch.device, gpu_ids: list[int]) -> torch.device:
    """Return the device where the module should be placed before optional DataParallel wrapping."""

    if device.type == "cuda" and gpu_ids:
        return torch.device(f"cuda:{gpu_ids[0]}")
    return device


def maybe_data_parallel(module: nn.Module, gpu_ids: list[int]) -> nn.Module:
    """Wrap a module in DataParallel when more than one CUDA device is requested."""

    if len(gpu_ids) <= 1:
        return module
    return nn.DataParallel(module, device_ids=gpu_ids, output_device=gpu_ids[0])


def unwrap_model(module: nn.Module) -> nn.Module:
    """Return the underlying module for state_dict/EMA operations."""

    parallel_types = (nn.DataParallel, DistributedDataParallel)
    return module.module if isinstance(module, parallel_types) else module
