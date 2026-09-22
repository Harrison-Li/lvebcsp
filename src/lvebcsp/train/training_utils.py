"""Shared optimizer schedules, early stopping and batch transfer for trainers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch_geometric.data import Data

from lvebcsp.common.chemistry import atom_types_to_element_ratios, element_counts_to_ratios


@dataclass
class EarlyStopping:
    """Track loss plateaus and decide when to stop training."""

    patience: int
    min_delta: float = 0.0
    warmup_epochs: int = 0
    ignore_warmup_metrics: bool = False
    best: float | None = None
    best_epoch: int = -1
    wait: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        self.patience = int(self.patience)
        self.min_delta = float(self.min_delta)
        self.warmup_epochs = int(self.warmup_epochs)
        self.ignore_warmup_metrics = bool(self.ignore_warmup_metrics)
        if self.patience < 1:
            raise ValueError("early_stopping.patience must be >= 1")
        if self.min_delta < 0:
            raise ValueError("early_stopping.min_delta must be >= 0")
        if self.warmup_epochs < 0:
            raise ValueError("early_stopping.warmup_epochs must be >= 0")

    def step(self, value: float, epoch: int) -> bool:
        value = float(value)
        epoch = int(epoch)
        if self.ignore_warmup_metrics and epoch < self.warmup_epochs:
            self.wait = 0
            self.stopped = False
            return False
        # Discard a pre-warmup best restored from a checkpoint written before
        # ``ignore_warmup_metrics`` was introduced.
        if (
            self.ignore_warmup_metrics
            and self.best is not None
            and self.best_epoch < self.warmup_epochs
        ):
            self.best = None
            self.best_epoch = -1
            self.wait = 0
            self.stopped = False
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.wait = 0
            self.stopped = False
            return False
        if epoch < self.warmup_epochs:
            return False
        self.wait += 1
        self.stopped = self.wait >= self.patience
        return self.stopped

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "warmup_epochs": self.warmup_epochs,
            "ignore_warmup_metrics": self.ignore_warmup_metrics,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "wait": self.wait,
            "stopped": self.stopped,
        }


def build_early_stopper(config: dict[str, Any]) -> EarlyStopping | None:
    """Build optional early stopping state from a training config."""

    stopping_cfg = config.get("early_stopping")
    if stopping_cfg in (None, False):
        return None
    if stopping_cfg is True:
        cfg: dict[str, Any] = {}
    elif isinstance(stopping_cfg, dict):
        cfg = dict(stopping_cfg)
    else:
        raise TypeError(f"early_stopping must be a bool, dict, or omitted; got {type(stopping_cfg).__name__}")
    if not bool(cfg.get("enabled", True)):
        return None
    return EarlyStopping(
        patience=int(cfg.get("patience", 20)),
        min_delta=float(cfg.get("min_delta", 0.0)),
        warmup_epochs=int(cfg.get("warmup_epochs", 0)),
        ignore_warmup_metrics=bool(cfg.get("ignore_warmup_metrics", False)),
    )


def ensure_ratio_in_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Ensure JEPA batches include normalized elemental ratios."""

    if "ratio" in batch:
        return batch
    if "element_counts" not in batch and "atom_types" not in batch:
        return batch
    out = dict(batch)
    if "element_counts" in out:
        out["ratio"] = element_counts_to_ratios(out["element_counts"])
    else:
        out["ratio"] = atom_types_to_element_ratios(out["atom_types"])
    return out


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors and PyG crystal graphs to device."""

    return {key: value.to(device, non_blocking=True) if isinstance(value, (torch.Tensor, Data)) else value for key, value in batch.items()}


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    max_epochs: int,
    steps_per_epoch: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
    """Build a scheduler; the metric interval steps after validation."""

    scheduler_cfg = config.get("scheduler")
    if scheduler_cfg in (None, False):
        return None, "epoch"
    if isinstance(scheduler_cfg, str):
        cfg: dict[str, Any] = {"type": scheduler_cfg}
    elif isinstance(scheduler_cfg, dict):
        cfg = dict(scheduler_cfg)
    else:
        raise TypeError(f"scheduler must be a string, dict, or falsey value; got {type(scheduler_cfg).__name__}")

    scheduler_type = str(cfg.get("type", "")).lower().replace("-", "_")
    if scheduler_type in {"none", "off", "false"}:
        return None, "epoch"
    if scheduler_type in {"reducelronplateau", "reduce_lr_on_plateau", "reduce_on_plateau"}:
        cfg.pop("type", None)
        cfg.pop("interval", None)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg), "metric"
    if scheduler_type not in {"linearwarmupcosineannealinglr", "linear_warmup_cosine_annealing_lr"}:
        raise ValueError(f"Unknown scheduler type {cfg.get('type')!r}")

    interval = str(cfg.get("interval", "epoch")).lower()
    if interval not in {"epoch", "step"}:
        raise ValueError(f"scheduler.interval must be 'epoch' or 'step', got {interval!r}")

    steps_per_epoch = max(int(steps_per_epoch), 1)
    total_steps = max(int(max_epochs), 1)
    if interval == "step":
        total_steps *= steps_per_epoch

    if "warmup_steps" in cfg:
        warmup_steps = int(cfg["warmup_steps"])
    elif "warmup_epochs" in cfg:
        warmup_steps = int(cfg["warmup_epochs"])
        if interval == "step":
            warmup_steps *= steps_per_epoch
    else:
        warmup_steps = int(round(total_steps * float(cfg.get("warmup_fraction", 0.1))))
    warmup_steps = min(max(warmup_steps, 0), total_steps)

    eta_min = float(cfg.get("eta_min", cfg.get("min_lr", 0.0)))
    warmup_start_lr = float(cfg.get("warmup_start_lr", 0.0))

    lr_lambdas = []
    for group in optimizer.param_groups:
        base_lr = float(group["lr"])
        min_factor = eta_min / base_lr if base_lr > 0 else 0.0
        start_factor = warmup_start_lr / base_lr if base_lr > 0 else 0.0
        min_factor = min(max(min_factor, 0.0), 1.0)
        start_factor = min(max(start_factor, 0.0), 1.0)

        def lr_lambda(
            step: int,
            *,
            min_factor: float = min_factor,
            start_factor: float = start_factor,
            warmup_steps: int = warmup_steps,
            total_steps: int = total_steps,
        ) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return start_factor + (1.0 - start_factor) * (step / warmup_steps)
            cosine_steps = max(total_steps - warmup_steps, 1)
            cosine_step = min(max(step - warmup_steps, 0), cosine_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_steps))
            return min_factor + (1.0 - min_factor) * cosine

        lr_lambdas.append(lr_lambda)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas), interval
