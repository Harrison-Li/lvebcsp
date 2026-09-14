"""Optional Weights & Biases logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def get_wandb_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return normalized W&B config, or an empty dict when disabled."""

    raw = config.get("wandb")
    if raw in (None, False):
        return {}
    if raw is True:
        return {"enabled": True}
    if not isinstance(raw, dict):
        raise TypeError(f"wandb must be a bool, dict, or omitted; got {type(raw).__name__}")
    cfg = dict(raw)
    if not bool(cfg.get("enabled", True)):
        return {}
    cfg["enabled"] = True
    return cfg


def init_wandb(
    config: dict[str, Any],
    *,
    job_type: str,
    output_dir: str | Path,
    default_name: str | None = None,
) -> Any | None:
    """Initialize a W&B run when enabled in config."""

    cfg = get_wandb_config(config)
    if not cfg:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging is enabled, but the `wandb` package is not installed. "
            "Install it with `pip install wandb` or set `wandb.enabled: false`."
        ) from exc

    init_kwargs: dict[str, Any] = {
        "project": cfg.get("project", "lvebcsp"),
        "job_type": str(cfg.get("job_type", job_type)),
        "config": config,
        "dir": str(output_dir),
    }
    optional_keys = ("entity", "group", "tags", "notes", "mode", "id", "resume")
    for key in optional_keys:
        if key in cfg:
            init_kwargs[key] = cfg[key]
    name = cfg.get("name", default_name)
    if name:
        init_kwargs["name"] = str(name)

    run = wandb.init(**init_kwargs)
    if bool(cfg.get("define_metrics", True)):
        run.define_metric("trainer/global_step")
        run.define_metric("*", step_metric="trainer/global_step")
    return run


def wandb_log_interval(config: dict[str, Any], default: int = 10) -> int:
    """Return batch logging interval for W&B, or 0 when disabled."""

    cfg = get_wandb_config(config)
    if not cfg:
        return 0
    return max(int(cfg.get("log_interval", default)), 0)


def maybe_watch_model(run: Any | None, model: Any, config: dict[str, Any]) -> None:
    """Optionally ask W&B to watch gradients/parameters for a model."""

    if run is None:
        return
    cfg = get_wandb_config(config)
    watch_cfg = cfg.get("watch", False)
    if not watch_cfg:
        return

    import wandb

    if watch_cfg is True:
        kwargs: dict[str, Any] = {"log": "gradients", "log_freq": 100}
    elif isinstance(watch_cfg, str):
        kwargs = {"log": watch_cfg}
    elif isinstance(watch_cfg, dict):
        kwargs = dict(watch_cfg)
    else:
        raise TypeError(f"wandb.watch must be a bool, string, or dict; got {type(watch_cfg).__name__}")
    wandb.watch(model, **kwargs)


def log_wandb(run: Any | None, metrics: dict[str, Any], *, step: int | None = None) -> None:
    """Log metrics to W&B, filtering out unavailable values."""

    if run is None:
        return
    payload = {name: value for name, value in metrics.items() if value is not None}
    if not payload:
        return
    if step is not None:
        payload.setdefault("trainer/global_step", int(step))
        run.log(payload, step=int(step))
    else:
        run.log(payload)


def finish_wandb(run: Any | None) -> None:
    """Finish a W&B run if one was started."""

    if run is not None:
        run.finish()
