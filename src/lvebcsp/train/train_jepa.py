"""Train crystal JEPA: independently perturbed molecular packing -> clean tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from lvebcsp.data.crystal_dataset import collate_crystal_graphs, context_parameter_range
from lvebcsp.common.config import load_config, save_config, select_device
from lvebcsp.common.gpu import primary_device, resolve_gpu_ids, unwrap_model
from lvebcsp.common.distributed import cleanup_distributed, is_distributed, is_main_process, setup_distributed
from lvebcsp.common.seed import seed_everything
from lvebcsp.common.wandb_logging import (
    finish_wandb,
    init_wandb,
    log_wandb,
    maybe_watch_model,
)


# Re-export shared helpers used by downstream trainers and existing callers.
from lvebcsp.train.jepa_setup import (
    build_jepa_head, build_jepa_from_config, load_jepa_weights,
    build_dataset_from_config, has_dataset_split, resolve_p_max,
    conditioning_contract,
)
from lvebcsp.train.training_utils import (
    EarlyStopping, build_early_stopper, build_lr_scheduler,
    batch_to_device, ensure_ratio_in_batch,
)


JEPA_OBJECTIVE = "mse_sigreg_context_target_slots_differences"
CONTEXT_VERSION = "independent_rigid_blocks_v1"

JEPA_LOG_METRICS = (
    ("pred", "loss_pred"),
    ("sigreg", "loss_sigreg"),
    ("sigreg_diff", "loss_sigreg_diff"),
    ("cos_prediction", "sim_prediction"),
    ("cos_context_target", "sim_diag"),
    ("cos_offdiag", "sim_offdiag"),
    ("ctx_std", "context_std"),
    ("tgt_std", "target_std"),
)
JEPA_WANDB_SPLIT_METRICS = (
    ("pred", "loss_pred"),
    ("sigreg", "loss_sigreg"),
    ("sigreg_diff", "loss_sigreg_diff"),
)
JEPA_METRIC_NAMES = (
    "loss_pred", "loss_sigreg", "loss_sigreg_diff", "sim_prediction", "sim_diag", "sim_offdiag", "context_std", "target_std",
)


def selected_jepa_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Return the small JEPA metric set exposed in logs."""

    return selected_jepa_metrics_by_spec(metrics, JEPA_LOG_METRICS)


def selected_jepa_metrics_by_spec(
    metrics: dict[str, float],
    metric_specs: tuple[tuple[str, str], ...],
) -> dict[str, float]:
    """Return compact JEPA metric names from a raw metric mapping."""

    return {
        log_name: float(metrics[raw_name])
        for log_name, raw_name in metric_specs
        if raw_name in metrics
    }


def selected_jepa_metrics_for_split(metrics: dict[str, float], split: str) -> dict[str, float]:
    """Return compact JEPA metric names with a train/valid suffix."""

    return {f"{name}_{split}": value for name, value in selected_jepa_metrics(metrics).items()}


def jepa_log_metric_fields(
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
) -> list[tuple[str, float]]:
    """Return train/valid JEPA metric fields in console-log order."""

    fields: list[tuple[str, float]] = []
    for log_name, raw_name in JEPA_LOG_METRICS:
        if raw_name in train_metrics:
            fields.append((f"{log_name}_train", float(train_metrics[raw_name])))
        if raw_name in valid_metrics:
            fields.append((f"{log_name}_valid", float(valid_metrics[raw_name])))
    return fields


def format_learning_rate(learning_rate: float) -> str:
    """Format a learning rate compactly for console logs."""

    mantissa, exponent = f"{learning_rate:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_jepa_epoch_log(
    epoch: int,
    train_loss: float,
    train_metrics: dict[str, float],
    valid_loss: float | None,
    valid_metrics: dict[str, float],
    learning_rate: float | None = None,
) -> str:
    """Format the compact JEPA epoch log line."""

    fields: list[tuple[str, float]] = [("train_loss", train_loss)]
    if valid_loss is not None:
        fields.append(("valid_loss", valid_loss))
    fields.extend(jepa_log_metric_fields(train_metrics, valid_metrics))
    metric_text = " ".join(f"{name}={value:.6f}" for name, value in fields)
    if learning_rate is not None:
        metric_text = f"{metric_text} lr={format_learning_rate(learning_rate)}"
    return f"epoch={epoch} {metric_text}"


def jepa_wandb_metrics(
    train_loss: float,
    train_metrics: dict[str, float],
    valid_loss: float | None,
    valid_metrics: dict[str, float],
    learning_rate: float | None = None,
) -> dict[str, float | None]:
    """Build the compact JEPA W&B payload."""

    payload: dict[str, float | None] = {
        "train/loss": train_loss,
        "valid/loss": valid_loss,
        "trainer/lr": learning_rate,
    }
    for log_name, raw_name in JEPA_WANDB_SPLIT_METRICS:
        if raw_name in train_metrics:
            payload[f"{log_name}_train"] = float(train_metrics[raw_name])
        if raw_name in valid_metrics:
            payload[f"{log_name}_valid"] = float(valid_metrics[raw_name])
    similarity_metrics = valid_metrics if valid_metrics else train_metrics
    for log_name, raw_name in JEPA_LOG_METRICS:
        if log_name.startswith("cos") and raw_name in similarity_metrics:
            payload[log_name] = float(similarity_metrics[raw_name])
    return payload


def reduce_epoch_statistics(
    total_loss: float | torch.Tensor,
    metric_totals: dict[str, float | torch.Tensor],
    samples: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Reduce loss and metrics across ranks, weighted by crystal count."""
    names = sorted(metric_totals)
    values = torch.stack([
        torch.as_tensor(value, device=device, dtype=torch.float64)
        for value in (total_loss, samples, *(metric_totals[name] for name in names))
    ])
    if is_distributed():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    averages = (values / values[1].clamp_min(1.0)).cpu().tolist()
    return averages[0], dict(zip(names, averages[2:]))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    max_epochs: int,
    grad_clip_norm: float = 0.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_interval: str = "epoch",
    max_batches: int | None = None,
    amp_dtype: torch.dtype | None = None,
    log_every_steps: int = 20,
) -> tuple[float, dict[str, float]]:
    """Run one standard PyTorch training epoch."""

    model.train()
    totals = torch.zeros(1 + len(JEPA_METRIC_NAMES), device=device, dtype=torch.float64)
    samples = 0
    progress = tqdm(
        loader,
        desc=f"JEPA epoch {epoch + 1}/{max_epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not is_main_process(),
    )
    for step, batch in enumerate(progress):
        if max_batches is not None and step >= max_batches:
            break
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch)
        loss = out["loss"]
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler is not None and scheduler_interval == "step":
            scheduler.step()

        batch_size = out["pred_emb"].size(0)
        values = torch.stack([loss.detach(), *(out["metrics"][name].detach() for name in JEPA_METRIC_NAMES)])
        totals.add_(values.double(), alpha=batch_size)
        samples += batch_size
        if is_main_process() and (step == 0 or (step + 1) % log_every_steps == 0):
            averages = (totals / samples).cpu().tolist()
            running_metrics = dict(zip(JEPA_METRIC_NAMES, averages[1:]))
            progress.set_postfix(
                train_loss=f"{averages[0]:.4f}",
                lr=format_learning_rate(float(optimizer.param_groups[0]["lr"])),
                **{
                    name: f"{value:.4f}"
                    for name, value in selected_jepa_metrics_for_split(running_metrics, "train").items()
                },
            )

    if scheduler is not None and scheduler_interval == "epoch":
        scheduler.step()

    return reduce_epoch_statistics(totals[0], dict(zip(JEPA_METRIC_NAMES, totals[1:])), samples, device)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader | None, device: torch.device,
    max_batches: int | None = None, amp_dtype: torch.dtype | None = None,
) -> tuple[float | None, dict[str, float]]:
    """Evaluate JEPA loss on a validation loader."""

    if loader is None:
        return None, {}
    model.eval()
    # Validation shards can have different lengths. Avoid DDP forward collectives.
    model = unwrap_model(model)
    totals = torch.zeros(1 + len(JEPA_METRIC_NAMES), device=device, dtype=torch.float64)
    samples = 0
    # Keep validation repeatable without modifying LeWM's SIGReg or training RNG.
    rng_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=rng_devices):
        batches = iter(loader)  # Start persistent workers before seeding SIGReg projections.
        torch.random.default_generator.manual_seed(0)
        if rng_devices:
            with torch.cuda.device(rng_devices[0]):
                torch.cuda.manual_seed(0)
        for step, batch in enumerate(batches):
            if max_batches is not None and step >= max_batches:
                break
            batch = batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(batch)
            batch_size = out["pred_emb"].size(0)
            values = torch.stack([out["loss"].detach(), *(out["metrics"][name].detach() for name in JEPA_METRIC_NAMES)])
            totals.add_(values.double(), alpha=batch_size)
            samples += batch_size
    return reduce_epoch_statistics(totals[0], dict(zip(JEPA_METRIC_NAMES, totals[1:])), samples, device)


def read_training_config(argv: list[str] | None = None) -> dict[str, Any]:
    """Apply CLI overrides and reject invalid runs before opening data or GPUs."""
    parser = argparse.ArgumentParser(description="Train crystal JEPA from graph manifests")
    parser.add_argument("--condition-source", choices=["conformer", "pxrd", "combined"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-dir", dest="graph_manifest_dir")
    parser.add_argument("--output-dir")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", help="Resume model, optimizer, scheduler, and epoch from a checkpoint")
    initialization.add_argument("--init-from", help="Start a new run from existing JEPA weights")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"))
    for name, help_text in (
        ("context-translation-std", "Gaussian translation std in angstroms: VALUE or MIN MAX, per block copy"),
        ("context-rotation-degrees", "Rotation angle in degrees: VALUE or MIN MAX, independent axis per block copy"),
    ):
        parser.add_argument(f"--{name}", type=float, nargs="+", help=help_text)
    for option in (
        "batch-size", "eval-batch-size", "num-workers", "log-every-steps", "max-epochs",
        "max-train-batches", "max-eval-batches", "max-train-samples", "max-valid-samples",
    ):
        parser.add_argument(f"--{option}", type=int)
    args = parser.parse_args(argv)
    for name in ("context_translation_std", "context_rotation_degrees"):
        value = getattr(args, name)
        if value is not None:
            if len(value) not in (1, 2):
                parser.error(f"--{name.replace('_', '-')} expects VALUE or MIN MAX")
            setattr(args, name, value[0] if len(value) == 1 else value)
    config = load_config(args.config)
    config.update({key: value for key, value in vars(args).items() if key != "config" and value is not None})
    # Persist the new default so checkpoints distinguish it from the former 0.01.
    config.setdefault("loss", {}).setdefault("mu_diff", 0.0)
    config["conditioning_contract"] = conditioning_contract(config)
    if args.resume:
        config.pop("init_from", None)
    if args.init_from:
        config.pop("resume", None)
    if config.get("resume") and config.get("init_from"):
        raise ValueError("Choose either resume or init_from.")
    if not config.get("graph_manifest_dir"):
        raise ValueError("Crystal JEPA requires graph_manifest_dir (or --manifest-dir).")
    for name, maximum in (("context_translation_std", float("inf")),
                          ("context_rotation_degrees", 180)):
        context_parameter_range(config.get(name, 0), name, maximum)
    if "context_mask_ratio" in config:
        raise ValueError("Atom masking was removed from JEPA. Remove context_mask_ratio from the config.")
    if config.get("prediction_target", "crystal_tokens") != "crystal_tokens":
        raise ValueError("JEPA now predicts crystal_tokens only; use a new config and --init-from for old weights.")
    if config.get("context_source", "crystal") != "crystal":
        raise ValueError("JEPA requires crystal context; representatives supply the condition only.")
    for name in ("batch_size", "eval_batch_size", "max_epochs", "log_every_steps",
                 "max_train_batches", "max_eval_batches", "max_train_samples", "max_valid_samples"):
        if config.get(name) is not None and int(config[name]) < 1:
            raise ValueError(f"{name} must be >= 1.")
    if int(config.get("num_workers", 4)) < 0:
        raise ValueError("num_workers must be >= 0.")
    config["graph_manifest_dir"] = str(Path(config["graph_manifest_dir"]).resolve())
    return config


def validate_resume(checkpoint: dict[str, Any], config: dict[str, Any], objective: str) -> None:
    """Require matching representations, corruption, and objective for optimizer resume."""
    advice = "Use --init-from and a new output directory."
    weights = checkpoint["model"]
    if any(key.startswith("context_encoder.readout") for key in weights):
        raise ValueError(f"Pooled checkpoints cannot resume token training. {advice}")
    if "context_encoder.local_encoder.lattice_presence_embedding.weight" not in weights:
        raise ValueError(f"Checkpoints without explicit lattice embeddings cannot resume. {advice}")
    if checkpoint.get("objective") != objective:
        raise ValueError(f"Checkpoint uses a different prediction/SIGReg objective. {advice}")
    if checkpoint.get("context_version") != CONTEXT_VERSION:
        raise ValueError(f"Checkpoint uses an older context-construction scheme. {advice}")
    previous = checkpoint.get("config", {})
    old_conditioning = previous.get("conditioning_contract", conditioning_contract(previous))
    if old_conditioning != conditioning_contract(config):
        raise ValueError(f"Cannot resume with a different conditioning/PXRD contract. {advice}")
    for name, old_default, current_default in (
        ("lambda_sig", .1, .1), ("mu_diff", .01, .0), ("sigreg_num_projections", 1024, 1024),
    ):
        if previous.get("loss", {}).get(name, old_default) != config.get("loss", {}).get(name, current_default):
            raise ValueError(f"Cannot resume with a different loss.{name}. {advice}")
    for name in ("context_translation_std", "context_rotation_degrees"):
        if context_parameter_range(previous.get(name, 0), name) != context_parameter_range(config.get(name, 0), name):
            raise ValueError(f"Cannot resume with a different {name}. {advice}")


def main(argv: list[str] | None = None) -> None:
    config = read_training_config(argv)
    output_dir = Path(config.get("output_dir", "outputs/jepa")).resolve()
    config["output_dir"] = str(output_dir)
    device = select_device(str(config.get("device", "auto")))
    rank, local_rank, world_size = setup_distributed("nccl" if device.type == "cuda" else "gloo")
    wandb_run = None
    try:
        if dist.is_initialized() and device.type == "cuda":
            device = torch.device("cuda", local_rank)
        elif not dist.is_initialized():
            gpu_ids = resolve_gpu_ids(config, device)
            if len(gpu_ids) > 1:
                raise ValueError("Launch one process per GPU with scripts/train_crystal_jepa.sh.")
            device = primary_device(device, gpu_ids)
            if device.type == "cuda":
                torch.cuda.set_device(device)

        precision = config.get("precision", "fp32")
        if precision == "auto":
            precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        amp_dtype = {"fp32": None, "bf16": torch.bfloat16}[precision]
        config["precision"] = precision
        seed = int(config.get("seed", 42))
        seed_everything(seed)
        dataset = build_dataset_from_config(config, split="train")
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True,
        ) if is_distributed() else None
        workers = int(config.get("num_workers", 4))
        loader_options = dict(
            num_workers=workers,
            collate_fn=collate_crystal_graphs,
            pin_memory=device.type == "cuda",
        )
        if workers > 0:
            # NCCL and forked workers are incompatible; retain spawned workers across epochs.
            loader_options.update(multiprocessing_context="spawn", persistent_workers=True)
        batch_size = int(config.get("batch_size", 16))
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=sampler is None,
            sampler=sampler, drop_last=True, **loader_options,
        )
        if not len(loader):
            raise ValueError("Training needs at least batch_size × world_size crystals for a full distributed batch.")
        val_loader = None
        validation = None
        if has_dataset_split(config, "valid"):
            val_dataset = build_dataset_from_config(config, split="valid")
            val_batch_size = int(config.get("eval_batch_size", batch_size))
            # Mix families once, identically on all ranks; retain the same batches every epoch.
            val_order = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(seed)).tolist()
            validation = dict(order="fixed_shuffle", seed=seed, samples=len(val_dataset),
                              batch_size=val_batch_size, world_size=world_size,
                              max_batches=config.get("max_eval_batches"))
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                sampler=val_order[rank::world_size],  # Every crystal once, without padding.
                **loader_options,
            )

        model = build_jepa_from_config(config).to(device)
        objective = JEPA_OBJECTIVE
        if config.get("init_from") and not config.get("resume"):
            checkpoint = torch.load(config["init_from"], map_location="cpu", weights_only=False)
            previous = checkpoint.get("config", {})
            changed_peak_version = (
                model.peak_encoder is not None
                and any(key.startswith("peak_encoder.") for key in checkpoint["model"])
                and conditioning_contract(previous).get("encoder_version") != conditioning_contract(config).get("encoder_version")
            )
            load_jepa_weights(model, checkpoint["model"], initialize_lattice=True, initialize_predictor=True,
                              initialize_conditioning=True, initialize_peak_encoder=changed_peak_version)
            del checkpoint
        if dist.is_initialized():
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
        max_epochs = int(config.get("max_epochs", 100))
        optimizer_steps_per_epoch = min(len(loader), config.get("max_train_batches") or len(loader))
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("learning_rate", 1e-4)),
            weight_decay=float(config.get("weight_decay", 1e-2)),
        )
        scheduler, scheduler_interval = build_lr_scheduler(
            opt, config, max_epochs=max_epochs, steps_per_epoch=optimizer_steps_per_epoch,
        )
        grad_clip_norm = float(config.get("gradient_clip_norm", 0.0))
        early_stopper = build_early_stopper(config)
        best, start_epoch, global_step = float("inf"), 0, 0
        if config.get("resume"):
            checkpoint = torch.load(config["resume"], map_location="cpu", weights_only=False)
            validate_resume(checkpoint, config, objective)
            unwrap_model(model).load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["optimizer"])
            same_validation = checkpoint.get("validation") == validation
            if scheduler is not None and checkpoint.get("scheduler") is not None and (same_validation or scheduler_interval != "metric"):
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint.get("global_step", start_epoch * optimizer_steps_per_epoch)
            if same_validation:
                best = checkpoint.get("best", checkpoint["monitor_loss"])
                if early_stopper is not None and checkpoint.get("early_stopping"):
                    for key in ("best", "best_epoch", "wait", "stopped"):
                        setattr(early_stopper, key, checkpoint["early_stopping"][key])
            elif is_main_process():
                print("Validation batches changed: reset best-loss, early-stopping, and plateau-scheduler tracking; "
                      "restored model, optimizer, and current learning rate.", flush=True)
            del checkpoint

        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, output_dir / "training_config.yaml")
            wandb_run = init_wandb(config, job_type="jepa", output_dir=output_dir, default_name=output_dir.name)
            maybe_watch_model(wandb_run, unwrap_model(model), config)
            print(
                f"device={device} world_size={world_size} precision={precision} "
                f"latent_shape=({unwrap_model(model).context_encoder.config.num_latents},{unwrap_model(model).d_jepa}) "
                f"prediction_target={unwrap_model(model).prediction_target} "
                f"batch_per_rank={batch_size} global_batch={batch_size * world_size} "
                f"context_translation_std={config.get('context_translation_std', 0.0)} "
                f"context_rotation_degrees={config.get('context_rotation_degrees', 0.0)} "
                f"train_samples={len(dataset)} valid_samples={len(val_loader.dataset) if val_loader is not None else 0} "
                f"start_epoch={start_epoch} global_step={global_step}", flush=True,
            )
        for epoch in range(start_epoch, max_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            seed_everything(seed + epoch * world_size + rank)
            epoch_loss, epoch_metrics = train_one_epoch(
                model,
                loader,
                opt,
                device,
                epoch=epoch,
                max_epochs=max_epochs,
                grad_clip_norm=grad_clip_norm,
                scheduler=scheduler,
                scheduler_interval=scheduler_interval,
                max_batches=config.get("max_train_batches"),
                amp_dtype=amp_dtype,
                log_every_steps=int(config.get("log_every_steps", 20)),
            )
            global_step += optimizer_steps_per_epoch
            val_loss, val_metrics = evaluate(
                model, val_loader, device, config.get("max_eval_batches"), amp_dtype=amp_dtype,
            )
            monitor_loss = val_loss if val_loss is not None else epoch_loss
            if scheduler is not None and scheduler_interval == "metric":
                # All ranks receive the same sample-weighted validation loss.
                scheduler.step(monitor_loss)
            is_best = monitor_loss < best
            if is_best:
                best = monitor_loss
            stop_now = early_stopper.step(monitor_loss, epoch) if early_stopper is not None else False
            if is_main_process():
                ckpt = {
                    "model": unwrap_model(model).state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                    "world_size": world_size,
                    "best": best,
                    "loss": epoch_loss,
                    "train_loss": epoch_loss,
                    "metrics": epoch_metrics,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "monitor_loss": monitor_loss,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "optimizer": opt.state_dict(),
                    "objective": objective,
                    "context_version": CONTEXT_VERSION,
                    "validation": validation,
                    "latent_shape": [unwrap_model(model).context_encoder.config.num_latents, unwrap_model(model).d_jepa],
                }
                if early_stopper is not None:
                    ckpt["early_stopping"] = early_stopper.state_dict()
                for name in (("last", "best") if is_best else ("last",)):
                    temporary = output_dir / f"{name}.ckpt.tmp"
                    torch.save(ckpt, temporary)
                    temporary.replace(output_dir / f"{name}.ckpt")
                learning_rate = float(opt.param_groups[0]["lr"])
                print(format_jepa_epoch_log(epoch, epoch_loss, epoch_metrics, val_loss, val_metrics, learning_rate), flush=True)
                log_wandb(
                    wandb_run,
                    jepa_wandb_metrics(epoch_loss, epoch_metrics, val_loss, val_metrics, learning_rate),
                    step=global_step,
                )
                if stop_now:
                    print(
                        f"early_stopping epoch={epoch} monitor_loss={monitor_loss:.6f} "
                        f"best={early_stopper.best:.6f} best_epoch={early_stopper.best_epoch} "
                        f"wait={early_stopper.wait}/{early_stopper.patience}", flush=True,
                    )
            if is_distributed():
                dist.barrier()
            if stop_now:
                break
        if is_main_process() and config.get("packing_eval_samples", 0) and val_loader is not None:
            from lvebcsp.eval.crystal_jepa import evaluate_packing
            best_ckpt = torch.load(output_dir / "best.ckpt", map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(best_ckpt["model"])
            report = evaluate_packing(unwrap_model(model), val_dataset, device,
                                      max_samples=config["packing_eval_samples"])
            (output_dir / "packing_valid.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        if is_distributed():
            dist.barrier()
    finally:
        finish_wandb(wandb_run)
        cleanup_distributed()


if __name__ == "__main__":
    main()
