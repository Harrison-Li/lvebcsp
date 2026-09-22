"""Single-device or torchrun training for crystal latent flow and reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from lvebcsp.common.config import load_config, save_config, select_device
from lvebcsp.common.distributed import cleanup_distributed, is_main_process, setup_distributed
from lvebcsp.common.gpu import unwrap_model
from lvebcsp.common.seed import seed_everything
from lvebcsp.data.crystal_latent import collate_crystal_latents
from lvebcsp.models.eaf import CrystalLatentDiffusion
from lvebcsp.train.train_jepa import (
    batch_to_device, build_dataset_from_config, build_early_stopper,
    build_jepa_from_config, build_lr_scheduler, has_dataset_split, load_jepa_weights,
)


METRICS = ("loss", "flow", "jepa", "target_std")


def build_crystal_latent(config, jepa_state=None):
    """Reuse the frozen JEPA encoder, projector and multiplicity conditioner."""
    if config.get("condition_source", "conformer") == "pxrd":
        raise ValueError("Crystal latent diffusion requires a pretrained conformer conditioner; "
                         "use a conformer or combined JEPA checkpoint.")
    jepa = build_jepa_from_config(config)
    if jepa_state is not None:
        load_jepa_weights(jepa, jepa_state)
    return CrystalLatentDiffusion(
        encoder=jepa.context_encoder, projector=jepa.projector,
        condition_encoder=jepa.condition_encoder, token_dim=jepa.d_jepa,
        loss_weights=config.get("loss"), **config.get("latent_diffusion", {}),
    )


def run_epoch(model, loader, device, *, optimizer=None, amp_dtype=None,
              scheduler=None, scheduler_interval="metric", max_batches=None,
              grad_clip=1.0, log_every=20):
    training = optimizer is not None
    model.train(training)
    if not training:
        # Validation shards can have different lengths, including zero batches.
        model = unwrap_model(model)
    totals = torch.zeros(len(METRICS) + 1, dtype=torch.float64, device=device)
    steps = 0
    progress = tqdm(loader, desc="Train" if training else "Valid", disable=not is_main_process())
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(progress):
            if max_batches is not None and step >= max_batches:
                break
            batch = batch_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(batch)
            if training:
                out["loss"].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None and scheduler_interval == "step":
                    scheduler.step()
            metrics = {"loss": out["loss"], **out["metrics"]}
            n = batch["atom_types"].size(0)
            totals[:-1] += torch.stack([metrics[key].detach() for key in METRICS]).double() * n
            totals[-1] += n
            steps += 1
            if is_main_process() and (step == 0 or (step + 1) % log_every == 0):
                progress.set_postfix(dict(zip(METRICS, (totals[:-1] / totals[-1]).cpu().tolist())))
    if dist.is_initialized():
        dist.all_reduce(totals)
    return dict(zip(METRICS, (totals[:-1] / totals[-1].clamp_min(1)).cpu().tolist())), steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/crystal_latent.yaml")
    parser.add_argument("--manifest-dir", dest="graph_manifest_dir")
    parser.add_argument("--output-dir")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-from", help="Initialize encoder/conditioner from a JEPA checkpoint")
    initialization.add_argument("--resume", help="Resume a crystal latent checkpoint and optimizer")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"))
    for option in ("batch-size", "eval-batch-size", "num-workers", "log-every-steps", "max-epochs",
                   "max-train-batches", "max-eval-batches", "max-train-samples", "max-valid-samples"):
        parser.add_argument(f"--{option}", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.update({key: value for key, value in vars(args).items() if key != "config" and value is not None})
    if not config.get("init_from") and not config.get("resume"):
        raise ValueError("Frozen JEPA flow training requires --init-from JEPA.ckpt or --resume EAF.ckpt.")
    config["graph_manifest_dir"] = str(Path(config["graph_manifest_dir"]).resolve())
    output = Path(config.get("output_dir", "outputs/crystal_latent/train_local")).resolve()
    config["output_dir"] = str(output)
    device = select_device(config.get("device", "auto"))
    rank, local_rank, world_size = setup_distributed("nccl" if device.type == "cuda" else "gloo")
    try:
        if device.type == "cuda":
            device = torch.device("cuda", local_rank if dist.is_initialized() else (device.index or 0))
            torch.cuda.set_device(device)
        precision = config.get("precision", "fp32")
        if precision == "auto":
            precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        config["precision"] = precision
        amp_dtype = {"fp32": None, "bf16": torch.bfloat16}[precision]
        seed = config.get("seed", 42)
        seed_everything(seed)
        jepa_state = None
        if config.get("init_from") and not config.get("resume"):
            jepa_checkpoint = torch.load(config["init_from"], map_location="cpu", weights_only=False)
            # Restore the representation architecture before constructing graphs:
            # the saved encoder cutoff must also be the dataset cutoff.
            for key in ("crystal_encoder", "d_jepa", "condition_encoder", "projector", "predictor", "pred_proj", "prediction_target",
                        "condition_source", "peak_encoder"):
                if key in jepa_checkpoint["config"]:
                    config[key] = jepa_checkpoint["config"][key]
            jepa_state = jepa_checkpoint["model"]
        # Downstream flow conditioning uses representatives only, even when the
        # frozen encoder came from a combined-input JEPA checkpoint.
        data_config = {**config, "condition_source": "conformer"}
        dataset = build_dataset_from_config(data_config)
        sampler = DistributedSampler(dataset, world_size, rank, seed=seed, drop_last=True)
        workers = config.get("num_workers", 8)
        loader_options = dict(collate_fn=collate_crystal_latents, num_workers=workers,
                              pin_memory=device.type == "cuda")
        if workers:
            loader_options.update(multiprocessing_context="spawn", persistent_workers=True)
        batch_size = config.get("batch_size", 80)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, **loader_options)
        if not len(loader):
            raise ValueError("Training needs at least batch_size * world_size crystals.")
        valid_loader = None
        if has_dataset_split(config, "valid"):
            valid = build_dataset_from_config(data_config, "valid")
            valid_loader = DataLoader(valid, batch_size=config.get("eval_batch_size", batch_size),
                                      sampler=range(rank, len(valid), world_size), **loader_options)
        model = build_crystal_latent(config, jepa_state).to(device)
        del jepa_state
        if config.get("init_from") and not config.get("resume"):
            del jepa_checkpoint
        if dist.is_initialized():
            model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                            broadcast_buffers=False)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=config.get("learning_rate", 1e-4),
                                      weight_decay=config.get("weight_decay", 0.01))
        epochs = config.get("max_epochs", 100)
        scheduler, interval = build_lr_scheduler(optimizer, config, epochs,
                                                  min(len(loader), config.get("max_train_batches") or len(loader)))
        early_stopper = build_early_stopper(config)
        best, start_epoch, global_step = float("inf"), 0, 0
        if config.get("resume"):
            checkpoint = torch.load(config["resume"], map_location="cpu", weights_only=False)
            unwrap_model(model).load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint["scheduler"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch, global_step, best = checkpoint["epoch"] + 1, checkpoint["global_step"], checkpoint["best"]
            if early_stopper and checkpoint.get("early_stopping"):
                for key in ("best", "best_epoch", "wait", "stopped"):
                    setattr(early_stopper, key, checkpoint["early_stopping"][key])
            del checkpoint
        if is_main_process():
            output.mkdir(parents=True, exist_ok=True)
            save_config(config, output / "training_config.yaml")
            print(f"device={device} world_size={world_size} precision={precision} "
                  f"latent_shape=({unwrap_model(model).num_latents},{unwrap_model(model).token_dim}) "
                  f"atoms_per_token={unwrap_model(model).decoder.max_atoms} "
                  f"batch_per_rank={batch_size} start_epoch={start_epoch} global_step={global_step}", flush=True)
        for epoch in range(start_epoch, epochs):
            sampler.set_epoch(epoch)
            seed_everything(seed + epoch * world_size + rank)
            train, steps = run_epoch(model, loader, device, optimizer=optimizer, amp_dtype=amp_dtype,
                                     scheduler=scheduler, scheduler_interval=interval,
                                     max_batches=config.get("max_train_batches"),
                                     grad_clip=config.get("gradient_clip_norm", 1.0),
                                     log_every=config.get("log_every_steps", 20))
            global_step += steps
            valid = None
            if valid_loader is not None:
                # Repeat validation noise without consuming training RNG.
                with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                    torch.manual_seed(0)
                    valid, _ = run_epoch(model, valid_loader, device, amp_dtype=amp_dtype,
                                         max_batches=config.get("max_eval_batches"),
                                         log_every=config.get("log_every_steps", 20))
            monitor = (valid or train)["loss"]
            if scheduler is not None and interval != "step":
                scheduler.step(monitor) if interval == "metric" else scheduler.step()
            improved = monitor < best
            best = min(best, monitor)
            stop = early_stopper.step(monitor, epoch) if early_stopper else False
            if is_main_process():
                checkpoint = {
                    "model": unwrap_model(model).state_dict(), "config": config,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch, "global_step": global_step, "best": best, "world_size": world_size,
                    "train_metrics": train, "valid_metrics": valid,
                    "early_stopping": early_stopper.state_dict() if early_stopper else None,
                    "latent_shape": [unwrap_model(model).num_latents, unwrap_model(model).token_dim],
                    "objective": "latent_flow_jepa_reconstruction",
                }
                for name in (("last", "best") if improved else ("last",)):
                    temporary = output / f"{name}.ckpt.tmp"
                    torch.save(checkpoint, temporary)
                    temporary.replace(output / f"{name}.ckpt")
                print(f"epoch={epoch} train={train} valid={valid} lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
            if stop:
                break
        if is_main_process() and device.type == "cuda":
            print(f"peak_cuda_allocated_mib={torch.cuda.max_memory_allocated(device) / 2**20:.0f} "
                  f"peak_cuda_reserved_mib={torch.cuda.max_memory_reserved(device) / 2**20:.0f}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
