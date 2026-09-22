"""Per-GPU SIGReg gradients and the distributed crystal training entry point."""

from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import yaml

from lvebcsp.data.crystal_dataset import collate_crystal_graphs
from lvebcsp.models.sigreg import SIGReg
from lvebcsp.train.train_jepa import (
    CONTEXT_VERSION, JEPA_OBJECTIVE, build_dataset_from_config, build_jepa_from_config,
    evaluate, main, read_training_config, validate_resume,
)
from test_crystal_graphs import make_row, write_lmdb


def check_local_gradient(rank, rendezvous, num_slots):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(13)
        reference = nn.Linear(5, 4)
        model = DistributedDataParallel(deepcopy(reference))
        batch_shape = (8,) if num_slots is None else (8, num_slots)
        x, target = torch.randn(*batch_shape, 5), torch.randn(*batch_shape, 4)
        prediction = model(x.chunk(2)[rank])
        # The upstream class uses independent random projections on each rank.
        sigreg = SIGReg(num_proj=16)
        torch.manual_seed(77 + rank)
        statistic = sigreg(prediction if num_slots is None else prediction.transpose(0, 1))
        loss = nn.functional.mse_loss(prediction, target.chunk(2)[rank]) + 0.1 * statistic
        loss.backward()

        expected_prediction = reference(x)
        expected_statistics, expected_losses = [], []
        for source_rank, (local_prediction, local_target) in enumerate(zip(expected_prediction.chunk(2), target.chunk(2))):
            torch.manual_seed(77 + source_rank)
            expected_samples = local_prediction if num_slots is None else local_prediction.transpose(0, 1)
            local_statistic = SIGReg(num_proj=16)(expected_samples)
            expected_statistics.append(local_statistic)
            expected_losses.append(nn.functional.mse_loss(local_prediction, local_target) + 0.1 * local_statistic)
        expected_loss = torch.stack(expected_losses).mean()
        expected_loss.backward()
        torch.testing.assert_close(statistic, expected_statistics[rank])
        for actual, expected in zip(model.module.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("num_slots", [None, 3], ids=["single_slot", "multiple_slots"])
def test_sigreg_ddp_matches_mean_local_gradients(tmp_path, num_slots):
    mp.spawn(check_local_gradient, args=(str(tmp_path / "rendezvous"), num_slots), nprocs=2, join=True)


@pytest.mark.parametrize("scheduler_config, scheduler_steps_per_epoch, valid_count, mu_diff, condition_source", [
    (dict(type="linear_warmup_cosine_annealing_lr", interval="step",
          warmup_steps=0, eta_min=1e-4), 2, 1, None, "conformer"),
    # Force a plateau after the first epoch to test a reduction on resume.
    (dict(type="reduce_on_plateau", factor=0.5, patience=0, threshold=1e6,
          threshold_mode="abs", min_lr=1e-6), 1, 5, .01, "conformer"),
    (dict(type="linear_warmup_cosine_annealing_lr", interval="step",
          warmup_steps=0, eta_min=1e-4), 2, 1, None, "pxrd"),
    (dict(type="linear_warmup_cosine_annealing_lr", interval="step",
          warmup_steps=0, eta_min=1e-4), 2, 1, None, "combined"),
], ids=["cosine", "plateau", "pxrd", "combined"])
def test_torchrun_crystal_training_and_resume(tmp_path, scheduler_config, scheduler_steps_per_epoch, valid_count, mu_diff, condition_source):
    source = tmp_path / "data.lmdb"
    rows = [make_row(f"CRYSTAL{i:02d}.cif") for i in range(8 + valid_count)]
    for i, row in enumerate(rows):
        row["lattice_matrix"] *= 1 + .02 * i  # Distinct embeddings make validation grouping observable.
    write_lmdb(source, rows)
    (tmp_path / "sources.json").write_text(json.dumps([{"path": str(source)}]))
    for split, rows in (("train", np.arange(8)), ("valid", np.arange(8, 8 + valid_count))):
        np.savez(
            tmp_path / f"{split}.npz", source=np.zeros(len(rows), dtype=int), row=rows,
            refcode=[f"CRYSTAL{i:02d}" for i in rows], family=[f"CRYSTAL{i}" for i in rows],
        )
    output = tmp_path / "run"
    config = dict(
        graph_manifest_dir=str(tmp_path), output_dir=str(output), device="cpu", precision="fp32",
        batch_size=2, eval_batch_size=2, num_workers=0, max_epochs=1, learning_rate=1e-3,
        d_jepa=16, random_block_geometry=False, context_translation_std=[.05, .75], context_rotation_degrees=[5, 90],
        crystal_encoder=dict(atom_dim=16, dim=16, heads=4, num_latents=4,
                             local_layers=1, latent_layers=1, cutoff=2.0),
        condition_encoder=dict(hidden_dim=8), predictor=dict(type="mlp", hidden_dim=16),
        loss=dict(sigreg_num_projections=8), early_stopping=dict(patience=3),
        scheduler=scheduler_config,
        condition_source=condition_source, peak_encoder=dict(hidden_dim=16),
        pxrd=dict(p_max=32, two_theta_range=[5., 45.], cache_dir=str(tmp_path / "pxrd")),
        wandb=dict(enabled=False),
    )
    if mu_diff is not None:
        config["loss"]["mu_diff"] = mu_diff
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    root = Path(__file__).resolve().parents[1]
    command = ["bash", str(root / "scripts/train_crystal_jepa.sh"), "--config", str(config_path),
               "--context-translation-std", "0.05", "0.75", "--context-rotation-degrees", "5", "90"]
    env = {**os.environ, "NPROC_PER_NODE": "2", "OMP_NUM_THREADS": "1"}
    first = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stdout + first.stderr
    assert first.stdout.count("epoch=0 train_loss=") == 1
    checkpoint = torch.load(output / "last.ckpt", weights_only=False)
    assert checkpoint["latent_shape"] == [4, 16]
    assert checkpoint["objective"] == JEPA_OBJECTIVE
    assert checkpoint["context_version"] == CONTEXT_VERSION
    assert checkpoint["config"]["loss"]["mu_diff"] == (mu_diff or 0.)
    if mu_diff:
        assert checkpoint["val_metrics"]["loss_sigreg_diff"] > 0
    else:
        assert checkpoint["val_metrics"]["loss_sigreg_diff"] == 0
    assert "sigreg_diff_train=" in first.stdout and "sigreg_diff_valid=" in first.stdout
    assert checkpoint["validation"] == dict(order="fixed_shuffle", seed=42, samples=valid_count,
                                            batch_size=2, world_size=2, max_batches=None)
    assert not any("readout" in key for key in checkpoint["model"])
    assert checkpoint["epoch"] == 0 and checkpoint["global_step"] == 2
    assert checkpoint["world_size"] == 2
    assert checkpoint["scheduler"]["last_epoch"] == scheduler_steps_per_epoch
    assert all(int(state["step"]) == 2 for state in checkpoint["optimizer"]["state"].values())
    assert not any(key.startswith("module.") for key in checkpoint["model"])
    assert (output / "best.ckpt").exists()

    # Cover both an empty rank and uneven full/partial batches, without duplicate samples.
    model = build_jepa_from_config(config)
    model.load_state_dict(checkpoint["model"])
    val_dataset = build_dataset_from_config(config, "valid")
    order = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(42)).tolist()
    val_loss, val_metrics = 0., {}
    for rank in range(2):
        shard = order[rank::2]
        if not shard:
            continue
        val_loader = DataLoader(val_dataset, batch_size=2, sampler=shard, collate_fn=collate_crystal_graphs)
        loss, metrics = evaluate(model, val_loader, torch.device("cpu"))
        val_loss += loss * len(shard) / len(order)
        for key, value in metrics.items():
            val_metrics[key] = val_metrics.get(key, 0.) + value * len(shard) / len(order)
    assert checkpoint["val_loss"] == pytest.approx(val_loss, rel=1e-5)
    assert checkpoint["val_metrics"] == pytest.approx(val_metrics, rel=1e-5, abs=1e-6)

    resumed = subprocess.run(
        [*command, "--resume", str(output / "last.ckpt"), "--max-epochs", "2"],
        env=env, cwd=root, capture_output=True, text=True, timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "start_epoch=1 global_step=2" in resumed.stdout
    assert resumed.stdout.count("epoch=1 train_loss=") == 1
    assert "epoch=0 train_loss=" not in resumed.stdout
    restored = torch.load(output / "last.ckpt", weights_only=False)
    assert restored["epoch"] == 1 and restored["global_step"] == 4
    assert restored["config"]["loss"]["mu_diff"] == (mu_diff or 0.)
    assert restored["scheduler"]["last_epoch"] == 2 * scheduler_steps_per_epoch
    assert all(int(state["step"]) == 4 for state in restored["optimizer"]["state"].values())
    assert restored["best"] <= checkpoint["best"]
    if scheduler_config["type"] == "reduce_on_plateau":
        assert restored["scheduler"]["best"] == pytest.approx(checkpoint["val_loss"])
        assert restored["optimizer"]["param_groups"][0]["lr"] == pytest.approx(config["learning_rate"] * 0.5)
        # Old ordered-validation loss histories must not trigger scheduling or stopping.
        legacy = deepcopy(restored)
        del legacy["validation"]
        legacy["best"] = legacy["scheduler"]["best"] = legacy["early_stopping"]["best"] = -1e6
        legacy["early_stopping"]["wait"] = config["early_stopping"]["patience"] - 1
        torch.save(legacy, tmp_path / "ordered.ckpt")
        migrated_dir = tmp_path / "shuffled"
        migrated = subprocess.run(
            [*command, "--resume", str(tmp_path / "ordered.ckpt"), "--max-epochs", "3", "--output-dir", str(migrated_dir)],
            env=env, cwd=root, capture_output=True, text=True, timeout=90,
        )
        assert migrated.returncode == 0, migrated.stdout + migrated.stderr
        assert "Validation batches changed" in migrated.stdout
        current = torch.load(migrated_dir / "last.ckpt", weights_only=False)
        assert current["epoch"] == 2 and current["global_step"] == 6
        assert all(int(state["step"]) == 6 for state in current["optimizer"]["state"].values())
        assert current["best"] == pytest.approx(current["val_loss"])
        assert current["scheduler"]["best"] == pytest.approx(current["val_loss"])
        assert current["optimizer"]["param_groups"][0]["lr"] == pytest.approx(restored["optimizer"]["param_groups"][0]["lr"])
        assert current["early_stopping"]["wait"] == 0 and not current["early_stopping"]["stopped"]
        assert (migrated_dir / "best.ckpt").exists()


@pytest.mark.parametrize("objective, old_mu", [
    (None, None), ("mse_sigreg_context_target_tokens", None),
    ("mse_sigreg_context_target_per_slot", None), (JEPA_OBJECTIVE, None), (JEPA_OBJECTIVE, .02),
    ("mse_sigreg_context_target_local_atoms_slot_differences", .01),
    ("masked_atom_set_mse_sigreg_context_target_local_atoms_slot_differences", .01),
    ("masked_atom_set_mse_sigreg_context_target_slots_differences", .01),
])
def test_incompatible_pretraining_requires_weight_initialization(tmp_path, objective, old_mu):
    source = tmp_path / "data.lmdb"
    write_lmdb(source, [make_row(f"CRYSTAL{i:02d}.cif") for i in range(2)])
    (tmp_path / "sources.json").write_text(json.dumps([{"path": str(source)}]))
    for split in ("train", "valid"):
        np.savez(
            tmp_path / f"{split}.npz", source=np.zeros(2, dtype=int), row=np.arange(2),
            refcode=["CRYSTAL00", "CRYSTAL01"], family=["CRYSTAL0", "CRYSTAL1"],
        )
    output = tmp_path / "run"
    config = dict(
        graph_manifest_dir=str(tmp_path), output_dir=str(output), device="cpu",
        batch_size=2, eval_batch_size=2, num_workers=0, max_epochs=1,
        d_jepa=4, random_block_geometry=False, context_translation_std=.25, context_rotation_degrees=30,
        crystal_encoder=dict(atom_dim=8, dim=8, heads=2, num_latents=2,
                             local_layers=1, latent_layers=1, cutoff=2.0),
        condition_encoder=dict(hidden_dim=4), predictor=dict(type="mlp", hidden_dim=8),
        loss=dict(sigreg_num_projections=8, mu_diff=.01), wandb=dict(enabled=False),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    checkpoint = {"model": build_jepa_from_config(config).state_dict(), "context_version": CONTEXT_VERSION}
    if objective is not None:
        checkpoint["objective"] = objective
    if old_mu is not None:
        checkpoint["config"] = deepcopy(config)
        checkpoint["config"]["loss"]["mu_diff"] = old_mu
    checkpoint_path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="Use --init-from"):
        main(["--config", str(config_path), "--resume", str(checkpoint_path)])
    assert not output.exists()

    main(["--config", str(config_path), "--init-from", str(checkpoint_path), "--context-translation-std", "0.25"])
    initialized = torch.load(output / "last.ckpt", weights_only=False)
    assert initialized["objective"] == JEPA_OBJECTIVE
    assert initialized["config"]["context_translation_std"] == .25
    assert initialized["epoch"] == 0 and initialized["global_step"] == 1
    assert initialized["optimizer"]["state"]


@pytest.mark.parametrize("previous_mu, current_mu, compatible", [
    (None, None, False), (None, .01, True), (None, 0., False),
    (.01, None, False), (0., None, True), (0., 0., True), (0., .01, False),
])
def test_resume_distinguishes_historical_and_current_difference_defaults(previous_mu, current_mu, compatible):
    previous_loss = {} if previous_mu is None else {"mu_diff": previous_mu}
    current_loss = {} if current_mu is None else {"mu_diff": current_mu}
    checkpoint = dict(
        model={"context_encoder.local_encoder.lattice_presence_embedding.weight": torch.ones(2, 8)},
        config={"loss": previous_loss}, objective=JEPA_OBJECTIVE, context_version=CONTEXT_VERSION,
    )
    if compatible:
        validate_resume(checkpoint, {"loss": current_loss}, JEPA_OBJECTIVE)
    else:
        with pytest.raises(ValueError, match="different loss.mu_diff.*Use --init-from"):
            validate_resume(checkpoint, {"loss": current_loss}, JEPA_OBJECTIVE)


def test_context_cli_validation_and_resume_guard(tmp_path):
    config = dict(graph_manifest_dir=str(tmp_path), context_translation_std=.25,
                  context_rotation_degrees=30)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    parsed = read_training_config(["--config", str(config_path), "--context-translation-std", ".1", ".7",
                                   "--context-rotation-degrees", "45"])
    assert parsed["context_translation_std"] == [.1, .7]
    assert parsed["context_rotation_degrees"] == 45
    assert parsed["loss"]["mu_diff"] == 0.
    with pytest.raises(SystemExit):
        read_training_config(["--config", str(config_path), "--context-rotation-degrees", "1", "2", "3"])
    with pytest.raises(ValueError, match="context_rotation_degrees"):
        read_training_config(["--config", str(config_path), "--context-rotation-degrees", "181"])
    checkpoint = dict(model={"context_encoder.local_encoder.lattice_presence_embedding.weight": torch.ones(2, 8)},
                      config={**config, "loss": {"mu_diff": 0.}},
                      objective=JEPA_OBJECTIVE, context_version=CONTEXT_VERSION)
    # A fixed scalar and a degenerate range define the same corruption.
    validate_resume(checkpoint, {**config, "context_translation_std": [.25, .25]}, JEPA_OBJECTIVE)
    for name in ("context_translation_std", "context_rotation_degrees"):
        with pytest.raises(ValueError, match=name):
            validate_resume(checkpoint, {**config, name: .5}, JEPA_OBJECTIVE)
    with pytest.raises(ValueError, match="context-construction"):
        validate_resume({**checkpoint, "context_version": None}, config, JEPA_OBJECTIVE)
    config_path.write_text(yaml.safe_dump({**config, "context_mask_ratio": .5}))
    with pytest.raises(ValueError, match="masking was removed"):
        read_training_config(["--config", str(config_path)])
    with pytest.raises(ValueError, match="crystal_tokens only"):
        build_jepa_from_config({"prediction_target": "masked_atoms"})
