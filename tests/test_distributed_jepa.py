"""Global SIGReg gradients and the distributed crystal training entry point."""

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
from lvebcsp.train.train_jepa import build_dataset_from_config, build_jepa_from_config, evaluate
from test_crystal_graphs import make_row, write_lmdb


def check_global_gradient(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(13)
        reference = nn.Linear(5, 4)
        model = DistributedDataParallel(deepcopy(reference))
        x, target = torch.randn(8, 5), torch.randn(8, 4)
        prediction = model(x.chunk(2)[rank])
        # Different seeds exercise broadcasting the rank-0 projection basis.
        sigreg = SIGReg(num_proj=16, seed=77 + rank)
        statistic = sigreg(prediction)
        loss = nn.functional.mse_loss(prediction, target.chunk(2)[rank]) + 0.1 * statistic
        loss.backward()

        expected_prediction = reference(x)
        expected_statistic = SIGReg(num_proj=16, seed=77).eval()(expected_prediction)
        expected_loss = nn.functional.mse_loss(expected_prediction, target) + 0.1 * expected_statistic
        expected_loss.backward()
        torch.testing.assert_close(statistic, expected_statistic)
        for actual, expected in zip(model.module.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


def test_sigreg_ddp_matches_global_batch_gradient(tmp_path):
    mp.spawn(check_global_gradient, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.mark.parametrize("scheduler_config, scheduler_steps_per_epoch", [
    (dict(type="linear_warmup_cosine_annealing_lr", interval="step",
          warmup_steps=0, eta_min=1e-4), 2),
    # Force a plateau after the first epoch to test a reduction on resume.
    (dict(type="reduce_on_plateau", factor=0.5, patience=0, threshold=1e6,
          threshold_mode="abs", min_lr=1e-6), 1),
], ids=["cosine", "plateau"])
def test_torchrun_crystal_training_and_resume(tmp_path, scheduler_config, scheduler_steps_per_epoch):
    source = tmp_path / "data.lmdb"
    write_lmdb(source, [make_row(f"CRYSTAL{i:02d}.cif") for i in range(9)])
    (tmp_path / "sources.json").write_text(json.dumps([{"path": str(source)}]))
    for split, rows in (("train", np.arange(8)), ("valid", np.array([8]))):
        np.savez(
            tmp_path / f"{split}.npz", source=np.zeros(len(rows), dtype=int), row=rows,
            refcode=[f"CRYSTAL{i:02d}" for i in rows], family=[f"CRYSTAL{i}" for i in rows],
        )
    output = tmp_path / "run"
    config = dict(
        graph_manifest_dir=str(tmp_path), output_dir=str(output), device="cpu", precision="fp32",
        batch_size=2, eval_batch_size=2, num_workers=0, max_epochs=1, learning_rate=1e-3,
        d_jepa=16, random_block_geometry=False,
        crystal_encoder=dict(atom_dim=16, dim=16, heads=4, num_latents=4,
                             local_layers=1, latent_layers=1, cutoff=2.0),
        condition_encoder=dict(hidden_dim=8), predictor=dict(type="mlp", hidden_dim=16),
        loss=dict(sigreg_num_projections=8), early_stopping=dict(patience=3),
        scheduler=scheduler_config,
        wandb=dict(enabled=False),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    root = Path(__file__).resolve().parents[1]
    command = ["bash", str(root / "scripts/train_crystal_jepa.sh"), "--config", str(config_path)]
    env = {**os.environ, "NPROC_PER_NODE": "2", "OMP_NUM_THREADS": "1"}
    first = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stdout + first.stderr
    assert first.stdout.count("epoch=0 train_loss=") == 1
    checkpoint = torch.load(output / "last.ckpt", weights_only=False)
    assert checkpoint["epoch"] == 0 and checkpoint["global_step"] == 2
    assert checkpoint["world_size"] == 2
    assert checkpoint["scheduler"]["last_epoch"] == scheduler_steps_per_epoch
    assert all(int(state["step"]) == 2 for state in checkpoint["optimizer"]["state"].values())
    assert not any(key.startswith("module.") for key in checkpoint["model"])
    assert (output / "best.ckpt").exists()

    # One validation sample leaves rank 1 empty: reduction must still finish and
    # agree with evaluating the unique sample outside the process group.
    model = build_jepa_from_config(config)
    model.load_state_dict(checkpoint["model"])
    val_loader = DataLoader(build_dataset_from_config(config, "valid"), collate_fn=collate_crystal_graphs)
    val_loss, val_metrics = evaluate(model, val_loader, torch.device("cpu"))
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
    assert restored["scheduler"]["last_epoch"] == 2 * scheduler_steps_per_epoch
    assert all(int(state["step"]) == 4 for state in restored["optimizer"]["state"].values())
    assert restored["best"] <= checkpoint["best"]
    if scheduler_config["type"] == "reduce_on_plateau":
        assert restored["scheduler"]["best"] == pytest.approx(checkpoint["val_loss"])
        assert restored["optimizer"]["param_groups"][0]["lr"] == pytest.approx(config["learning_rate"] * 0.5)
