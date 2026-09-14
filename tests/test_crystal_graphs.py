"""Crystal graph collation, split isolation, and training integration."""

import gzip
import json
import pickle

import lmdb
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from lvebcsp.data.crystal_dataset import CrystalDataset, collate_crystal_graphs
from lvebcsp.data.crystal_splits import build_splits, refcode_family
from lvebcsp.eval.crystal_jepa import perturb_packing
from lvebcsp.train.train_jepa import build_jepa_from_config, build_dataset_from_config, train_one_epoch, evaluate


def make_row(name):
    return dict(
        cif_filename=name,
        mol_atom_type_2=["C", "O"],
        mol_atom_pos_2=np.array([[[0, 0, 0], [1.2, 0, 0]], [[0, 0, 0], [0, 1.2, 0]]]),
        mol2_group_slices=[(0, 2)], mol_group_id_2=[1], repeat_dict_2={1: 2},
        crystal_atom_type=["C", "O", "C", "O"],
        crystal_atom_pos=np.array([[0.9, 0, 0], [0.1, 0, 0], [0.4, 0.5, 0.5], [0.6, 0.5, 0.5]]),
        lattice_matrix=np.eye(3) * 6,
        crystal_group_id2=np.array([1, 1, 2, 2]),
    )


def write_lmdb(path, rows):
    with lmdb.open(str(path), subdir=False, map_size=10_000_000) as env:
        with env.begin(write=True) as txn:
            for i, row in enumerate(rows):
                txn.put(f"{i:08d}".encode(), gzip.compress(pickle.dumps(row), mtime=0))


@pytest.fixture
def crystal_dataset(tmp_path):
    source = tmp_path / "data.lmdb"
    write_lmdb(source, [make_row("ALPHA01_std_2_a.cif")])
    (tmp_path / "sources.json").write_text(json.dumps([{"path": str(source)}]))
    np.savez(tmp_path / "train.npz", source=[0], row=[0], refcode=["ALPHA01"], family=["ALPHA"])
    return CrystalDataset(tmp_path / "train.npz", cutoff=2.0)


def test_graph_geometry_multiplicities_and_collation(crystal_dataset):
    item = crystal_dataset[0]
    assert item["multiplicity"].tolist() == [2]
    assert item["context"][0].num_nodes == 2
    assert not item["context"][0].periodic.item()
    torch.testing.assert_close(item["context"][0].pos.mean(dim=0), torch.zeros(3))
    target = item["target"]
    src, dst = target.edge_index
    displacement = target.pos[src] + target.edge_shifts.float() @ target.cell[0] - target.pos[dst]
    assert torch.any(target.edge_shifts != 0)
    torch.testing.assert_close(displacement.norm(dim=-1), torch.full((4,), 1.2), atol=1e-6, rtol=1e-5)
    batch = collate_crystal_graphs([item, item])
    assert isinstance(batch["context"], Batch)
    assert batch["block_batch"].tolist() == [0, 1]
    assert batch["multiplicity"].tolist() == [2, 2]
    assert batch["target"].cell.shape == (2, 3, 3)
    torch.testing.assert_close(batch["target"].edge_index[:, 4:], target.edge_index + 4)
    assert torch.equal(batch["target"].batch[batch["target"].edge_index[0]], batch["target"].batch[batch["target"].edge_index[1]])


def test_clean_splits_and_graph_training(tmp_path):
    paths = [tmp_path / "train.lmdb", tmp_path / "test.lmdb"]
    rows = [make_row(f"{family}{number:02d}_std_2_a.cif")
            for family in ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA", "IOTA", "KAPPA"]
            for number in [1, 2]]
    write_lmdb(paths[0], rows)
    write_lmdb(paths[1], [rows[0]])
    out = tmp_path / "splits"
    report = build_splits(paths, out, workers=1, valid_fraction=0.2, test_fraction=0.2)
    assert report["exact_duplicates_removed"] == 1
    families = []
    for split in ["train", "valid", "test"]:
        with np.load(out / f"{split}.npz") as manifest:
            families.append(set(manifest["family"]))
    assert not families[0] & families[1] and not families[0] & families[2] and not families[1] & families[2]
    assert refcode_family("ALPHA01") == refcode_family("ALPHA02")
    config = dict(graph_manifest_dir=str(out), d_jepa=16,
                  crystal_encoder=dict(atom_dim=16, dim=16, num_latents=4, heads=4, local_layers=1, latent_layers=1),
                  condition_encoder=dict(hidden_dim=8), loss=dict(sigreg_num_projections=8))
    dataset = build_dataset_from_config(config)
    assert isinstance(dataset, CrystalDataset)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_crystal_graphs, num_workers=1)
    model = build_jepa_from_config(config)
    assert model.stop_gradient is False
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"), epoch=0, max_epochs=1, max_batches=1)
    assert np.isfinite(loss) and np.isfinite(metrics["loss_pred"])
    val_loader = DataLoader(build_dataset_from_config(config, "valid"), batch_size=2, collate_fn=collate_crystal_graphs)
    result1 = evaluate(model, val_loader, torch.device("cpu"), max_batches=1)
    result2 = evaluate(model, val_loader, torch.device("cpu"), max_batches=1)
    assert result1 == result2


def test_packing_perturbation_keeps_internal_periodic_geometry(crystal_dataset):
    item = crystal_dataset[0]
    changed = perturb_packing(item["target"], 6.0, np.random.default_rng(4))
    cell = changed.cell[0]
    for a, b in [(0, 1), (2, 3)]:
        fractional = (changed.pos[a] - changed.pos[b]) @ torch.linalg.inv(cell)
        distance = ((fractional - fractional.round()) @ cell).norm()
        torch.testing.assert_close(distance, torch.tensor(1.2), atol=1e-6, rtol=1e-5)
    assert not torch.allclose(changed.pos, item["target"].pos)
