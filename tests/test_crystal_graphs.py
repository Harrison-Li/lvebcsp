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
from lvebcsp.data.crystal_filter import filter_splits
from lvebcsp.data.crystal_splits import build_splits, refcode_family
from lvebcsp.eval.crystal_jepa import evaluate_packing, perturb_packing
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


@pytest.mark.parametrize("translation,rotation", [(.05, 5), ([.05, .75], [5, 90]), (.75, 90)])
def test_rigid_context_corruption_preserves_target_and_periodic_molecules(crystal_dataset, translation, rotation):
    from pymatgen.core import Structure
    from lvebcsp.data.preprocessing_utils import build_crystal_graph

    clean = CrystalDataset(crystal_dataset.path, cutoff=2)[0]
    dataset = CrystalDataset(crystal_dataset.path, cutoff=2,
                             context_translation_std=translation, context_rotation_degrees=rotation)
    rng = torch.get_rng_state()
    item = dataset[0]
    assert torch.equal(rng, torch.get_rng_state())
    target, context = item['target'], item['crystal_context']
    for key in ('pos', 'cell', 'z', 'edge_index', 'edge_shifts', 'atom_map'):
        torch.testing.assert_close(target[key], clean['target'][key])
    torch.testing.assert_close(context.pos, dataset[0]['crystal_context'].pos)
    assert torch.equal(context.z, target.z)
    torch.testing.assert_close(context.cell, target.cell)
    assert not torch.allclose(context.pos, target.pos)
    crystal = Structure(context.cell[0].numpy(), context.z.tolist(), context.pos.numpy(), coords_are_cartesian=True)
    expected = build_crystal_graph(crystal, graph_method='radius', cutoff=2)
    assert set(map(tuple, np.column_stack((context.edge_index.T.numpy(), context.edge_shifts.numpy())))) == set(
        map(tuple, np.column_stack((expected['edge_indices'], expected['to_jimages']))))
    # First molecule crosses the boundary: rotating wrapped positions would break its bond.
    assert crystal.get_distance(0, 1) == pytest.approx(1.2, abs=2e-6)
    assert crystal.get_distance(2, 3) == pytest.approx(1.2, abs=2e-6)
    assert context.num_nodes == target.num_nodes == 4
    assert "masked" not in target and "masked" not in context
    config = dict(d_jepa=16, context_translation_std=translation,
                  context_rotation_degrees=rotation, crystal_encoder=dict(atom_dim=16, dim=16, output_dim=16,
                  num_latents=4, heads=4, local_layers=1, latent_layers=1),
                  condition_encoder=dict(hidden_dim=8), loss=dict(sigreg_num_projections=8))
    model = build_jepa_from_config(config)
    loss = model(collate_crystal_graphs([item, item]))['loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize('kwargs', [dict(context_translation_std=-1), dict(context_translation_std=float('nan')),
                                    dict(context_rotation_degrees=181), dict(context_rotation_degrees=float('inf')),
                                    dict(context_translation_std=[.75, .05]), dict(context_translation_std=[]),
                                    dict(context_rotation_degrees=[0, 181]), dict(context_rotation_degrees=[5]),
                                    dict(context_translation_std=[0, float('nan')])])
def test_invalid_context_corruption(crystal_dataset, kwargs):
    with pytest.raises(ValueError, match='context_'):
        CrystalDataset(crystal_dataset.path, **kwargs)


def test_context_ranges_resample_for_training_and_allow_zero(crystal_dataset, monkeypatch):
    import lvebcsp.data.crystal_dataset as module

    sampled = []
    perturb = module.perturb_context
    def record(graph, cutoff, translation, rotation, generator=None):
        sampled.append((translation, rotation))
        return perturb(graph, cutoff, translation, rotation, generator)
    monkeypatch.setattr(module, 'perturb_context', record)
    dataset = CrystalDataset(crystal_dataset.path, cutoff=2, random_context=True,
                             context_translation_std=[.05, .75], context_rotation_degrees=[5, 90])
    for _ in range(8):
        dataset[0]
    assert all(.05 <= t <= .75 and 5 <= r <= 90 for t, r in sampled)
    assert len(set(sampled)) == 8
    disabled = CrystalDataset(crystal_dataset.path, context_translation_std=[0, 0], context_rotation_degrees=[0, 0])
    clean = disabled[0]
    torch.testing.assert_close(clean['crystal_context'].pos, clean['target'].pos)
    assert clean['crystal_context'].pos.data_ptr() != clean['target'].pos.data_ptr()
    assert len(sampled) == 8


@pytest.mark.parametrize("translation, rotation", [(0.3, 0), (0, 60)])
@pytest.mark.parametrize("cutoff", [0.2, 2.0])
def test_independent_rigid_motions_preserve_wrapped_molecules(crystal_dataset, translation, rotation, cutoff):
    from pymatgen.core import Structure

    dataset = CrystalDataset(crystal_dataset.path, cutoff=cutoff,
                             context_translation_std=translation, context_rotation_degrees=rotation)
    item = dataset[0]
    context, target = item["crystal_context"], item["target"]
    structure = Structure(context.cell[0].numpy(), context.z.tolist(), context.frac_coords.numpy())
    assert context.num_nodes == target.num_nodes == 4
    torch.testing.assert_close(context.block_instance_id, target.block_instance_id)
    torch.testing.assert_close(context.atom_map, target.atom_map)
    torch.testing.assert_close(context.cell, target.cell)
    assert not torch.allclose(context.pos, target.pos)
    for first, second in ((0, 1), (2, 3)):
        assert structure.get_distance(first, second) == pytest.approx(1.2, abs=2e-6)
    if translation:
        displacement = ((context.frac_coords - target.frac_coords + .5) % 1 - .5) @ target.cell[0]
        torch.testing.assert_close(displacement[0], displacement[1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(displacement[2], displacement[3], atol=1e-6, rtol=1e-5)
        assert not torch.allclose(displacement[0], displacement[2])
    else:
        bonds = ((context.frac_coords[[1, 3]] - context.frac_coords[[0, 2]] + .5) % 1 - .5) @ target.cell[0]
        # Identical input molecules now have independent orientations, even with no radius edges.
        assert not torch.allclose(bonds[0], bonds[1])
        assert bonds[:, 1:].abs().sum() > 0


def test_context_sampling_is_stable_by_record_and_does_not_mutate_target(crystal_dataset):
    dataset = CrystalDataset(crystal_dataset.path, cutoff=2,
                             context_translation_std=[.05, .75], context_rotation_degrees=[5, 90])
    state = torch.get_rng_state()
    first = dataset[0]
    assert torch.equal(torch.get_rng_state(), state)
    clean = crystal_dataset[0]["target"]
    for key, value in clean.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(first["target"][key], value)
    # Filtering or reordering a manifest must not change the evaluation transform.
    for name in ("source", "row", "refcode", "family"):
        setattr(dataset, name, np.repeat(getattr(dataset, name), 3))
    torch.rand(5)
    for index in range(3):
        torch.testing.assert_close(dataset[index]["crystal_context"].pos, first["crystal_context"].pos)
    dataset.random_context = True
    assert len({tuple(dataset[0]["crystal_context"].pos.flatten().tolist()) for _ in range(5)}) == 5


def test_zero_perturbation_and_single_block_translation(crystal_dataset):
    from lvebcsp.data.crystal_dataset import perturb_context

    graph = crystal_dataset[0]["target"].subgraph(torch.tensor([0, 1]))
    graph.num_atoms = torch.tensor([2])
    identity = perturb_context(graph, 2, 0, 0)
    torch.testing.assert_close(identity.pos, graph.pos)
    identity.pos.add_(1)
    assert not torch.allclose(identity.pos, graph.pos)
    # Independent translation should still move a single molecule; do not subtract its own shift.
    changed = perturb_context(graph, 2, .3, 0, torch.Generator().manual_seed(7))
    assert not torch.allclose(changed.pos, graph.pos)
    assert changed.num_nodes == graph.num_nodes


def test_packing_cli_restores_context_perturbations(crystal_dataset, tmp_path, monkeypatch):
    import sys
    from lvebcsp.eval.crystal_jepa import main

    config = dict(graph_manifest_dir=str(tmp_path), d_jepa=8,
                  crystal_encoder=dict(atom_dim=8, dim=8, heads=2, num_latents=2,
                                       local_layers=1, latent_layers=1, cutoff=2),
                  condition_encoder=dict(hidden_dim=4), loss=dict(sigreg_num_projections=8),
                  context_translation_std=[.1, .4], context_rotation_degrees=[15, 60])
    (tmp_path / "test.npz").write_bytes(crystal_dataset.path.read_bytes())
    checkpoint, output = tmp_path / "model.ckpt", tmp_path / "packing.json"
    torch.save(dict(model=build_jepa_from_config(config).state_dict(), config=config), checkpoint)
    monkeypatch.setattr(sys, "argv", ["packing", "--checkpoint", str(checkpoint), "--output", str(output),
                                     "--max-samples", "1", "--max-families", "0"])
    main()
    report = json.loads(output.read_text())
    assert report["context_translation_std"] == [.1, .4]
    assert report["context_rotation_degrees"] == [15, 60]
    assert report["native_energy"] > 0 and report["samples"] == 1


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
    config = dict(graph_manifest_dir=str(out), d_jepa=16, context_translation_std=[.05, .75], context_rotation_degrees=[5, 90],
                  crystal_encoder=dict(atom_dim=16, dim=16, num_latents=4, heads=4, local_layers=1, latent_layers=1),
                  condition_encoder=dict(hidden_dim=8), loss=dict(sigreg_num_projections=8))
    dataset = build_dataset_from_config(config)
    assert isinstance(dataset, CrystalDataset)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_crystal_graphs, num_workers=1)
    model = build_jepa_from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"), epoch=0, max_epochs=1, max_batches=1)
    assert np.isfinite(loss) and np.isfinite(metrics["loss_pred"])
    val_loader = DataLoader(build_dataset_from_config(config, "valid"), batch_size=2,
                           collate_fn=collate_crystal_graphs, num_workers=1, persistent_workers=True)
    rng_before = torch.get_rng_state()
    result1 = evaluate(model, val_loader, torch.device("cpu"), max_batches=1)
    result2 = evaluate(model, val_loader, torch.device("cpu"), max_batches=1)
    assert result1 == result2
    assert torch.equal(torch.get_rng_state(), rng_before)
    report = evaluate_packing(model, val_loader.dataset, torch.device("cpu"), max_samples=2, max_families=0)
    assert report["context_translation_std"] == [.05, .75]
    assert report["context_rotation_degrees"] == [5, 90]
    assert report["samples"] == 2
    assert report["native_energy"] is not None


def test_packing_perturbation_keeps_internal_periodic_geometry(crystal_dataset):
    item = crystal_dataset[0]
    changed = perturb_packing(item["target"], 6.0, np.random.default_rng(4))
    cell = changed.cell[0]
    for a, b in [(0, 1), (2, 3)]:
        fractional = (changed.pos[a] - changed.pos[b]) @ torch.linalg.inv(cell)
        distance = ((fractional - fractional.round()) @ cell).norm()
        torch.testing.assert_close(distance, torch.tensor(1.2), atol=1e-6, rtol=1e-5)
    assert not torch.allclose(changed.pos, item["target"].pos)


def test_filter_excludes_incomplete_records_and_preserves_permutations(tmp_path):
    rows = [make_row(f"RECORD{i:02d}.cif") for i in range(5)]
    rows[1]["repeat_dict_2"][1] = 1  # Only two of the four atoms are represented.
    rows[2]["crystal_group_id2"][:] = 1  # Same total composition, wrong copies.
    rows[3]["crystal_atom_type"][0] = "N"  # Same total atom count, wrong species.
    permutation = [0, 2, 1, 3]
    for key in ("crystal_atom_type", "crystal_atom_pos", "crystal_group_id2"):
        rows[4][key] = np.asarray(rows[4][key])[permutation]
    source = tmp_path / "data.lmdb"
    write_lmdb(source, rows)
    original_bytes = source.read_bytes()
    manifests, output = tmp_path / "original", tmp_path / "filtered"
    manifests.mkdir()
    (manifests / "sources.json").write_text(json.dumps([{"path": str(source)}]))
    for split, indices in (("train", [0, 1, 2, 3, 4]), ("valid", [1, 4]), ("test", [0, 3])):
        np.savez(manifests / f"{split}.npz", source=np.zeros(len(indices), dtype=int), row=indices,
                 refcode=[f"RECORD{i:02d}" for i in indices], family=[f"FAMILY{i}" for i in indices])
    report = filter_splits(manifests, output, workers=2)
    assert report["splits"] == {
        "train": {"input": 5, "kept": 2, "excluded": 3},
        "valid": {"input": 2, "kept": 1, "excluded": 1},
        "test": {"input": 2, "kept": 1, "excluded": 1},
    }
    with np.load(output / "train.npz") as data:
        assert data["row"].tolist() == [0, 4]
        assert data["refcode"].tolist() == ["RECORD00", "RECORD04"]
    with np.load(manifests / "train.npz") as data:
        assert data["row"].tolist() == [0, 1, 2, 3, 4]
    assert source.read_bytes() == original_bytes
    rejected = [json.loads(line) for line in (output / "excluded.jsonl").read_text().splitlines()]
    assert [row["row"] for row in rejected if row["split"] == "train"] == [1, 2, 3]
    dataset = CrystalDataset(output / "train.npz", cutoff=2.0)
    from lvebcsp.data.crystal_latent import collate_crystal_latents
    batch = collate_crystal_latents([dataset[0], dataset[1]])
    torch.testing.assert_close(batch["atom_types"][0], batch["atom_types"][1])
    for graph, item in zip(batch["target"].to_data_list(), [dataset[0], dataset[1]]):
        torch.testing.assert_close(graph.pos, item["target"].pos)
