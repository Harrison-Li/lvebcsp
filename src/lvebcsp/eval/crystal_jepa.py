"""Evaluate packing sensitivity and within-family retrieval on held-out crystals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ase import Atoms
from ase.data import covalent_radii
import numpy as np
import networkx as nx
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import torch

from lvebcsp.data.crystal_dataset import CrystalDataset, graph_to_data, collate_crystal_graphs
from lvebcsp.data.preprocessing_utils import build_crystal_graph
from lvebcsp.data.crystal_pxrd import PEAK_FIELDS


def atoms_to_graph(atoms, cutoff):
    crystal = Structure(np.asarray(atoms.cell), atoms.numbers, atoms.positions, coords_are_cartesian=True)
    return graph_to_data(build_crystal_graph(crystal, graph_method="radius", cutoff=cutoff))


def perturb_packing(graph, cutoff, rng, translation_std=0.5):
    """Translate building-block instances independently, preserving internal geometry.

    These are synthetic packing perturbations, not relaxed or verified polymorphs.
    block_instance_id labels each atom's block copy within the source cell.
    It is used only here, never as an encoder feature.
    """
    atoms = Atoms(numbers=graph.z.cpu().numpy(), positions=graph.pos.cpu().numpy(),
                  cell=graph.cell[0].cpu().numpy(), pbc=True)
    _, groups = np.unique(graph.block_instance_id.cpu().numpy(), return_inverse=True)
    shifts = rng.normal(scale=translation_std, size=(groups.max() + 1, 3))
    shifts -= shifts.mean(axis=0)
    atoms.positions += shifts[groups]
    atoms.wrap()
    return atoms_to_graph(atoms, cutoff)


@torch.no_grad()
def evaluate_packing(model, dataset, device, *, max_samples=32, max_families=16, seed=42):
    """Report sensitivity separately from candidate-ranking quality; no pass threshold."""
    model.eval()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
    invariance, sensitivity, native_wins, native_energy, perturbed_energy = [], [], [], [], []
    for index in chosen:
        item = dataset[int(index)]
        batch = collate_crystal_graphs([item])
        query = model.encode_ctx(batch["context"].to(device), batch["multiplicity"].to(device), batch["block_batch"].to(device),
                                 crystal_context=batch["crystal_context"].to(device),
                                 **{key: batch[key].to(device) for key in PEAK_FIELDS if key in batch})
        graph = item["target"]
        target = model.encode_tgt(graph.to(device))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotation[:, 0] *= np.linalg.det(rotation)
        rotated = Atoms(numbers=graph.z.cpu().numpy(),
                        positions=graph.pos.cpu().numpy() @ rotation + rng.normal(size=3),
                        cell=graph.cell[0].cpu().numpy() @ rotation, pbc=True)
        rotated_z = model.encode_tgt(atoms_to_graph(rotated, dataset.cutoff).to(device))
        invariance.append(float((target - rotated_z).square().mean()))
        if graph.block_instance_id.unique().numel() < 2:
            continue
        changed_z = model.encode_tgt(perturb_packing(graph, dataset.cutoff, rng).to(device))
        sensitivity.append(float((target - changed_z).square().mean()))
        original_score = float((query - target).square().mean())
        changed_score = float((query - changed_z).square().mean())
        native_energy.append(original_score)
        perturbed_energy.append(changed_score)
        native_wins.append(float(original_score < changed_score) + 0.5 * float(original_score == changed_score))

    # Compare different refcodes within a source family and the same block
    # composition/multiplicities. Equivalent crystal structures are filtered out.
    family_rows = defaultdict(dict)
    for index, (family, refcode) in enumerate(zip(dataset.family, dataset.refcode)):
        family_rows[str(family)].setdefault(str(refcode), index)
    families = [family for family, rows in family_rows.items() if len(rows) > 1]
    rng.shuffle(families)
    matcher = StructureMatcher(primitive_cell=True, scale=False, attempt_supercell=False)
    reciprocal_ranks, top1, separations, groups_report = [], [], [], []
    for family in families[:max_families]:
        comparable = defaultdict(list)
        for index in list(family_rows[family].values())[:16]:
            item = dataset[index]
            block_signatures = []
            for g, m in zip(item["context"], item["multiplicity"]):
                numbers = g.z.numpy()
                distances = np.linalg.norm(g.pos.numpy()[:, None] - g.pos.numpy()[None, :], axis=-1)
                radii = covalent_radii[numbers]
                bonds = np.triu((distances > 0) & (distances < 1.2 * (radii[:, None] + radii[None, :])), k=1)
                block_graph = nx.Graph()
                block_graph.add_nodes_from((i, {"element": str(z)}) for i, z in enumerate(numbers))
                block_graph.add_edges_from(zip(*np.where(bonds)))
                block_signatures.append((nx.weisfeiler_lehman_graph_hash(block_graph, node_attr="element"), int(m)))
            signature = tuple(sorted(block_signatures))
            comparable[signature].append(item)
        for items in comparable.values():
            distinct, structures = [], []
            for item in items:
                g = item["target"]
                structure = Structure(g.cell[0].numpy(), g.z.tolist(), g.pos.numpy(), coords_are_cartesian=True)
                if any(matcher.fit(structure, previous) for previous in structures):
                    continue
                distinct.append(item)
                structures.append(structure)
            if len(distinct) < 2:
                continue
            batch = collate_crystal_graphs(distinct)
            target = model.encode_tgt(batch["target"].to(device))
            target = target.flatten(1)
            query = model.encode_ctx(batch["context"].to(device), batch["multiplicity"].to(device), batch["block_batch"].to(device),
                                     crystal_context=batch["crystal_context"].to(device) if "crystal_context" in batch else None,
                                     **{key: batch[key].to(device) for key in PEAK_FIELDS if key in batch}).flatten(1)
            energies = (query[:, None] - target[None, :]).square().mean(dim=-1)
            own = energies.diag()[:, None]
            # Mid-rank handles ties without depending on row order.
            ranks = 1 + (energies < own).sum(dim=-1) + 0.5 * ((energies == own).sum(dim=-1) - 1)
            reciprocal_ranks.extend((1.0 / ranks).cpu().tolist())
            best = energies.min(dim=-1, keepdim=True).values
            wins = ((own == best).float() / (energies == best).sum(dim=-1, keepdim=True)).flatten()
            top1.extend(wins.cpu().tolist())
            distances = (target[:, None] - target[None, :]).square().mean(dim=-1)
            separations.extend(distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)].cpu().tolist())
            groups_report.append({"family": family, "refcodes": [item["refcode"] for item in distinct],
                                  "candidate_count": len(distinct)})

    return {
        "prediction_target": model.prediction_target,
        "context_translation_std": dataset.context_translation_std,
        "context_rotation_degrees": dataset.context_rotation_degrees,
        "samples": len(invariance), "packing_perturbations": len(sensitivity),
        "rigid_transform_mse": float(np.mean(invariance)) if invariance else None,
        "packing_change_mse": float(np.mean(sensitivity)) if sensitivity else None,
        "native_vs_perturbed_win_rate": float(np.mean(native_wins)) if native_wins else None,
        "native_energy": float(np.mean(native_energy)) if native_energy else None,
        "perturbed_energy": float(np.mean(perturbed_energy)) if perturbed_energy else None,
        "within_family_queries": len(top1),
        "within_family_top1": float(np.mean(top1)) if top1 else None,
        "within_family_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        "within_family_target_mse": float(np.mean(separations)) if separations else None,
        "groups": groups_report,
        "notes": ["Crystal contexts measure recovery from perturbed reference packing with a known lattice; they do not measure molecule-only CSP.",
                  "Synthetic perturbations translate building-block instances; they are not verified polymorphs.",
                  "Within-family candidates share inferred heavy-atom connectivity hashes and multiplicities and differ under StructureMatcher. Bond orders/hydrogens and experimental polymorph labels are not verified.",
                  "Native-vs-perturbed and exact-refcode retrieval are diagnostics; a shared context can have multiple compatible targets."],
    }


def main():
    from lvebcsp.train.jepa_setup import build_jepa_from_config, load_jepa_weights, graph_pxrd_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-families", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_jepa_from_config(config).to(device)
    load_jepa_weights(model, checkpoint["model"])
    manifest = Path(args.manifest_dir or config["graph_manifest_dir"]) / f"{args.split}.npz"
    dataset = CrystalDataset(manifest, cutoff=model.context_encoder.config.cutoff,
                             context_translation_std=config.get("context_translation_std", 0.0),
                             context_rotation_degrees=config.get("context_rotation_degrees", 0.0),
                             pxrd_config=graph_pxrd_config(config) if model.peak_encoder is not None else None)
    report = evaluate_packing(model, dataset, device, max_samples=args.max_samples, max_families=args.max_families)
    report.update(checkpoint=str(Path(args.checkpoint).resolve()), split=str(manifest.resolve()), epoch=checkpoint.get("epoch"),
                  latent_shape=[model.context_encoder.config.num_latents, model.d_jepa],
                  source_had_pooled_readout=any(k.startswith("context_encoder.readout") for k in checkpoint["model"]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
