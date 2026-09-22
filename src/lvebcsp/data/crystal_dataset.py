"""Copyright (c) Meta Platforms, Inc. and affiliates.

Tensor construction in graph_to_data is copied from CrystalDataset.__getitem__
in all-atom-diffusion-transformer, commit b9ce505f170597a7c8ca50d13ce8e15df21cf8c9.
Changes: z/edge fields for UniversalEncoder; unused diffusion fields omitted.
LMDB loading and representative building-block batches are project adapters.
Upstream credits CDVAE. See third_party/all-atom-diffusion-transformer.
"""

import gzip
import json
import os
from pathlib import Path
import pickle

import lmdb
import numpy as np
import torch
from ase.data import covalent_radii
from pymatgen.core import Structure
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from lvebcsp.data.preprocessing_utils import process_one, build_crystal_graph
from lvebcsp.data.crystal_pxrd import PEAK_FIELDS, pxrd_settings, target_pxrd


# One handle per source per worker: all three manifests can reference one LMDB.
_ENVIRONMENTS = {}
_ENV_PID = os.getpid()


def _open_lmdb(path):
    global _ENV_PID
    if _ENV_PID != os.getpid():
        for env in _ENVIRONMENTS.values():
            env.close()
        _ENVIRONMENTS.clear()
        _ENV_PID = os.getpid()
    if path not in _ENVIRONMENTS:
        _ENVIRONMENTS[path] = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)
    return _ENVIRONMENTS[path]


def graph_to_data(graph_arrays):
    """Shared tensor-construction block from upstream CrystalDataset.__getitem__."""
    atom_types = torch.LongTensor(graph_arrays["atom_types"])
    frac_coords = torch.Tensor(graph_arrays["frac_coords"])
    cell = torch.Tensor(graph_arrays["cell"].copy()).unsqueeze(0)
    num_atoms = torch.LongTensor([graph_arrays["num_atoms"]])
    edge_index = torch.LongTensor(graph_arrays["edge_indices"].T).contiguous()
    to_jimages = torch.LongTensor(graph_arrays["to_jimages"])

    # Cartesian coordinates (NOTE do not zero-center prior to graph construction)
    pos = torch.einsum(
        "bi,bij->bj", frac_coords, torch.repeat_interleave(cell, num_atoms, dim=0)
    )

    return Data(
        z=atom_types,
        pos=pos,
        frac_coords=frac_coords,
        cell=cell,
        num_atoms=num_atoms,
        num_nodes=graph_arrays["num_atoms"],  # special attribute used for PyG batching
        edge_index=edge_index,
        edge_shifts=to_jimages,
        periodic=torch.tensor([True]),
    )


def context_parameter_range(value, name, maximum=float("inf")):
    """Validate a fixed corruption strength or a uniformly sampled [min, max]."""
    bounds = value if isinstance(value, (list, tuple)) else (value, value)
    if (len(bounds) != 2 or not np.isfinite(bounds).all()
            or not 0 <= bounds[0] <= bounds[1] <= maximum):
        raise ValueError(f"{name} must be a finite nonnegative scalar or [min, max] range"
                         f" with min <= max <= {maximum}.")
    return tuple(float(bound) for bound in bounds)


def _sample_range(bounds, generator):
    low, high = bounds
    return low if low == high else low + (high - low) * torch.rand((), generator=generator).item()


def _molecular_bonds(crystal, groups):
    """Find covalent image offsets independently of the encoder's radius cutoff."""
    radii = covalent_radii[np.asarray(crystal.atomic_numbers)]
    receivers, sources, images, distances = crystal.get_neighbor_list(2.4 * radii.max())
    bonds = ((groups[sources] == groups[receivers]) & (distances > .2)
             & (distances < 1.2 * (radii[sources] + radii[receivers])))
    adjacency = [[] for _ in crystal]
    for receiver, source, shift in zip(receivers[bonds], sources[bonds], images[bonds]):
        adjacency[receiver].append((int(source), shift.astype(int)))
    return adjacency


def _unwrap_block(members, adjacency, pos, cell):
    """Return one finite molecule, or None for disconnected/periodic networks."""
    root = int(members[0])
    offsets = {root: np.zeros(3, dtype=int)}
    stack = [root]
    while stack:
        atom = stack.pop()
        for neighbor, image in adjacency[atom]:
            proposed = offsets[atom] + image
            if neighbor in offsets:
                if not np.array_equal(offsets[neighbor], proposed):
                    return None
            else:
                offsets[neighbor] = proposed
                stack.append(neighbor)
    if len(offsets) != len(members):
        return None
    return pos[members] + np.array([offsets[int(i)] for i in members]) @ cell


def perturb_context(graph, cutoff, translation_std, rotation_degrees, generator=None):
    """Create Cc from Ct using an independent rigid transform per block copy.

    Rotate each finite molecule about its unwrapped centroid, translate it in
    Cartesian angstroms, wrap into the unchanged cell, then rebuild all radius
    edges. The target graph is never modified. A disconnected or periodically
    bonded block has no unique finite molecular pose and is translated only.
    """
    if translation_std == 0 and rotation_degrees == 0:
        return graph.clone()
    pos = graph.pos.numpy().astype(float)
    cell = graph.cell[0].numpy().astype(float)
    groups = graph.block_instance_id.numpy()
    crystal = Structure(cell, graph.z.tolist(), pos, coords_are_cartesian=True)
    adjacency = _molecular_bonds(crystal, groups) if rotation_degrees else None
    unique = np.unique(groups)
    shifts = torch.randn((len(unique), 3), generator=generator).numpy() * translation_std
    axes = torch.randn((len(unique), 3), generator=generator).numpy()
    changed = pos.copy()
    for group, shift, axis in zip(unique, shifts, axes):
        members = np.flatnonzero(groups == group)
        unwrapped = _unwrap_block(members, adjacency, pos, cell) if rotation_degrees else None
        if unwrapped is not None:
            center = unwrapped.mean(axis=0)
            rotation = Rotation.from_rotvec(axis / max(np.linalg.norm(axis), 1e-12) * np.deg2rad(rotation_degrees))
            changed[members] = rotation.apply(unwrapped - center) + center
        changed[members] += shift
    crystal = Structure(cell, graph.z.tolist(), changed, coords_are_cartesian=True, to_unit_cell=True)
    result = graph_to_data(build_crystal_graph(crystal, graph_method="radius", cutoff=cutoff))
    result.block_instance_id = graph.block_instance_id.clone()
    result.atom_map = graph.atom_map.clone()
    return result


class CrystalDataset(Dataset):
    """Clean targets, complete perturbed contexts, and representative conditions.

    Like upstream, preprocessing produces graph_arrays and __getitem__ creates
    PyG Data. The existing LMDB is the cache; records are loaded lazily instead
    of retaining more than a million structures and their edges in memory.
    """

    def __init__(self, path, cutoff=6.0, random_block_geometry=False, max_samples=None,
                 random_context=False,
                 context_translation_std=0.0, context_rotation_degrees=0.0, pxrd_config=None):
        super().__init__()
        translation_range = context_parameter_range(context_translation_std, "context_translation_std")
        rotation_range = context_parameter_range(context_rotation_degrees, "context_rotation_degrees", 180)
        self.path = Path(path).resolve()
        self.sources = [source["path"] for source in json.loads(self.path.with_name("sources.json").read_text())]
        with np.load(self.path) as manifest:
            self.source, self.row = manifest["source"], manifest["row"]
            self.refcode, self.family = manifest["refcode"], manifest["family"]
        if max_samples is not None and max_samples < len(self.row):
            selected = np.linspace(0, len(self.row) - 1, max_samples, dtype=int)
            self.source, self.row = self.source[selected], self.row[selected]
            self.refcode, self.family = self.refcode[selected], self.family[selected]
        self.cutoff = cutoff
        self.random_block_geometry = random_block_geometry
        self.random_context = random_context
        self.context_translation_std = context_translation_std
        self.context_rotation_degrees = context_rotation_degrees
        self.translation_range, self.rotation_range = translation_range, rotation_range
        self.pxrd_config = None if pxrd_config is None else dict(pxrd_config)
        if self.pxrd_config is not None:
            pxrd_settings(self.pxrd_config)

    def __len__(self):
        return len(self.row)

    def __getitem__(self, index):
        env = _open_lmdb(self.sources[int(self.source[index])])
        with env.begin() as txn:
            payload = txn.get(f"{self.row[index]:08d}".encode())
        data_dict = process_one(pickle.loads(gzip.decompress(payload)), self.cutoff)
        target = graph_to_data(data_dict["graph_arrays"])
        target.block_instance_id = torch.LongTensor(data_dict["graph_arrays"]["block_instance_id"])
        target.atom_map = torch.LongTensor(data_dict["graph_arrays"]["atom_map"])
        blocks = []
        for block in data_dict["blocks"]:
            geometries = block["geometries"]
            geometry_index = torch.randint(len(geometries), ()).item() if self.random_block_geometry else 0
            pos = torch.Tensor(geometries[geometry_index])
            pos = pos - pos.mean(dim=0)
            distances = torch.cdist(pos, pos)
            neighbors = (distances > 0) & (distances < self.cutoff)
            neighbors.fill_diagonal_(False)
            edge_index = neighbors.nonzero().T.contiguous()
            blocks.append(Data(z=torch.LongTensor(block["atom_types"]), pos=pos,
                               atom_map=torch.LongTensor(block["atom_map"]),
                               edge_index=edge_index, periodic=torch.tensor([False])))
        item = {
            "context": blocks, "target": target,
            "multiplicity": torch.LongTensor(data_dict["multiplicity"]),
            "material_id": data_dict["material_id"],
            "refcode": str(self.refcode[index]), "family": str(self.family[index]),
        }
        generator = None if self.random_context else torch.Generator().manual_seed(
            (int(self.source[index]) << 32) + int(self.row[index]))
        translation = _sample_range(self.translation_range, generator)
        rotation = _sample_range(self.rotation_range, generator)
        context = (perturb_context(target, self.cutoff, translation, rotation, generator)
                   if translation or rotation else target.clone())
        item["crystal_context"] = context
        if self.pxrd_config is not None:
            item.update(target_pxrd(target, self.pxrd_config))
        return item

    def __repr__(self):
        return f"CrystalDataset(path={self.path}, size={len(self)})"


def collate_crystal_graphs(items):
    """Keep representative conditions, context crystals, and targets in separate batches."""
    context = Batch.from_data_list([graph for item in items for graph in item["context"]])
    target = Batch.from_data_list([item["target"] for item in items])
    # Compute padding sizes in the CPU worker, avoiding GPU scalar reads later.
    context.max_num_nodes = int((context.ptr[1:] - context.ptr[:-1]).max())
    target.max_num_nodes = int((target.ptr[1:] - target.ptr[:-1]).max())
    batch = {
        "context": context,
        "target": target,
        "multiplicity": torch.cat([item["multiplicity"] for item in items]),
        "block_batch": torch.repeat_interleave(torch.arange(len(items)), torch.tensor([len(item["context"]) for item in items])),
        **{key: [item[key] for item in items] for key in ("material_id", "refcode", "family")},
    }
    if "crystal_context" in items[0]:
        crystals = Batch.from_data_list([item["crystal_context"] for item in items])
        crystals.max_num_nodes = int((crystals.ptr[1:] - crystals.ptr[:-1]).max())
        batch["crystal_context"] = crystals
    if any(any(key in item for key in PEAK_FIELDS) for item in items):
        if not all(all(key in item for key in PEAK_FIELDS) for item in items):
            raise ValueError("Every item in a PXRD batch must provide peak_d, peak_i, and peak_mask")
        for key in PEAK_FIELDS:
            batch[key] = torch.stack([item[key] for item in items])
    return batch
