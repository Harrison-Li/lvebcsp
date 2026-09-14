"""Copyright (c) Meta Platforms, Inc. and affiliates.

Tensor construction in graph_to_data is copied from CrystalDataset.__getitem__
in all-atom-diffusion-transformer, commit b9ce505f170597a7c8ca50d13ce8e15df21cf8c9.
Changes: z/edge fields for CrystalEncoder; unused diffusion fields omitted.
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
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from lvebcsp.data.preprocessing_utils import process_one


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


class CrystalDataset(Dataset):
    """Periodic crystal targets with representative building-block contexts.

    Like upstream, preprocessing produces graph_arrays and __getitem__ creates
    PyG Data. The existing LMDB is the cache; records are loaded lazily instead
    of retaining more than a million structures and their edges in memory.
    """

    def __init__(self, path, cutoff=6.0, random_block_geometry=False, max_samples=None):
        super().__init__()
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

    def __len__(self):
        return len(self.row)

    def __getitem__(self, index):
        env = _open_lmdb(self.sources[int(self.source[index])])
        with env.begin() as txn:
            payload = txn.get(f"{self.row[index]:08d}".encode())
        data_dict = process_one(pickle.loads(gzip.decompress(payload)), self.cutoff)
        target = graph_to_data(data_dict["graph_arrays"])
        target.block_instance_id = torch.LongTensor(data_dict["graph_arrays"]["block_instance_id"])
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
                               edge_index=edge_index, periodic=torch.tensor([False])))
        return {
            "context": blocks, "target": target,
            "multiplicity": torch.LongTensor(data_dict["multiplicity"]),
            "material_id": data_dict["material_id"],
            "refcode": str(self.refcode[index]), "family": str(self.family[index]),
        }

    def __repr__(self):
        return f"CrystalDataset(path={self.path}, size={len(self)})"


def collate_crystal_graphs(items):
    """Batch representative graphs separately from the B complete crystals."""
    return {
        "context": Batch.from_data_list([graph for item in items for graph in item["context"]]),
        "target": Batch.from_data_list([item["target"] for item in items]),
        "multiplicity": torch.cat([item["multiplicity"] for item in items]),
        "block_batch": torch.repeat_interleave(torch.arange(len(items)), torch.tensor([len(item["context"]) for item in items])),
        **{key: [item[key] for item in items] for key in ("material_id", "refcode", "family")},
    }
