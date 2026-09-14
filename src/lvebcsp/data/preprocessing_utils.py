"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Adapted from:
- CDVAE: https://github.com/txie-93/cdvae
- DiffCSP: https://github.com/jiaor17/DiffCSP

build_crystal_graph is copied from all-atom-diffusion-transformer at
b9ce505f170597a7c8ca50d13ce8e15df21cf8c9, with a radius-neighbor branch and
without the canonical-cell assertion. process_one adapts our LMDB row schema.
See third_party/all-atom-diffusion-transformer for the source and license.
"""

import numpy as np
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Molecule, Structure


CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


def build_crystal_graph(crystal, graph_method="crystalnn", cutoff=6.0):
    """Upstream crystal graph extraction, with radius edges for UniversalEncoder."""

    if graph_method == "crystalnn":
        try:
            crystal_graph = StructureGraph.from_local_env_strategy(crystal, CrystalNN)
        except:
            crystalNN_tmp = local_env.CrystalNN(
                distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=10
            )
            crystal_graph = StructureGraph.from_local_env_strategy(crystal, crystalNN_tmp)
    elif graph_method in ("none", "radius"):
        pass
    else:
        raise NotImplementedError

    cell = crystal.lattice.matrix
    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers

    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]
    # Preserve source cells, including rotated cells; no canonical-basis assertion.

    edge_indices, to_jimages = [], []
    if graph_method == "radius":
        receivers, sources, to_jimages, _ = crystal.get_neighbor_list(cutoff)
        edge_indices = np.stack([sources, receivers], axis=-1)
        to_jimages = to_jimages.astype(np.int64)
    elif graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return {
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "cell": cell,
        "lattices": lattice_parameters,
        "lengths": lengths,
        "angles": angles,
        "edge_indices": edge_indices,
        "to_jimages": to_jimages,
        "num_atoms": num_atoms,
    }


def process_one(row, cutoff=6.0):
    """Adapt the current CCDC LMDB schema to crystal and building-block arrays.

    No cell reduction or standardization: atom groups and copy counts refer to
    the source cell. Context coordinates contain no target poses or lattice.
    """
    crystal = Structure(row["lattice_matrix"], row["crystal_atom_type"],
                        np.asarray(row["crystal_atom_pos"]) % 1.0)
    graph_arrays = build_crystal_graph(crystal, graph_method="radius", cutoff=cutoff)
    graph_arrays["block_instance_id"] = np.asarray(row["crystal_group_id2"])
    blocks = []
    for start, end in row["mol2_group_slices"]:
        geometries = np.asarray(row["mol_atom_pos_2"][:, start:end], dtype=np.float32)
        atom_types = np.array(Molecule(row["mol_atom_type_2"][start:end], geometries[0]).atomic_numbers)
        blocks.append({"atom_types": atom_types, "geometries": geometries})
    return {
        "material_id": str(row["cif_filename"]),
        "graph_arrays": graph_arrays,
        "blocks": blocks,
        "multiplicity": np.array([row["repeat_dict_2"][group] for group in row["mol_group_id_2"]]),
    }
