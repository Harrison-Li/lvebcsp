"""Crystal graph batches and target-free, fixed-composition sampling conditions."""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch


def collate_crystal_latents(items):
    """Expand the supplied species/counts; targets need no atom correspondence."""
    context = Batch.from_data_list([graph for item in items for graph in item["context"]])
    context.max_num_nodes = int((context.ptr[1:] - context.ptr[:-1]).max())
    species = [torch.cat([graph.z.repeat(int(count))
                         for graph, count in zip(item["context"], item["multiplicity"])])
               for item in items]
    atom_types = pad_sequence(species, batch_first=True)
    batch = {
        "context": context,
        "multiplicity": torch.cat([item["multiplicity"] for item in items]),
        "block_batch": torch.repeat_interleave(
            torch.arange(len(items)), torch.tensor([len(item["context"]) for item in items])),
        "atom_types": atom_types,
        "atom_mask": atom_types != 0,
    }
    if "target" in items[0]:
        for item, z in zip(items, species):
            if not torch.equal(torch.bincount(z, minlength=119),
                               torch.bincount(item["target"].z, minlength=119)):
                raise ValueError(f"{item.get('material_id', 'crystal')}: supplied composition "
                                 "does not match the target crystal.")
        target = Batch.from_data_list([item["target"] for item in items])
        target.max_num_nodes = int((target.ptr[1:] - target.ptr[:-1]).max())
        batch["target"] = target
    return batch
