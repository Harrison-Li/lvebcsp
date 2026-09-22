"""Sample crystal tokens, decode once, and save tensors and CIF candidates."""

import argparse
from pathlib import Path

import torch
from pymatgen.core import Structure

from lvebcsp.common.config import select_device
from lvebcsp.common.seed import seed_everything
from lvebcsp.data.crystal_dataset import CrystalDataset
from lvebcsp.data.crystal_latent import collate_crystal_latents
from lvebcsp.train.train_crystal_latent import build_crystal_latent
from lvebcsp.train.train_jepa import batch_to_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--condition", help="torch.save item with context graphs and multiplicity; no target")
    source.add_argument("--manifest-dir", help="Read only the representatives/counts from a dataset item")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/crystal_latent/samples")
    args = parser.parse_args(argv)
    device = select_device(args.device)
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_crystal_latent(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if args.condition:
        item = torch.load(args.condition, map_location="cpu", weights_only=False)
    else:
        item = CrystalDataset(Path(args.manifest_dir) / f"{args.split}.npz",
                              cutoff=model.encoder.config.cutoff)[args.index]
    condition = {"context": item["context"], "multiplicity": item["multiplicity"]}
    batch = batch_to_device(collate_crystal_latents([condition] * args.num_samples), device)
    result = model.sample(batch, steps=args.steps)
    result = {key: value.cpu() for key, value in result.items()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result, output / "samples.pt")
    saved = 0
    for i in range(args.num_samples):
        mask = result["atom_mask"][i]
        if not mask.any():
            print(f"sample_{i:03d}: no predicted atoms; raw outputs are in samples.pt")
            continue
        crystal = Structure(result["cell"][i].numpy(), result["atom_types"][i, mask].tolist(),
                            result["frac_coords"][i, mask].numpy())
        crystal.to(filename=str(output / f"sample_{i:03d}.cif"))
        saved += 1
    print(f"Saved {saved}/{args.num_samples} candidate CIFs and all raw outputs to {output.resolve()}")


if __name__ == "__main__":
    main()
