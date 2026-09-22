"""Crystal-token reconstruction with the copied dlmcsp matching protocol.

The matcher, strict RMSD < 0.5 rule, case metrics, and CSV writer are reused
unchanged from mp20_realpxrd_solver.py. That file is copied from the local
dlmcsp/src/lvebcsp/eval/mp20_realpxrd_solver.py implementation.

Only the model/data adapter changes: encode the target crystal into 32 tokens
and decode once with its representative/count condition. No diffusion rollout
or decoder noise is used. Coordinates and lattice are predicted; composition is supplied.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from lvebcsp.common.config import select_device
from lvebcsp.common.seed import seed_everything
from lvebcsp.data.crystal_dataset import CrystalDataset
from lvebcsp.data.crystal_latent import collate_crystal_latents
from lvebcsp.eval.mp20_realpxrd_solver import (
    REALPXRD_SOLVER_MATCHER, RMSD_MATCH_THRESHOLD, case_metrics_from_rmsd,
    mean_or_none, rms_dist_or_none, rmsd_is_match, save_per_case_csv,
)
from lvebcsp.inference.export_cif import export_structure_to_cif
from lvebcsp.losses.validity import has_usable_lattice
from lvebcsp.train.train_crystal_latent import build_crystal_latent
from lvebcsp.train.train_jepa import batch_to_device


def match_reconstruction(reference, candidate, roundtrip):
    """Use the solver's matcher for a prediction and a target roundtrip control."""
    matcher = StructureMatcher(**REALPXRD_SOLVER_MATCHER)
    gt_primitive = reference.get_primitive_structure()
    control_rmsd = rms_dist_or_none(matcher, roundtrip, gt_primitive)
    valid = len(candidate) > 0 and has_usable_lattice(candidate)
    error = None
    try:
        rmsd = rms_dist_or_none(matcher, candidate, gt_primitive) if valid else None
    except Exception as exc:
        # Keep matcher failures in the denominator, as in the copied evaluator.
        rmsd, error = None, f"{type(exc).__name__}: {exc}"
    return {
        **case_metrics_from_rmsd([rmsd], top_k=1),
        "rmsd": rmsd, "num_evals": 1, "valid_candidates": int(valid),
        "finite_rmsd_candidates": int(rmsd is not None),
        "roundtrip_match": rmsd_is_match(control_rmsd), "roundtrip_rmsd": control_rmsd,
        "matcher_error": error,
    }, gt_primitive


def evaluate_reconstruction(checkpoint_path, manifest_dir=None, output_dir="outputs/crystal_latent/reconstruction",
                            *, split="valid", max_cases=1000, batch_size=32, num_workers=4,
                            seed=42, device="auto", save_cifs=16):
    """Measure one deterministic reconstruction per randomly selected crystal.

    max_cases <= 0 evaluates the full split. Record the exact sample indices and
    checkpoint hash so a checkpoint updated during training stays identifiable.
    The roundtrip control uses target coordinates and the source cell;
    the scored reconstruction uses the decoder's predicted coordinates/cell.
    """
    seed_everything(seed)
    device = select_device(device)
    checkpoint_path, output = Path(checkpoint_path).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Keep one file handle across loading/hashing if training replaces best.ckpt.
    with checkpoint_path.open("rb") as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
        handle.seek(0)
        checkpoint_sha256 = hashlib.sha256(handle.read()).hexdigest()
    config = checkpoint["config"]
    model = build_crystal_latent(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    epoch, global_step = checkpoint.get("epoch"), checkpoint.get("global_step")
    del checkpoint
    manifest_dir = Path(manifest_dir or config["graph_manifest_dir"]).resolve()
    dataset = CrystalDataset(manifest_dir / f"{split}.npz", cutoff=model.encoder.config.cutoff)
    count = len(dataset) if max_cases <= 0 else min(max_cases, len(dataset))
    indices = np.sort(np.random.default_rng(seed).choice(len(dataset), count, replace=False))
    (output / "indices.json").write_text(json.dumps(indices.tolist()) + "\n")
    options = {"multiprocessing_context": "spawn"} if num_workers else {}
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                        num_workers=num_workers, collate_fn=list, **options)
    per_case, started = [], time.monotonic()
    print(f"checkpoint_epoch={epoch} global_step={global_step} split={split} cases={count} "
          f"device={device} mode=clean_token_reconstruction", flush=True)
    progress = tqdm(total=count, desc="Reconstruct", unit="crystal")
    for items in loader:
        cpu_batch = collate_crystal_latents(items)
        batch = batch_to_device(cpu_batch, device)
        with torch.inference_mode():
            condition = model.encode_condition(batch)
            tokens = model.encode(batch["target"])
            prediction = model.decode(tokens, condition, batch["atom_types"])
            embedding_mse = (model.encode(model.generated_graph(prediction)) - tokens).square().mean((1, 2)).cpu()
        prediction = {key: value.cpu() for key, value in prediction.items()}
        for i, item in enumerate(items):
            index = int(indices[len(per_case)])
            target = item["target"]
            reference = Structure(target.cell[0].numpy(), target.z.tolist(), target.frac_coords.numpy())
            roundtrip = Structure(target.cell[0].numpy(), target.z.tolist(), target.frac_coords.remainder(1).numpy())
            predicted_mask = prediction["atom_mask"][i]
            candidate = Structure(prediction["cell"][i].numpy(),
                                  prediction["atom_types"][i, predicted_mask].tolist(),
                                  prediction["frac_coords"][i, predicted_mask].numpy())
            metrics, gt_primitive = match_reconstruction(reference, candidate, roundtrip)
            case_dir, pred_path, gt_path = None, None, None
            if metrics["matched"] or len(per_case) < save_cifs:
                case_dir = output / f"case_{index:06d}"
                gt_path = export_structure_to_cif(gt_primitive, case_dir / "gt_primitive.cif")
                if len(candidate):
                    pred_path = export_structure_to_cif(candidate, case_dir / "reconstruction.cif")
            per_case.append({
                "index": index, "material_id": item["material_id"], "family": item["family"],
                "source": int(dataset.source[index]), "row": int(dataset.row[index]),
                "atoms": len(reference), "primitive_atoms": len(gt_primitive),
                "predicted_atoms": len(candidate),
                "jepa_embedding_mse": float(embedding_mse[i]),
                "saved_candidate_index": 0 if pred_path else None,
                "best_candidate_cif": str(pred_path) if pred_path else None,
                "gt_primitive_cif": str(gt_path) if gt_path else None,
                "case_dir": str(case_dir) if case_dir else None, **metrics,
            })
        progress.update(len(items))
        progress.set_postfix(matches=sum(row["matched"] for row in per_case),
                             roundtrip=sum(row["roundtrip_match"] for row in per_case))
    progress.close()
    total = len(per_case)
    matched = sum(row["matched"] for row in per_case)
    summary = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": epoch, "global_step": global_step,
        "manifest_dir": str(manifest_dir), "split": split, "seed": seed,
        "selection": "uniform random records without replacement",
        "num_cases": total, "unique_families": len({row["family"] for row in per_case}),
        "mode": "clean_token_reconstruction", "latent_shape": [model.num_latents, model.token_dim],
        "decoder_noise": 0.0, "num_evals": 1, "num_diffusion_steps": 0,
        "predicted_lattice": True, "atom_types": "supplied", "atom_count": "supplied",
        "atoms_per_token": model.decoder.max_atoms,
        "matcher": dict(REALPXRD_SOLVER_MATCHER), "rmsd_threshold": RMSD_MATCH_THRESHOLD,
        "structure_matcher_rms_dist_units": "normalized_by_average_free_length_per_atom",
        "matcher_primitive_cell": True, "matcher_scale": True,
        "matched_cases": matched, "match_rate": matched / total if total else None,
        "top_1_match_rate": matched / total if total else None,
        "mean_rmsd": mean_or_none([row["best_rmsd"] for row in per_case]),
        "candidate_validity_rate": mean_or_none([row["valid_candidates"] for row in per_case]),
        "candidate_validity_definition": "nonempty candidate and copied solver's usable-lattice guard",
        "roundtrip_match_rate": mean_or_none([row["roundtrip_match"] for row in per_case]),
        "mean_jepa_embedding_mse": mean_or_none([row["jepa_embedding_mse"] for row in per_case]),
        "atom_count_accuracy": mean_or_none([row["predicted_atoms"] == row["atoms"] for row in per_case]),
        "matcher_errors": sum(row["matcher_error"] is not None for row in per_case),
        "seconds": time.monotonic() - started,
        "per_case_metrics_csv": str(save_per_case_csv(per_case, output / "per_case_metrics.csv")),
        "per_case_metrics_json": str(output / "per_case_metrics.json"),
    }
    (output / "per_case_metrics.json").write_text(json.dumps(per_case, indent=2, allow_nan=False) + "\n")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-dir")
    parser.add_argument("--output-dir", default="outputs/crystal_latent/reconstruction")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--max-cases", type=int, default=1000, help="Use 0 for the full split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-cifs", type=int, default=16,
                        help="Save the first N cases as examples; matched CIFs are always saved")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    evaluate_reconstruction(args.checkpoint, args.manifest_dir, args.output_dir, split=args.split,
                            max_cases=args.max_cases, batch_size=args.batch_size, num_workers=args.num_workers,
                            seed=args.seed, device=args.device, save_cifs=args.save_cifs)


if __name__ == "__main__":
    main()
