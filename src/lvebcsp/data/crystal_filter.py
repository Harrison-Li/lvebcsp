"""Exclude incomplete building-block records from existing crystal manifests."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import pickle

import lmdb
import numpy as np
from tqdm import tqdm


def _check_chunk(path, indices, rows):
    rejected = []
    with lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False) as env:
        with env.begin() as txn:
            for index, row_id in zip(indices, rows):
                row = pickle.loads(gzip.decompress(txn.get(f"{row_id:08d}".encode())))
                expected = Counter()
                for (start, end), group in zip(row["mol2_group_slices"], row["mol_group_id_2"]):
                    expected[tuple(sorted(row["mol_atom_type_2"][start:end]))] += row["repeat_dict_2"][group]
                copies = defaultdict(list)
                for group, species in zip(row["crystal_group_id2"], row["crystal_atom_type"]):
                    copies[group].append(species)
                actual = Counter(tuple(sorted(species)) for species in copies.values())
                if expected != actual:
                    rejected.append({
                        "index": int(index), "row": int(row_id), "material_id": str(row["cif_filename"]),
                        "expected_atoms": sum(len(species) * count for species, count in expected.items()),
                        "target_atoms": len(row["crystal_atom_type"]),
                        "reason": "Representative compositions/multiplicities do not match target copies",
                    })
    return rejected


def filter_splits(manifest_dir, output_dir, *, workers=8):
    """Filter once before DDP sampling; preserve split membership and record order.

    Check the original representative table without repairing missing blocks.
    Atom permutations and independently numbered atom maps are allowed. Source
    LMDBs and input manifests are read-only; exclusions are recorded as JSONL.
    """
    manifest_dir, output_dir = Path(manifest_dir).resolve(), Path(output_dir).resolve()
    if manifest_dir == output_dir:
        raise ValueError("Use a separate output directory for filtered manifests.")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = json.loads((manifest_dir / "sources.json").read_text())
    report = {"input_dir": str(manifest_dir), "output_dir": str(output_dir), "splits": {}}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        with (output_dir / "excluded.jsonl").open("w") as exclusions:
            for split in ("train", "valid", "test"):
                path = manifest_dir / f"{split}.npz"
                if not path.exists():
                    continue
                with np.load(path) as data:
                    manifest = {key: data[key] for key in data.files}
                keep = np.ones(len(manifest["row"]), dtype=bool)
                futures = {}
                for source, metadata in enumerate(sources):
                    indices = np.flatnonzero(manifest["source"] == source)
                    for start in range(0, len(indices), 5000):
                        chunk = indices[start:start + 5000]
                        future = executor.submit(_check_chunk, metadata["path"], chunk, manifest["row"][chunk])
                        futures[future] = (source, len(chunk))
                rejected = []
                with tqdm(total=len(keep), desc=f"Filter {split}", unit="crystal", mininterval=2) as progress:
                    for future in as_completed(futures):
                        source, checked = futures[future]
                        for record in future.result():
                            keep[record["index"]] = False
                            rejected.append({"split": split, "source": source, **record})
                        progress.update(checked)
                for record in sorted(rejected, key=lambda record: record["index"]):
                    exclusions.write(json.dumps(record) + "\n")
                np.savez_compressed(output_dir / f"{split}.npz", **{key: value[keep] for key, value in manifest.items()})
                report["splits"][split] = {"input": len(keep), "kept": int(keep.sum()), "excluded": len(rejected)}
                print(f"{split}: {report['splits'][split]}", flush=True)
    (output_dir / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    (output_dir / "filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="outputs/crystal_jepa/splits")
    parser.add_argument("--output-dir", default="outputs/crystal_latent/splits")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    filter_splits(args.manifest_dir, args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()
