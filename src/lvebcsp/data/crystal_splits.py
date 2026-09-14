"""Prepare crystal split manifests using the current source's refcode families."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import re

import lmdb
import numpy as np


def refcode_family(refcode: str) -> str:
    """Group numbered refcodes and all cell/augmentation variants together."""
    return re.sub(r"\d+$", "", refcode)


def _scan_chunk(path: str, start: int, end: int):
    records = []
    with lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False) as env:
        with env.begin() as txn:
            for index in range(start, end):
                payload = txn.get(f"{index:08d}".encode())
                row = pickle.loads(gzip.decompress(payload))
                refcode = Path(row["cif_filename"]).stem.split("_std_")[0]
                records.append((index, refcode, hashlib.sha256(payload).hexdigest()))
    return records


def audit_splits(output_dir):
    """Verify the saved manifests against scanned record hashes and source families."""
    output_dir = Path(output_dir)
    sources = json.loads((output_dir / "sources.json").read_text())
    digests = []
    for source in range(len(sources)):
        with np.load(output_dir / f"source_{source}_index.npz") as cache:
            digests.append(cache["digest"].astype("S64"))
    families, records = [], []
    within_split_duplicates = 0
    for split in ["train", "valid", "test"]:
        with np.load(output_dir / f"{split}.npz") as manifest:
            hashes = np.empty(len(manifest["row"]), dtype="S64")
            for source, lookup in enumerate(digests):
                selected = manifest["source"] == source
                hashes[selected] = lookup[manifest["row"][selected]]
            families.append(set(manifest["family"]))
            records.append(set(hashes))
            within_split_duplicates += len(hashes) - len(records[-1])
    return {
        "family_overlap": sum(len(families[a] & families[b]) for a, b in [(0, 1), (0, 2), (1, 2)]),
        "exact_record_overlap": sum(len(records[a] & records[b]) for a, b in [(0, 1), (0, 2), (1, 2)]),
        "within_split_exact_duplicates": within_split_duplicates,
        "audited_unique_records": len(set.union(*records)),
    }


def build_splits(paths, output_dir, *, seed=42, valid_fraction=0.05, test_fraction=0.05, workers=8):
    """Hash whole source families into train/valid/test; deduplicate compressed records.

    Refcode families are a source-identity grouping, not a chemical-identity proof.
    Source order determines which copy survives when a record is repeated.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, all_records = [], []
    for source, path in enumerate(map(Path, paths)):
        path = path.resolve()
        with lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False) as env:
            length = env.stat()["entries"]
        fingerprint = dict(path=str(path), entries=length, size=path.stat().st_size,
                           mtime_ns=path.stat().st_mtime_ns)
        sources.append(fingerprint)
        cache = output_dir / f"source_{source}_index.npz"
        cache_info = cache.with_suffix(".json")
        if cache.exists() and cache_info.exists() and json.loads(cache_info.read_text()) == fingerprint:
            with np.load(cache) as data:
                records = list(zip(data["row"], data["refcode"], data["digest"]))
        else:
            records = []
            jobs = [(str(path), start, min(start + 10000, length)) for start in range(0, length, 10000)]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_scan_chunk, *job) for job in jobs]
                for future in as_completed(futures):
                    records.extend(future.result())
                    print(f"{path.name}: indexed {len(records):,}/{length:,}", flush=True)
            records.sort()
            np.savez_compressed(cache, row=np.array([r[0] for r in records], dtype=np.int64),
                                refcode=np.array([r[1] for r in records]),
                                digest=np.array([r[2] for r in records]))
            cache_info.write_text(json.dumps(fingerprint, indent=2) + "\n")
        all_records.extend((source, int(index), str(refcode), str(digest)) for index, refcode, digest in records)

    seen, unique = {}, []
    duplicate_count = 0
    for record in all_records:
        source, index, refcode, digest = record
        if digest in seen:
            duplicate_count += 1
        else:
            seen[digest] = (source, index)
            unique.append(record)
    family_split = {}
    for _, _, refcode, _ in unique:
        family = refcode_family(refcode)
        if family not in family_split:
            value = int.from_bytes(hashlib.sha256(f"{seed}:{family}".encode()).digest()[:8], "big") / 2**64
            family_split[family] = "test" if value < test_fraction else "valid" if value < test_fraction + valid_fraction else "train"

    report = {"seed": seed, "grouping": "refcode without trailing digits, before _std_ suffix",
              "sources": sources, "input_rows": len(all_records), "exact_duplicates_removed": duplicate_count,
              "unique_rows": len(unique), "splits": {}}
    for split in ["train", "valid", "test"]:
        chosen = [r for r in unique if family_split[refcode_family(r[2])] == split]
        np.savez_compressed(output_dir / f"{split}.npz",
                            source=np.array([r[0] for r in chosen], dtype=np.int16),
                            row=np.array([r[1] for r in chosen], dtype=np.int64),
                            refcode=np.array([r[2] for r in chosen]),
                            family=np.array([refcode_family(r[2]) for r in chosen]))
        report["splits"][split] = {"rows": len(chosen), "families": len({refcode_family(r[2]) for r in chosen}),
                                    "refcodes": len({r[2] for r in chosen})}
    (output_dir / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    report.update(audit_splits(output_dir))
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build_splits(args.sources, args.output_dir, seed=args.seed, valid_fraction=args.valid_fraction,
                 test_fraction=args.test_fraction, workers=args.workers)


if __name__ == "__main__":
    main()
