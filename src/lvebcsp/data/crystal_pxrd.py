"""Target PXRD for graph JEPA, cached by geometry and simulation settings."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from pymatgen.core import Structure
import torch

from lvebcsp.data.xrd import normalize_intensity, pattern_to_di, simulate_pxrd, wavelength_to_angstrom


PEAK_FIELDS = ("peak_d", "peak_i", "peak_mask")


def pxrd_settings(config):
    """Canonical simulation contract; paths are excluded from cache identity."""
    angles = tuple(float(x) for x in config.get("two_theta_range", (5., 80.)))
    p_max = int(config.get("p_max", 2048))
    threshold = float(config.get("min_intensity", 1e-4))
    if len(angles) != 2 or not 0 < angles[0] < angles[1] < 180:
        raise ValueError("PXRD two_theta_range must satisfy 0 < min < max < 180")
    if p_max < 1 or not 0 < threshold <= 1:
        raise ValueError("PXRD requires p_max >= 1 and 0 < min_intensity <= 1")
    return dict(version="target_pxrd_v1", pymatgen=version("pymatgen"), p_max=p_max,
                wavelength=wavelength_to_angstrom(config.get("wavelength", "CuKa")),
                two_theta_range=angles, min_intensity=threshold)


def target_pxrd(graph, config):
    """Simulate the exact target cell, never the perturbed context or its edges.

    Cache writes are atomic across dataset workers/ranks. Different arrangements,
    scan settings, or pymatgen versions get distinct entries. The first p_max
    surviving peaks in increasing angle are retained, as in pattern_to_di.
    """
    settings = pxrd_settings(config)
    arrays = {name: graph[name].detach().cpu().contiguous().numpy() for name in ("z", "pos", "cell")}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for name, array in arrays.items():
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    key = digest.hexdigest()
    cache = (Path(config["cache_dir"]).expanduser() / key[:2] / f"{key}.npz"
             if config.get("cache_dir") else None)
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return {name: torch.from_numpy(saved[name].copy()) for name in (*PEAK_FIELDS, "pxrd_num_peaks")}
    crystal = Structure(arrays["cell"].reshape(3, 3), arrays["z"].tolist(), arrays["pos"],
                        coords_are_cartesian=True)
    pattern = simulate_pxrd(crystal, settings["wavelength"], settings["two_theta_range"])
    peaks = pattern_to_di(pattern, settings["p_max"], settings["min_intensity"], settings["wavelength"])
    if not peaks.peak_mask.any():
        raise ValueError("Target PXRD has no peaks in the configured scan range")
    count = int((normalize_intensity(pattern.y) >= settings["min_intensity"]).sum())
    result = {name: getattr(peaks, name) for name in PEAK_FIELDS}
    result["pxrd_num_peaks"] = torch.tensor(count)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(temporary, **{name: value.numpy() for name, value in result.items()})
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
    return result
