"""Simple raw PXRD preprocessing into d-I peak lists."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from scipy.signal import find_peaks, savgol_filter

from lvebcsp.data.xrd import DIPeakList, normalize_intensity, pad_peak_list, two_theta_to_d, wavelength_to_angstrom


def read_xy_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a whitespace-delimited two-column .xy file."""

    data = np.loadtxt(Path(path), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")
    return data[:, 0], data[:, 1]


def read_csv_peak_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV peak file with two_theta/d and intensity columns."""

    import pandas as pd

    frame = pd.read_csv(path)
    lower = {column.lower(): column for column in frame.columns}
    angle_col = lower.get("two_theta") or lower.get("2theta") or lower.get("theta") or lower.get("d")
    intensity_col = lower.get("intensity") or lower.get("i")
    if angle_col is None or intensity_col is None:
        raise ValueError("CSV must contain two_theta or d plus intensity columns")
    return frame[angle_col].to_numpy(dtype=np.float64), frame[intensity_col].to_numpy(dtype=np.float64)


def baseline_correction_simple(intensity: np.ndarray, quantile: float = 0.05) -> np.ndarray:
    """Subtract a low quantile baseline and clip to non-negative values."""

    values = np.asarray(intensity, dtype=np.float64)
    if values.size == 0:
        return values
    baseline = np.quantile(values[np.isfinite(values)], quantile)
    return np.clip(values - baseline, 0.0, None)


def subtract_polynomial_background(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    poly_order: int,
    smoothing_window: int = 3,
) -> np.ndarray:
    """Subtract a fitted polynomial background and return 0-100 corrected intensity."""

    x = np.asarray(two_theta, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    if y.size == 0:
        return y

    valid = np.isfinite(x) & np.isfinite(y)
    corrected = np.zeros_like(y, dtype=np.float64)
    if valid.sum() == 0:
        return corrected

    xv = x[valid]
    yv = y[valid]
    window = max(1, int(smoothing_window))
    if window > 1:
        kernel = np.ones(window, dtype=np.float64) / float(window)
        left = window // 2
        right = window - 1 - left
        yv = np.convolve(np.pad(yv, (left, right), mode="edge"), kernel, mode="valid")

    order = min(max(0, int(poly_order)), max(0, xv.size - 1))
    try:
        background = Polynomial.fit(xv, yv, order)(xv)
    except Exception:
        residual = baseline_correction_simple(yv)
    else:
        residual = np.clip(yv - background, 0.0, None)

    max_value = float(np.nanmax(residual)) if residual.size else 0.0
    if max_value <= 0.0:
        return corrected
    corrected[valid] = residual / max_value * 100.0
    return corrected


def correct_background(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    method: str = "simple",
    *,
    poly_order: int = 5,
    smoothing_window: int = 3,
) -> np.ndarray:
    """Apply the selected PXRD background correction."""

    method = method.lower()
    if method == "none":
        return np.clip(np.asarray(intensity, dtype=np.float64), 0.0, None)
    if method == "simple":
        return baseline_correction_simple(intensity)
    if method == "polynomial":
        return subtract_polynomial_background(
            two_theta,
            intensity,
            poly_order=poly_order,
            smoothing_window=smoothing_window,
        )
    raise ValueError(f"Unknown background correction method: {method}")


def smooth_pxrd(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    window_deg: float = 0.4,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky-Golay smooth a PXRD profile using an angular-width window.

    PXRD sources in this project have very different sampling intervals.  An
    angular window therefore gives consistent smoothing while a fixed number
    of samples can under-smooth dense profiles and erase peaks in sparse ones.
    """

    x = np.asarray(two_theta, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    if y.size < 3 or float(window_deg) <= 0.0:
        return y.copy()

    positive_steps = np.diff(x)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0.0)]
    if positive_steps.size == 0:
        return y.copy()
    step_deg = float(np.median(positive_steps))
    window_length = max(1, int(round(float(window_deg) / step_deg)))
    if window_length % 2 == 0:
        window_length += 1

    order = max(0, int(polyorder))
    minimum = order + 2
    if minimum % 2 == 0:
        minimum += 1
    window_length = max(window_length, minimum)
    maximum = y.size if y.size % 2 == 1 else y.size - 1
    window_length = min(window_length, maximum)
    if window_length <= order:
        return y.copy()
    return savgol_filter(y, window_length=window_length, polyorder=order)


def local_max_peak_pick(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    p_max: int,
    min_prominence: float = 0.01,
    min_distance_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick prominence-filtered local maxima and keep the strongest peaks."""

    x = np.asarray(two_theta, dtype=np.float64)
    raw_y = np.asarray(intensity, dtype=np.float64)
    if x.shape != raw_y.shape:
        raise ValueError("two_theta and intensity must have equal length")
    valid = np.isfinite(x) & np.isfinite(raw_y)
    x = x[valid]
    raw_y = raw_y[valid]
    if x.size:
        order = np.argsort(x, kind="stable")
        x = x[order]
        raw_y = raw_y[order]
    y = normalize_intensity(raw_y)
    limit = max(0, int(p_max))
    if limit == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if x.size < 3:
        keep = min(limit, x.size)
        return x[:keep], y[:keep]

    peak_idx, _ = find_peaks(y, prominence=max(0.0, float(min_prominence)))
    if peak_idx.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    separation = max(0.0, float(min_distance_deg))
    if separation > 0.0 and peak_idx.size > 1:
        ranked = peak_idx[np.argsort(y[peak_idx], kind="stable")[::-1]]
        kept: list[int] = []
        for idx in ranked:
            if all(abs(float(x[idx] - x[other])) >= separation for other in kept):
                kept.append(int(idx))
        peak_idx = np.asarray(kept, dtype=np.int64)

    strongest = peak_idx[np.argsort(y[peak_idx], kind="stable")[-limit:]]
    order = np.argsort(x[strongest])
    strongest = strongest[order]
    return x[strongest], y[strongest]


def preprocess_raw_pxrd(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    p_max: int = 128,
    wavelength: str | float = "CuKa",
    smooth: bool = True,
    smoothing_window_deg: float = 0.4,
    smoothing_polyorder: int = 2,
    min_prominence: float = 0.01,
    min_peak_distance_deg: float = 0.1,
    background: str = "simple",
    background_poly_order: int = 5,
    background_smoothing_window: int = 3,
) -> DIPeakList:
    """Convert raw sampled PXRD intensities to a padded d-I peak list."""

    x = np.asarray(two_theta, dtype=np.float64)
    raw_y = np.asarray(intensity, dtype=np.float64)
    if x.shape != raw_y.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    valid = np.isfinite(x) & np.isfinite(raw_y)
    x = x[valid]
    raw_y = raw_y[valid]
    if x.size:
        order = np.argsort(x, kind="stable")
        x = x[order]
        raw_y = raw_y[order]
    y = correct_background(
        x,
        raw_y,
        method=background,
        poly_order=background_poly_order,
        smoothing_window=background_smoothing_window,
    )
    if smooth:
        y = smooth_pxrd(
            x,
            y,
            window_deg=smoothing_window_deg,
            polyorder=smoothing_polyorder,
        )
    peak_two_theta, peak_i = local_max_peak_pick(
        x,
        y,
        p_max=p_max,
        min_prominence=min_prominence,
        min_distance_deg=min_peak_distance_deg,
    )
    peak_i = normalize_intensity(peak_i)
    peak_d = two_theta_to_d(peak_two_theta, wavelength_to_angstrom(wavelength))
    order = np.argsort(peak_d)[::-1]
    return pad_peak_list(peak_d[order], peak_i[order], p_max)
