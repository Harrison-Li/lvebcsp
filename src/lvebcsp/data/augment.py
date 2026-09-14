"""Spectrum-level PXRD augmentation for JEPA context views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lvebcsp.data.xrd import DIPeakList, pattern_to_di


def _as_float_array(values: Any) -> np.ndarray:
    """Convert spectrum arrays to one-dimensional float64 arrays."""

    return np.asarray(values, dtype=np.float64).reshape(-1)


def _rng_uniform(
    rng: np.random.Generator | None,
    low: float,
    high: float,
    size: tuple[int, ...] | int | None = None,
) -> np.ndarray | float:
    if rng is None:
        return np.random.uniform(low, high, size=size)
    return rng.uniform(low, high, size=size)


def _rng_normal(
    rng: np.random.Generator | None,
    loc: float,
    scale: float,
    size: tuple[int, ...] | int,
) -> np.ndarray:
    if rng is None:
        return np.random.normal(loc, scale, size=size)
    return rng.normal(loc, scale, size=size)


def filter_spectrum(
    two_theta: Any,
    intensity: Any,
    *,
    intensity_threshold: float = 5.0,
    two_theta_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep finite spectrum points inside the angle range and above threshold."""

    x = _as_float_array(two_theta)
    y = _as_float_array(intensity)
    if x.shape != y.shape:
        raise ValueError(
            f"two_theta and intensity must have equal shape, got {x.shape} and {y.shape}"
        )

    keep = np.isfinite(x) & np.isfinite(y) & (y > float(intensity_threshold))
    if two_theta_range is not None:
        t_min, t_max = two_theta_range
        keep &= (x >= float(t_min)) & (x <= float(t_max))
    return x[keep], y[keep]


def augment_spectrum(
    two_theta: Any,
    intensity: Any,
    noise_level: float = 0.05,
    shift_range: float = 0.1,
    scale_range: tuple[float, float] = (0.8, 1.2),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Gaussian intensity noise, two-theta peak shifts, and global scaling."""

    x = _as_float_array(two_theta)
    y = _as_float_array(intensity)
    if x.shape != y.shape:
        raise ValueError(
            f"two_theta and intensity must have equal shape, got {x.shape} and {y.shape}"
        )
    if x.size == 0:
        return x.copy(), y.copy()

    max_intensity = float(np.max(y))
    noise_scale = max(float(noise_level), 0.0) * max(max_intensity, 0.0)
    intensity_aug = y + _rng_normal(rng, 0.0, noise_scale, size=y.shape)

    shift = _rng_uniform(rng, -float(shift_range), float(shift_range), size=x.shape)
    two_theta_aug = x + shift

    scale_min, scale_max = scale_range
    scale_factor = float(_rng_uniform(rng, float(scale_min), float(scale_max)))
    intensity_aug = intensity_aug * scale_factor
    intensity_aug = np.where(intensity_aug < 0.0, y, intensity_aug)

    aug_max = float(np.max(intensity_aug)) if intensity_aug.size else 0.0
    if aug_max > 0.0:
        intensity_aug = intensity_aug / aug_max * 100.0
    else:
        intensity_aug = np.zeros_like(intensity_aug)

    return two_theta_aug, intensity_aug


@dataclass
class PXRDDataAugmenter:
    """Augment two-theta/intensity spectra before d-I conversion."""

    enabled: bool = True
    noise_level: float = 0.05
    shift_range: float = 0.1
    scale_range: tuple[float, float] = (0.8, 1.2)
    intensity_threshold: float = 5.0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "PXRDDataAugmenter":
        """Construct from a config dictionary."""

        if not config:
            return cls()
        fields = cls.__dataclass_fields__
        kwargs = {key: value for key, value in config.items() if key in fields}
        if "scale_range" in kwargs:
            kwargs["scale_range"] = tuple(float(value) for value in kwargs["scale_range"])
        return cls(**kwargs)

    def to_di(
        self,
        two_theta: Any,
        intensity: Any,
        *,
        p_max: int,
        wavelength: str | float = "CuKa",
        two_theta_range: tuple[float, float] | None = None,
        augment: bool = True,
        rng: np.random.Generator | None = None,
    ) -> DIPeakList:
        """Convert a clean or augmented spectrum to padded d-I descriptors."""

        x, y = filter_spectrum(
            two_theta,
            intensity,
            intensity_threshold=self.intensity_threshold,
            two_theta_range=two_theta_range,
        )
        if augment and self.enabled:
            x, y = augment_spectrum(
                x,
                y,
                noise_level=self.noise_level,
                shift_range=self.shift_range,
                scale_range=self.scale_range,
                rng=rng,
            )
            x, y = filter_spectrum(
                x,
                y,
                intensity_threshold=self.intensity_threshold,
                two_theta_range=two_theta_range,
            )
        return pattern_to_di((x, y), p_max=p_max, min_intensity=0.0, wavelength=wavelength)
