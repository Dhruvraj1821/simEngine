"""Tiny text visualization helpers. No third-party dependencies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BLOCKS = "▁▂▃▄▅▆▇█"


def resample(values: Sequence[float], width: int) -> np.ndarray:
    """Shrink a series to at most `width` points by averaging equal buckets."""
    if width < 1:
        raise ValueError("width must be at least 1")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must not be empty")
    if arr.size <= width:
        return arr
    return np.array([bucket.mean() for bucket in np.array_split(arr, width)])


def sparkline(values: Sequence[float], width: int = 60) -> str:
    """One-line block-character chart, scaled to this series' own min and max."""
    arr = resample(values, width)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-12 * max(1.0, abs(hi)):
        return BLOCKS[len(BLOCKS) // 2] * arr.size
    levels = np.floor((arr - lo) / (hi - lo) * (len(BLOCKS) - 1) + 0.5).astype(int)
    return "".join(BLOCKS[i] for i in levels)