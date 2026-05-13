"""POV image transforms (numpy only; no MuJoCo)."""

from __future__ import annotations

import numpy as np


def pov_rotate_bgr(bgr: np.ndarray, rotate: str) -> np.ndarray:
    """Rotate POV image for OpenCV (``numpy.rot90``; returns C-contiguous array)."""
    r = str(rotate).strip().lower()
    if r in ("", "none"):
        return np.ascontiguousarray(bgr)
    img = np.asarray(bgr)
    if r == "ccw90":
        out = np.rot90(img, k=1, axes=(0, 1))
    elif r == "cw90":
        out = np.rot90(img, k=-1, axes=(0, 1))
    elif r == "180":
        out = np.rot90(img, k=2, axes=(0, 1))
    elif r == "transpose":
        out = np.transpose(img, (1, 0, 2))
    else:
        return np.ascontiguousarray(bgr)
    return np.ascontiguousarray(out)
