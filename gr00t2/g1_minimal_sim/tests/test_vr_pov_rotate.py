"""``pov_rotate_bgr`` pixel transforms (numpy only)."""

from __future__ import annotations

import numpy as np

from legacy_vr_code.pov_image import pov_rotate_bgr


def test_pov_rotate_none_contiguous() -> None:
    bgr = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    out = pov_rotate_bgr(bgr, "none")
    assert out.shape == bgr.shape
    assert out.flags["C_CONTIGUOUS"]


def test_pov_rotate_ccw90_shape() -> None:
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    out = pov_rotate_bgr(bgr, "ccw90")
    assert out.shape == (6, 4, 3)


def test_pov_rotate_cw90_inverse_of_ccw() -> None:
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 255, size=(5, 7, 3), dtype=np.uint8)
    a = pov_rotate_bgr(pov_rotate_bgr(bgr, "ccw90"), "cw90")
    assert np.array_equal(a, bgr)


def test_pov_rotate_180() -> None:
    bgr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    out = pov_rotate_bgr(bgr, "180")
    assert out.shape == bgr.shape
    assert np.array_equal(out[0, 0], bgr[1, 1])


def test_pov_rotate_transpose() -> None:
    bgr = np.zeros((3, 5, 3), dtype=np.uint8)
    out = pov_rotate_bgr(bgr, "transpose")
    assert out.shape == (5, 3, 3)
