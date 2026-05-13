#!/usr/bin/env python3
"""Generate tileable wood-plank-style albedo PNG for stylish_diner floor (warm browns, matte-friendly).

Avoids obvious periodic stripes: tileable value noise + wavy grain along plank run, no crisp groove grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _tileable_value_noise(
    x: np.ndarray, y: np.ndarray, grid_n: int, rng: np.random.Generator
) -> np.ndarray:
    """Bilinear value noise on a torus [0,1)^2. x,y shape (H,W)."""
    grid = rng.random((grid_n, grid_n))
    fx = x * grid_n
    fy = y * grid_n
    gx0 = np.floor(fx).astype(np.int64) % grid_n
    gy0 = np.floor(fy).astype(np.int64) % grid_n
    gx1 = (gx0 + 1) % grid_n
    gy1 = (gy0 + 1) % grid_n
    tx = fx - np.floor(fx)
    ty = fy - np.floor(fy)
    v00 = grid[gy0, gx0]
    v01 = grid[gy1, gx0]
    v10 = grid[gy0, gx1]
    v11 = grid[gy1, gx1]
    v0 = v00 * (1.0 - tx) + v10 * tx
    v1 = v01 * (1.0 - tx) + v11 * tx
    return v0 * (1.0 - ty) + v1 * ty


def _fbm_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """2-octave tileable fbm-style blend (coarse + medium)."""
    rng = np.random.default_rng(seed)
    n0 = _tileable_value_noise(x, y, 5, rng)
    rng = np.random.default_rng(seed + 101)
    n1 = _tileable_value_noise(x, y, 13, rng)
    return 0.55 * n0 + 0.45 * n1


def make_floor_texture(size: int = 2048, seed: int = 42) -> np.ndarray:
    """RGB float [0,1], seamless on [0,1]^2 (torus)."""
    y = np.linspace(0.0, 1.0, size, endpoint=False)
    x = np.linspace(0.0, 1.0, size, endpoint=False)
    x, y = np.meshgrid(x, y)

    # Organic low-frequency tone (no parallel stripe family)
    blob = _fbm_noise(x, y, seed)
    # Slow wavy modulation — breaks axis-aligned repetition
    warp = 0.18 * np.sin(2 * np.pi * (3 * x + 2 * y)) + 0.12 * np.cos(2 * np.pi * (2 * x - 5 * y))
    base = 0.48 + 0.14 * blob + 0.06 * warp

    # Long grain streaks mostly along x (plank run), phase-warped so lines are not ruler-straight
    phase_y = 1.7 * np.sin(2 * np.pi * 4 * y) + 0.9 * np.sin(2 * np.pi * 7 * y + 2 * np.pi * x)
    grain_lines = np.sin(2 * np.pi * (31 * x + phase_y * 0.08))
    grain_lines += 0.45 * np.sin(2 * np.pi * (53 * x + 0.12 * phase_y))
    grain_lines *= 0.022

    # Fine high-freq grain (integer Fourier modes only → tiles)
    fine = (
        0.012 * np.sin(2 * np.pi * (19 * x + 11 * y))
        + 0.010 * np.sin(2 * np.pi * (29 * y + 7 * x))
        + 0.008 * np.sin(2 * np.pi * (47 * x - 23 * y))
    )

    # Very soft wavy “wear” — not a crisp plank groove grid
    seam = 0.97 + 0.03 * np.sin(2 * np.pi * (9 * y + 1.2 * np.sin(2 * np.pi * 6 * x)))
    seam *= 0.985 + 0.015 * np.sin(2 * np.pi * (11 * x + 0.8 * np.sin(2 * np.pi * 5 * y)))

    r = np.clip(base * 0.93 + grain_lines + fine + 0.02, 0.0, 1.0) * seam
    g = np.clip(base * 0.51 + grain_lines * 0.75 + fine * 0.65, 0.0, 1.0) * seam
    b = np.clip(base * 0.27 + grain_lines * 0.5 + fine * 0.45, 0.0, 1.0) * seam

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        action="append",
        dest="outs",
        required=True,
        help="Output path(s); specify multiple --out for preview + g1 copies.",
    )
    args = p.parse_args()
    rgb = make_floor_texture(args.size, seed=args.seed)
    img = Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB")
    for path in args.outs:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, format="PNG", optimize=True)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
