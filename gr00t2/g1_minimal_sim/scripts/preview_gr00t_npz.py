#!/usr/bin/env python3
"""Inspect P2 ``episode_*.npz`` logs from :mod:`gr00t_teleop_logger`.

Prints shapes/dtypes and optional Matplotlib trajectory plots + ego-view frame previews.

**Run** (numpy-only summary works in any env; plots need ``matplotlib``)::

    python scripts/preview_gr00t_npz.py /path/to/episode_000.npz
    python scripts/preview_gr00t_npz.py /path/to/episode_000.npz --plot-dir /tmp/gr00t_plots

From ``g1_minimal_sim/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _decode_task(z: np.lib.npyio.NpzFile) -> str:
    if "annotation_human_task_description_bytes" not in z.files:
        return ""
    raw = z["annotation_human_task_description_bytes"]
    return raw.tobytes().decode("utf-8", errors="replace")


def summarize_npz(path: Path) -> str:
    """Human-readable summary (no plotting)."""
    lines: list[str] = []
    with np.load(path, allow_pickle=False) as z:
        lines.append(f"file: {path}")
        lines.append(f"arrays: {sorted(z.files)}")
        if "T" in z.files:
            lines.append(f"T (steps): {int(z['T'])}")
        task = _decode_task(z)
        if task:
            lines.append(f"task: {task!r}")
        for name in sorted(z.files):
            arr = z[name]
            if name == "annotation_human_task_description_bytes":
                continue
            shape = getattr(arr, "shape", ())
            dtype = getattr(arr, "dtype", type(arr))
            lines.append(f"  {name}: shape={shape} dtype={dtype}")
    meta = path.parent / f"{path.stem}_metadata.json"
    if meta.is_file():
        lines.append(f"metadata: {meta}")
        try:
            meta_obj = json.loads(meta.read_text(encoding="utf-8"))
            for k in ("schema", "action_keys", "obs_state_keys", "include_video", "notes"):
                if k in meta_obj:
                    lines.append(f"  meta.{k}: {meta_obj[k]!r}")
        except json.JSONDecodeError as e:
            lines.append(f"  (metadata JSON parse error: {e})")
    return "\n".join(lines)


def plot_npz(path: Path, out_dir: Path) -> list[Path]:
    """Save trajectory PNGs under ``out_dir``. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "plotting requires matplotlib (`pip install matplotlib`)"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with np.load(path, allow_pickle=False) as z:
        if "wall_time_s" not in z.files:
            raise ValueError("NPZ missing wall_time_s — not a gr00t_teleop_logger export?")
        t_axis = np.asarray(z["wall_time_s"], dtype=np.float64)
        stem = path.stem

        def _save(fig: plt.Figure, suffix: str) -> None:
            outp = out_dir / f"{stem}_{suffix}.png"
            fig.savefig(outp, dpi=120, bbox_inches="tight")
            plt.close(fig)
            written.append(outp)

        # Actions overview
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        if "action_navigate_command" in z.files:
            nav = np.asarray(z["action_navigate_command"], dtype=np.float64)
            for i, lab in enumerate(("vx", "vy", "vyaw")):
                axes[0].plot(t_axis, nav[:, i], label=lab)
            axes[0].set_ylabel("navigate_cmd")
            axes[0].legend(loc="upper right", fontsize=8)
            axes[0].grid(True, alpha=0.3)
        if "action_base_height_command" in z.files:
            bh = np.asarray(z["action_base_height_command"], dtype=np.float64).reshape(-1)
            axes[1].plot(t_axis, bh)
            axes[1].set_ylabel("base_height_cmd")
            axes[1].grid(True, alpha=0.3)
        if "action_waist" in z.files:
            w = np.asarray(z["action_waist"], dtype=np.float64)
            for i in range(min(3, w.shape[1])):
                axes[2].plot(t_axis, w[:, i], label=f"j{i}")
            axes[2].set_ylabel("waist cmd")
            axes[2].set_xlabel("wall_time_s")
            axes[2].legend(loc="upper right", fontsize=8)
            axes[2].grid(True, alpha=0.3)
        _save(fig, "trajectory_actions")

        # Arms: measured state (symmetric layout)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        if "state_left_arm" in z.files:
            sa = np.asarray(z["state_left_arm"], dtype=np.float64)
            for j in range(sa.shape[1]):
                axes[0].plot(t_axis, sa[:, j], alpha=0.7, label=f"q{j}" if j < 7 else None)
            axes[0].set_ylabel("state_left_arm")
            axes[0].grid(True, alpha=0.3)
        if "state_right_arm" in z.files:
            sa = np.asarray(z["state_right_arm"], dtype=np.float64)
            for j in range(sa.shape[1]):
                axes[1].plot(t_axis, sa[:, j], alpha=0.7)
            axes[1].set_ylabel("state_right_arm")
            axes[1].set_xlabel("wall_time_s")
            axes[1].grid(True, alpha=0.3)
        _save(fig, "trajectory_arm_state")

        # Ego RGB samples
        if "ego_view_image" in z.files:
            vid = np.asarray(z["ego_view_image"], dtype=np.uint8)
            if vid.ndim == 4 and vid.shape[0] > 0:
                idxs = [0, vid.shape[0] // 2, vid.shape[0] - 1]
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, ix in zip(axes, idxs):
                    ax.imshow(vid[ix])
                    ax.set_title(f"frame {ix}")
                    ax.axis("off")
                plt.suptitle("ego_view_image samples")
                _save(fig, "ego_view_samples")

    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("npz", type=Path, help="Path to episode_*.npz")
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If set, write trajectory PNGs here (needs matplotlib)",
    )
    args = p.parse_args()
    npz_path = args.npz.resolve()
    if not npz_path.is_file():
        print(f"error: not a file: {npz_path}", flush=True)
        return 1

    print(summarize_npz(npz_path), flush=True)

    if args.plot_dir is not None:
        try:
            outs = plot_npz(npz_path, args.plot_dir.resolve())
            print("\nwrote:", flush=True)
            for o in outs:
                print(f"  {o}", flush=True)
        except RuntimeError as e:
            print(f"error: {e}", flush=True)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
