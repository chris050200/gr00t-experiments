#!/usr/bin/env python3
"""Quick QC for P2 GR00T NPZ recordings.

Checks structural consistency across all ``episode_*.npz`` files under a directory.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATE_KEYS = [
    "state_left_leg",
    "state_right_leg",
    "state_waist",
    "state_left_arm",
    "state_right_arm",
    "state_left_hand",
    "state_right_hand",
]
ACTION_KEYS = [
    "action_left_arm",
    "action_right_arm",
    "action_left_hand",
    "action_right_hand",
    "action_waist",
    "action_base_height_command",
    "action_navigate_command",
]
BASE_KEYS = [
    "T",
    "wall_time_s",
    "step_index",
    "annotation_human_task_description_bytes",
]


@dataclass
class QCSummary:
    episodes: int = 0
    total_steps: int = 0
    min_steps: int = 1_000_000_000
    max_steps: int = 0
    video_episodes: int = 0


def _require(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in z.files:
        raise KeyError(f"missing key {key}")
    return np.asarray(z[key])


def _check_episode(path: Path, *, require_video: bool) -> tuple[int, bool]:
    with np.load(path, allow_pickle=False) as z:
        for k in BASE_KEYS + STATE_KEYS + ACTION_KEYS:
            _require(z, k)
        t = int(np.asarray(z["T"]).reshape(()))
        if t <= 0:
            raise ValueError("T must be > 0")
        wall = _require(z, "wall_time_s")
        if wall.ndim != 1 or wall.shape[0] != t:
            raise ValueError(f"wall_time_s shape mismatch: {wall.shape} vs T={t}")
        step_index = _require(z, "step_index")
        if step_index.ndim != 1 or step_index.shape[0] != t:
            raise ValueError(f"step_index shape mismatch: {step_index.shape} vs T={t}")

        for k in STATE_KEYS + ACTION_KEYS:
            arr = _require(z, k)
            if arr.ndim != 2 or arr.shape[0] != t:
                raise ValueError(f"{k} shape mismatch: {arr.shape} vs T={t}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{k} has non-finite values")

        has_video = "ego_view_image" in z.files
        if require_video and not has_video:
            raise ValueError("video required but ego_view_image missing")
        if has_video:
            vid = np.asarray(z["ego_view_image"])
            if vid.ndim != 4 or vid.shape[0] != t or vid.shape[-1] != 3:
                raise ValueError(f"ego_view_image invalid shape: {vid.shape}")
            if vid.dtype != np.uint8:
                raise ValueError(f"ego_view_image must be uint8, got {vid.dtype}")

    return t, has_video


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recordings_root", type=Path, help="Directory containing episode_*.npz")
    p.add_argument("--require-video", action="store_true", help="Fail if any episode has no video")
    args = p.parse_args()

    root = args.recordings_root.resolve()
    files = sorted(root.rglob("episode_*.npz"))
    if not files:
        print(f"error: no episode_*.npz under {root}", file=sys.stderr)
        return 2

    summary = QCSummary()
    failures: list[tuple[Path, str]] = []
    for fp in files:
        try:
            t, has_video = _check_episode(fp, require_video=bool(args.require_video))
            summary.episodes += 1
            summary.total_steps += t
            summary.min_steps = min(summary.min_steps, t)
            summary.max_steps = max(summary.max_steps, t)
            summary.video_episodes += int(has_video)
        except Exception as e:  # explicit per-file report
            failures.append((fp, str(e)))

    if failures:
        print("QC failed:")
        for fp, err in failures:
            print(f"  - {fp}: {err}")
        return 1

    print(
        f"QC OK: episodes={summary.episodes}, total_steps={summary.total_steps}, "
        f"min_steps={summary.min_steps}, max_steps={summary.max_steps}, "
        f"video_episodes={summary.video_episodes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
