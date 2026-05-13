#!/usr/bin/env python3
"""Smoke load a dataset with GR00T's ``LeRobotEpisodeLoader``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_isaac_gr00t_on_path() -> Path:
    root = (Path(__file__).resolve().parents[2] / "Isaac-GR00T").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Expected sibling Isaac-GR00T repo at {root}")
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_root", type=Path, help="LeRobot dataset root")
    p.add_argument(
        "--embodiment",
        type=str,
        default="unitree_g1",
        help="Embodiment config key from gr00t.configs.data.embodiment_configs.MODALITY_CONFIGS",
    )
    p.add_argument("--video-backend", type=str, default="decord", help="LeRobot video backend")
    p.add_argument("--episode-index", type=int, default=0, help="Episode index to load")
    p.add_argument(
        "--disable-video-modality",
        action="store_true",
        help="Load with state/action/language only (useful for --no-video export smoke)",
    )
    args = p.parse_args()

    _ensure_isaac_gr00t_on_path()
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    if args.embodiment not in MODALITY_CONFIGS:
        print(f"error: unknown embodiment {args.embodiment!r}", file=sys.stderr)
        return 2
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        print(f"error: dataset path not found: {dataset_root}", file=sys.stderr)
        return 3

    modality_cfg = dict(MODALITY_CONFIGS[args.embodiment])
    if args.disable_video_modality and "video" in modality_cfg:
        modality_cfg.pop("video", None)

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_root,
        modality_configs=modality_cfg,
        video_backend=str(args.video_backend),
    )
    if len(loader) == 0:
        print("error: dataset has 0 episodes", file=sys.stderr)
        return 4
    epi = int(args.episode_index)
    if epi < 0 or epi >= len(loader):
        print(f"error: episode-index out of range [0, {len(loader)-1}]", file=sys.stderr)
        return 5
    df = loader[epi]
    print(
        f"smoke OK: episodes={len(loader)} episode={epi} rows={len(df)} "
        f"columns={list(df.columns)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
