#!/usr/bin/env python3
"""Collect many headless GR00T NPZ episodes (diner-friendly defaults).

Each subprocess runs ``play_g1_gear_wbc.py`` with:

- ``--headless``
- ``--headless-scripted-demo wiggle``
- ``--gr00t-record-dir`` (per-episode folder)
- ``--hands`` (default on)
- ``--gr00t-record-video`` (default on; use ``--no-record-video`` to disable)
- Per-episode ``--wiggle-seed`` so trajectories differ; optional arm noise + reduced loco wobble

Example (from ``g1_minimal_sim/``, WholeBodyControl venv python):

    python scripts/collect_dummy_gr00t_batch.py /data/g1_wiggle_diner \\
        --overwrite --episodes 300 --steps 600
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_root", type=Path, help="Output root; each episode gets its own folder")
    p.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of episodes to collect (default 300)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Headless MuJoCo steps per episode (default 600)",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run play_g1_gear_wbc.py (use WholeBodyControl venv)",
    )
    p.add_argument(
        "--play-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "play_g1_gear_wbc.py",
        help="Path to play_g1_gear_wbc.py",
    )
    p.add_argument(
        "--scene",
        choices=("stylish_diner", "floor", "table_pnp", "table_pnp_apple"),
        default="stylish_diner",
        help="MJCF scene (default stylish_diner)",
    )
    p.add_argument(
        "--no-hands",
        action="store_true",
        help="Disable articulated hands (not recommended for GR00T logging)",
    )
    p.add_argument(
        "--no-record-video",
        action="store_true",
        help="Omit --gr00t-record-video (faster, smaller; not recommended for VLA finetune)",
    )
    p.add_argument(
        "--wiggle-seed-base",
        type=int,
        default=10_000_007,
        help="Per-episode seed is base + stride * index (default 10000007)",
    )
    p.add_argument(
        "--wiggle-seed-stride",
        type=int,
        default=100_003,
        help="Multiplier for episode index added to --wiggle-seed-base",
    )
    p.add_argument(
        "--wiggle-arm-noise-rad",
        type=float,
        default=0.04,
        help="Passed to play.py --wiggle-arm-noise-rad (default 0.04 rad)",
    )
    p.add_argument(
        "--wiggle-loco-scale",
        type=float,
        default=0.3,
        help="Passed to play.py --wiggle-loco-scale (default 0.3)",
    )
    p.add_argument(
        "--no-wiggle-noise",
        action="store_true",
        help="Set --wiggle-arm-noise-rad 0 (sinusoids + seeded episode params only)",
    )
    p.add_argument(
        "--task-prefix",
        type=str,
        default="stylish_diner scripted wiggle",
        help="Task text prefix for annotation.human.task_description",
    )
    p.add_argument("--overwrite", action="store_true", help="Delete existing out_root first")
    args = p.parse_args()

    out_root = args.out_root.resolve()
    play_script = args.play_script.resolve()
    if not play_script.is_file():
        print(f"error: missing play script: {play_script}", file=sys.stderr)
        return 2
    if out_root.exists():
        if not args.overwrite:
            print(f"error: output exists: {out_root} (pass --overwrite)", file=sys.stderr)
            return 3
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sigma = 0.0 if args.no_wiggle_noise else float(args.wiggle_arm_noise_rad)

    for i in range(int(args.episodes)):
        ep_dir = out_root / f"episode_{i:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        task = f"{args.task_prefix} #{i:06d}"
        ep_seed = int(args.wiggle_seed_base) + int(args.wiggle_seed_stride) * int(i)
        cmd = [
            str(args.python),
            str(play_script),
            "--headless",
            "--max-steps",
            str(int(args.steps)),
            "--headless-scripted-demo",
            "wiggle",
            "--scene",
            str(args.scene),
            "--gr00t-record-dir",
            str(ep_dir),
            "--gr00t-task",
            task,
            "--wiggle-seed",
            str(ep_seed),
            "--wiggle-arm-noise-rad",
            str(sigma),
            "--wiggle-loco-scale",
            str(float(args.wiggle_loco_scale)),
        ]
        if not args.no_hands:
            cmd.append("--hands")
        if not args.no_record_video:
            cmd.append("--gr00t-record-video")
        print(f"[{i+1}/{args.episodes}] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"error: episode {i} collection failed with code {proc.returncode}", file=sys.stderr)
            return int(proc.returncode)
        npz = ep_dir / "episode_000.npz"
        if not npz.is_file():
            print(f"error: missing expected file {npz}", file=sys.stderr)
            return 4

    print(f"done: wrote {args.episodes} episodes under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
