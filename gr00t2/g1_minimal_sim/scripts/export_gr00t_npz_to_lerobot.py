#!/usr/bin/env python3
"""Export P2 ``episode_*.npz`` logs to a GR00T/LeRobot-v2 style dataset.

This converter targets the ``unitree_g1`` modality contract:
- video: ``ego_view``
- state: ``left_leg, right_leg, waist, left_arm, right_arm, left_hand, right_hand``
- action: ``left_arm, right_arm, left_hand, right_hand, waist, base_height_command, navigate_command``
- language: ``annotation.human.task_description`` (task-index encoded in parquet)

Input NPZ files are produced by :mod:`gr00t_teleop_logger`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STATE_KEYS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
]
ACTION_KEYS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "base_height_command",
    "navigate_command",
]

DATA_PATH_PATTERN = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_PATTERN = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


@dataclass
class Episode:
    npz_path: Path
    task: str
    timestamps_s: np.ndarray  # (T,)
    state_by_key: dict[str, np.ndarray]  # key -> (T, Dk)
    action_by_key: dict[str, np.ndarray]  # key -> (T, Dk)
    video: np.ndarray | None  # (T, H, W, 3) uint8

    @property
    def length(self) -> int:
        return int(self.timestamps_s.shape[0])


def _decode_task(npz: Any) -> str:
    raw = npz["annotation_human_task_description_bytes"]
    return np.asarray(raw, dtype=np.uint8).tobytes().decode("utf-8", errors="replace")


def _require_array(npz: Any, key: str, *, ndim: int | None = None) -> np.ndarray:
    if key not in npz.files:
        raise KeyError(f"missing key {key!r}")
    arr = np.asarray(npz[key])
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{key!r} expected ndim={ndim}, got shape={arr.shape}")
    return arr


def _stack_dim(arr: np.ndarray, key: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{key!r} must be (T, D), got {out.shape}")
    return out


def load_episode(npz_path: Path) -> Episode:
    with np.load(npz_path, allow_pickle=False) as z:
        t_steps = int(np.asarray(z["T"]).reshape(()))
        ts = _require_array(z, "wall_time_s", ndim=1).astype(np.float64)
        if ts.shape[0] != t_steps:
            raise ValueError(f"{npz_path}: T={t_steps} but wall_time_s={ts.shape[0]}")
        ts = ts - float(ts[0])
        task = _decode_task(z)

        state_by_key: dict[str, np.ndarray] = {}
        action_by_key: dict[str, np.ndarray] = {}
        for k in STATE_KEYS:
            arr = _stack_dim(_require_array(z, f"state_{k}"), f"state_{k}")
            if arr.shape[0] != t_steps:
                raise ValueError(f"{npz_path}: state_{k} length mismatch")
            state_by_key[k] = arr
        for k in ACTION_KEYS:
            arr = _stack_dim(_require_array(z, f"action_{k}"), f"action_{k}")
            if arr.shape[0] != t_steps:
                raise ValueError(f"{npz_path}: action_{k} length mismatch")
            action_by_key[k] = arr

        video = None
        if "ego_view_image" in z.files:
            vid = np.asarray(z["ego_view_image"])
            if vid.ndim != 4 or vid.shape[0] != t_steps or vid.shape[-1] != 3:
                raise ValueError(f"{npz_path}: invalid ego_view_image shape {vid.shape}")
            if vid.dtype != np.uint8:
                vid = np.clip(vid, 0, 255).astype(np.uint8)
            video = vid

    return Episode(
        npz_path=npz_path,
        task=task,
        timestamps_s=ts,
        state_by_key=state_by_key,
        action_by_key=action_by_key,
        video=video,
    )


def apply_relative_arm_actions(ep: Episode) -> None:
    for key in ("left_arm", "right_arm"):
        ep.action_by_key[key] = ep.action_by_key[key] - ep.state_by_key[key]


def concat_by_keys(d: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    return np.concatenate([d[k] for k in keys], axis=1).astype(np.float32)


def modality_slices(keys: list[str], arrays_by_key: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    st = 0
    for k in keys:
        dim = int(arrays_by_key[k].shape[1])
        out[k] = {"start": st, "end": st + dim}
        st += dim
    return out


def compute_stats(arr: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def write_mp4_with_ffmpeg(path: Path, frames: np.ndarray, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; install ffmpeg or run export with --no-video")
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be uint8 (T,H,W,3), got {frames.shape} {frames.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    if t == 0:
        raise ValueError("cannot encode empty video")
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {path} (code {proc.returncode}):\n{proc.stderr.decode(errors='replace')}"
        )


def iter_npz_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"no episode_*.npz under {root}")
    return files


def export_dataset(
    episodes: list[Episode],
    out_root: Path,
    *,
    fps: int,
    chunk_size: int,
    write_video: bool,
) -> None:
    import pandas as pd

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta").mkdir(exist_ok=True)
    (out_root / "data").mkdir(exist_ok=True)
    if write_video:
        (out_root / "videos").mkdir(exist_ok=True)

    tasks: list[str] = []
    for ep in episodes:
        if ep.task not in tasks:
            tasks.append(ep.task)
    task_to_idx = {t: i for i, t in enumerate(tasks)}

    episodes_rows: list[dict[str, Any]] = []
    global_index = 0
    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    video_shape: tuple[int, int, int] | None = None

    state_dims = modality_slices(STATE_KEYS, episodes[0].state_by_key)
    action_dims = modality_slices(ACTION_KEYS, episodes[0].action_by_key)

    for epi, ep in enumerate(episodes):
        task_idx = int(task_to_idx[ep.task])
        obs_state = concat_by_keys(ep.state_by_key, STATE_KEYS)
        act = concat_by_keys(ep.action_by_key, ACTION_KEYS)
        all_obs.append(obs_state)
        all_act.append(act)

        t = ep.length
        if t <= 0:
            raise ValueError(f"{ep.npz_path}: empty episode")
        frame_index = np.arange(t, dtype=np.int64)
        done = np.zeros((t,), dtype=bool)
        done[-1] = True

        df = pd.DataFrame(
            {
                "observation.state": [obs_state[i] for i in range(t)],
                "action": [act[i] for i in range(t)],
                "timestamp": ep.timestamps_s.astype(np.float64),
                "episode_index": np.full((t,), epi, dtype=np.int64),
                "frame_index": frame_index,
                "index": np.arange(global_index, global_index + t, dtype=np.int64),
                "task_index": np.full((t,), task_idx, dtype=np.int64),
                "annotation.human.task_description": np.full((t,), task_idx, dtype=np.int64),
                "next.reward": np.zeros((t,), dtype=np.float32),
                "next.done": done,
            }
        )
        global_index += t

        chunk = epi // chunk_size
        parquet_rel = DATA_PATH_PATTERN.format(episode_chunk=chunk, episode_index=epi)
        parquet_path = out_root / parquet_rel
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

        if write_video:
            if ep.video is None:
                raise ValueError(f"{ep.npz_path}: missing ego_view_image but video export requested")
            cur_shape = (int(ep.video.shape[1]), int(ep.video.shape[2]), int(ep.video.shape[3]))
            if video_shape is None:
                video_shape = cur_shape
            elif video_shape != cur_shape:
                raise ValueError(f"inconsistent video shape: {video_shape} vs {cur_shape}")
            video_rel = VIDEO_PATH_PATTERN.format(
                episode_chunk=chunk,
                video_key="observation.images.ego_view",
                episode_index=epi,
            )
            write_mp4_with_ffmpeg(out_root / video_rel, ep.video, fps=fps)

        episodes_rows.append({"episode_index": epi, "tasks": [ep.task], "length": t})

    obs_all = np.concatenate(all_obs, axis=0)
    act_all = np.concatenate(all_act, axis=0)
    stats = {
        "observation.state": compute_stats(obs_all),
        "action": compute_stats(act_all),
    }

    with (out_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_rows:
            f.write(json.dumps(row) + "\n")
    with (out_root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    modality: dict[str, Any] = {
        "state": state_dims,
        "action": action_dims,
        "annotation": {"human.task_description": {"original_key": "annotation.human.task_description"}},
    }
    if write_video:
        modality["video"] = {"ego_view": {"original_key": "observation.images.ego_view"}}
    else:
        modality["video"] = {}
    (out_root / "meta" / "modality.json").write_text(
        json.dumps(modality, indent=2),
        encoding="utf-8",
    )

    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [int(obs_all.shape[1])]},
        "action": {"dtype": "float32", "shape": [int(act_all.shape[1])]},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    if write_video and video_shape is not None:
        features["observation.images.ego_view"] = {
            "dtype": "video",
            "shape": [video_shape[0], video_shape[1], video_shape[2]],
            "names": ["height", "width", "channel"],
        }

    info: dict[str, Any] = {
        "codebase_version": "g1_minimal_sim_npz_export_v1",
        "robot_type": "unitree_g1",
        "fps": int(fps),
        "chunks_size": int(chunk_size),
        "data_path": DATA_PATH_PATTERN,
        "video_path": VIDEO_PATH_PATTERN if write_video else None,
        "features": features,
        "total_episodes": len(episodes_rows),
        "total_frames": int(obs_all.shape[0]),
    }
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _is_parquet_engine_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401

            return True
        except Exception:
            return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recordings_root", type=Path, help="Root containing episode_*.npz files")
    p.add_argument("out_dataset_root", type=Path, help="Output dataset directory")
    p.add_argument("--fps", type=int, default=30, help="Dataset/video fps")
    p.add_argument("--chunk-size", type=int, default=1000, help="LeRobot chunk size")
    p.add_argument(
        "--relative-arms",
        action="store_true",
        help="Convert left/right arm actions to relative deltas against current state arm joints",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MP4 export and omit video modality (state/action-only dataset)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output directory before writing",
    )
    args = p.parse_args()

    if not _is_parquet_engine_available():
        print(
            "error: pandas parquet engine missing (install pyarrow or fastparquet in this env)",
            file=sys.stderr,
        )
        return 2

    in_root = args.recordings_root.resolve()
    out_root = args.out_dataset_root.resolve()
    files = iter_npz_files(in_root)
    if out_root.exists():
        if not args.overwrite:
            print(
                f"error: output exists: {out_root} (pass --overwrite)",
                file=sys.stderr,
            )
            return 3
        shutil.rmtree(out_root)

    episodes = [load_episode(pth) for pth in files]
    if args.relative_arms:
        for ep in episodes:
            apply_relative_arm_actions(ep)
    write_video = not bool(args.no_video)
    export_dataset(
        episodes,
        out_root,
        fps=int(args.fps),
        chunk_size=int(args.chunk_size),
        write_video=write_video,
    )
    print(
        f"exported {len(episodes)} episodes -> {out_root} "
        f"(video={'on' if write_video else 'off'}, relative_arms={'on' if args.relative_arms else 'off'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
