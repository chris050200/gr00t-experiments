"""Save/load Tier-A ``policy.get_action`` action chunks for Variant A oracle replay.

``rollout_policy.py`` writes ``replan_NNNNN.npz`` files; ``run_gr00t_stylish_diner_inference.py``
loads them with ``--replay-oracle-action-chunks``.  Action dict keys use dots
(``action.left_arm``); ``.`` is escaped in NPZ array names so ``numpy.savez`` works.

This module is imported from both ``g1_minimal_sim`` and ``Isaac-GR00T`` (via ``sys.path``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_DOT_ESC = "__DOT__"


def encode_action_npz_key(action_key: str) -> str:
    return action_key.replace(".", _DOT_ESC)


def decode_action_npz_key(file_key: str) -> str:
    return file_key.replace(_DOT_ESC, ".")


def save_replan_chunk(out_dir: Path, replan_idx: int, actions: dict[str, Any]) -> dict[str, Any]:
    """Write one ``replan_{idx:05d}.npz`` with float32 (T,D) arrays; return chunk metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for k in sorted(actions.keys()):
        arr = np.asarray(actions[k])
        if arr.ndim == 3:
            if int(arr.shape[0]) != 1:
                raise ValueError(
                    "oracle action chunks: expected batch size 1 in action tensors "
                    f"(use --n_envs 1 for recording). Got shape {arr.shape} for {k!r}."
                )
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(
                f"oracle action chunks: expected (T,D) for {k!r} after batch strip, got {arr.shape}"
            )
        payload[encode_action_npz_key(k)] = np.asarray(arr, dtype=np.float32)
    path = out_dir / f"replan_{replan_idx:05d}.npz"
    if payload:
        np.savez_compressed(path, **payload)
        t0 = int(next(iter(payload.values())).shape[0])
    else:
        t0 = 0
    return {
        "file": path.name,
        "action_keys": list(actions.keys()),
        "T": t0,
    }


def load_replan_chunk(path: Path) -> dict[str, np.ndarray]:
    """Load ``replan_*.npz`` into ``action.*`` keys (float32 (T,D))."""
    z = np.load(path)
    return {decode_action_npz_key(str(k)): np.asarray(z[k], dtype=np.float32) for k in z.files}


def infer_horizons_from_vector_obs(obs: dict[str, Any]) -> tuple[int, int, list[str], list[str]]:
    """Read state/video temporal horizons from batched vector-env observations."""
    state_keys = sorted(k for k in obs if str(k).startswith("state."))
    video_keys = sorted(k for k in obs if str(k).startswith("video."))
    if not state_keys:
        raise ValueError("cannot infer state horizon: no state.* keys in observations")
    s0 = np.asarray(obs[state_keys[0]])
    if s0.ndim < 3:
        raise ValueError(
            f"expected batched state (B,T,D) for {state_keys[0]!r}, got shape {s0.shape}"
        )
    state_h = int(s0.shape[1])
    video_h = state_h
    if video_keys:
        v0 = np.asarray(obs[video_keys[0]])
        if v0.ndim >= 3:
            video_h = int(v0.shape[1])
    return state_h, video_h, state_keys, video_keys


def write_oracle_manifest(
    out_dir: Path,
    *,
    env_name: str,
    first_observations: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> None:
    """Write ``manifest.json`` next to ``replan_*.npz`` files."""
    sh, vh, sk, vk = infer_horizons_from_vector_obs(first_observations)
    manifest: dict[str, Any] = {
        "version": 1,
        "env_name": str(env_name),
        "state_horizon": sh,
        "video_horizon": vh,
        "state_keys": sk,
        "video_keys": vk,
        "n_chunks": len(chunks),
        "chunks": chunks,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def sorted_replan_paths(root: Path) -> list[Path]:
    """Return ``replan_*.npz`` sorted by numeric index."""
    paths = list(root.glob("replan_*.npz"))

    def _key(p: Path) -> int:
        stem = p.stem  # replan_00000
        if not stem.startswith("replan_"):
            return -1
        return int(stem.split("_", 1)[1])

    return sorted(paths, key=_key)
