"""Serialize policy I/O for A/B debugging (oracle rollout vs minimal-sim client).

Writes ``manifest.json`` plus ``obs/*.npy`` and ``action/*.npy`` under a directory.
Language / tuple fields are recorded in the manifest only (not as .npy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return str(x)


def _strip_batch0(x: np.ndarray) -> np.ndarray:
    """Drop leading batch dim when it is 1 (vector-env or client convention)."""
    if x.ndim >= 1 and x.shape[0] == 1:
        return np.asarray(x[0])
    return x


def _obs_filename(key: str) -> str:
    safe = key.replace("/", "_").replace(".", "_").replace(" ", "_")
    if not safe.endswith(".npy"):
        safe += ".npy"
    return safe


def _action_filename(key: str) -> str:
    safe = key.replace("/", "_").replace(".", "_").replace(" ", "_")
    if not safe.endswith(".npy"):
        safe += ".npy"
    return safe


def _maybe_save_ego_preview(obs_dir: Path, policy_obs: dict[str, Any]) -> None:
    """Save first timestep of ``video.ego_view`` as PNG when present."""
    key = "video.ego_view"
    if key not in policy_obs:
        return
    val = policy_obs[key]
    if not isinstance(val, np.ndarray):
        return
    arr = _strip_batch0(val)
    if arr.ndim == 4:
        frame = np.asarray(arr[0], dtype=np.uint8)
    elif arr.ndim == 3:
        frame = np.asarray(arr, dtype=np.uint8)
    else:
        return
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(obs_dir / "video_ego_view_t0.png"), frame)
    except Exception:
        pass


def dump_policy_roundtrip_pack(
    out_dir: Path | str,
    *,
    side: str,
    policy_obs: dict[str, Any],
    actions: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one policy request/response cycle to ``out_dir`` for diffing.

    Parameters
    ----------
    out_dir
        Directory to create (parents created too).
    side
        Short label, e.g. ``"rollout_policy"`` or ``"minimal_sim"``.
    policy_obs
        Flat observation dict passed to ``get_action`` (video/state/language keys).
    actions
        Flat action dict returned from ``get_action`` (may be ``None`` to dump obs-only).
    extra
        Arbitrary JSON-serializable metadata (scene, step, arm_mode, …).
    """
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    obs_dir = root / "obs"
    act_dir = root / "action"
    obs_dir.mkdir(exist_ok=True)
    if actions:
        act_dir.mkdir(exist_ok=True)

    manifest: dict[str, Any] = {
        "side": side,
        "extra": _json_safe(extra or {}),
        "obs_keys": [],
        "action_keys": [],
    }

    for key, val in sorted(policy_obs.items()):
        entry: dict[str, Any] = {"key": key}
        if isinstance(val, np.ndarray):
            arr = _strip_batch0(val)
            path = obs_dir / _obs_filename(key)
            np.save(path, arr, allow_pickle=False)
            entry["dtype"] = str(arr.dtype)
            entry["shape"] = list(arr.shape)
            entry["file"] = str(path.relative_to(root))
        elif isinstance(val, (tuple, list)) and (
            len(val) == 0 or isinstance(val[0], str)
        ):
            entry["strings"] = [str(s) for s in val]
        else:
            entry["repr"] = repr(val)[:500]
        manifest["obs_keys"].append(entry)

    if actions:
        for key, val in sorted(actions.items()):
            entry = {"key": key}
            if isinstance(val, np.ndarray):
                arr = _strip_batch0(val)
                path = act_dir / _action_filename(key)
                np.save(path, arr, allow_pickle=False)
                entry["dtype"] = str(arr.dtype)
                entry["shape"] = list(arr.shape)
                entry["file"] = str(path.relative_to(root))
            else:
                entry["repr"] = repr(val)[:500]
            manifest["action_keys"].append(entry)

    _maybe_save_ego_preview(obs_dir, policy_obs)

    man_path = root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root
