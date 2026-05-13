"""Smoke tests for dummy collection/export pipeline scripts."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np


def _load_script_module(name: str, rel: str):
    import sys

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_stub_npz(path: Path, *, with_video: bool) -> None:
    t = 4
    task = "dummy task"
    payload: dict[str, np.ndarray] = {
        "T": np.int32(t),
        "step_index": np.arange(t, dtype=np.int32),
        "wall_time_s": np.linspace(0.0, 0.03 * (t - 1), t, dtype=np.float64),
        "annotation_human_task_description_bytes": np.frombuffer(
            task.encode("utf-8"), dtype=np.uint8
        ),
        "state_left_leg": np.zeros((t, 6), dtype=np.float32),
        "state_right_leg": np.zeros((t, 6), dtype=np.float32),
        "state_waist": np.zeros((t, 3), dtype=np.float32),
        "state_left_arm": np.ones((t, 7), dtype=np.float32),
        "state_right_arm": np.ones((t, 7), dtype=np.float32) * 2.0,
        "state_left_hand": np.zeros((t, 7), dtype=np.float32),
        "state_right_hand": np.zeros((t, 7), dtype=np.float32),
        "action_left_arm": np.ones((t, 7), dtype=np.float32) * 1.5,
        "action_right_arm": np.ones((t, 7), dtype=np.float32) * 2.5,
        "action_left_hand": np.zeros((t, 7), dtype=np.float32),
        "action_right_hand": np.zeros((t, 7), dtype=np.float32),
        "action_waist": np.zeros((t, 3), dtype=np.float32),
        "action_base_height_command": np.ones((t, 1), dtype=np.float32) * 0.74,
        "action_navigate_command": np.zeros((t, 3), dtype=np.float32),
    }
    if with_video:
        payload["ego_view_image"] = np.zeros((t, 16, 24, 3), dtype=np.uint8)
    np.savez_compressed(path, **payload)


def test_qc_checker_accepts_stub() -> None:
    mod = _load_script_module("check_gr00t_npz_dataset", "check_gr00t_npz_dataset.py")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "episode_000.npz"
        _write_stub_npz(p, with_video=True)
        steps, has_video = mod._check_episode(p, require_video=True)
        assert steps == 4
        assert has_video is True


def test_export_loader_and_relative_arm_conversion() -> None:
    mod = _load_script_module("export_gr00t_npz_to_lerobot", "export_gr00t_npz_to_lerobot.py")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "episode_000.npz"
        _write_stub_npz(p, with_video=False)
        ep = mod.load_episode(p)
        assert ep.length == 4
        assert ep.action_by_key["left_arm"].shape == (4, 7)
        mod.apply_relative_arm_actions(ep)
        # left arm action 1.5 - state 1.0
        assert np.allclose(ep.action_by_key["left_arm"], 0.5)
        # right arm action 2.5 - state 2.0
        assert np.allclose(ep.action_by_key["right_arm"], 0.5)
