"""Smoke: ``scripts/preview_gr00t_npz.py`` summary + optional plot (matplotlib)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_preview_module():
    path = _ROOT / "scripts" / "preview_gr00t_npz.py"
    spec = importlib.util.spec_from_file_location("preview_gr00t_npz", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_npz(path: Path) -> None:
    """Schema-compatible stub (matches :meth:`gr00t_teleop_logger.Gr00tTeleopEpisodeLogger.save_npz`)."""
    T = 5
    task = "pytest stub episode"
    np.savez_compressed(
        path,
        T=np.int32(T),
        episode_index=np.int32(0),
        step_index=np.arange(T, dtype=np.int32),
        wall_time_s=np.linspace(0.0, 0.2 * (T - 1), T),
        action_navigate_command=np.zeros((T, 3), dtype=np.float32),
        action_base_height_command=np.ones((T, 1), dtype=np.float32) * 0.74,
        action_waist=np.zeros((T, 3), dtype=np.float32),
        action_left_arm=np.zeros((T, 7), dtype=np.float32),
        action_right_arm=np.zeros((T, 7), dtype=np.float32),
        action_left_hand=np.zeros((T, 7), dtype=np.float32),
        action_right_hand=np.zeros((T, 7), dtype=np.float32),
        state_left_arm=np.zeros((T, 7), dtype=np.float64),
        state_right_arm=np.zeros((T, 7), dtype=np.float64),
        annotation_human_task_description_bytes=np.frombuffer(
            task.encode("utf-8"), dtype=np.uint8
        ),
    )


def test_summarize_npz_stub() -> None:
    mod = _load_preview_module()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "episode_000.npz"
        _minimal_npz(p)
        text = mod.summarize_npz(p)
    assert "T (steps): 5" in text
    assert "pytest stub episode" in text
    assert "action_navigate_command" in text


def test_plot_npz_writes_pngs() -> None:
    pytest.importorskip("matplotlib")
    mod = _load_preview_module()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        p = td_path / "episode_000.npz"
        _minimal_npz(p)
        # optional video key for third plot
        z = dict(np.load(p, allow_pickle=False))
        T = int(z["T"])
        z["ego_view_image"] = np.zeros((T, 32, 48, 3), dtype=np.uint8)
        z["ego_view_image"][:, :, :, 0] = np.arange(T, dtype=np.uint8)[:, None, None]
        np.savez_compressed(p, **{k: z[k] for k in z})
        out = td_path / "plots"
        written = mod.plot_npz(p, out)
        assert len(written) >= 2
        assert all(x.suffix == ".png" for x in written)
        assert all(x.is_file() for x in written)
