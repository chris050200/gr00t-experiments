"""Unit tests for ``oracle_action_chunk_io`` (Variant A NPZ chunk I/O)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import oracle_action_chunk_io as oio


def test_encode_decode_action_npz_key_roundtrip() -> None:
    k = "action.left_arm"
    enc = oio.encode_action_npz_key(k)
    assert "." not in enc
    assert oio.decode_action_npz_key(enc) == k


def test_save_load_replan_chunk_roundtrip(tmp_path: Path) -> None:
    T, d_la, d_nav = 4, 7, 3
    actions = {
        "action.left_arm": np.random.randn(1, T, d_la).astype(np.float32),
        "action.navigate_command": np.random.randn(1, T, d_nav).astype(np.float32),
    }
    meta = oio.save_replan_chunk(tmp_path, 3, actions)
    assert meta["T"] == T
    assert meta["file"] == "replan_00003.npz"
    loaded = oio.load_replan_chunk(tmp_path / meta["file"])
    assert set(loaded.keys()) == set(actions.keys())
    np.testing.assert_allclose(
        loaded["action.left_arm"], actions["action.left_arm"][0], rtol=0, atol=1e-5
    )


def test_save_replan_chunk_rejects_batch_not_one(tmp_path: Path) -> None:
    bad = {"action.left_arm": np.zeros((2, 3, 7), dtype=np.float32)}
    with pytest.raises(ValueError, match="batch size 1"):
        oio.save_replan_chunk(tmp_path, 0, bad)


def test_sorted_replan_paths_order(tmp_path: Path) -> None:
    (tmp_path / "replan_00010.npz").write_bytes(b"")
    (tmp_path / "replan_00002.npz").write_bytes(b"")
    (tmp_path / "replan_00000.npz").write_bytes(b"")
    (tmp_path / "other.npz").write_bytes(b"")
    paths = oio.sorted_replan_paths(tmp_path)
    assert [p.name for p in paths] == ["replan_00000.npz", "replan_00002.npz", "replan_00010.npz"]


def test_write_oracle_manifest(tmp_path: Path) -> None:
    B, H, D = 2, 5, 3
    first_obs = {
        "state.foo": np.zeros((B, H, D), dtype=np.float32),
        "video.bar": np.zeros((B, H - 1, 2, 2, 3), dtype=np.uint8),
    }
    chunks = [{"file": "replan_00000.npz", "T": 30, "action_keys": ["action.left_arm"]}]
    oio.write_oracle_manifest(
        tmp_path, env_name="test_env", first_observations=first_obs, chunks=chunks
    )
    mpath = tmp_path / "manifest.json"
    assert mpath.is_file()
    data = json.loads(mpath.read_text(encoding="utf-8"))
    assert data["env_name"] == "test_env"
    assert data["state_horizon"] == H
    assert data["video_horizon"] == H - 1
    assert data["n_chunks"] == 1
    assert data["chunks"] == chunks
