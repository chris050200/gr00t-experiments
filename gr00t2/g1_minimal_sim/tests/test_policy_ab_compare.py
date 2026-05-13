"""Tests for ``policy_ab_compare.load_pack`` / ``diff_pack_roots``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from policy_ab_compare import diff_pack_roots, load_pack  # noqa: E402
from policy_ab_dump import dump_policy_roundtrip_pack  # noqa: E402


def _tiny_obs(
    *, t: int = 2, h: int = 64, w: int = 64, c: int = 3, state_val: float = 0.1
) -> dict[str, np.ndarray]:
    return {
        "video.ego_view": np.full((1, t, h, w, c), 42, dtype=np.uint8),
        "state.left_arm": np.full((1, t, 7), state_val, dtype=np.float32),
    }


def _tiny_actions() -> dict[str, np.ndarray]:
    return {"action.left_arm": np.zeros((1, 4, 7), dtype=np.float32)}


def test_load_pack_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    dump_policy_roundtrip_pack(
        root,
        side="test",
        policy_obs=_tiny_obs(),
        actions=_tiny_actions(),
        extra={"k": 1},
    )
    _, obs, act = load_pack(root)
    assert "video.ego_view" in obs
    _, vid = obs["video.ego_view"]
    assert vid.shape == (2, 64, 64, 3)
    assert "action.left_arm" in act


def test_diff_pack_roots_never_exits_zero(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    dump_policy_roundtrip_pack(
        a,
        side="a",
        policy_obs=_tiny_obs(h=8, w=8, state_val=0.5),
        actions=_tiny_actions(),
        extra={},
    )
    dump_policy_roundtrip_pack(
        b,
        side="b",
        policy_obs=_tiny_obs(h=8, w=8, state_val=0.9),
        actions=_tiny_actions(),
        extra={},
    )
    rc = diff_pack_roots(a, b, label_a="a", label_b="b", exit_on="never")
    assert rc == 0


def test_diff_pack_roots_strict_fails_on_state(tmp_path: Path) -> None:
    a = tmp_path / "a2"
    b = tmp_path / "b2"
    dump_policy_roundtrip_pack(
        a,
        side="a",
        policy_obs=_tiny_obs(h=4, w=4, state_val=0.0),
        actions=_tiny_actions(),
        extra={},
    )
    dump_policy_roundtrip_pack(
        b,
        side="b",
        policy_obs=_tiny_obs(h=4, w=4, state_val=1.0),
        actions=_tiny_actions(),
        extra={},
    )
    rc = diff_pack_roots(
        a, b, label_a="a", label_b="b", state_atol=1e-6, exit_on="strict"
    )
    assert rc == 1
