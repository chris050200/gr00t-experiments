"""Tests for ``policy_obs_audit``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from policy_ab_dump import dump_policy_roundtrip_pack
from policy_obs_audit import (
    classify_rollout_only_obs_key,
    rollout_only_keys,
)


def test_classify_q() -> None:
    cat, _ = classify_rollout_only_obs_key("q")
    assert cat == "vecenv_lowlevel"


def test_classify_ego_view_image() -> None:
    cat, _ = classify_rollout_only_obs_key("ego_view_image")
    assert cat == "parallel_image_alias"


def test_classify_tpp_view() -> None:
    cat, _ = classify_rollout_only_obs_key("video.tpp_view")
    assert cat == "optional_third_person_video"


def test_rollout_only_keys_from_manifests(tmp_path: Path) -> None:
    def _pack(root: Path, *, with_tpp: bool) -> None:
        obs: dict[str, np.ndarray] = {
            "state.left_arm": np.zeros((1, 1, 7), dtype=np.float32),
            "video.ego_view": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8),
        }
        if with_tpp:
            obs["video.tpp_view"] = np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)
        obs["annotation.human.task_description"] = ("x",)
        dump_policy_roundtrip_pack(
            root,
            side="t",
            policy_obs=obs,
            actions={"action.left_arm": np.zeros((1, 2, 7), dtype=np.float32)},
            extra={},
        )

    ro = tmp_path / "ro"
    mi = tmp_path / "mi"
    _pack(ro, with_tpp=True)
    _pack(mi, with_tpp=False)
    only = rollout_only_keys(ro, mi)
    assert "video.tpp_view" in only
    assert "state.left_arm" not in only
