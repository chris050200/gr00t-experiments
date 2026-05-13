"""Tests for hybrid policy_obs construction from ``policy_ab_dump`` pairs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from policy_ab_compare import (
    build_policy_obs_from_ab_dumps,
    max_abs_action_diff,
    tagged_obs_to_batched_policy_obs,
)


def _fake_modality() -> dict[str, SimpleNamespace]:
    return {
        "video": SimpleNamespace(modality_keys=["ego_view", "tpp_view"]),
        "state": SimpleNamespace(modality_keys=["left_arm"]),
        "language": SimpleNamespace(modality_keys=["annotation.human.task_description"]),
    }


def test_tagged_obs_to_batched_policy_obs_adds_batch_dim() -> None:
    tagged = {
        "state.left_arm": ("array", np.zeros((2, 7), dtype=np.float32)),
        "video.ego_view": ("array", np.zeros((2, 4, 4, 3), dtype=np.uint8)),
        "annotation.human.task_description": ("strings", ("task a",)),
    }
    batched = tagged_obs_to_batched_policy_obs(tagged)
    assert batched["state.left_arm"].shape == (1, 2, 7)
    assert batched["video.ego_view"].shape == (1, 2, 4, 4, 3)
    assert batched["annotation.human.task_description"] == ("task a",)


def test_build_hybrid_ego_swaps_only_ego() -> None:
    modality = _fake_modality()
    obs_min = {
        "state.left_arm": ("array", np.ones((1, 7), dtype=np.float32) * 3.0),
        "video.ego_view": ("array", np.zeros((1, 2, 2, 3), dtype=np.uint8)),
        "video.tpp_view": ("array", np.full((1, 2, 2, 3), 7, dtype=np.uint8)),
        "annotation.human.task_description": ("strings", ("from_min",)),
    }
    obs_roll = {
        "state.left_arm": ("array", np.ones((1, 7), dtype=np.float32) * 9.0),
        "video.ego_view": ("array", np.full((1, 2, 2, 3), 200, dtype=np.uint8)),
        "video.tpp_view": ("array", np.full((1, 2, 2, 3), 50, dtype=np.uint8)),
        "annotation.human.task_description": ("strings", ("from_roll",)),
    }
    h = build_policy_obs_from_ab_dumps(modality, obs_min, obs_roll, "hybrid_ego")
    assert h["state.left_arm"][1][0, 0] == 3.0
    assert h["annotation.human.task_description"] == (
        "strings",
        ("from_min",),
    )
    assert h["video.ego_view"][1][0, 0, 0, 0] == 200
    assert h["video.tpp_view"][1][0, 0, 0, 0] == 7


def test_build_hybrid_ego_falls_back_rollout_for_tpp_when_minimal_missing() -> None:
    modality = _fake_modality()
    obs_min = {
        "state.left_arm": ("array", np.zeros((1, 7), dtype=np.float32)),
        "video.ego_view": ("array", np.zeros((1, 2, 2, 3), dtype=np.uint8)),
        "annotation.human.task_description": ("strings", ("t",)),
    }
    obs_roll = {
        "state.left_arm": ("array", np.zeros((1, 7), dtype=np.float32)),
        "video.ego_view": ("array", np.zeros((1, 2, 2, 3), dtype=np.uint8)),
        "video.tpp_view": ("array", np.full((1, 2, 2, 3), 42, dtype=np.uint8)),
        "annotation.human.task_description": ("strings", ("t",)),
    }
    h = build_policy_obs_from_ab_dumps(modality, obs_min, obs_roll, "hybrid_ego")
    assert h["video.tpp_view"][1][0, 0, 0, 0] == 42


def test_max_abs_action_diff() -> None:
    ref = {"action.a": np.array([[0.0, 2.0]], dtype=np.float32)}
    got = {"action.a": np.array([[0.0, 2.05]], dtype=np.float32)}
    d = max_abs_action_diff(ref, got)
    assert abs(d["action.a"] - 0.05) < 1e-6
