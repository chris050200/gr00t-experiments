"""Tests for ``policy_ab_dump`` (policy I/O serialization for A/B)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy_ab_dump import dump_policy_roundtrip_pack


def test_dump_policy_roundtrip_pack_writes_manifest_and_arrays(tmp_path: Path) -> None:
    obs = {
        "video.ego_view": np.zeros((1, 1, 4, 6, 3), dtype=np.uint8),
        "state.left_arm": np.ones((1, 1, 7), dtype=np.float32),
        "annotation.human.task_description": ("hello world",),
    }
    actions = {"action.left_arm": np.linspace(0, 1, 30 * 7, dtype=np.float32).reshape(1, 30, 7)}
    out = dump_policy_roundtrip_pack(
        tmp_path / "pack",
        side="test",
        policy_obs=obs,
        actions=actions,
        extra={"k": 1},
    )
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["side"] == "test"
    assert man["extra"] == {"k": 1}
    la = np.load(out / "obs" / "state_left_arm.npy", allow_pickle=False)
    assert la.shape == (1, 7)
    aa = np.load(out / "action" / "action_left_arm.npy", allow_pickle=False)
    assert aa.shape == (30, 7)
    assert (out / "obs" / "video_ego_view_t0.png").is_file()
