"""Unit tests for arm IK per-step limiting (numpy only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("mujoco")

from gear_wbc_arm_ik import _limit_arm_ik_step  # noqa: E402


def test_limit_arm_ik_step_per_joint_clip() -> None:
    q_cur = np.zeros(14, dtype=np.float64)
    q_sol = np.ones(14, dtype=np.float64) * 0.5
    q_out, d = _limit_arm_ik_step(
        q_cur, q_sol, dq_max=0.1, l2_cap=0.0, step_blend=1.0
    )
    assert np.allclose(q_out, 0.1 * np.ones(14), atol=1e-6)
    assert d["ik_arm_delta_raw_l2"] > 1.0
    assert abs(d["ik_arm_delta_out_l2"] - 0.1 * np.sqrt(14)) < 1e-5


def test_limit_arm_ik_step_l2_cap_after_clip() -> None:
    q_cur = np.zeros(14, dtype=np.float64)
    q_sol = np.ones(14, dtype=np.float64) * 0.2
    q_out, d = _limit_arm_ik_step(
        q_cur, q_sol, dq_max=0.2, l2_cap=0.5, step_blend=1.0
    )
    assert float(np.linalg.norm(q_out - q_cur)) <= 0.500001
    assert d["ik_arm_delta_out_l2"] <= 0.500001


def test_limit_arm_ik_step_blend() -> None:
    q_cur = np.zeros(14, dtype=np.float64)
    q_sol = np.ones(14, dtype=np.float64) * 0.1
    q_out, _ = _limit_arm_ik_step(
        q_cur, q_sol, dq_max=1.0, l2_cap=0.0, step_blend=0.5
    )
    assert np.allclose(q_out, 0.05 * np.ones(14), atol=1e-7)
