"""HMD → mocap pose helper (legacy pose-stream POV preview)."""

from __future__ import annotations

import numpy as np
import mujoco

from legacy_vr_code.vr_teleop_test_streaming import _pov_pose_for_mocap


def test_pov_pose_world_frame_identity():
    p = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    pp, qq = _pov_pose_for_mocap(
        p, q, np.zeros(3), 0.0, True, np.zeros(3, dtype=np.float64)
    )
    assert np.allclose(pp, p)
    assert np.allclose(qq, q, atol=1e-6)


def test_pov_pose_eye_offset_world():
    p = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    off = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    pp, qq = _pov_pose_for_mocap(p, q, np.zeros(3), 0.0, True, off)
    assert np.allclose(pp, off)
    assert np.allclose(qq, q, atol=1e-6)


def test_pov_pose_root_translation_only():
    """Pure translation of root: HMD world at t; root-relative position should be origin."""
    hmd = np.array([2.0, 1.0, 1.5], dtype=np.float64)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    t_root = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    pp, qq = _pov_pose_for_mocap(hmd, q, t_root, 0.0, False, np.zeros(3))
    assert np.allclose(pp, [0.0, 0.0, 1.5], atol=1e-6)
    assert np.allclose(qq, q, atol=1e-6)
