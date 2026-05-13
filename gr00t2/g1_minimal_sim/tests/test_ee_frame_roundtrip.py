"""Round-trip ref-body ↔ world EE pose helpers (no MuJoCo scene)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ee_frame  # noqa: E402
from vr_teleop.vr_stream_torso_compose import (  # noqa: E402
    torso_xmat_with_yaw_offset,
    torso_xquat_with_receiver_yaw,
)


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.standard_normal(4)
    return ee_frame.normalize_quat(q)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = ee_frame.normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def test_quat_wxyz_to_rotmat_matches_manual_formula():
    rng = np.random.default_rng(4)
    for _ in range(20):
        q = _random_unit_quat(rng)
        R_man = _quat_to_R(q)
        R_fn = ee_frame.quat_wxyz_to_rotmat(q)
        np.testing.assert_allclose(R_fn, R_man, atol=1e-14, rtol=1e-14)


def test_pose_roundtrip_numeric():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q_ref = _random_unit_quat(rng)
        R_ref = _quat_to_R(q_ref)
        xpos = rng.standard_normal(3) * 0.5
        p_w = rng.standard_normal(3) * 0.3
        q_w = _random_unit_quat(rng)

        p_b, q_b = ee_frame.ee_pose_ref_from_world(xpos, R_ref, q_ref, p_w, q_w)
        p_w2, q_w2 = ee_frame.ee_pose_world_from_ref(xpos, R_ref, q_ref, p_b, q_b)

        np.testing.assert_allclose(p_w2, p_w, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            _quat_to_R(q_w2), _quat_to_R(q_w), atol=1e-5, rtol=1e-5
        )


def test_compose_matches_quat_mul():
    rng = np.random.default_rng(1)
    q_ref = _random_unit_quat(rng)
    R_ref = _quat_to_R(q_ref)
    q_ee = _random_unit_quat(rng)
    xpos = np.zeros(3)
    p_ref = rng.standard_normal(3) * 0.2
    p_w, q_w = ee_frame.ee_pose_world_from_ref(xpos, R_ref, q_ref, p_ref, q_ee)
    q_exp = ee_frame.quat_mul(q_ref, q_ee)
    np.testing.assert_allclose(
        _quat_to_R(q_w), _quat_to_R(q_exp), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(p_w, R_ref @ p_ref, atol=1e-6)


def test_intrinsic_zyx_zero_is_identity():
    q = ee_frame.intrinsic_zyx_deg_to_quat_wxyz(0.0, 0.0, 0.0)
    np.testing.assert_allclose(q, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-9)


def test_intrinsic_zyx_unit_quaternion():
    q = ee_frame.intrinsic_zyx_deg_to_quat_wxyz(-90.0, 180.0, 12.0)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-9


def test_vr_torso_yaw_offset_ref_roundtrip():
    """Encode/decode must use the same R basis as q_ref (VR receiver yaw)."""
    rng = np.random.default_rng(42)
    for yaw_deg in (-37.0, 0.0, 55.5):
        for _ in range(20):
            q_torso = _random_unit_quat(rng)
            R_torso = _quat_to_R(q_torso)
            xpos = rng.standard_normal(3) * 0.4
            R_vr = torso_xmat_with_yaw_offset(R_torso, yaw_deg)
            q_ref = torso_xquat_with_receiver_yaw(R_torso, q_torso, yaw_deg)
            p_w = rng.standard_normal(3) * 0.25
            q_w = _random_unit_quat(rng)
            p_b, q_b = ee_frame.ee_pose_ref_from_world(xpos, R_vr, q_ref, p_w, q_w)
            p_w2, q_w2 = ee_frame.ee_pose_world_from_ref(xpos, R_vr, q_ref, p_b, q_b)
            np.testing.assert_allclose(p_w2, p_w, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(
                _quat_to_R(q_w2), _quat_to_R(q_w), atol=1e-5, rtol=1e-5
            )
