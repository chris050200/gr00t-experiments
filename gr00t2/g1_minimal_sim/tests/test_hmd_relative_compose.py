"""Torso compose + world→torso round-trip for hmd_relative UDP semantics."""

import numpy as np

from ee_frame import (
    ee_pose_ref_from_world,
    ee_pose_world_from_ref,
    normalize_quat,
    quat_mul,
)


def test_hmd_relative_identity_torso_matches_ref_frame() -> None:
    xpos = np.zeros(3, dtype=np.float64)
    xmat = np.eye(3, dtype=np.float64)
    qb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    p_in = np.array([0.1, -0.2, 0.35], dtype=np.float64)
    q_in = normalize_quat(np.array([0.924, 0.0, 0.383, 0.0], dtype=np.float64))
    p_w = xpos + xmat @ p_in
    q_w = normalize_quat(quat_mul(qb, q_in))
    p_b, q_b = ee_pose_ref_from_world(xpos, xmat, qb, p_w, q_w)
    assert np.allclose(p_b, p_in, atol=1e-6)
    assert np.allclose(q_b, q_in, atol=1e-5) or np.allclose(q_b, -q_in, atol=1e-5)


def test_hmd_relative_world_ref_roundtrip() -> None:
    """UDP ``hmd_relative`` path: world_from_ref then ref_from_world → original torso target."""
    xpos = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    c = np.sqrt(0.5)
    qb = np.array([c, 0.0, 0.0, c], dtype=np.float64)
    xmat = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    p_in = np.array([0.1, -0.2, 0.35], dtype=np.float64)
    q_in = normalize_quat(np.array([0.924, 0.0, 0.383, 0.0], dtype=np.float64))
    p_w, q_w = ee_pose_world_from_ref(xpos, xmat, qb, p_in, q_in)
    p_b, q_b = ee_pose_ref_from_world(xpos, xmat, qb, p_w, q_w)
    assert np.allclose(p_b, p_in, atol=1e-6)
    assert np.allclose(q_b, q_in, atol=1e-5) or np.allclose(q_b, -q_in, atol=1e-5)


def test_hmd_relative_offset_torso() -> None:
    xpos = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    # 90 deg about Z
    c = np.sqrt(0.5)
    qb = np.array([c, 0.0, 0.0, c], dtype=np.float64)
    xmat = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    p_in = np.array([0.0, 0.2, 0.0], dtype=np.float64)
    q_in = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    p_w = xpos + xmat @ p_in
    q_w = normalize_quat(quat_mul(qb, q_in))
    p_b, q_b = ee_pose_ref_from_world(xpos, xmat, qb, p_w, q_w)
    assert np.allclose(p_b, p_in, atol=1e-6)
    assert np.allclose(q_b, q_in, atol=1e-6)
