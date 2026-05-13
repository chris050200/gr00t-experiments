"""Unit tests for legacy OpenVR→MuJoCo mapping (numpy only, no OpenVR)."""

from __future__ import annotations

import numpy as np

from legacy_vr_code.vr_steamvr_mujoco_mapping import (
    R_STEAMVR_LINEAR_TO_MJ,
    T_STEAMVR_TO_MJ_QUAT_WXYZ,
    openvr_mat34_to_Rt,
    openvr_pose_to_mujoco,
    openvr_pose_to_mujoco_custom,
    quat_from_matrix_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    steamvr_linear_to_mujoco_pos,
    steamvr_rotmat_to_mujoco_quat_wxyz,
)


def test_linear_map_matches_minus_z_minus_x_y() -> None:
    v = np.array([1.0, 2.0, 3.0])
    got = steamvr_linear_to_mujoco_pos(v)
    assert np.allclose(got, [-3.0, -1.0, 2.0])
    assert np.allclose(R_STEAMVR_LINEAR_TO_MJ @ v, got)


def test_linear_map_equivalent_to_matrix() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = rng.standard_normal(3)
        assert np.allclose(steamvr_linear_to_mujoco_pos(t), R_STEAMVR_LINEAR_TO_MJ @ t)


def test_T_matches_rotation_matrix_quaternion() -> None:
    """Fixed T should match quat(R_linear) up to sign (double cover)."""
    q_R = quat_from_matrix_wxyz(R_STEAMVR_LINEAR_TO_MJ)
    dot = abs(float(np.dot(q_R, T_STEAMVR_TO_MJ_QUAT_WXYZ)))
    assert dot > 1.0 - 1e-6


def test_identity_openvr_pose() -> None:
    R = np.eye(3)
    t = np.zeros(3)
    pos, quat = steamvr_linear_to_mujoco_pos(t), steamvr_rotmat_to_mujoco_quat_wxyz(R)
    assert np.allclose(pos, 0.0)
    # q_vr = identity → q_mj = T
    assert abs(float(np.dot(quat_normalize_wxyz(quat), T_STEAMVR_TO_MJ_QUAT_WXYZ))) > 1.0 - 1e-6


def test_steamvr_rotmat_matches_matrix_compose() -> None:
    """``steamvr_rotmat_to_mujoco_quat_wxyz`` == ``quat_from_matrix(R_lin @ R_ovr)``."""
    # Small fixed orthogonal set (no random QR — avoids flaky OpenMP/MKL in some CI sandboxes).
    mats = [
        np.eye(3),
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    for R_ovr in mats:
        R_mj = R_STEAMVR_LINEAR_TO_MJ @ R_ovr
        q_exp = quat_from_matrix_wxyz(R_mj)
        q_got = steamvr_rotmat_to_mujoco_quat_wxyz(R_ovr)
        dot = abs(float(np.dot(q_exp, q_got)))
        assert dot > 1.0 - 1e-5, f"quat mismatch dot={dot}"


def test_openvr_mat34_to_Rt_and_pose() -> None:
    class FakeM:
        m = [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 5.0],
        ]

    R, t = openvr_mat34_to_Rt(FakeM())
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, [2.0, 3.0, 5.0])
    pos, quat = openvr_pose_to_mujoco(FakeM())
    pos_c, quat_c = openvr_pose_to_mujoco_custom(FakeM(), R_STEAMVR_LINEAR_TO_MJ)
    assert np.allclose(pos, [-5.0, -2.0, 3.0])
    assert np.allclose(pos_c, pos)
    assert abs(float(np.dot(quat, quat_c))) > 1.0 - 1e-6
    assert abs(float(np.dot(quat, T_STEAMVR_TO_MJ_QUAT_WXYZ))) > 1.0 - 1e-6


def test_quat_mul_matches_matrix_compose_fixed() -> None:
    """Hamilton product matches matrix multiply for one fixed pair."""
    Ra = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Rb = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    qa, qb = quat_from_matrix_wxyz(Ra), quat_from_matrix_wxyz(Rb)
    q_ab = quat_mul_wxyz(qa, qb)
    q_Rab = quat_from_matrix_wxyz(Ra @ Rb)
    dot = abs(float(np.dot(quat_normalize_wxyz(q_ab), q_Rab)))
    assert dot > 1.0 - 1e-5
