"""Unit tests for new ``vr_teleop.mapping`` frame conventions."""

from __future__ import annotations

import numpy as np

from vr_teleop.mapping import (
    R_OPENVR_TO_MJ,
    T_OPENVR_TO_MJ_QUAT_WXYZ,
    openvr_pose_to_mujoco,
    openvr_rotmat_to_mujoco_quat_wxyz,
    openvr_to_mujoco_pos,
    quat_from_matrix_wxyz,
)


def test_linear_map_matches_forward_left_up_convention() -> None:
    v = np.array([1.0, 2.0, 3.0])  # openvr: x right, y up, z backward
    got = openvr_to_mujoco_pos(v)
    assert np.allclose(got, [-3.0, -1.0, 2.0])  # x forward, y left, z up
    assert np.allclose(R_OPENVR_TO_MJ @ v, got)


def test_transform_quat_matches_transform_matrix() -> None:
    qR = quat_from_matrix_wxyz(R_OPENVR_TO_MJ)
    dot = abs(float(np.dot(qR, T_OPENVR_TO_MJ_QUAT_WXYZ)))
    assert dot > 1.0 - 1e-6


def test_identity_pose_maps_to_fixed_transform() -> None:
    q_mj = openvr_rotmat_to_mujoco_quat_wxyz(np.eye(3))
    dot = abs(float(np.dot(q_mj, T_OPENVR_TO_MJ_QUAT_WXYZ)))
    assert dot > 1.0 - 1e-6


def test_openvr_pose_to_mujoco_fake_mat34() -> None:
    class FakeMat:
        m = [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 5.0],
        ]

    pos, quat = openvr_pose_to_mujoco(FakeMat())
    assert np.allclose(pos, [-5.0, -2.0, 3.0])
    dot = abs(float(np.dot(quat, T_OPENVR_TO_MJ_QUAT_WXYZ)))
    assert dot > 1.0 - 1e-6

