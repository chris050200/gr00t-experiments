"""VR stream device world pose + torso-frame EE mapping (numpy only)."""

import numpy as np

from ee_frame import ee_pose_ref_from_world, ee_pose_world_from_ref
from vr_teleop.openvr_stream import PoseState, StreamState
from vr_teleop.vr_stream_torso_compose import (
    VrTorsoVizCalib,
    compose_world_draw_pose,
    stream_device_world_pose,
    torso_xmat_with_yaw_offset,
    torso_xquat_with_receiver_yaw,
)


def test_torso_xquat_matches_rotmat_from_torso_xmat() -> None:
    """Quaternion helper must match the rotated torso basis used in compose."""
    torso_p = np.array([0.1, -0.2, 0.9], dtype=np.float64)
    c = np.sqrt(0.5)
    qb = np.array([c, 0.0, 0.0, c], dtype=np.float64)  # yaw 90
    R_torso = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    hmd_p = np.array([0.05, -0.03, 1.25], dtype=np.float64)
    hmd_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    left_p = np.array([0.25, -0.05, 1.15], dtype=np.float64)
    left_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    yaw_deg = -40.0
    R_vr = torso_xmat_with_yaw_offset(R_torso, yaw_deg)
    q_ref = torso_xquat_with_receiver_yaw(R_torso, qb, yaw_deg)

    p_w, q_w = compose_world_draw_pose(torso_p, R_vr, hmd_p, hmd_q, left_p, left_q)
    pl_b, ql_b = ee_pose_ref_from_world(torso_p, R_torso, q_ref, p_w, q_w)
    p_back, q_back = ee_pose_world_from_ref(torso_p, R_torso, q_ref, pl_b, ql_b)
    assert np.allclose(p_back, p_w, atol=1e-7)
    assert np.allclose(q_back, q_w, atol=1e-6) or np.allclose(q_back, -q_w, atol=1e-6)


def test_stream_device_world_matches_compose_helper() -> None:
    st = StreamState(
        seq=1,
        stamp=1.0,
        hmd=PoseState(valid=True, pos=np.array([0.0, 0.0, 1.4]), quat=np.array([1.0, 0.0, 0.0, 0.0])),
        left=PoseState(valid=True, pos=np.array([0.2, 0.0, 0.0]), quat=np.array([1.0, 0.0, 0.0, 0.0])),
        right=PoseState(valid=True, pos=np.array([-0.2, 0.0, 0.0]), quat=np.array([1.0, 0.0, 0.0, 0.0])),
        left_stick_xy=np.zeros(2),
        right_stick_xy=np.zeros(2),
        left_trigger=0.0,
        right_trigger=0.0,
        left_input_valid=False,
        right_input_valid=False,
    )
    calib = VrTorsoVizCalib()
    torso_p = np.array([1.0, 2.0, 0.8], dtype=np.float64)
    R_torso = np.eye(3, dtype=np.float64)
    yaw_deg = -90.0
    R_vr = torso_xmat_with_yaw_offset(R_torso, yaw_deg)

    po = stream_device_world_pose(
        st.left,
        stream=st,
        torso_xpos=torso_p,
        torso_xmat=R_torso,
        yaw_offset_deg=yaw_deg,
        calib=calib,
        calib_orient_mode="full",
    )
    assert calib.is_ready()
    p_exp, q_exp = compose_world_draw_pose(
        torso_p,
        R_vr,
        calib.hmd0_pos,
        calib.hmd0_quat,
        st.left.pos,
        st.left.quat,
    )
    assert po.valid
    assert np.allclose(po.pos, p_exp, atol=1e-9)
    assert np.allclose(po.quat, q_exp, atol=1e-9) or np.allclose(po.quat, -q_exp, atol=1e-9)

