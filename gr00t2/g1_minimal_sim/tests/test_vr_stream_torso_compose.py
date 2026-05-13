"""Torso-mounted stream pose composition (numpy only)."""

import numpy as np

from vr_teleop.openvr_stream import PoseState, default_stream_state
from vr_teleop.vr_stream_torso_compose import (
    VrTorsoVizCalib,
    compose_world_draw_pose,
    pose_for_overlay_device,
    torso_xmat_with_yaw_offset,
)


def test_compose_hmd_matches_torso_when_stream_unchanged() -> None:
    """At calibration, HMD stream pose equals hmd0; composed HMD world pose equals torso."""
    torso_p = np.array([0.5, -0.25, 0.88], dtype=np.float64)
    c = np.sqrt(0.5)
    qb = np.array([c, 0.0, 0.0, c], dtype=np.float64)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    hmd_p = np.array([0.1, 0.2, 1.5], dtype=np.float64)
    hmd_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    p_w, q_w = compose_world_draw_pose(torso_p, R, hmd_p, hmd_q, hmd_p, hmd_q)
    assert np.allclose(p_w, torso_p, atol=1e-9)
    assert np.allclose(q_w, qb, atol=1e-6) or np.allclose(q_w, -qb, atol=1e-6)


def test_compose_translates_with_torso_frozen_stream() -> None:
    hmd_p = np.array([0.0, 0.0, 1.4], dtype=np.float64)
    hmd_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t0 = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    t1 = t0 + np.array([0.3, -0.1, 0.0], dtype=np.float64)
    p0, _ = compose_world_draw_pose(t0, R, hmd_p, hmd_q, hmd_p, hmd_q)
    p1, _ = compose_world_draw_pose(t1, R, hmd_p, hmd_q, hmd_p, hmd_q)
    assert np.allclose(p1 - p0, t1 - t0, atol=1e-9)


def test_compose_controller_offset_rotates_with_torso_yaw() -> None:
    """Left controller +0.2 m stream X from HMD; after 90° torso yaw, offset maps to world +Y."""
    hmd_p = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    hmd_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    left_p = np.array([0.2, 0.0, 0.0], dtype=np.float64)
    left_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    R0 = np.eye(3, dtype=np.float64)
    t0 = np.array([1.0, 1.0, 0.75], dtype=np.float64)
    p_l0, _ = compose_world_draw_pose(t0, R0, hmd_p, hmd_q, left_p, left_q)
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    p_l1, _ = compose_world_draw_pose(t0, Rz, hmd_p, hmd_q, left_p, left_q)
    assert np.allclose(p_l0, t0 + np.array([0.2, 0.0, 0.0]), atol=1e-9)
    assert np.allclose(p_l1, t0 + np.array([0.0, 0.2, 0.0]), atol=1e-9)


def test_torso_yaw_offset_rotates_basis_not_translation() -> None:
    hmd_p = np.zeros(3, dtype=np.float64)
    hmd_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dev_p = np.array([0.2, 0.0, 0.0], dtype=np.float64)
    dev_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    torso_pos = np.array([2.0, -3.0, 0.85], dtype=np.float64)
    R_torso = np.eye(3, dtype=np.float64)
    p_no, _ = compose_world_draw_pose(torso_pos, R_torso, hmd_p, hmd_q, dev_p, dev_q)
    R_off = torso_xmat_with_yaw_offset(R_torso, 90.0)
    p_off, _ = compose_world_draw_pose(torso_pos, R_off, hmd_p, hmd_q, dev_p, dev_q)
    # Origin stays anchored at torso world position.
    assert np.allclose(p_no - torso_pos, [0.2, 0.0, 0.0], atol=1e-9)
    assert np.allclose(p_off - torso_pos, [0.0, 0.2, 0.0], atol=1e-9)


def test_fixed_world_z_ignores_torso_vertical_motion() -> None:
    calib = VrTorsoVizCalib()
    calib.hmd0_pos = np.array([0.0, 0.0, 1.4], dtype=np.float64)
    calib.hmd0_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t_lo = np.array([0.0, 0.0, 0.7], dtype=np.float64)
    t_hi = np.array([0.0, 0.0, 0.95], dtype=np.float64)
    dev = PoseState(valid=True, pos=calib.hmd0_pos.copy(), quat=calib.hmd0_quat.copy())
    p0 = pose_for_overlay_device(
        dev,
        torso_xpos=t_lo,
        torso_xmat=R,
        calib=calib,
        device_index=0,
        z_mode="fixed_world",
        z_fixed_world_m=1.05,
    ).pos
    p1 = pose_for_overlay_device(
        dev,
        torso_xpos=t_hi,
        torso_xmat=R,
        calib=calib,
        device_index=0,
        z_mode="fixed_world",
        z_fixed_world_m=1.05,
    ).pos
    assert abs(float(p0[2]) - 1.05) < 1e-9
    assert abs(float(p1[2]) - 1.05) < 1e-9
    assert abs(float(p1[0]) - float(p0[0])) < 1e-9 and abs(float(p1[1]) - float(p0[1])) < 1e-9


def test_maybe_calibrate_from_stream_state() -> None:
    st = default_stream_state()
    st.hmd = PoseState(valid=True, pos=np.array([0.1, 0.2, 1.3]), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    c = VrTorsoVizCalib()
    assert not c.is_ready()
    c.maybe_calibrate(st)
    assert c.is_ready()
    assert np.allclose(c.hmd0_pos, st.hmd.pos)
    c.reset()
    assert not c.is_ready()


def test_maybe_calibrate_yaw_only_drops_pitch_roll() -> None:
    # q = yaw(30) * pitch(20) * roll(-15), scalar-first (wxyz)
    ry, rp, rr = np.deg2rad(30.0), np.deg2rad(20.0), np.deg2rad(-15.0)
    cy, sy = np.cos(0.5 * ry), np.sin(0.5 * ry)
    cp, sp = np.cos(0.5 * rp), np.sin(0.5 * rp)
    cr, sr = np.cos(0.5 * rr), np.sin(0.5 * rr)
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    st = default_stream_state()
    st.hmd = PoseState(valid=True, pos=np.array([0.1, 0.2, 1.3]), quat=q)

    c_full = VrTorsoVizCalib()
    c_full.maybe_calibrate(st, orient_mode="full")
    assert c_full.hmd0_quat is not None

    c_yaw = VrTorsoVizCalib()
    c_yaw.maybe_calibrate(st, orient_mode="yaw_only")
    assert c_yaw.hmd0_quat is not None

    # Yaw-only calibration must zero x/y quaternion components while preserving z yaw sign.
    qy = c_yaw.hmd0_quat
    assert abs(float(qy[1])) < 1e-9
    assert abs(float(qy[2])) < 1e-9
    assert abs(float(qy[3])) > 1e-6
    # Full calibration keeps the original tilt components.
    qf = c_full.hmd0_quat
    assert abs(float(qf[1])) > 1e-6 or abs(float(qf[2])) > 1e-6


def test_frozen_calib_z_sticky() -> None:
    calib = VrTorsoVizCalib()
    calib.hmd0_pos = np.zeros(3, dtype=np.float64)
    calib.hmd0_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    dev = PoseState(valid=True, pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    p0 = pose_for_overlay_device(
        dev, torso_xpos=t, torso_xmat=R, calib=calib, device_index=1, z_mode="frozen_calib", z_fixed_world_m=0.0
    ).pos
    t2 = t + np.array([0.0, 0.0, 0.5], dtype=np.float64)
    p1 = pose_for_overlay_device(
        dev, torso_xpos=t2, torso_xmat=R, calib=calib, device_index=1, z_mode="frozen_calib", z_fixed_world_m=0.0
    ).pos
    assert abs(float(p0[2]) - float(p1[2])) < 1e-9
    assert float(p0[2]) == 0.8
