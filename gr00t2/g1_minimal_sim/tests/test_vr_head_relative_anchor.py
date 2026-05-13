"""VR IK anchor modes: current headset-relative compose (numpy only)."""

import numpy as np

from vr_teleop.vr_stream_torso_compose import (
    compose_head_relative_controller_world_pose,
    compose_world_draw_pose,
    normalize_head_orient_mode,
    torso_xmat_with_yaw_offset,
    vr_ik_pelvis_mount_matrix,
)


def test_head_relative_invariant_to_common_stream_translation() -> None:
    """Same delta on HMD + controller leaves composed world palm (torso anchor, mount=I)."""
    torso_p = np.array([0.4, -0.1, 0.9], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    hmd_p = np.array([1.0, 0.0, 1.2], dtype=np.float64)
    hmd_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dev_p = np.array([1.15, 0.05, 1.18], dtype=np.float64)
    dev_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([3.0, -2.0, 0.5], dtype=np.float64)
    p0, q0 = compose_head_relative_controller_world_pose(
        torso_p, R, None, hmd_p, hmd_q, dev_p, dev_q
    )
    p1, q1 = compose_head_relative_controller_world_pose(
        torso_p, R, None, hmd_p + v, hmd_q, dev_p + v, dev_q
    )
    assert np.allclose(p0, p1, atol=1e-9)
    assert np.allclose(q0, q1, atol=1e-9) or np.allclose(q0, -q1, atol=1e-9)


def test_head_relative_matches_legacy_when_hmd_equals_hmd0() -> None:
    """At calib instant (dev relative to frozen hmd0 == inv(hmd)*dev when hmd==hmd0)."""
    torso_p = np.array([0.2, 0.3, 0.85], dtype=np.float64)
    R_torso = np.eye(3, dtype=np.float64)
    yaw_deg = -25.0
    R_vr = torso_xmat_with_yaw_offset(R_torso, yaw_deg)
    hmd0_p = np.array([0.05, -0.02, 1.35], dtype=np.float64)
    hmd0_q = np.array([0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float64)  # pitch ~45°
    left_p = np.array([0.28, -0.1, 1.22], dtype=np.float64)
    left_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    p_legacy, q_legacy = compose_world_draw_pose(
        torso_p, R_vr, hmd0_p, hmd0_q, left_p, left_q
    )
    p_new, q_new = compose_head_relative_controller_world_pose(
        torso_p,
        R_vr,
        None,
        hmd0_p,
        hmd0_q,
        left_p,
        left_q,
    )
    assert np.allclose(p_legacy, p_new, atol=1e-8)
    assert np.allclose(q_legacy, q_new, atol=1e-6) or np.allclose(q_legacy, -q_new, atol=1e-6)


def test_pelvis_mount_identity_when_bodies_aligned() -> None:
    """If pelvis == torso pose, M is identity and head-relative matches torso-only."""
    p = np.array([0.0, 0.0, 0.95], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    yaw = 0.0
    M = vr_ik_pelvis_mount_matrix(p, R, p, R, yaw)
    assert np.allclose(M, np.eye(4), atol=1e-9)
    hmd_p = np.array([0.0, 0.0, 1.4], dtype=np.float64)
    hmd_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dev_p = np.array([0.2, 0.0, 1.4], dtype=np.float64)
    dev_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    p_t, q_t = compose_head_relative_controller_world_pose(p, R, None, hmd_p, hmd_q, dev_p, dev_q)
    p_p, q_p = compose_head_relative_controller_world_pose(p, R, M, hmd_p, hmd_q, dev_p, dev_q)
    assert np.allclose(p_t, p_p, atol=1e-9)
    assert np.allclose(q_t, q_p, atol=1e-9) or np.allclose(q_t, -q_p, atol=1e-9)


def test_normalize_vr_ik_anchor_mode_aliases() -> None:
    from vr_stream_ik_mode import normalize_vr_ik_anchor_mode

    assert normalize_vr_ik_anchor_mode("LEGACY") == "legacy"
    assert normalize_vr_ik_anchor_mode("head-relative-torso") == "torso"
    assert normalize_vr_ik_anchor_mode("pelvis_head") == "pelvis"


def test_world_pose_overlay_hmd_torso_anchor_at_torso() -> None:
    """HMD arrow (device 0) in torso anchor mode sits at torso origin with torso R_vr."""
    from vr_teleop.openvr_stream import PoseState, StreamState
    from vr_teleop.vr_stream_torso_compose import VrTorsoVizCalib, world_pose_stream_device_for_anchor

    torso_p = np.array([0.5, -0.2, 0.91], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    hmd_p = np.array([0.1, 0.2, 1.5], dtype=np.float64)
    hmd_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    st = StreamState(
        seq=1,
        stamp=1.0,
        hmd=PoseState(valid=True, pos=hmd_p, quat=hmd_q),
        left=PoseState(valid=True, pos=hmd_p + np.array([0.1, 0.0, 0.0]), quat=hmd_q),
        right=PoseState(valid=True, pos=hmd_p - np.array([0.1, 0.0, 0.0]), quat=hmd_q),
        left_stick_xy=np.zeros(2),
        right_stick_xy=np.zeros(2),
        left_trigger=0.0,
        right_trigger=0.0,
        left_input_valid=False,
        right_input_valid=False,
    )
    calib = VrTorsoVizCalib()
    calib.maybe_calibrate(st, orient_mode="full")
    assert calib.is_ready()
    po = world_pose_stream_device_for_anchor(
        0,
        st.hmd,
        stream=st,
        anchor_mode="torso",
        torso_xpos=torso_p,
        torso_R_vr=R,
        calib=calib,
        yaw_offset_deg=0.0,
        pelvis_xpos=None,
        pelvis_xmat=None,
        mount_T=None,
    )
    assert po.valid
    assert np.allclose(po.pos, torso_p, atol=1e-9)


def test_head_relative_yaw_only_ignores_hmd_pitch_roll_in_translation() -> None:
    """Yaw-only HMD mode keeps controller translation invariant to HMD pitch/roll."""
    torso_p = np.array([0.0, 0.0, 0.9], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    hmd_p = np.array([0.0, 0.0, 1.5], dtype=np.float64)
    # Two HMD quats with same yaw (0), different pitch.
    hmd_q_flat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    hmd_q_pitch = np.array([0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float64)  # +45 deg pitch
    dev_p = np.array([0.25, 0.0, 1.35], dtype=np.float64)
    dev_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    p0, _ = compose_head_relative_controller_world_pose(
        torso_p,
        R,
        None,
        hmd_p,
        hmd_q_flat,
        dev_p,
        dev_q,
        head_orient_mode="yaw_only",
    )
    p1, _ = compose_head_relative_controller_world_pose(
        torso_p,
        R,
        None,
        hmd_p,
        hmd_q_pitch,
        dev_p,
        dev_q,
        head_orient_mode="yaw_only",
    )
    assert np.allclose(p0, p1, atol=1e-8)


def test_normalize_head_orient_mode_aliases() -> None:
    assert normalize_head_orient_mode("FULL") == "full"
    assert normalize_head_orient_mode("yaw") == "yaw_only"
    assert normalize_head_orient_mode("yaw-only") == "yaw_only"
