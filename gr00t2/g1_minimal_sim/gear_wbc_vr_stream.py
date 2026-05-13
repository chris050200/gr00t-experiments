"""OpenVR UDP stream: stick locomotion + VR IK ``ee_*`` targets (torso compose)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from arm_ik import reset_ee_to_home
from ee_frame import (
    ee_pose_ref_from_world,
    intrinsic_zyx_deg_to_quat_wxyz,
    normalize_quat,
    quat_mul,
)
from gear_wbc_pd import clamp_ref_target_to_anchor_radius
from vr_teleop.openvr_stream import PoseState, StreamState, sticks_to_loco_cmd
from vr_teleop.vr_stream_torso_compose import (
    compose_head_relative_controller_world_pose,
    normalize_head_orient_mode,
    stream_device_world_pose,
    torso_xmat_with_yaw_offset,
    torso_xquat_with_receiver_yaw,
    vr_ik_pelvis_mount_matrix,
)

VR_STREAM_STALE_S = 0.35
VR_IK_ANCHOR_RADIUS_M = 0.6

# Controller → G1 palm orientation (intrinsic Z–Y′–X″ deg; post-multiply on composed world quat).
# Edit here if a different headset / sender changes the grip–palm relationship.
VR_IK_HAND_INTRINSIC_YAW_DEG = 90.0
VR_IK_HAND_INTRINSIC_PITCH_DEG = 180.0
VR_IK_HAND_INTRINSIC_ROLL_DEG = 0.0
VR_IK_HAND_OFFSET_QUAT_WXYZ = intrinsic_zyx_deg_to_quat_wxyz(
    VR_IK_HAND_INTRINSIC_YAW_DEG,
    VR_IK_HAND_INTRINSIC_PITCH_DEG,
    VR_IK_HAND_INTRINSIC_ROLL_DEG,
).astype(np.float64)


def snapshot_vr_receiver(rt: Any) -> tuple[StreamState | None, float]:
    recv = rt._vr_teleop_receiver
    if recv is None:
        return None, 0.0
    return recv.snapshot()


def apply_vr_stream_locomotion(rt: Any, vr_snap: StreamState | None, vr_t_rx: float) -> None:
    """Update ``loco_cmd`` from sticks; reset on stale stream (and EE home when VR IK + no keyboard)."""
    if rt._vr_teleop_receiver is None:
        return
    now = time.monotonic()
    if vr_t_rx > 0.0:
        age = now - vr_t_rx
        with rt.cmd_lock:
            if vr_snap is not None and age < VR_STREAM_STALE_S:
                rt.control_dict["loco_cmd"][:] = sticks_to_loco_cmd(
                    vr_snap.left_stick_xy,
                    vr_snap.right_stick_xy,
                )
            elif age >= VR_STREAM_STALE_S:
                rt.control_dict["loco_cmd"][:] = rt.config["cmd_init"]
                if (
                    rt.vr_ik_mode != "off"
                    and (not rt.teleop)
                    and rt.n_arm > 0
                    and "ee_left_pos" in rt.control_dict
                ):
                    reset_ee_to_home(rt.control_dict)


def apply_vr_stream_ik_targets(
    rt: Any,
    vr_snap: StreamState | None,
    vr_t_rx: float,
) -> dict[str, Any] | None:
    """Compose stream → torso-frame ``ee_*`` (+ grippers). Returns trace dict or ``None``."""
    if (
        rt._vr_teleop_receiver is None
        or rt.vr_ik_mode == "off"
        or vr_snap is None
        or vr_t_rx <= 0.0
        or (time.monotonic() - vr_t_rx) >= VR_STREAM_STALE_S
        or rt._vr_stream_calib is None
        or rt.n_arm <= 0
        or rt.left_ik is None
    ):
        return None

    ti = rt.torso_index
    xpos = rt.data.xpos[ti]
    xmat = rt.data.xmat[ti].reshape(3, 3)
    xquat = np.asarray(rt.data.xquat[ti], dtype=np.float64)
    q_ref = torso_xquat_with_receiver_yaw(xmat, xquat, rt.vr_receiver_yaw_deg)
    R_vr = torso_xmat_with_yaw_offset(xmat, rt.vr_receiver_yaw_deg)
    calib = rt._vr_stream_calib
    assert calib is not None
    calib.maybe_calibrate(vr_snap, orient_mode=rt.vr_stream_calib_orient)
    if not calib.is_ready():
        return None

    anchor_mode = str(getattr(rt, "vr_ik_anchor_mode", "legacy"))
    head_orient_mode = normalize_head_orient_mode(
        getattr(rt, "vr_head_orient_mode", "yaw_only")
    )
    if anchor_mode != "legacy" and not vr_snap.hmd.valid:
        return None

    if anchor_mode == "pelvis" and rt._vr_ik_mount_mat4 is None:
        pi = int(rt.pelvis_index)
        px = np.asarray(rt.data.xpos[pi], dtype=np.float64).reshape(3)
        pm = np.asarray(rt.data.xmat[pi], dtype=np.float64).reshape(3, 3)
        rt._vr_ik_mount_mat4 = vr_ik_pelvis_mount_matrix(
            px,
            pm,
            np.asarray(xpos, dtype=np.float64).reshape(3),
            np.asarray(xmat, dtype=np.float64).reshape(3, 3),
            float(rt.vr_receiver_yaw_deg),
        )

    vr_ik_trace: dict[str, Any] = {
        "vr_ik_active": True,
        "vr_ik_anchor_mode": anchor_mode,
        "vr_head_orient_mode": head_orient_mode,
        "vr_ik_z_offset_m": float(getattr(rt, "vr_ik_z_offset_m", 0.0)),
        "vr_ik_pelvis_mount_ready": bool(rt._vr_ik_mount_mat4 is not None),
        "torso_xpos": np.asarray(xpos, dtype=np.float64).reshape(3).tolist(),
        "torso_xmat": np.asarray(xmat, dtype=np.float64).reshape(3, 3).reshape(-1).tolist(),
        "torso_q_ref": np.asarray(q_ref, dtype=np.float64).reshape(4).tolist(),
        "torso_R_vr": np.asarray(R_vr, dtype=np.float64).reshape(3, 3).reshape(-1).tolist(),
    }
    with rt.cmd_lock:
        for side, dev in (("left", vr_snap.left), ("right", vr_snap.right)):
            if anchor_mode == "legacy":
                pw = stream_device_world_pose(
                    dev,
                    stream=vr_snap,
                    torso_xpos=xpos,
                    torso_xmat=xmat,
                    yaw_offset_deg=rt.vr_receiver_yaw_deg,
                    calib=calib,
                    calib_orient_mode=rt.vr_stream_calib_orient,
                )
            else:
                if not dev.valid:
                    continue
                h = vr_snap.hmd
                if anchor_mode == "torso":
                    pw_p, pw_q = compose_head_relative_controller_world_pose(
                        xpos,
                        R_vr,
                        None,
                        h.pos,
                        h.quat,
                        dev.pos,
                        dev.quat,
                        head_orient_mode=head_orient_mode,
                    )
                elif anchor_mode == "pelvis":
                    pi = int(rt.pelvis_index)
                    px = rt.data.xpos[pi]
                    pm = rt.data.xmat[pi].reshape(3, 3)
                    R_p = torso_xmat_with_yaw_offset(pm, rt.vr_receiver_yaw_deg)
                    assert rt._vr_ik_mount_mat4 is not None
                    pw_p, pw_q = compose_head_relative_controller_world_pose(
                        px,
                        R_p,
                        rt._vr_ik_mount_mat4,
                        h.pos,
                        h.quat,
                        dev.pos,
                        dev.quat,
                        head_orient_mode=head_orient_mode,
                    )
                else:
                    raise ValueError(f"unknown vr_ik_anchor_mode {anchor_mode!r}")
                pw = PoseState(valid=True, pos=pw_p, quat=pw_q)
            if not pw.valid:
                continue
            pw_p = np.asarray(pw.pos, dtype=np.float64).reshape(3).copy()
            pw_p[2] += float(getattr(rt, "vr_ik_z_offset_m", 0.0))
            q_raw = np.asarray(pw.quat, dtype=np.float64).reshape(4)
            q_use = normalize_quat(quat_mul(q_raw, VR_IK_HAND_OFFSET_QUAT_WXYZ))
            # Use R_vr (same basis as q_ref) for position, not raw torso xmat — otherwise
            # heading-dependent orientation/position mismatch when vr_receiver_yaw_deg ≠ 0.
            pr, qr = ee_pose_ref_from_world(xpos, R_vr, q_ref, pw_p, q_use)
            pr_rvr = (R_vr.T @ (pw_p - np.asarray(xpos, dtype=np.float64).reshape(3))).tolist()
            pr_raw = (
                np.asarray(xmat, dtype=np.float64).reshape(3, 3).T
                @ (pw_p - np.asarray(xpos, dtype=np.float64).reshape(3))
            ).tolist()
            if side == "left":
                vr_ik_trace["ik_pw_left_pos"] = pw_p.astype(float).tolist()
                vr_ik_trace["ik_pw_left_quat"] = np.asarray(pw.quat, dtype=np.float64).reshape(4).tolist()
                vr_ik_trace["ik_pr_left_pos"] = np.asarray(pr, dtype=np.float64).reshape(3).tolist()
                vr_ik_trace["ik_qr_left_quat"] = np.asarray(qr, dtype=np.float64).reshape(4).tolist()
                vr_ik_trace["ik_pr_left_pos_Rvr"] = pr_rvr
                vr_ik_trace["ik_pr_left_pos_raw_xmat"] = pr_raw
                anchor = np.asarray(
                    rt.control_dict.get("_ee_left_pos_home", np.zeros(3)),
                    dtype=np.float32,
                )
                pr_clamped = clamp_ref_target_to_anchor_radius(
                    pr,
                    anchor,
                    VR_IK_ANCHOR_RADIUS_M,
                )
                d0 = float(np.linalg.norm(np.asarray(pr, dtype=np.float64) - anchor))
                d1 = float(
                    np.linalg.norm(np.asarray(pr_clamped, dtype=np.float64) - anchor)
                )
                vr_ik_trace["ik_left_anchor"] = anchor.astype(float).tolist()
                vr_ik_trace["ik_left_dist_before"] = d0
                vr_ik_trace["ik_left_dist_after"] = d1
                rt.control_dict["ee_left_pos"][:] = pr_clamped
                rt.control_dict["ee_left_quat"][:] = qr.astype(np.float32)
            else:
                vr_ik_trace["ik_pw_right_pos"] = pw_p.astype(float).tolist()
                vr_ik_trace["ik_pw_right_quat"] = np.asarray(pw.quat, dtype=np.float64).reshape(4).tolist()
                vr_ik_trace["ik_pr_right_pos"] = np.asarray(pr, dtype=np.float64).reshape(3).tolist()
                vr_ik_trace["ik_qr_right_quat"] = np.asarray(qr, dtype=np.float64).reshape(4).tolist()
                vr_ik_trace["ik_pr_right_pos_Rvr"] = pr_rvr
                vr_ik_trace["ik_pr_right_pos_raw_xmat"] = pr_raw
                anchor = np.asarray(
                    rt.control_dict.get("_ee_right_pos_home", np.zeros(3)),
                    dtype=np.float32,
                )
                pr_clamped = clamp_ref_target_to_anchor_radius(
                    pr,
                    anchor,
                    VR_IK_ANCHOR_RADIUS_M,
                )
                d0 = float(np.linalg.norm(np.asarray(pr, dtype=np.float64) - anchor))
                d1 = float(
                    np.linalg.norm(np.asarray(pr_clamped, dtype=np.float64) - anchor)
                )
                vr_ik_trace["ik_right_anchor"] = anchor.astype(float).tolist()
                vr_ik_trace["ik_right_dist_before"] = d0
                vr_ik_trace["ik_right_dist_after"] = d1
                rt.control_dict["ee_right_pos"][:] = pr_clamped
                rt.control_dict["ee_right_quat"][:] = qr.astype(np.float32)
        if rt.has_hands and vr_snap.left_input_valid:
            rt.control_dict["gripper_left"] = float(
                np.clip(1.0 - float(vr_snap.left_trigger), 0.0, 1.0)
            )
        if rt.has_hands and vr_snap.right_input_valid:
            rt.control_dict["gripper_right"] = float(
                np.clip(1.0 - float(vr_snap.right_trigger), 0.0, 1.0)
            )
        vr_ik_trace["ee_left_pos"] = np.asarray(
            rt.control_dict["ee_left_pos"], dtype=np.float32
        ).reshape(3).tolist()
        vr_ik_trace["ee_left_quat"] = np.asarray(
            rt.control_dict["ee_left_quat"], dtype=np.float32
        ).reshape(4).tolist()
        vr_ik_trace["ee_right_pos"] = np.asarray(
            rt.control_dict["ee_right_pos"], dtype=np.float32
        ).reshape(3).tolist()
        vr_ik_trace["ee_right_quat"] = np.asarray(
            rt.control_dict["ee_right_quat"], dtype=np.float32
        ).reshape(4).tolist()
    return vr_ik_trace
