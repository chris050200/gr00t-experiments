"""JSONL ``vr_compare`` rows for VR stream vs overlay vs IK (flight recorder)."""

from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np

from arm_ik import body_palm_pose, ee_world_targets_for_ik
from ee_frame import ee_pose_world_from_ref
from gear_wbc_vr_stream import (
    VR_IK_HAND_INTRINSIC_PITCH_DEG,
    VR_IK_HAND_INTRINSIC_ROLL_DEG,
    VR_IK_HAND_INTRINSIC_YAW_DEG,
    VR_IK_HAND_OFFSET_QUAT_WXYZ,
)
from vr_teleop.openvr_stream import PoseState, StreamState
from vr_teleop.vr_stream_torso_compose import (
    pose_for_overlay_device,
    torso_xmat_with_yaw_offset,
    torso_xquat_with_receiver_yaw,
    world_pose_stream_device_for_anchor,
)


def quat_geodesic_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    """Smallest rotation angle (deg) between two orientations, wxyz unit quaternions."""
    d = float(
        abs(
            float(
                np.dot(
                    np.asarray(qa, dtype=np.float64).reshape(4),
                    np.asarray(qb, dtype=np.float64).reshape(4),
                )
            )
        )
    )
    d = min(1.0, max(0.0, d))
    return float(2.0 * np.degrees(np.arccos(d)))


def trace_pose_dict(ps: PoseState) -> dict[str, Any]:
    return {
        "valid": bool(ps.valid),
        "pos": ps.pos.astype(float).tolist(),
        "quat": ps.quat.astype(float).tolist(),
    }


def trace_should_emit(rt: Any) -> bool:
    if rt._trace is None:
        return False
    hz = float(rt._trace_hz)
    if hz <= 0.0:
        return True
    now = time.monotonic()
    min_dt = 1.0 / hz
    if (now - rt._trace_last_emit_mono) < min_dt:
        return False
    rt._trace_last_emit_mono = now
    return True


def _forensic_should_trigger(rt: Any, row: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    pos = max(
        float(row.get("err_meas_minus_ik_tgt_world_left_l2", 0.0)),
        float(row.get("err_meas_minus_ik_tgt_world_right_l2", 0.0)),
    )
    quat = max(
        float(row.get("err_quat_meas_minus_ik_tgt_world_left_deg", 0.0)),
        float(row.get("err_quat_meas_minus_ik_tgt_world_right_deg", 0.0)),
    )
    ikd = row.get("ik")
    raw = float(ikd.get("ik_arm_delta_raw_l2", 0.0)) if isinstance(ikd, dict) else 0.0
    scale = float(ikd.get("ik_step_limit_scale", 1.0)) if isinstance(ikd, dict) else 1.0
    cond = bool(
        (pos > float(getattr(rt, "_forensic_trigger_pos_err_m", 0.15)))
        or (quat > float(getattr(rt, "_forensic_trigger_quat_err_deg", 45.0)))
        or (
            raw > float(getattr(rt, "_forensic_trigger_raw_step_l2", 1.0))
            and scale <= float(getattr(rt, "_forensic_trigger_step_scale_max", 0.40))
        )
    )
    return cond, {
        "trigger_pos_err_m": pos,
        "trigger_quat_err_deg": quat,
        "trigger_ik_raw_step_l2": raw,
        "trigger_ik_step_scale": scale,
    }


def maybe_emit_forensic_burst(rt: Any, row: dict[str, Any]) -> None:
    if rt._trace is None or not bool(getattr(rt, "_forensic_burst_enable", False)):
        return
    rt._forensic_row_index += 1
    idx = int(rt._forensic_row_index)
    snap = dict(row)
    snap["_forensic_row_index"] = idx
    rt._forensic_ring.append(snap)
    if rt._forensic_cooldown_left > 0:
        rt._forensic_cooldown_left -= 1

    trigger, metrics = _forensic_should_trigger(rt, row)
    if (
        not rt._forensic_active
        and trigger
        and rt._forensic_cooldown_left <= 0
    ):
        rt._forensic_active = True
        rt._forensic_episode_id += 1
        rt._forensic_post_left = int(rt._forensic_post_steps)
        rt._trace.write(
            {
                "kind": "teleop_forensic_event",
                "event": "trigger_start",
                "episode_id": int(rt._forensic_episode_id),
                "row_index": idx,
                "sim_time": float(row.get("sim_time", 0.0)),
                **metrics,
            }
        )
        for r in rt._forensic_ring:
            out = dict(r)
            out["kind"] = "teleop_forensic_row"
            out["episode_id"] = int(rt._forensic_episode_id)
            out["phase"] = "pre"
            rt._trace.write(out)

    if rt._forensic_active:
        out = dict(row)
        out["kind"] = "teleop_forensic_row"
        out["episode_id"] = int(rt._forensic_episode_id)
        out["phase"] = "post"
        out["_forensic_row_index"] = idx
        rt._trace.write(out)
        rt._forensic_post_left -= 1
        if rt._forensic_post_left <= 0:
            rt._trace.write(
                {
                    "kind": "teleop_forensic_event",
                    "event": "trigger_end",
                    "episode_id": int(rt._forensic_episode_id),
                    "row_index": idx,
                    "sim_time": float(row.get("sim_time", 0.0)),
                }
            )
            rt._forensic_active = False
            rt._forensic_cooldown_left = int(rt._forensic_cooldown_steps)


def _trace_step_delta(rt: Any, name: str, cur: np.ndarray | None) -> float | None:
    """Norm since previous sample for flight-recorder deltas (uses ``rt._trace_prev``)."""
    if cur is None:
        return None
    key = f"_prev_{name}"
    prev = rt._trace_prev.get(key)
    rt._trace_prev[key] = np.asarray(cur, dtype=np.float64).copy()
    if prev is None:
        return None
    return float(np.linalg.norm(np.asarray(cur, dtype=np.float64) - prev))


def append_post_mj_ik_debug(
    rt: Any,
    row: dict[str, Any],
    ik_details: dict[str, Any] | None,
    q_sol: np.ndarray | None,
) -> None:
    """After ``mj_step``: IK key deltas, measured palms vs targets, torso ω, arm τ stats."""
    if isinstance(ik_details, dict):
        for k in (
            "ik_pw_left_pos",
            "ik_pw_right_pos",
            "ik_pr_left_pos",
            "ik_pr_right_pos",
            "ik_pr_left_pos_Rvr",
            "ik_pr_right_pos_Rvr",
            "ik_pr_left_pos_raw_xmat",
            "ik_pr_right_pos_raw_xmat",
            "ee_left_pos",
            "ee_right_pos",
            "ik_world_left_from_ee",
            "ik_world_right_from_ee",
        ):
            v = ik_details.get(k)
            if isinstance(v, list) and len(v) == 3:
                row[f"delta_{k}"] = _trace_step_delta(
                    rt, k, np.asarray(v, dtype=np.float64)
                )

    if q_sol is not None:
        row["delta_ik_q_sol_l2"] = _trace_step_delta(
            rt, "ik_q_sol", np.asarray(q_sol, dtype=np.float64).reshape(-1)
        )

    ti = int(rt.torso_index)
    if rt.left_ik is not None and rt.right_ik is not None:
        pml, qml = body_palm_pose(rt.data, rt.left_ik)
        pmr, qmr = body_palm_pose(rt.data, rt.right_ik)
        row["meas_palm_world_left_pos"] = pml.astype(float).tolist()
        row["meas_palm_world_left_quat"] = qml.astype(float).tolist()
        row["meas_palm_world_right_pos"] = pmr.astype(float).tolist()
        row["meas_palm_world_right_quat"] = qmr.astype(float).tolist()
        with rt.cmd_lock:
            if "ee_left_pos" in rt.control_dict:
                tl_p = np.asarray(rt.control_dict["ee_left_pos"], dtype=np.float64).reshape(3)
                tl_q = np.asarray(rt.control_dict["ee_left_quat"], dtype=np.float64).reshape(4)
                tr_p = np.asarray(rt.control_dict["ee_right_pos"], dtype=np.float64).reshape(3)
                tr_q = np.asarray(rt.control_dict["ee_right_quat"], dtype=np.float64).reshape(4)
            else:
                tl_p = tl_q = tr_p = tr_q = None
        if tl_p is not None:
            ti_e = int(rt.torso_index)
            xm_e = np.asarray(rt.data.xmat[ti_e], dtype=np.float64).reshape(3, 3)
            tw_l, tw_ql, tw_r, tw_qr = ee_world_targets_for_ik(
                rt.data,
                rt.torso_index,
                tl_p,
                tl_q,
                tr_p,
                tr_q,
                ref_quat_wxyz=rt._ee_decode_quat(xm_e, rt.data.xquat[ti_e]),
                ref_rotmat=rt._ee_ref_rotmat(xm_e, rt.data.xquat[ti_e]),
            )
            row["ik_tgt_world_left_pos_from_ee"] = tw_l.astype(float).tolist()
            row["ik_tgt_world_left_quat_from_ee"] = tw_ql.astype(float).tolist()
            row["ik_tgt_world_right_pos_from_ee"] = tw_r.astype(float).tolist()
            row["ik_tgt_world_right_quat_from_ee"] = tw_qr.astype(float).tolist()
            row["err_meas_minus_ik_tgt_world_left_l2"] = float(np.linalg.norm(pml - tw_l))
            row["err_meas_minus_ik_tgt_world_right_l2"] = float(np.linalg.norm(pmr - tw_r))
            if isinstance(ik_details, dict):
                pwl = ik_details.get("ik_pw_left_pos")
                pwr = ik_details.get("ik_pw_right_pos")
                if isinstance(pwl, list) and len(pwl) == 3:
                    row["err_meas_minus_ik_pw_world_left_l2"] = float(
                        np.linalg.norm(pml - np.asarray(pwl, dtype=np.float64).reshape(3))
                    )
                if isinstance(pwr, list) and len(pwr) == 3:
                    row["err_meas_minus_ik_pw_world_right_l2"] = float(
                        np.linalg.norm(pmr - np.asarray(pwr, dtype=np.float64).reshape(3))
                    )
            row["err_quat_meas_minus_ik_tgt_world_left_deg"] = float(
                quat_geodesic_deg(qml, tw_ql)
            )
            row["err_quat_meas_minus_ik_tgt_world_right_deg"] = float(
                quat_geodesic_deg(qmr, tw_qr)
            )
            if isinstance(ik_details, dict):
                sx = ik_details.get("ik_solve_torso_xpos")
                sm = ik_details.get("ik_solve_torso_xmat")
                sq = ik_details.get("ik_solve_torso_xquat")
                if (
                    isinstance(sx, list)
                    and len(sx) == 3
                    and isinstance(sm, list)
                    and len(sm) == 9
                    and isinstance(sq, list)
                    and len(sq) == 4
                ):
                    xp_s = np.asarray(sx, dtype=np.float64).reshape(3)
                    xm_s = np.asarray(sm, dtype=np.float64).reshape(3, 3)
                    xq_s = np.asarray(sq, dtype=np.float64).reshape(4)
                    xq_dec = rt._ee_decode_quat(xm_s, xq_s)
                    xm_dec = rt._ee_ref_rotmat(xm_s, xq_s)
                    tw_ls, _ = ee_pose_world_from_ref(xp_s, xm_dec, xq_dec, tl_p, tl_q)
                    tw_rs, _ = ee_pose_world_from_ref(xp_s, xm_dec, xq_dec, tr_p, tr_q)
                    row["dbg_ik_tgt_world_pos_post_minus_ik_solve_left_l2"] = float(
                        np.linalg.norm(tw_l - tw_ls)
                    )
                    row["dbg_ik_tgt_world_pos_post_minus_ik_solve_right_l2"] = float(
                        np.linalg.norm(tw_r - tw_rs)
                    )

    v6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(rt.model, rt.data, mujoco.mjtObj.mjOBJ_BODY, ti, v6, 0)
    omega = np.asarray(v6[3:6], dtype=np.float64).reshape(3)
    row["torso_omega_world"] = omega.astype(float).tolist()
    row["torso_omega_norm"] = float(np.linalg.norm(omega))

    pre = rt._trace_arm_tau_pre_clip
    post = rt._trace_arm_tau_post_clip
    if pre is not None and post is not None and pre.shape == post.shape:
        row["arm_tau_pre_clip_l2"] = float(np.linalg.norm(pre))
        row["arm_tau_post_clip_l2"] = float(np.linalg.norm(post))
        delta = np.abs(post - pre)
        row["arm_tau_n_clip"] = int(np.sum(delta > 1e-9))
        row["arm_tau_max_abs_clip_delta"] = float(np.max(delta)) if delta.size else 0.0


def emit_teleop_ik_row(
    rt: Any,
    ik_details: dict[str, Any] | None,
    q_sol: np.ndarray | None,
) -> None:
    """JSONL row for keyboard / UDP / OpenVR teleop (no OpenVR UDP stream receiver)."""
    if rt._trace is None or not trace_should_emit(rt):
        return
    now = time.monotonic()
    ti = int(rt.torso_index)
    xpos = np.asarray(rt.data.xpos[ti], dtype=np.float64).reshape(3)
    xmat = np.asarray(rt.data.xmat[ti], dtype=np.float64).reshape(3, 3)
    xquat = np.asarray(rt.data.xquat[ti], dtype=np.float64).reshape(4)
    row: dict[str, Any] = {
        "kind": "teleop_ik",
        "t_wall": float(now),
        "sim_time": float(rt.data.time),
        "counter": int(rt.counter),
        "arm_ik_solver": "v2",
        "teleop": bool(rt.teleop),
        "openvr_teleop": bool(rt.openvr_teleop),
        "udp_teleop": bool(rt._udp_receiver is not None),
        "pelvis_pos": rt.data.qpos[0:3].astype(float).tolist(),
        "pelvis_quat": rt.data.qpos[3:7].astype(float).tolist(),
        "torso_xpos": xpos.astype(float).tolist(),
        "torso_xmat": xmat.astype(float).reshape(-1).tolist(),
        "torso_xquat": xquat.astype(float).tolist(),
    }
    with rt.cmd_lock:
        cd = rt.control_dict
        row["loco_cmd"] = np.asarray(cd["loco_cmd"], dtype=np.float64).reshape(-1).astype(float).tolist()
        row["height_cmd"] = float(cd["height_cmd"])
        row["rpy_cmd"] = np.asarray(cd["rpy_cmd"], dtype=np.float64).reshape(-1).astype(float).tolist()
        row["freq_cmd"] = float(cd.get("freq_cmd", 0.75))
        if "ee_left_pos" in cd:
            row["ee_left_pos"] = np.asarray(cd["ee_left_pos"], dtype=np.float64).tolist()
            row["ee_left_quat"] = np.asarray(cd["ee_left_quat"], dtype=np.float64).tolist()
            row["ee_right_pos"] = np.asarray(cd["ee_right_pos"], dtype=np.float64).tolist()
            row["ee_right_quat"] = np.asarray(cd["ee_right_quat"], dtype=np.float64).tolist()
    row["ik"] = ik_details
    row["ik_q_sol"] = None if q_sol is None else q_sol.astype(float).tolist()
    append_post_mj_ik_debug(rt, row, ik_details, q_sol)
    rt._trace.write(row)
    maybe_emit_forensic_burst(rt, row)


def emit_vr_compare_row(
    rt: Any,
    *,
    vr_snap: StreamState | None,
    vr_t_rx: float,
    ik_details: dict[str, Any] | None,
    q_sol: np.ndarray | None,
) -> None:
    if not trace_should_emit(rt):
        return
    now = time.monotonic()
    age = float(now - vr_t_rx) if vr_t_rx > 0.0 else float("nan")
    ti = int(rt.torso_index)
    xpos = np.asarray(rt.data.xpos[ti], dtype=np.float64).reshape(3)
    xmat = np.asarray(rt.data.xmat[ti], dtype=np.float64).reshape(3, 3)
    xquat = np.asarray(rt.data.xquat[ti], dtype=np.float64).reshape(4)
    q_ref = torso_xquat_with_receiver_yaw(xmat, xquat, rt.vr_receiver_yaw_deg)
    R_vr = torso_xmat_with_yaw_offset(xmat, rt.vr_receiver_yaw_deg)
    calib = rt._vr_stream_calib

    row: dict[str, Any] = {
        "kind": "vr_compare",
        "t_wall": float(now),
        "sim_time": float(rt.data.time),
        "counter": int(rt.counter),
        "arm_ik_solver": "v2",
        "vr_ik_mode": str(rt.vr_ik_mode),
        "vr_ik_anchor_mode": str(getattr(rt, "vr_ik_anchor_mode", "legacy")),
        "vr_head_orient_mode": str(getattr(rt, "vr_head_orient_mode", "yaw_only")),
        "vr_receiver_yaw_deg": float(rt.vr_receiver_yaw_deg),
        "vr_ik_hand_yaw_deg": float(VR_IK_HAND_INTRINSIC_YAW_DEG),
        "vr_ik_hand_pitch_deg": float(VR_IK_HAND_INTRINSIC_PITCH_DEG),
        "vr_ik_hand_roll_deg": float(VR_IK_HAND_INTRINSIC_ROLL_DEG),
        "vr_ik_hand_quat_offset_wxyz": np.asarray(
            VR_IK_HAND_OFFSET_QUAT_WXYZ, dtype=np.float64
        )
        .reshape(4)
        .astype(float)
        .tolist(),
        "vr_stream_calib_orient": str(rt.vr_stream_calib_orient),
        "vr_debug_stream": bool(rt.vr_debug_stream),
        "vr_debug_torso_z_mode": str(rt.vr_debug_torso_z_mode),
        "vr_debug_z_offset_m": float(rt.vr_debug_z_offset_m),
        "pelvis_pos": rt.data.qpos[0:3].astype(float).tolist(),
        "pelvis_quat": rt.data.qpos[3:7].astype(float).tolist(),
        "torso_xpos": xpos.astype(float).tolist(),
        "torso_xmat": xmat.astype(float).reshape(-1).tolist(),
        "torso_xquat": xquat.astype(float).tolist(),
        "torso_q_ref": q_ref.astype(float).tolist(),
        "torso_R_vr": R_vr.astype(float).reshape(-1).tolist(),
        "vr_recv_mono": float(vr_t_rx),
        "vr_age_s": age,
        "vr_stream": None
        if vr_snap is None
        else {
            "seq": int(vr_snap.seq),
            "stamp": float(vr_snap.stamp),
            "hmd": trace_pose_dict(vr_snap.hmd),
            "left": trace_pose_dict(vr_snap.left),
            "right": trace_pose_dict(vr_snap.right),
            "sticks": {
                "left": vr_snap.left_stick_xy.astype(float).tolist(),
                "right": vr_snap.right_stick_xy.astype(float).tolist(),
            },
            "trigger": {
                "left": float(vr_snap.left_trigger),
                "right": float(vr_snap.right_trigger),
            },
            "inputs_valid": {
                "left": bool(vr_snap.left_input_valid),
                "right": bool(vr_snap.right_input_valid),
            },
        },
        "calib_ready": bool(calib is not None and calib.is_ready()),
        "calib_hmd0_pos": None
        if calib is None or calib.hmd0_pos is None
        else calib.hmd0_pos.astype(float).tolist(),
        "calib_hmd0_quat": None
        if calib is None or calib.hmd0_quat is None
        else calib.hmd0_quat.astype(float).tolist(),
        "overlay": {},
        "ik": ik_details,
        "ik_q_sol": None if q_sol is None else q_sol.astype(float).tolist(),
    }

    overlay: dict[str, Any] = {}
    if vr_snap is not None and calib is not None:
        calib.maybe_calibrate(vr_snap, orient_mode=rt.vr_stream_calib_orient)
        anchor_tr = str(getattr(rt, "vr_ik_anchor_mode", "legacy"))
        pi_tr = int(rt.pelvis_index)
        if pi_tr >= 0:
            ppos_tr = np.asarray(rt.data.xpos[pi_tr], dtype=np.float64).reshape(3)
            pxm_tr = np.asarray(rt.data.xmat[pi_tr], dtype=np.float64).reshape(3, 3)
        else:
            ppos_tr, pxm_tr = None, None
        mount_tr = getattr(rt, "_vr_ik_mount_mat4", None)
        devices = (("hmd", vr_snap.hmd, 0), ("left", vr_snap.left, 1), ("right", vr_snap.right, 2))
        for name, dev, idx in devices:
            if not dev.valid:
                overlay[name] = {"valid": False}
                continue
            if calib.is_ready():
                if anchor_tr == "legacy":
                    po = pose_for_overlay_device(
                        dev,
                        torso_xpos=xpos,
                        torso_xmat=R_vr,
                        calib=calib,
                        device_index=idx,
                        z_mode=rt.vr_debug_torso_z_mode,
                        z_fixed_world_m=float(rt.vr_debug_z_offset_m),
                    )
                else:
                    po0 = world_pose_stream_device_for_anchor(
                        idx,
                        dev,
                        stream=vr_snap,
                        anchor_mode=anchor_tr,
                        torso_xpos=xpos,
                        torso_R_vr=R_vr,
                        calib=calib,
                        yaw_offset_deg=float(rt.vr_receiver_yaw_deg),
                        pelvis_xpos=ppos_tr,
                        pelvis_xmat=pxm_tr,
                        mount_T=mount_tr,
                        head_orient_mode=str(getattr(rt, "vr_head_orient_mode", "yaw_only")),
                    )
                    if po0.valid:
                        pz = calib.apply_z_mode(
                            idx,
                            po0.pos,
                            rt.vr_debug_torso_z_mode,
                            float(rt.vr_debug_z_offset_m),
                        )
                        po = PoseState(valid=True, pos=pz, quat=po0.quat)
                    else:
                        po = PoseState(
                            valid=False,
                            pos=np.asarray(dev.pos, dtype=np.float64).reshape(3).copy(),
                            quat=np.asarray(dev.quat, dtype=np.float64).reshape(4).copy(),
                        )
                overlay[name] = trace_pose_dict(po)
            else:
                overlay[name] = {
                    "valid": True,
                    "note": "pre_calib_raw_plus_z_offset",
                    "pos": (np.asarray(dev.pos, dtype=np.float64).reshape(3)).tolist(),
                    "quat": (np.asarray(dev.quat, dtype=np.float64).reshape(4)).tolist(),
                }
    row["overlay"] = overlay

    if vr_snap is not None and calib is not None and calib.is_ready():
        for side, dev, okey in (
            ("left", vr_snap.left, "left"),
            ("right", vr_snap.right, "right"),
        ):
            if not dev.valid:
                continue
            pw0 = world_pose_stream_device_for_anchor(
                1 if side == "left" else 2,
                dev,
                stream=vr_snap,
                anchor_mode=str(getattr(rt, "vr_ik_anchor_mode", "legacy")),
                torso_xpos=xpos,
                torso_R_vr=R_vr,
                calib=calib,
                yaw_offset_deg=float(rt.vr_receiver_yaw_deg),
                pelvis_xpos=ppos_tr if pi_tr >= 0 else None,
                pelvis_xmat=pxm_tr if pi_tr >= 0 else None,
                mount_T=mount_tr,
                head_orient_mode=str(getattr(rt, "vr_head_orient_mode", "yaw_only")),
            )
            if pw0.valid:
                pz_w = calib.apply_z_mode(
                    1 if side == "left" else 2,
                    pw0.pos,
                    rt.vr_debug_torso_z_mode,
                    float(rt.vr_debug_z_offset_m),
                )
                pw_dbg = PoseState(valid=True, pos=pz_w, quat=pw0.quat)
            else:
                pw_dbg = pw0
            op = overlay.get(okey)
            if isinstance(op, dict) and op.get("valid") and pw_dbg.valid:
                op_p = np.asarray(op["pos"], dtype=np.float64).reshape(3)
                row[f"dbg_pw_minus_overlay_{side}_l2"] = float(
                    np.linalg.norm(pw_dbg.pos - op_p)
                )
            if isinstance(ik_details, dict):
                ikw = ik_details.get(f"ik_world_{side}_from_ee")
                if isinstance(ikw, list) and len(ikw) == 3 and pw_dbg.valid:
                    row[f"dbg_pw_minus_ik_world_{side}_l2"] = float(
                        np.linalg.norm(
                            np.asarray(pw_dbg.pos, dtype=np.float64).reshape(3)
                            - np.asarray(ikw, dtype=np.float64).reshape(3)
                        )
                    )
                if (
                    isinstance(op, dict)
                    and op.get("valid")
                    and isinstance(ikw, list)
                    and len(ikw) == 3
                ):
                    row[f"dbg_overlay_minus_ik_world_{side}_l2"] = float(
                        np.linalg.norm(
                            np.asarray(op["pos"], dtype=np.float64).reshape(3)
                            - np.asarray(ikw, dtype=np.float64).reshape(3)
                        )
                    )

    if vr_snap is not None and vr_snap.left.valid:
        row["delta_stream_left_pos"] = _trace_step_delta(
            rt, "stream_left_pos", vr_snap.left.pos
        )
    if vr_snap is not None and vr_snap.right.valid:
        row["delta_stream_right_pos"] = _trace_step_delta(
            rt, "stream_right_pos", vr_snap.right.pos
        )

    append_post_mj_ik_debug(rt, row, ik_details, q_sol)

    rt._trace.write(row)
