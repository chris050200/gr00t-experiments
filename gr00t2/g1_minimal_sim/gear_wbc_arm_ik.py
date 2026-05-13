"""Teleop / VR IK: torso-frame ``ee_*`` → dual-arm joint targets."""

from __future__ import annotations

from typing import Any

import numpy as np

from arm_ik import body_palm_pose, ee_world_targets_for_ik
from arm_ik_v2 import solve_dual_arm_ik_v2
from ee_frame import ee_pose_world_from_ref, normalize_quat, quat_conj, quat_mul
from hand_gripper import (
    SL_LEFT_ARM,
    SL_LEFT_HAND,
    SL_RIGHT_ARM,
    SL_RIGHT_HAND,
    hand_target_q,
)


# Per-arm joint order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
# wrist_roll, wrist_pitch, wrist_yaw (see ``arm_ik.LEFT_ARM_JOINTS``).
def _limit_arm_ik_step(
    q_cur: np.ndarray,
    q_sol: np.ndarray,
    *,
    dq_max: float,
    l2_cap: float,
    step_blend: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Limit change from current arm qpos to IK output (anti-spike / explosion guard).

    Applied once per sim step after ``solve_dual_arm_ik_v2``. Order: per-joint clip,
    optional global L2 rescale, optional blend toward current configuration.
    """
    qc = np.asarray(q_cur, dtype=np.float64).reshape(-1)
    qs = np.asarray(q_sol, dtype=np.float64).reshape(-1)
    if qc.shape != qs.shape:
        raise ValueError(f"q_cur and q_sol shape mismatch {qc.shape} vs {qs.shape}")
    delta = qs - qc
    raw_l2 = float(np.linalg.norm(delta))
    dm = float(dq_max)
    if dm > 0.0:
        delta = np.clip(delta, -dm, dm)
    lc = float(l2_cap)
    if lc > 0.0:
        n = float(np.linalg.norm(delta))
        if n > lc and n > 1e-12:
            delta *= lc / n
    beta = float(np.clip(step_blend, 0.0, 1.0))
    q_mid = qc + delta
    q_out = qc + beta * (q_mid - qc)
    out_l2 = float(np.linalg.norm(q_out - qc))
    diag = {
        "ik_arm_delta_raw_l2": raw_l2,
        "ik_arm_delta_out_l2": out_l2,
    }
    return q_out.astype(np.float32), diag


_ARM_IK_POSTURE_JOINT_WEIGHTS = np.concatenate(
    [
        np.array(
            [0.18, 0.38, 1.0, 0.10, 0.16, 0.16, 0.16],
            dtype=np.float64,
        ),
        np.array(
            [0.18, 0.38, 1.0, 0.10, 0.16, 0.16, 0.16],
            dtype=np.float64,
        ),
    ]
)


def _arm_q_neutral_14(rt: Any) -> np.ndarray:
    """14-vector (L then R arm joints) from ``_arm_target_q_home`` in ``arm_target_q`` order."""
    home = np.asarray(rt.control_dict["_arm_target_q_home"], dtype=np.float64).reshape(-1)
    if rt.has_hands:
        return np.concatenate([home[SL_LEFT_ARM], home[SL_RIGHT_ARM]], dtype=np.float64)
    return home[:14].copy()


def _quat_geodesic_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    q1 = normalize_quat(np.asarray(qa, dtype=np.float64).reshape(4))
    q2 = normalize_quat(np.asarray(qb, dtype=np.float64).reshape(4))
    d = float(abs(float(np.dot(q1, q2))))
    d = min(1.0, max(0.0, d))
    return float(2.0 * np.degrees(np.arccos(d)))


def _lock_ee_quats_to_home_under_teleop(rt: Any) -> None:
    """Must run under ``rt.cmd_lock``. World-fixed palm orientation for keyboard teleop.

    Constant **torso-frame** quats would still change **world** palm orientation when the
    torso yaws while walking. We store world anchors at init (and refresh on ``z``) and each
    step set ``ee_*_quat`` so ``quat_mul(q_torso, q_ee)`` matches those anchors (same basis as
    ``ee_pose_world_from_ref`` / ``_ee_decode_quat``).
    """
    if not bool(getattr(rt, "_lock_ee_orient", False)):
        return
    if not bool(getattr(rt, "teleop", False)):
        return
    if bool(getattr(rt, "vr_teleop", False)) and getattr(rt, "vr_ik_mode", "off") != "off":
        return
    cd = rt.control_dict
    if "_ee_left_quat_world_anchor" not in cd:
        return
    ti = int(rt.torso_index)
    xm = np.asarray(rt.data.xmat[ti], dtype=np.float64).reshape(3, 3)
    xq = np.asarray(rt.data.xquat[ti], dtype=np.float64).reshape(4)
    qr = normalize_quat(rt._ee_decode_quat(xm, xq))
    Ql = normalize_quat(np.asarray(cd["_ee_left_quat_world_anchor"], dtype=np.float64).reshape(4))
    Qr = normalize_quat(np.asarray(cd["_ee_right_quat_world_anchor"], dtype=np.float64).reshape(4))
    cd["ee_left_quat"][:] = quat_mul(quat_conj(qr), Ql).astype(np.float32)
    cd["ee_right_quat"][:] = quat_mul(quat_conj(qr), Qr).astype(np.float32)


def apply_arm_ik_step(
    rt: Any,
    vr_ik_trace: dict[str, Any] | None,
) -> np.ndarray | None:
    """Solve IK and write ``arm_target_q`` arm slices (+ hands). Returns commanded arm ``q`` (post-clamp) for trace."""
    with rt.cmd_lock:
        _lock_ee_quats_to_home_under_teleop(rt)
        tl_p = rt.control_dict["ee_left_pos"].copy()
        tl_q = rt.control_dict["ee_left_quat"].copy()
        tr_p = rt.control_dict["ee_right_pos"].copy()
        tr_q = rt.control_dict["ee_right_quat"].copy()
    ti_ik = int(rt.torso_index)
    xmat_ik = rt.data.xmat[ti_ik]
    xquat_ik = rt.data.xquat[ti_ik]
    xm = np.asarray(xmat_ik, dtype=np.float64).reshape(3, 3)
    ee_dec_q = rt._ee_decode_quat(xm, xquat_ik)
    ee_dec_R = rt._ee_ref_rotmat(xm, xquat_ik)
    wp_l, wq_l, wp_r, wq_r = ee_world_targets_for_ik(
        rt.data,
        rt.torso_index,
        tl_p,
        tl_q,
        tr_p,
        tr_q,
        ref_quat_wxyz=ee_dec_q,
        ref_rotmat=ee_dec_R,
    )
    lock_on = bool(getattr(rt, "_lock_ee_orient", False) and getattr(rt, "teleop", False))
    lock_anchor_err_l_deg = float("nan")
    lock_anchor_err_r_deg = float("nan")
    lock_anchor_l = None
    lock_anchor_r = None
    if lock_on:
        with rt.cmd_lock:
            cd = rt.control_dict
            if "_ee_left_quat_world_anchor" in cd and "_ee_right_quat_world_anchor" in cd:
                lock_anchor_l = normalize_quat(
                    np.asarray(cd["_ee_left_quat_world_anchor"], dtype=np.float64).reshape(4)
                )
                lock_anchor_r = normalize_quat(
                    np.asarray(cd["_ee_right_quat_world_anchor"], dtype=np.float64).reshape(4)
                )
                lock_anchor_err_l_deg = _quat_geodesic_deg(lock_anchor_l, wq_l)
                lock_anchor_err_r_deg = _quat_geodesic_deg(lock_anchor_r, wq_r)
    pml, qml = body_palm_pose(rt.data, rt.left_ik)
    pmr, qmr = body_palm_pose(rt.data, rt.right_ik)
    err_pos_max = float(
        max(
            np.linalg.norm(np.asarray(wp_l, dtype=np.float64).reshape(3) - pml),
            np.linalg.norm(np.asarray(wp_r, dtype=np.float64).reshape(3) - pmr),
        )
    )
    err_quat_max_deg = float(
        max(_quat_geodesic_deg(wq_l, qml), _quat_geodesic_deg(wq_r, qmr))
    )
    q_cur = np.asarray(rt.data.qpos[rt.arm_qpos_ids], dtype=np.float64).reshape(-1)
    posture_w = float(rt.arm_ik_posture_weight)
    q_sol = solve_dual_arm_ik_v2(
        rt.model,
        rt.data,
        rt.left_ik,
        rt.right_ik,
        wp_l,
        wq_l,
        wp_r,
        wq_r,
        arm_qpos_ids=rt.arm_qpos_ids,
        rot_weight=rt.arm_ik_rot_weight,
        q_neutral=_arm_q_neutral_14(rt),
        posture_weight=posture_w,
        per_joint_weight=_ARM_IK_POSTURE_JOINT_WEIGHTS,
        q_prev=rt._ik_q_prev,
        temporal_weight=float(rt.arm_ik_temporal_weight),
    )
    dq_max = float(getattr(rt, "arm_ik_frame_dq_max", 0.16))
    l2_cap = float(getattr(rt, "arm_ik_frame_dq_l2_cap", 1.2))
    step_blend = float(getattr(rt, "arm_ik_step_blend", 1.0))
    q_out, step_diag = _limit_arm_ik_step(
        q_cur,
        q_sol,
        dq_max=dq_max,
        l2_cap=l2_cap,
        step_blend=step_blend,
    )
    rt._ik_q_prev = np.asarray(q_out, dtype=np.float64).copy()
    raw_l2 = float(step_diag["ik_arm_delta_raw_l2"])
    out_l2 = float(step_diag["ik_arm_delta_out_l2"])
    step_scale = float(out_l2 / raw_l2) if raw_l2 > 1e-9 else 1.0
    if vr_ik_trace is not None:
        ti = int(rt.torso_index)
        xp = np.asarray(rt.data.xpos[ti], dtype=np.float64).reshape(3)
        xm = np.asarray(rt.data.xmat[ti], dtype=np.float64).reshape(3, 3)
        xq = np.asarray(rt.data.xquat[ti], dtype=np.float64).reshape(4)
        tl_pd = np.asarray(tl_p, dtype=np.float64).reshape(3)
        tl_qd = np.asarray(tl_q, dtype=np.float64).reshape(4)
        tr_pd = np.asarray(tr_p, dtype=np.float64).reshape(3)
        tr_qd = np.asarray(tr_q, dtype=np.float64).reshape(4)
        xq_ikw = rt._ee_decode_quat(xm, xq)
        xm_dec = rt._ee_ref_rotmat(xm, xq)
        ikw_l, ikw_ql = ee_pose_world_from_ref(xp, xm_dec, xq_ikw, tl_pd, tl_qd)
        ikw_r, ikw_qr = ee_pose_world_from_ref(xp, xm_dec, xq_ikw, tr_pd, tr_qd)
        vr_ik_trace["ik_world_left_from_ee"] = ikw_l.astype(float).tolist()
        vr_ik_trace["ik_world_left_quat_from_ee"] = ikw_ql.astype(float).tolist()
        vr_ik_trace["ik_world_right_from_ee"] = ikw_r.astype(float).tolist()
        vr_ik_trace["ik_world_right_quat_from_ee"] = ikw_qr.astype(float).tolist()
        vr_ik_trace["ik_dbg_target_quat_vs_recon_left_deg"] = _quat_geodesic_deg(wq_l, ikw_ql)
        vr_ik_trace["ik_dbg_target_quat_vs_recon_right_deg"] = _quat_geodesic_deg(wq_r, ikw_qr)
        vr_ik_trace["ik_dbg_meas_pre_vs_target_left_deg"] = _quat_geodesic_deg(qml, wq_l)
        vr_ik_trace["ik_dbg_meas_pre_vs_target_right_deg"] = _quat_geodesic_deg(qmr, wq_r)
        vr_ik_trace["ik_dbg_lock_mode_active"] = bool(lock_on)
        vr_ik_trace["ik_dbg_lock_anchor_vs_target_left_deg"] = float(lock_anchor_err_l_deg)
        vr_ik_trace["ik_dbg_lock_anchor_vs_target_right_deg"] = float(lock_anchor_err_r_deg)
        if lock_anchor_l is not None and lock_anchor_r is not None:
            vr_ik_trace["ik_dbg_lock_anchor_world_left_quat"] = lock_anchor_l.astype(float).tolist()
            vr_ik_trace["ik_dbg_lock_anchor_world_right_quat"] = lock_anchor_r.astype(float).tolist()
        vr_ik_trace["ik_dbg_target_world_left_quat"] = (
            np.asarray(wq_l, dtype=np.float64).reshape(4).astype(float).tolist()
        )
        vr_ik_trace["ik_dbg_target_world_right_quat"] = (
            np.asarray(wq_r, dtype=np.float64).reshape(4).astype(float).tolist()
        )
        vr_ik_trace["ik_solve_torso_xpos"] = xp.astype(float).tolist()
        vr_ik_trace["ik_solve_torso_xmat"] = xm.astype(float).reshape(-1).tolist()
        vr_ik_trace["ik_solve_torso_xquat"] = xq.astype(float).tolist()
        vr_ik_trace["ik_posture_weight_effective"] = float(posture_w)
        vr_ik_trace["ik_arm_delta_raw_l2"] = raw_l2
        vr_ik_trace["ik_arm_delta_out_l2"] = out_l2
        vr_ik_trace["ik_step_limit_scale"] = float(step_scale)
        vr_ik_trace["ik_track_err_pos_max_m"] = float(err_pos_max)
        vr_ik_trace["ik_track_err_quat_max_deg"] = float(err_quat_max_deg)
    with rt.cmd_lock:
        atq = rt.control_dict["arm_target_q"]
        if rt.has_hands:
            atq[SL_LEFT_ARM] = q_out[:7]
            atq[SL_RIGHT_ARM] = q_out[7:14]
            gl = float(rt.control_dict["gripper_left"])
            gr = float(rt.control_dict["gripper_right"])
        else:
            atq[: q_out.shape[0]] = q_out
            gl = gr = 1.0
    if rt.has_hands:
        hl = hand_target_q(rt.model, "left", gl)
        hr = hand_target_q(rt.model, "right", gr)
        with rt.cmd_lock:
            rt.control_dict["arm_target_q"][SL_LEFT_HAND] = hl
            rt.control_dict["arm_target_q"][SL_RIGHT_HAND] = hr
    return np.asarray(q_out, dtype=np.float32).copy()
