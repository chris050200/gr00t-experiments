"""Dual-arm IK v2: single stacked damped least-squares step (no L/R alternation).

Same palm targets and ``arm_qpos_ids`` contract as the former dual-arm IK in ``arm_ik`` (v1 removed).
Rotation error uses ``R_err = R_tgt @ R_cur.T`` mapped to an axis–angle-like 3-vector
(vee skew symmetric part), with quaternion hemisphere alignment to reduce flips.
"""

from __future__ import annotations

import mujoco
import numpy as np

from arm_ik import ArmSideIK, _clamp_qpos, body_palm_pose
from arm_ik_posture import augment_task_with_posture


def _quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _rot_error_from_quat(
    q_tgt: np.ndarray,
    q_cur: np.ndarray,
    *,
    max_angle_rad: float,
) -> np.ndarray:
    """Shortest-arc orientation residual as axis-angle vector (world frame)."""
    qt = _normalize_quat(q_tgt)
    qc = _normalize_quat(q_cur)
    q_err = _normalize_quat(_quat_mul(qt, _quat_conj(qc)))
    # Keep shortest arc to avoid the long-way quaternion branch.
    if float(q_err[0]) < 0.0:
        q_err = -q_err
    w = float(np.clip(q_err[0], -1.0, 1.0))
    v = q_err[1:]
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3, dtype=np.float64)
    ang = float(2.0 * np.arctan2(s, w))
    if max_angle_rad > 0.0:
        ang = float(np.clip(ang, -max_angle_rad, max_angle_rad))
    axis = v / s
    return axis * ang


def _align_quat_hemisphere(q_tgt: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q_tgt, dtype=np.float64).reshape(4)
    q0 = np.asarray(q_ref, dtype=np.float64).reshape(4)
    if float(np.dot(q0, q1)) < 0.0:
        return -q1
    return q1


def solve_dual_arm_ik_v2(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left: ArmSideIK,
    right: ArmSideIK,
    target_left_pos: np.ndarray,
    target_left_quat: np.ndarray,
    target_right_pos: np.ndarray,
    target_right_quat: np.ndarray,
    *,
    arm_qpos_ids: np.ndarray,
    outer_iters: int = 8,
    damping: float = 0.15,
    pos_weight: float = 1.0,
    rot_weight: float = 0.6,
    rot_error_max_rad: float = 0.6,
    max_dq: float = 0.25,
    q_neutral: np.ndarray | None = None,
    posture_weight: float = 0.0,
    per_joint_weight: np.ndarray | None = None,
    q_prev: np.ndarray | None = None,
    temporal_weight: float = 0.0,
) -> np.ndarray:
    """Stacked 12×14 Jacobian IK; restores ``data.qpos`` after.

    Optional **posture** rows (``arm_ik_posture.augment_task_with_posture``) bias the nullspace
    toward ``q_neutral``. Optional **temporal** rows pull toward ``q_prev`` (last step's IK
    output) to reduce elbow-flip / discontinuous jumps. Both default off for unit tests.
    """
    q_save = np.asarray(data.qpos, dtype=np.float64).copy()
    cols_l = left.dof_ids.astype(np.int32)
    cols_r = right.dof_ids.astype(np.int32)
    lam = float(damping) ** 2
    try:
        q_arm = np.asarray(data.qpos[arm_qpos_ids], dtype=np.float64).copy()

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)

        for _ in range(int(outer_iters)):
            data.qpos[arm_qpos_ids] = q_arm
            mujoco.mj_forward(model, data)

            p_l, q_l = body_palm_pose(data, left)
            p_r, q_r = body_palm_pose(data, right)
            qt_l = _align_quat_hemisphere(
                np.asarray(target_left_quat, dtype=np.float64).reshape(4), q_l
            )
            qt_r = _align_quat_hemisphere(
                np.asarray(target_right_quat, dtype=np.float64).reshape(4), q_r
            )
            e_p_l = (np.asarray(target_left_pos, dtype=np.float64).reshape(3) - p_l) * pos_weight
            e_p_r = (np.asarray(target_right_pos, dtype=np.float64).reshape(3) - p_r) * pos_weight
            e_r_l = _rot_error_from_quat(
                qt_l, q_l, max_angle_rad=float(rot_error_max_rad)
            ) * rot_weight
            e_r_r = _rot_error_from_quat(
                qt_r, q_r, max_angle_rad=float(rot_error_max_rad)
            ) * rot_weight
            e_task = np.concatenate([e_p_l, e_r_l, e_p_r, e_r_r], axis=0)

            # mj_jac requires `point` in WORLD coordinates, not body-local.
            # `p_l` / `p_r` are the palm world positions just computed by body_palm_pose,
            # so the position-Jacobian column uses the correct lever arm R_body @ palm_local.
            mujoco.mj_jac(model, data, jacp, jacr, p_l, left.body_id)
            Jl = np.vstack(
                [
                    jacp[:, cols_l] * pos_weight,
                    jacr[:, cols_l] * rot_weight,
                ]
            )
            mujoco.mj_jac(model, data, jacp, jacr, p_r, right.body_id)
            Jr = np.vstack(
                [
                    jacp[:, cols_r] * pos_weight,
                    jacr[:, cols_r] * rot_weight,
                ]
            )

            n = 14
            J = np.zeros((12, n), dtype=np.float64)
            J[0:6, 0:7] = Jl
            J[6:12, 7:14] = Jr

            J_use, e_use = augment_task_with_posture(
                J,
                e_task,
                q_arm,
                q_neutral,
                posture_weight,
                per_joint_weight,
            )
            if q_prev is not None and float(temporal_weight) > 0.0:
                wt = np.sqrt(float(temporal_weight))
                qpv = np.asarray(q_prev, dtype=np.float64).reshape(n)
                J_use = np.vstack([J_use, wt * np.eye(n, dtype=np.float64)])
                e_use = np.concatenate([e_use, wt * (qpv - q_arm)], axis=0)

            H = J_use.T @ J_use + lam * np.eye(n, dtype=np.float64)
            g = J_use.T @ e_use

            # Tiny jitter for numerical SPD stability (some MKL builds are strict).
            h_eps = float(lam) * 1e-6 + 1e-12
            dq = np.linalg.solve(H + h_eps * np.eye(n, dtype=np.float64), g)
            dq[:7] = np.clip(dq[:7], -max_dq, max_dq)
            dq[7:] = np.clip(dq[7:], -max_dq, max_dq)
            q_arm = q_arm + dq
            _clamp_qpos(model, left.joint_ids, q_arm[:7])
            _clamp_qpos(model, right.joint_ids, q_arm[7:])

        return q_arm.astype(np.float32)
    finally:
        data.qpos[:] = q_save
        mujoco.mj_forward(model, data)
