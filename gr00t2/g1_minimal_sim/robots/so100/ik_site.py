"""Damped least-squares IK for a MuJoCo site (world-frame Jacobian and targets)."""

from __future__ import annotations

import numpy as np
import mujoco


def quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_wxyz_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_wxyz_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(axis))
    if na < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / na
    ha = 0.5 * float(angle)
    s = np.sin(ha)
    return np.array([np.cos(ha), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (world from body) -> quaternion wxyz."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(R))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return quat_wxyz_normalize(np.array([w, x, y, z], dtype=np.float64))


def mat_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = quat_wxyz_normalize(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotvec_from_mat(R: np.ndarray) -> np.ndarray:
    """Rotation vector ``r`` with ``|r| <= pi`` such that ``exp([r]_x) ≈ R`` for small angles."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-10:
        return np.zeros(3, dtype=np.float64)
    rx = (R[2, 1] - R[1, 2]) * 0.5
    ry = (R[0, 2] - R[2, 0]) * 0.5
    rz = (R[1, 0] - R[0, 1]) * 0.5
    denom = float(np.sin(theta))
    if abs(denom) < 1e-10:
        return np.zeros(3, dtype=np.float64)
    k = np.array([rx, ry, rz], dtype=np.float64) / denom
    return (k * theta).astype(np.float64)


def base_to_world_pose(
    data: mujoco.MjData,
    base_body_id: int,
    p_des_base: np.ndarray,
    quat_des_base_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """EE pose expressed in Base frame -> world frame.

    ``p_des_base`` is the EE origin in Base coordinates.
    ``quat_des_base_wxyz`` maps EE frame vectors into Base frame (same convention as ``xmat``).
    """
    R_wb = data.xmat[base_body_id].reshape(3, 3)
    p_b = data.xpos[base_body_id]
    p_w = p_b + R_wb @ np.asarray(p_des_base, dtype=np.float64).reshape(3)
    R_ee_b = mat_from_quat_wxyz(quat_des_base_wxyz)
    R_w = R_wb @ R_ee_b
    return p_w, R_w


def fk_site_pose_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_body_id: int,
    site_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Current site pose relative to Base: position in base coords, orientation quat wxyz (EE from base)."""
    mujoco.mj_forward(model, data)
    R_wb = data.xmat[base_body_id].reshape(3, 3)
    p_b = data.xpos[base_body_id]
    p_ee = data.site_xpos[site_id]
    R_ee_w = data.site_xmat[site_id].reshape(3, 3)
    p_rel = R_wb.T @ (p_ee - p_b)
    R_rel = R_wb.T @ R_ee_w
    q_rel = mat_to_quat_wxyz(R_rel)
    return p_rel, q_rel


def fk_site_pose_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Current site pose in **world**: origin ``site_xpos``, orientation quat wxyz from ``site_xmat``."""
    mujoco.mj_forward(model, data)
    p_w = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    R_w = data.site_xmat[site_id].reshape(3, 3)
    q_w = mat_to_quat_wxyz(R_w)
    return p_w, q_w


def clip_qpos_to_limits(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        lo, hi = model.jnt_range[j]
        if lo < hi:
            data.qpos[adr] = float(np.clip(data.qpos[adr], lo, hi))


def ik_step_site(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    site_id: int,
    p_des_world: np.ndarray,
    R_des_world: np.ndarray,
    arm_nv: int = 5,
    damping: float = 0.035,
    pos_gain: float = 2.5,
    rot_gain: float = 0.45,
    max_dq: float = 0.22,
    pos_only: bool = False,
) -> None:
    """One damped IK step toward ``(p_des_world, R_des_world)`` for ``site_id``.

    Only the first ``arm_nv`` velocity coordinates are used (SO-100: arm without Jaw for ``ee_site``).

    If ``pos_only`` is True, only **position** error is minimized (3×``arm_nv`` Jacobian). Use when
    orientation is less important than reducing **XYZ** error on a 5-DoF arm.
    """
    mujoco.mj_forward(model, data)
    p = data.site_xpos[site_id].copy()
    R = data.site_xmat[site_id].reshape(3, 3)
    e_p = pos_gain * (np.asarray(p_des_world, dtype=np.float64).reshape(3) - p)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    Jp = jacp[:, :arm_nv]

    if pos_only:
        e = e_p.reshape(3)
        J = Jp
        dim = 3
    else:
        R_err = R_des_world @ R.T
        e_r = rot_gain * rotvec_from_mat(R_err)
        e = np.concatenate([e_p, e_r], axis=0)
        J = np.vstack([jacp, jacr])[:, :arm_nv]
        dim = 6

    jjt = J @ J.T
    a_mat = jjt + (damping**2) * np.eye(dim, dtype=np.float64)
    dq5 = J.T @ np.linalg.solve(a_mat, e)
    dq5 = np.clip(dq5, -max_dq, max_dq)

    for i in range(arm_nv):
        data.qpos[i] = float(data.qpos[i] + dq5[i])

    clip_qpos_to_limits(model, data)
    mujoco.mj_forward(model, data)
