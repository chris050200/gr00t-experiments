"""SteamVR / OpenVR device poses → MuJoCo teleop frame (numpy only).

Matches the proven mapping from the standalone debug visualizer:

- **Linear:** OpenVR room position ``(x,y,z)`` → MuJoCo ``[-z, -x, y]`` (Z-up, +X forward, +Y left).
- **Rotation:** ``quat_mj = quat_mul(T_STEAMVR_TO_MJ_QUAT_WXYZ, quat_from_matrix(R_openvr))``
  with fixed ``T = [0.5, 0.5, -0.5, -0.5]`` (wxyz), equivalent to left-multiplying the
  OpenVR 3×3 by the same linear map as for positions.

No ``transforms3d`` dependency — use this module on the VR PC with only ``numpy`` + ``openvr``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Fixed quaternion (wxyz) from the working debug script: same as composing the linear R_fix.
T_STEAMVR_TO_MJ_QUAT_WXYZ = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float64)

# 3×3: t_mj = R_STEAMVR_LINEAR_TO_MJ @ t_openvr  ==  [-z, -x, y]
R_STEAMVR_LINEAR_TO_MJ = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product (wxyz), same as ``transforms3d.quaternions.qmult`` for scalar-first."""
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


def quat_from_matrix_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) → unit quaternion wxyz (Shepperd, same as openvr_teleop_client)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(R)
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
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
    return quat_normalize_wxyz(np.array([w, x, y, z], dtype=np.float64))


def steamvr_linear_to_mujoco_pos(pos_steamvr: np.ndarray) -> np.ndarray:
    """Map OpenVR linear coordinates to MuJoCo teleop world (meters)."""
    x_vr, y_vr, z_vr = np.asarray(pos_steamvr, dtype=np.float64).reshape(3)
    return np.array([-z_vr, -x_vr, y_vr], dtype=np.float64)


def steamvr_rotmat_to_mujoco_quat_wxyz(R_steamvr: np.ndarray) -> np.ndarray:
    """OpenVR device rotation matrix → MuJoCo body quaternion wxyz (debug-script path)."""
    q_vr = quat_from_matrix_wxyz(R_steamvr)
    return quat_normalize_wxyz(quat_mul_wxyz(T_STEAMVR_TO_MJ_QUAT_WXYZ, q_vr))


def openvr_mat34_to_Rt(M: Any) -> tuple[np.ndarray, np.ndarray]:
    """``HmdMatrix34_t`` (``M.m`` row lists) → ``R`` (3,3), ``t`` (3,) SteamVR frame."""
    m = M.m
    R = np.array(
        [
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]],
        ],
        dtype=np.float64,
    )
    t = np.array([m[0][3], m[1][3], m[2][3]], dtype=np.float64)
    return R, t


def openvr_pose_to_mujoco(
    M: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """One OpenVR ``mDeviceToAbsoluteTracking`` matrix → ``(pos_mj, quat_wxyz_mj)``."""
    return openvr_pose_to_mujoco_custom(M, R_STEAMVR_LINEAR_TO_MJ)


def openvr_pose_to_mujoco_custom(
    M: Any,
    R_linear: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """OpenVR mat34 → MJ pos/quat with ``t_mj = R_linear @ t`` and ``R_mj = R_linear @ R_ovr``.

    Default ``R_linear`` is ``R_STEAMVR_LINEAR_TO_MJ`` (same as ``openvr_pose_to_mujoco``).
    Use a custom 3×3 for ``openvr_teleop_client --axis-map``.
    """
    R, t = openvr_mat34_to_Rt(M)
    R_lin = np.asarray(R_linear, dtype=np.float64).reshape(3, 3)
    t_mj = R_lin @ np.asarray(t, dtype=np.float64).reshape(3)
    R_mj = R_lin @ R
    return t_mj, quat_from_matrix_wxyz(R_mj)


def quat_fix_from_linear_map() -> np.ndarray:
    """Quaternion wxyz equivalent to ``R_STEAMVR_LINEAR_TO_MJ`` (for cross-checks)."""
    return quat_from_matrix_wxyz(R_STEAMVR_LINEAR_TO_MJ)
