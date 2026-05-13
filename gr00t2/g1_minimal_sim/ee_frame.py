"""Reference-body ↔ world transforms for teleop EE targets (numpy only, no MuJoCo)."""

from __future__ import annotations

import numpy as np


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix **R** with columns = body axes in world frame (matches MuJoCo ``xmat``).

    Use this when pairing orientation (`data.xquat`) with translation so **R** and **qr**
    describe the **same** SO(3) element — avoids mixing ``data.xmat`` with ``data.xquat``
    when the two can drift apart numerically.
    """
    w, x, y, z = normalize_quat(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternions), wxyz."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2, both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return normalize_quat(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def intrinsic_zyx_deg_to_quat_wxyz(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Unit quaternion (wxyz) for intrinsic Z–Y′–X″ (Tait–Bryan) angles in degrees.

    Matches rotation matrix ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (radians internally).
    Intended for **post-multiply** on a streamed device orientation:
    ``quat_mul(q_device, q_offset)`` applies this offset in the device's **local** axes.
    """
    ay = np.radians(float(yaw_deg))
    ap = np.radians(float(pitch_deg))
    ar = np.radians(float(roll_deg))
    hz, hy, hx = 0.5 * ay, 0.5 * ap, 0.5 * ar
    qz = normalize_quat(
        np.array([np.cos(hz), 0.0, 0.0, np.sin(hz)], dtype=np.float64)
    )
    qy = normalize_quat(
        np.array([np.cos(hy), 0.0, np.sin(hy), 0.0], dtype=np.float64)
    )
    qx = normalize_quat(
        np.array([np.cos(hx), np.sin(hx), 0.0, 0.0], dtype=np.float64)
    )
    return normalize_quat(quat_mul(quat_mul(qz, qy), qx))


def ee_pose_ref_from_world(
    xpos: np.ndarray,
    xmat: np.ndarray,
    q_ref_wxyz: np.ndarray,
    p_world: np.ndarray,
    q_world_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Palm pose in reference-body frame from world (MuJoCo wxyz)."""
    R = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    pw = np.asarray(p_world, dtype=np.float64).reshape(3)
    xp = np.asarray(xpos, dtype=np.float64).reshape(3)
    p_ref = R.T @ (pw - xp)
    qr = normalize_quat(q_ref_wxyz)
    qw = normalize_quat(q_world_wxyz)
    q_ee = quat_mul(quat_conj(qr), qw)
    return p_ref, normalize_quat(q_ee)


def ee_pose_world_from_ref(
    xpos: np.ndarray,
    xmat: np.ndarray,
    q_ref_wxyz: np.ndarray,
    p_ref: np.ndarray,
    q_ee_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World palm pose from reference-body-frame target."""
    R = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    pr = np.asarray(p_ref, dtype=np.float64).reshape(3)
    xp = np.asarray(xpos, dtype=np.float64).reshape(3)
    p_world = xp + R @ pr
    qr = normalize_quat(q_ref_wxyz)
    qe = normalize_quat(q_ee_wxyz)
    q_world = quat_mul(qr, qe)
    return p_world, normalize_quat(q_world)
