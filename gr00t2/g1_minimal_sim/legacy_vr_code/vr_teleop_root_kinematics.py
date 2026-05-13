"""Planar root motion + pose unwrapping for legacy pose-stream receive (numpy only)."""

from __future__ import annotations

import numpy as np


def Rz(yaw: float) -> np.ndarray:
    """Rotation about +Z (MJ up), yaw CCW from +X toward +Y."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_axis_deadzone(v: float, dz: float) -> float:
    av = abs(v)
    if av < dz:
        return 0.0
    return float(np.copysign((av - dz) / max(1.0 - dz, 1e-6), v))


def integrate_root_planar(
    root_x: float,
    root_y: float,
    yaw: float,
    left_x: float,
    left_y: float,
    right_x: float,
    dt: float,
    *,
    move_scale: float,
    yaw_left_scale: float,
    yaw_right_scale: float,
    invert_forward: bool,
) -> tuple[float, float, float]:
    """Joystick → planar ``(x, y, yaw)``. ``left_y`` GLFW: stick up = negative."""
    lx, ly, rx = float(left_x), float(left_y), float(right_x)
    fwd_raw = -ly if not invert_forward else ly
    fwd = float(fwd_raw * move_scale)
    dyaw = float((-lx * yaw_left_scale + rx * yaw_right_scale) * dt)
    yaw_n = yaw + dyaw
    c, s = np.cos(yaw_n), np.sin(yaw_n)
    root_x_n = root_x + c * fwd * dt
    root_y_n = root_y + s * fwd * dt
    return root_x_n, root_y_n, yaw_n


def world_pose_to_root_planar(
    p_w: np.ndarray,
    R_w: np.ndarray,
    t_root: np.ndarray,
    yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Device pose in world → yaw-stripped root frame (translation + Rz(-yaw))."""
    R_inv = Rz(-yaw)
    p_d = R_inv @ (
        np.asarray(p_w, dtype=np.float64).reshape(3)
        - np.asarray(t_root, dtype=np.float64).reshape(3)
    )
    Rw = np.asarray(R_w, dtype=np.float64).reshape(3, 3)
    R_d = R_inv @ Rw
    return p_d, R_d


def compose_root_planar_to_world(
    p_d: np.ndarray,
    R_d: np.ndarray,
    t_root: np.ndarray,
    yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``world_pose_to_root_planar``: ``p_w = t_root + Rz(yaw) @ p_d``, ``R_w = Rz(yaw) @ R_d``."""
    R_yaw = Rz(yaw)
    p_w = np.asarray(t_root, dtype=np.float64).reshape(3) + R_yaw @ np.asarray(
        p_d, dtype=np.float64
    ).reshape(3)
    Rd = np.asarray(R_d, dtype=np.float64).reshape(3, 3)
    R_w = R_yaw @ Rd
    return p_w, R_w
