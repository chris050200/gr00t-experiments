"""Shared MuJoCo passive-viewer arrow geoms (HMD / controllers)."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .openvr_stream import PoseState

# Match vr_lab.run_vr_lab device arrows.
VR_DEVICE_ARROW_SIZE = np.array([0.012, 0.022, 0.20], dtype=np.float64)
RGBA_HMD = np.array([0.2, 0.45, 1.0, 1.0], dtype=np.float32)
RGBA_LEFT = np.array([0.1, 1.0, 0.25, 1.0], dtype=np.float32)
RGBA_RIGHT = np.array([1.0, 0.55, 0.1, 1.0], dtype=np.float32)


def mjv_mat_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    mat9 = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat9, np.asarray(quat, dtype=np.float64).reshape(4))
    return mat9


def init_pose_arrow_geom(
    geom: Any,
    pose: PoseState,
    rgba: np.ndarray,
    *,
    size: np.ndarray | None = None,
) -> None:
    """One ``mjGEOM_ARROW`` at ``pose``; dim when not ``pose.valid``."""
    if size is None:
        size = VR_DEVICE_ARROW_SIZE
    alpha = 1.0 if pose.valid else 0.18
    color = np.asarray(rgba, dtype=np.float32).copy()
    color[3] = alpha
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.asarray(size, dtype=np.float64).reshape(3),
        pos=np.asarray(pose.pos, dtype=np.float64).reshape(3),
        mat=mjv_mat_from_quat_wxyz(pose.quat),
        rgba=color,
    )
