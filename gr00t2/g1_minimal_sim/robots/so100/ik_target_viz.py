"""Red arrow for desired EE pose in the passive MuJoCo viewer (``user_scn``)."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

# Similar scale to ``vr_teleop/viewer_arrows.VR_DEVICE_ARROW_SIZE`` (shaft, head, length).
_IK_ARROW_SIZE = np.array([0.01, 0.018, 0.18], dtype=np.float64)
_IK_ARROW_RGBA = np.array([1.0, 0.15, 0.15, 1.0], dtype=np.float32)


def mjv_mat9_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    mat9 = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat9, np.asarray(quat, dtype=np.float64).reshape(4))
    return mat9


def fill_ik_target_arrow(
    geoms: Any,
    geom_start: int,
    pos_world: np.ndarray,
    quat_world_wxyz: np.ndarray,
    *,
    enabled: bool,
) -> int:
    """Draw one red ``mjGEOM_ARROW`` at desired world pose. Returns next free geom index."""
    if not enabled:
        return geom_start
    mujoco.mjv_initGeom(
        geoms[geom_start],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=_IK_ARROW_SIZE,
        pos=np.asarray(pos_world, dtype=np.float64).reshape(3),
        mat=mjv_mat9_from_quat_wxyz(quat_world_wxyz),
        rgba=_IK_ARROW_RGBA,
    )
    return geom_start + 1
