"""World-space IK palm targets in the passive MuJoCo viewer (``user_scn`` geoms)."""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from arm_ik import ee_world_targets_for_ik

_EE_TARGET_ARROW_SIZE = np.array([0.018, 0.032, 0.32], dtype=np.float64)
_EE_TARGET_LEFT_ARROW_RGBA = np.array([0.05, 0.95, 0.25, 0.95], dtype=np.float32)
_EE_TARGET_RIGHT_ARROW_RGBA = np.array([1.0, 0.40, 0.08, 0.95], dtype=np.float32)


def _mjv_mat9_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    mat9 = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat9, np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    return mat9


def fill_ee_target_geoms(
    geoms: Any,
    geom_start: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control_dict: Mapping[str, Any],
    torso_index: int,
    *,
    enabled: bool,
    n_arm: int,
    sphere_radius: float = 0.045,
    ref_quat_wxyz: np.ndarray | None = None,
    ref_rotmat: np.ndarray | None = None,
) -> int:
    """Write target spheres + orientation arrows; return next free geom index."""
    del model
    if not enabled or n_arm <= 0:
        return geom_start
    need = (
        "ee_left_pos",
        "ee_left_quat",
        "ee_right_pos",
        "ee_right_quat",
    )
    if not all(k in control_dict for k in need):
        return geom_start

    tl_p = np.asarray(control_dict["ee_left_pos"], dtype=np.float32)
    tl_q = np.asarray(control_dict["ee_left_quat"], dtype=np.float32)
    tr_p = np.asarray(control_dict["ee_right_pos"], dtype=np.float32)
    tr_q = np.asarray(control_dict["ee_right_quat"], dtype=np.float32)

    if ref_quat_wxyz is not None:
        wp_l, wq_l, wp_r, wq_r = ee_world_targets_for_ik(
            data,
            torso_index,
            tl_p,
            tl_q,
            tr_p,
            tr_q,
            ref_quat_wxyz=ref_quat_wxyz,
            ref_rotmat=ref_rotmat,
        )
    else:
        wp_l, wq_l, wp_r, wq_r = ee_world_targets_for_ik(
            data, torso_index, tl_p, tl_q, tr_p, tr_q
        )

    mat = np.eye(3, dtype=np.float64).flatten()
    sz = np.array([sphere_radius, 0.0, 0.0], dtype=np.float64)
    pos_l = np.asarray(wp_l, dtype=np.float64).reshape(3)
    pos_r = np.asarray(wp_r, dtype=np.float64).reshape(3)

    mujoco.mjv_initGeom(
        geoms[geom_start],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=sz,
        pos=pos_l,
        mat=mat,
        rgba=np.array([0.15, 0.95, 0.25, 0.82], dtype=np.float32),
    )
    mujoco.mjv_initGeom(
        geoms[geom_start + 1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=sz,
        pos=pos_r,
        mat=mat,
        rgba=np.array([1.0, 0.4, 0.08, 0.82], dtype=np.float32),
    )
    mujoco.mjv_initGeom(
        geoms[geom_start + 2],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=_EE_TARGET_ARROW_SIZE,
        pos=pos_l,
        mat=_mjv_mat9_from_quat_wxyz(wq_l),
        rgba=_EE_TARGET_LEFT_ARROW_RGBA,
    )
    mujoco.mjv_initGeom(
        geoms[geom_start + 3],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=_EE_TARGET_ARROW_SIZE,
        pos=pos_r,
        mat=_mjv_mat9_from_quat_wxyz(wq_r),
        rgba=_EE_TARGET_RIGHT_ARROW_RGBA,
    )
    return geom_start + 4


def draw_ee_targets_in_viewer(
    viewer: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control_dict: Mapping[str, Any],
    torso_index: int,
    *,
    enabled: bool,
    n_arm: int,
    sphere_radius: float = 0.045,
    ref_quat_wxyz: np.ndarray | None = None,
    ref_rotmat: np.ndarray | None = None,
) -> None:
    """Target markers: green/orange spheres + large orientation arrows.

    Positions match ``ee_world_targets_for_ik`` (torso-frame ``ee_*`` → world).
    Call after ``step_physics`` / ``mj_step``, before ``viewer.sync()``, under the
    same threading rules as other ``user_scn`` updates (``viewer.lock()``).
    """
    with viewer.lock():
        n = fill_ee_target_geoms(
            viewer.user_scn.geoms,
            0,
            model,
            data,
            control_dict,
            torso_index,
            enabled=enabled,
            n_arm=n_arm,
            sphere_radius=sphere_radius,
            ref_quat_wxyz=ref_quat_wxyz,
            ref_rotmat=ref_rotmat,
        )
        viewer.user_scn.ngeom = n
