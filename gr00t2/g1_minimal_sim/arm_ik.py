"""Dual-arm damped least-squares IK in MuJoCo (position + orientation, quaternion targets).

Teleop stores palm targets in a **reference body** frame (e.g. ``torso_link``): position is the
palm point in that body's axes; orientation ``ee_*_quat`` (wxyz) satisfies
``R_world_palm = R_world_ref @ R(ee_quat)``. Each control step converts to **world** for IK.

Palm point is fixed in each ``*_wrist_yaw_link`` body (from ``g1_gear_wbc.xml`` palm geom).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ee_frame import (
    ee_pose_ref_from_world,
    ee_pose_world_from_ref,
    normalize_quat as _normalize_quat,
    quat_mul,
    quat_wxyz_to_rotmat,
)

LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class ArmSideIK:
    body_id: int
    palm_local: np.ndarray  # (3,) in body frame
    dof_ids: np.ndarray  # (7,) column indices into nv
    qpos_ids: np.ndarray  # (7,) indices into full qpos
    joint_ids: np.ndarray  # (7,) for jnt_range


def build_g1_arm_ik_specs(model: mujoco.MjModel) -> tuple[ArmSideIK, ArmSideIK]:
    def spec(joint_names: tuple[str, ...], body_name: str, palm_local: np.ndarray) -> ArmSideIK:
        qpos_ids = []
        dof_ids = []
        joint_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint not found: {name}")
            qpos_ids.append(int(model.jnt_qposadr[jid]))
            dof_ids.append(int(model.jnt_dofadr[jid]))
            joint_ids.append(jid)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body not found: {body_name}")
        return ArmSideIK(
            body_id=bid,
            palm_local=np.asarray(palm_local, dtype=np.float64),
            dof_ids=np.asarray(dof_ids, dtype=np.int32),
            qpos_ids=np.asarray(qpos_ids, dtype=np.int32),
            joint_ids=np.asarray(joint_ids, dtype=np.int32),
        )

    left = spec(LEFT_ARM_JOINTS, "left_wrist_yaw_link", [0.0415, 0.003, 0.0])
    right = spec(RIGHT_ARM_JOINTS, "right_wrist_yaw_link", [0.0415, -0.003, 0.0])
    return left, right


def body_palm_pose(
    data: mujoco.MjData, spec: ArmSideIK
) -> tuple[np.ndarray, np.ndarray]:
    """World position (3,) and quaternion wxyz (4,) of palm point."""
    bid = spec.body_id
    xmat = data.xmat[bid].reshape(3, 3)
    pos = data.xpos[bid] + xmat @ spec.palm_local
    quat = np.asarray(data.xquat[bid], dtype=np.float64).copy()
    return pos, quat


def ee_world_targets_for_ik(
    data: mujoco.MjData,
    ref_body_id: int,
    ee_left_pos: np.ndarray,
    ee_left_quat: np.ndarray,
    ee_right_pos: np.ndarray,
    ee_right_quat: np.ndarray,
    *,
    ref_quat_wxyz: np.ndarray | None = None,
    ref_rotmat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert stored ref-body EE targets to world frame for IK.

    ``ref_quat_wxyz`` must match the quaternion basis used when **encoding** ``ee_*``
    (e.g. VR stream IK uses ``torso_xquat_with_receiver_yaw``). If ``None``, uses
    ``data.xquat[ref_body_id]`` (keyboard / UDP / default init).

    When ``ref_quat_wxyz`` is set (VR yaw-corrected decode), pass ``ref_rotmat`` equal to
    the same rotation matrix used when encoding (e.g. ``torso_xmat_with_yaw_offset``).
    If ``ref_rotmat`` is omitted, ``R`` is derived from ``ref_quat_wxyz`` via
    :func:`ee_frame.quat_wxyz_to_rotmat` (not ``data.xmat``), so position and orientation
    use one SO(3) representation.
    """
    xpos = data.xpos[ref_body_id]
    if ref_quat_wxyz is None:
        qb = np.asarray(data.xquat[ref_body_id], dtype=np.float64)
        R_pos = quat_wxyz_to_rotmat(_normalize_quat(qb))
    else:
        qb = np.asarray(ref_quat_wxyz, dtype=np.float64).reshape(4)
        R_pos = (
            np.asarray(ref_rotmat, dtype=np.float64).reshape(3, 3)
            if ref_rotmat is not None
            else quat_wxyz_to_rotmat(_normalize_quat(qb))
        )
    pl_w, ql_w = ee_pose_world_from_ref(xpos, R_pos, qb, ee_left_pos, ee_left_quat)
    pr_w, qr_w = ee_pose_world_from_ref(xpos, R_pos, qb, ee_right_pos, ee_right_quat)
    return (
        pl_w.astype(np.float32),
        ql_w.astype(np.float32),
        pr_w.astype(np.float32),
        qr_w.astype(np.float32),
    )


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / n
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _clamp_qpos(model: mujoco.MjModel, joint_ids: np.ndarray, q: np.ndarray) -> None:
    for i, jid in enumerate(joint_ids):
        lo, hi = model.jnt_range[jid]
        if lo < hi:
            q[i] = float(np.clip(q[i], lo, hi))


def init_ee_control_dict(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left: ArmSideIK,
    right: ArmSideIK,
    control_dict: dict[str, Any],
    ref_body_id: int,
) -> None:
    """Set ``ee_*`` and ``_*_home`` from current FK in **ref body** frame (call after setup)."""
    mujoco.mj_forward(model, data)
    xpos = data.xpos[ref_body_id]
    qb = np.asarray(data.xquat[ref_body_id], dtype=np.float64)
    R_ref = quat_wxyz_to_rotmat(_normalize_quat(qb))

    p_l, q_l = body_palm_pose(data, left)
    p_r, q_r = body_palm_pose(data, right)
    pl_b, ql_b = ee_pose_ref_from_world(xpos, R_ref, qb, p_l, q_l)
    pr_b, qr_b = ee_pose_ref_from_world(xpos, R_ref, qb, p_r, q_r)

    control_dict["ee_left_pos"] = pl_b.astype(np.float32)
    control_dict["ee_left_quat"] = ql_b.astype(np.float32)
    control_dict["ee_right_pos"] = pr_b.astype(np.float32)
    control_dict["ee_right_quat"] = qr_b.astype(np.float32)
    control_dict["_ee_left_pos_home"] = control_dict["ee_left_pos"].copy()
    control_dict["_ee_left_quat_home"] = control_dict["ee_left_quat"].copy()
    control_dict["_ee_right_pos_home"] = control_dict["ee_right_pos"].copy()
    control_dict["_ee_right_quat_home"] = control_dict["ee_right_quat"].copy()
    # World palm quaternions (MuJoCo wxyz) at init — used by --lock-ee-orient to keep palms
    # world-stable while walking (torso-frame q_e is updated each step from these anchors).
    control_dict["_ee_left_quat_world_anchor"] = np.asarray(q_l, dtype=np.float64).reshape(4).copy()
    control_dict["_ee_right_quat_world_anchor"] = np.asarray(q_r, dtype=np.float64).reshape(4).copy()


_POS_STEP = 0.012
_ANG_STEP = 0.055


def apply_ik_arm_teleop_key(control_dict: dict[str, Any], k: str) -> None:
    """Keyboard deltas for EE targets in **reference body** axes (e.g. torso).

    Position keys move palm targets along ref-body X/Y/Z. Orientation keys apply small rotations
    about ref-body axes to both hands' ``ee_*_quat`` (left-multiply in ref frame).
    """
    if "ee_left_pos" not in control_dict:
        return

    def rot_both(axis: np.ndarray, sign: float) -> None:
        dq = quat_from_axis_angle(axis, sign * _ANG_STEP)
        control_dict["ee_left_quat"][:] = quat_mul(
            dq, np.asarray(control_dict["ee_left_quat"], dtype=np.float64)
        )
        control_dict["ee_right_quat"][:] = quat_mul(
            dq, np.asarray(control_dict["ee_right_quat"], dtype=np.float64)
        )

    # Left position (ref body)
    if k == "i":
        control_dict["ee_left_pos"][0] -= _POS_STEP
    elif k == "k":
        control_dict["ee_left_pos"][0] += _POS_STEP
    elif k == "j":
        control_dict["ee_left_pos"][1] -= _POS_STEP
    elif k == "l":
        control_dict["ee_left_pos"][1] += _POS_STEP
    elif k == "u":
        control_dict["ee_left_pos"][2] += _POS_STEP
    elif k == "p":
        control_dict["ee_left_pos"][2] -= _POS_STEP
    # Right position (ref body)
    elif k == "r":
        control_dict["ee_right_pos"][0] -= _POS_STEP
    elif k == "t":
        control_dict["ee_right_pos"][0] += _POS_STEP
    elif k == "f":
        control_dict["ee_right_pos"][1] -= _POS_STEP
    elif k == "g":
        control_dict["ee_right_pos"][1] += _POS_STEP
    elif k == "v":
        control_dict["ee_right_pos"][2] += _POS_STEP
    elif k == "b":
        control_dict["ee_right_pos"][2] -= _POS_STEP
    # Both orientations: ref-body-fixed small rotations
    elif k == ",":
        rot_both(np.array([1.0, 0.0, 0.0]), -1.0)
    elif k == ".":
        rot_both(np.array([1.0, 0.0, 0.0]), 1.0)
    elif k == ";":
        rot_both(np.array([0.0, 1.0, 0.0]), -1.0)
    elif k == "'":
        rot_both(np.array([0.0, 1.0, 0.0]), 1.0)
    elif k == "[":
        rot_both(np.array([0.0, 0.0, 1.0]), -1.0)
    elif k == "]":
        rot_both(np.array([0.0, 0.0, 1.0]), 1.0)


def reset_ee_to_home(control_dict: dict[str, Any]) -> None:
    if "_ee_left_pos_home" not in control_dict:
        return
    control_dict["ee_left_pos"][:] = control_dict["_ee_left_pos_home"]
    control_dict["ee_left_quat"][:] = control_dict["_ee_left_quat_home"]
    control_dict["ee_right_pos"][:] = control_dict["_ee_right_pos_home"]
    control_dict["ee_right_quat"][:] = control_dict["_ee_right_quat_home"]
