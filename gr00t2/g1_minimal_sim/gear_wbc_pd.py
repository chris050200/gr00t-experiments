"""PD control helpers and VR IK anchor clamping."""

from __future__ import annotations

import mujoco
import numpy as np

ARM_TAU_CLIP_NM = 30.0
# Slightly higher clip while walking + IK: moving world targets and base shake need more headroom
# before PD saturates (saturation reads as "hands diverge while spheres stay put").
ARM_TAU_CLIP_WALK_IK_NM = 40.0
# Friction-grasp profile: wrists need more torque headroom to sustain palm-pinch contact load
# without back-driving. Selected by ``--arm-gain-profile grasp`` / runtime override.
ARM_TAU_CLIP_GRASP_NM = 40.0

# Per-arm joint order (left & right symmetric):
#   0: shoulder_pitch, 1: shoulder_roll, 2: shoulder_yaw, 3: elbow,
#   4: wrist_roll,     5: wrist_pitch,   6: wrist_yaw
# (matches ``arm_ik.LEFT_ARM_JOINTS`` / ``RIGHT_ARM_JOINTS`` and the kinematic-tree qpos slice.)
ARM_PD_KP_DEFAULT_PER_JOINT = np.array(
    [45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0], dtype=np.float32
)
ARM_PD_KD_DEFAULT_PER_JOINT = np.array(
    [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2], dtype=np.float32
)
# Friction-grasp profile: stiffer elbow + wrists (esp. wrist_pitch, the contact-loaded one)
# so palm-pinch can sustain normal force against ``target_block`` without back-driving.
ARM_PD_KP_GRASP_PER_JOINT = np.array(
    [45.0, 45.0, 45.0, 60.0, 90.0, 120.0, 60.0], dtype=np.float32
)
ARM_PD_KD_GRASP_PER_JOINT = np.array(
    [1.2, 1.2, 1.2, 1.5, 2.0, 2.5, 1.6], dtype=np.float32
)
# Hand-finger PD (when has_hands): unchanged from the historical uniform 45 / 0.35.
HAND_PD_KP_DEFAULT = 45.0
HAND_PD_KD_DEFAULT = 0.35


def pd_control(target_q, q, kp, target_dq, dq, kd) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_arm_pd_per_joint(
    n_u: int,
    *,
    has_hands: bool,
    kp_per_arm_joint: np.ndarray,
    kd_per_arm_joint: np.ndarray,
    kd_walk_scale: float,
    walking: bool,
    hand_kp: float = HAND_PD_KP_DEFAULT,
    hand_kd: float = HAND_PD_KD_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile per-arm-joint kp/kd into the ``arm_target_q`` / ``qpos[7+num_act:]`` slice layout.

    Without hands (``n_u = 14``): ``[LEFT_ARM(7), RIGHT_ARM(7)]`` — both arms get
    ``kp_per_arm_joint`` / ``kd_per_arm_joint``.

    With hands (``n_u = 28``): ``[LEFT_ARM(7), LEFT_HAND(7), RIGHT_ARM(7), RIGHT_HAND(7)]``
    in MuJoCo kinematic tree order. Hand slots use ``hand_kp`` / ``hand_kd``.

    ``kd`` is multiplied by ``kd_walk_scale`` when ``walking`` is true (matches the historical
    ``_ARM_KD_WALK_SCALE`` behaviour for the arm slice; hand kd is unaffected).
    """
    kp_arm = np.asarray(kp_per_arm_joint, dtype=np.float32).reshape(7).copy()
    kd_arm = np.asarray(kd_per_arm_joint, dtype=np.float32).reshape(7).copy()
    if walking:
        kd_arm = kd_arm * float(kd_walk_scale)
    if has_hands:
        if n_u != 28:
            raise ValueError(f"has_hands=True expects n_u=28, got {n_u}")
        kp = np.empty(28, dtype=np.float32)
        kd = np.empty(28, dtype=np.float32)
        kp[0:7] = kp_arm
        kp[7:14] = float(hand_kp)
        kp[14:21] = kp_arm
        kp[21:28] = float(hand_kp)
        kd[0:7] = kd_arm
        kd[7:14] = float(hand_kd)
        kd[14:21] = kd_arm
        kd[21:28] = float(hand_kd)
        return kp, kd
    if n_u != 14:
        raise ValueError(f"has_hands=False expects n_u=14, got {n_u}")
    kp = np.concatenate([kp_arm, kp_arm]).astype(np.float32)
    kd = np.concatenate([kd_arm, kd_arm]).astype(np.float32)
    return kp, kd


def left_right_wrist_pitch_indices(*, has_hands: bool) -> tuple[int, int]:
    """Indices of ``left_wrist_pitch`` and ``right_wrist_pitch`` inside the ``n_u`` arm slice.

    Used by the live ``--print-arm-tau`` readout to surface the most contact-loaded wrist DoF.
    """
    if has_hands:
        return 5, 14 + 5
    return 5, 7 + 5


def clamp_ref_target_to_anchor_radius(
    target_ref: np.ndarray, anchor_ref: np.ndarray, max_radius_m: float
) -> np.ndarray:
    """Clamp a torso-frame target to a sphere around an anchor (same frame)."""
    p = np.asarray(target_ref, dtype=np.float64).reshape(3)
    a = np.asarray(anchor_ref, dtype=np.float64).reshape(3)
    r = float(max_radius_m)
    if r <= 0.0:
        return a.astype(np.float32)
    d = p - a
    n = float(np.linalg.norm(d))
    if n <= r or n < 1e-9:
        return p.astype(np.float32)
    return (a + (r / n) * d).astype(np.float32)


def arm_actuator_qpos_slice_indices(
    model: mujoco.MjModel, num_act: int, n_joints: int
) -> np.ndarray:
    """Map ``ctrl[num_act + i]`` slot *i* → slice index *s* into ``arm_target_q[s]`` / ``qpos[7+num_act+s]``.

    Actuator order in MJCF can differ from kinematic ``qpos`` order (right hand: motors are
    thumb→index→middle; ``qpos`` is thumb→middle→index). Without this, finger torques hit the wrong joints.
    """
    n_u = n_joints - num_act
    base_q = 7 + num_act
    out = np.empty(n_u, dtype=np.int32)
    for slot in range(n_u):
        ai = num_act + slot
        jid = int(model.actuator_trnid[ai, 0])
        qadr = int(model.jnt_qposadr[jid])
        out[slot] = int(qadr - base_q)
    return out
