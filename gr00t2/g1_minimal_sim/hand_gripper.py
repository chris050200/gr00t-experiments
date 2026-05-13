"""Simple open/close hand poses for G1 MJCF with articulated hand joints (g1_gear_wbc_hands.xml)."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

LEFT_HAND_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
# Must match MuJoCo **qpos** order under ``right_wrist_yaw`` (kinematic tree): thumb chain,
# then middle, then index — same pattern as the left hand. (The <actuator> block lists
# index before middle; qpos does not follow that order.)
RIGHT_HAND_JOINT_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)

# Per key tap for 9/0 and =/- (open / close). Smaller = finer squeeze control (~56 taps
# full range at 0.018 vs ~14 at 0.07). Override with ``--grip-key-step`` / ``grip_key_step``.
GRIP_KEY_STEP_DEFAULT = 0.018

# ``arm_target_q`` / ``qpos[7+num_act:]`` when hands are enabled: tree order is
# left_arm(7), left_hand(7), right_arm(7), right_hand(7) — not [both arms then both hands].
SL_LEFT_ARM = slice(0, 7)
SL_LEFT_HAND = slice(7, 14)
SL_RIGHT_ARM = slice(14, 21)
SL_RIGHT_HAND = slice(21, 28)

# When True, :class:`GearWBCRuntime` PD-tracks ``arm_target_q`` hand slices from GR00T
# (``action.{left,right}_hand``) instead of overwriting them each substep from
# ``gripper_left`` / ``gripper_right`` via :func:`hand_target_q`. Set/cleared in
# ``run_gr00t_stylish_diner_inference._apply_action_step``.
GR00T_POLICY_HAND_PD_SOURCE_KEY = "_gr00t_use_policy_hand_targets"


def hand_perm_policy_to_mj(side: str, policy_joint_names: list[str]) -> np.ndarray:
    """Permutation that reorders a policy-ordered hand vector into MuJoCo qpos order.

    Returns indices ``perm`` such that
    ``arm_target_q[SL_{side}_HAND][i] = action.{side}_hand[perm[i]]``.

    The GR00T policy ordering for ``action.left_hand`` / ``action.right_hand`` is
    governed by ``RobotModel.get_joint_group_indices("{side}_hand")`` (sorted DoF
    index order, which on the G1 puts **index → middle → thumb**). MuJoCo's
    kinematic-tree ``qpos`` order under the wrist puts **thumb → middle → index**
    (see ``LEFT_HAND_JOINT_NAMES`` / ``RIGHT_HAND_JOINT_NAMES`` above). Writing
    ``arm_target_q[SL_LEFT_HAND] = action.left_hand`` *without* this permutation
    silently routes index commands onto thumb joints. Discovered May 2026 while
    A/B testing the CloudWalk apple-to-plate checkpoint against
    ``rollout_policy.py`` — see ``memory-bank/gr00t_compatible_data_collection.md``.

    Parameters
    ----------
    side : str
        ``"left"`` or ``"right"``.
    policy_joint_names : list[str]
        Joint names in the order the policy emits them (i.e. iterate
        ``robot_model.joint_names[i] for i in get_joint_group_indices("{side}_hand")``).

    Raises
    ------
    ValueError
        If any MuJoCo-side joint name is missing from ``policy_joint_names``.
    """
    if side == "left":
        mj_names = LEFT_HAND_JOINT_NAMES
    elif side == "right":
        mj_names = RIGHT_HAND_JOINT_NAMES
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if len(policy_joint_names) != len(mj_names):
        raise ValueError(
            f"policy_joint_names has {len(policy_joint_names)} entries, "
            f"expected {len(mj_names)} for side={side!r}"
        )
    name_to_policy_idx: dict[str, int] = {
        str(n): i for i, n in enumerate(policy_joint_names)
    }
    perm = np.empty(len(mj_names), dtype=np.int64)
    for i, mj_name in enumerate(mj_names):
        if mj_name not in name_to_policy_idx:
            raise ValueError(
                f"joint {mj_name!r} (MuJoCo qpos order for {side} hand) is not in "
                f"policy_joint_names={list(policy_joint_names)!r}"
            )
        perm[i] = int(name_to_policy_idx[mj_name])
    return perm


def _open_closed_for_joint(model: mujoco.MjModel, joint_id: int) -> tuple[float, float]:
    lo, hi = float(model.jnt_range[joint_id, 0]), float(model.jnt_range[joint_id, 1])
    if lo > hi:
        lo, hi = hi, lo
    limited = bool(model.jnt_limited[joint_id])
    if not limited:
        return 0.0, 0.0
    if lo <= 0.0 <= hi:
        q_open = 0.0
        raw_closed = lo if abs(lo) >= abs(hi) else hi
        q_closed = float(raw_closed * 0.82)
    else:
        q_open = lo + 0.15 * (hi - lo)
        # Stay slightly inside the limit to reduce PD slam / overshoot on close.
        q_closed = lo + 0.82 * (hi - lo)
    return q_open, q_closed


def hand_target_q(model: mujoco.MjModel, side: str, g: float) -> np.ndarray:
    """Interpolate each finger joint between open (g=1) and closed (g=0)."""
    g = float(np.clip(g, 0.0, 1.0))
    names = LEFT_HAND_JOINT_NAMES if side == "left" else RIGHT_HAND_JOINT_NAMES
    out = np.zeros(7, dtype=np.float32)
    for i, name in enumerate(names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Hand joint missing from model: {name}")
        q_open, q_closed = _open_closed_for_joint(model, jid)
        out[i] = np.float32(g * q_open + (1.0 - g) * q_closed)
    return out


def init_gripper_control_dict(
    control_dict: dict[str, Any], *, grip_key_step: float | None = None
) -> None:
    control_dict["gripper_left"] = 1.0
    control_dict["gripper_right"] = 1.0
    step = float(GRIP_KEY_STEP_DEFAULT if grip_key_step is None else grip_key_step)
    if not (0.0 < step <= 1.0):
        raise ValueError("grip_key_step must be in (0, 1]")
    control_dict["_grip_key_step"] = step
    control_dict[GR00T_POLICY_HAND_PD_SOURCE_KEY] = False


def reset_gripper_control_dict(control_dict: dict[str, Any]) -> None:
    if "gripper_left" in control_dict:
        control_dict["gripper_left"] = 1.0
        control_dict["gripper_right"] = 1.0
        control_dict[GR00T_POLICY_HAND_PD_SOURCE_KEY] = False


_HY_NO_HANDS_WARNED = False


def apply_gripper_teleop_key(control_dict: dict[str, Any], k: str) -> None:
    """Map gripper keys onto ``control_dict`` for articulated G1 hands.

    Keys h/y operate on the articulated grippers (hard close / hard open both).
    On scenes without articulated fingers (``--hands`` not set / unsupported),
    they no-op with a one-time warning rather than triggering any kinematic
    grasp-assist cheat — physics-only is the contract.
    """
    if "gripper_left" not in control_dict:
        if k in ("h", "y"):
            global _HY_NO_HANDS_WARNED
            if not _HY_NO_HANDS_WARNED:
                print(
                    "[hand_gripper] h/y ignored: scene has no articulated grippers. "
                    "Pass --hands (or use --scene with hands support) to enable physics gripping."
                )
                _HY_NO_HANDS_WARNED = True
        return

    step = float(control_dict.get("_grip_key_step", GRIP_KEY_STEP_DEFAULT))

    def bump(side: str, delta: float) -> None:
        key = "gripper_left" if side == "left" else "gripper_right"
        control_dict[key] = float(np.clip(control_dict[key] + delta, 0.0, 1.0))

    # Incremental open/close (like arm teleop): left 9/0, right =/-.
    # Fast bilateral presets help when a teleop grasp needs to be latched immediately.
    if k == "9":
        bump("left", step)
    elif k == "0":
        bump("left", -step)
    elif k == "=":
        bump("right", step)
    elif k == "-":
        bump("right", -step)
    elif k == "h":
        control_dict["gripper_left"] = 0.0
        control_dict["gripper_right"] = 0.0
    elif k == "y":
        control_dict["gripper_left"] = 1.0
        control_dict["gripper_right"] = 1.0
