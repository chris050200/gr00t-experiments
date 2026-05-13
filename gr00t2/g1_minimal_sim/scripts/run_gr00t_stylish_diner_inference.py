#!/usr/bin/env python3
"""Run GR00T policy-server inference inside g1_minimal_sim (modular scene).

Canonical **teleop → dataset → training → inference** map (modalities, server
wrapper, ``decode_action`` / state alignment): ``g1_minimal_sim/memory-bank/
gr00t_compatible_data_collection.md`` §6.4.

Preferred CLI entrypoint: ``scripts/run_gr00t_inference.py`` (same implementation).

This bridges:
- g1_minimal_sim runtime/scene (`GearWBCRuntime` via Gym env)
- GR00T policy server (`run_gr00t_server.py --use-sim-policy-wrapper`)

------------------------------------------------------------------------------
Control-cadence contract (READ THIS BEFORE TUNING ``--plan-hz`` / ``--n-action-steps``)
------------------------------------------------------------------------------

Training data is sampled at a fixed plan rate (``Gr00tTeleopEpisodeLogger``'s
default ``sample_hz=50.0``). Each ``action.*`` row in the dataset therefore
represents the policy target after **one** plan period (``1 / plan_hz``) of
physics evolution. The upstream eval pipeline matches this implicitly because
``SyncEnv`` is ``control_freq=50`` and one ``MultiStepWrapper`` slice equals
one ``SyncEnv.step`` (i.e., 20 ms).

In this minimal-sim runtime ``env.step()`` advances **one** ``mj_step`` —
which is ``model.opt.timestep`` (5 ms with the diner YAML's ``simulation_dt:
0.005``). To consume each plan step at the trained cadence we substep
``n_substeps = round(1 / (plan_hz * sim_dt)) = 4`` times per plan step.
``--plan-hz`` makes this overridable; passing ``--plan-hz`` smaller than
``50`` simulates a slower wall-clock (each plan step held longer); passing
``--plan-hz`` larger than ``50`` recreates the historical 4x-too-fast
behaviour and is useful for re-confirming the bug if needed.

The plan-Hz contract is shared with ``Gr00tTeleopEpisodeLogger.sample_hz``;
keeping these aligned is the inference-time half of the relative/absolute
arm contract documented below. ``tests/test_run_gr00t_inference_cadence.py``
pins the substep math and the logger default.

------------------------------------------------------------------------------
Saved MP4 timing (``--save-video`` / ``--video-fps``)
------------------------------------------------------------------------------

The loops append **one RGB frame per plan step** (not per ``mj_step``), so each
frame covers ``≈ 1 / plan_hz`` simulated seconds. The MP4 ``fps`` metadata is
**playback** speed: if ``fps`` is much lower than ``plan_hz``, the clip looks
like slow motion vs training video (usually one frame per logged sample at the
same ``sample_hz``). Default ``--video-fps`` follows ``--plan-hz`` via
``resolve_video_fps``; override only for hosts that cap encode FPS.

------------------------------------------------------------------------------
Action representation contract (READ THIS BEFORE TUNING ``--arm-action-mode``)
------------------------------------------------------------------------------

The full pipeline for ``action.left_arm`` / ``action.right_arm`` is:

1. ``gr00t_teleop_logger.py`` defaults to ``relative_arm_actions=True`` and writes
   ``action.{side}_arm = arm_target_q - state.{side}_arm`` to NPZ.
2. ``scripts/export_gr00t_npz_to_lerobot.py`` defaults ``--relative-arms`` OFF,
   so it passes those deltas through to parquet AS IS — i.e. the LeRobot
   dataset's ``action.{side}_arm`` column is **already a delta vs current
   state**, not an absolute joint target.
3. ``Isaac-GR00T/gr00t/experiment/launch_finetune.py`` always sets
   ``config.model.use_relative_action = True``. Training applies
   ``StateActionProcessor._convert_to_relative_action(action, state[-1])`` on
   top of the on-disk data, so the model learns deltas in normalized space
   with ``meta/relative_stats.json`` computed over the same transform.
4. At eval, the policy server's ``Gr00tPolicy._get_action`` calls
   ``self.processor.decode_action`` which delegates to
   ``StateActionProcessor.unapply_action``. For any RELATIVE-rep action key,
   when ``use_relative_action=True`` is baked into the saved
   ``processor_config.json`` (it is for any checkpoint produced by
   ``launch_finetune.py``), ``unapply_action`` itself calls
   ``_convert_to_absolute_action(action, state[-1])`` →
   ``JointActionChunk.to_absolute_chunking(...)`` which does
   ``absolute_joints = reference_frame.joints + relative_pose.joints``.
   **Net result: the policy server returns ABSOLUTE arm targets**, NOT
   deltas. All non-RELATIVE keys (``left_hand`` / ``right_hand`` /
   ``waist`` / ``base_height_command`` / ``navigate_command``) come back
   as absolute values either way.

NPZ replay sees a DIFFERENT representation. The recorded ``action.left_arm``
in NPZ is a true delta (``arm_target_q − state``) because the logger writes
it that way. Replay must add ``state_now`` back to recover the recorded
``arm_target_q``.

So the correct ``--arm-action-mode`` depends on the BRANCH:

- **Live policy** (PolicyClient → ``unapply_action`` already absolutized) →
  ``absolute``: take ``action.{side}_arm`` as-is, write directly to
  ``arm_target_q[SL_{SIDE}_ARM]``.
- **NPZ replay** (deltas stored on disk) → ``relative``: add ``state_now``
  back before writing to ``arm_target_q[SL_{SIDE}_ARM]``.

The default ``--arm-action-mode auto`` resolves to the correct option per
branch (live=absolute, replay=relative, oracle replay of Tier-A chunks=
absolute). Override only if you changed the logger / exporter / launcher
defaults so the on-disk or server-side representation no longer matches.

Historical note: prior to May 2026 this script defaulted to ``relative`` in
both branches, which silently DOUBLE-ADDED state on live policy output
(server already added ``state[-1]``, script added ``state_now`` on top) and
produced compounding arm targets that saturated joints 2–4 to ~11 rad
within ~50 plan steps — the "pretzel" symptom.

------------------------------------------------------------------------------
Examples (from g1_minimal_sim/, policy server already running)
------------------------------------------------------------------------------

Stylish diner + default prompt for that scene::

    python scripts/run_gr00t_inference.py \\
      --scene stylish_diner \\
      --policy-host 127.0.0.1 --policy-port 2000 \\
      --max-steps 1500 --debug-policy-prints \\
      --save-video /tmp/g1_diner_infer.mp4

Explicit prompt (``--task`` is an alias for ``--prompt``)::

    python scripts/run_gr00t_inference.py \\
      --scene stylish_diner \\
      --prompt "pick up the blue cube" \\
      --policy-host 127.0.0.1 --policy-port 2000

Table PnP (apple/plate MJCF) + DC-style instruction::

    python scripts/run_gr00t_inference.py \\
      --scene table_pnp \\
      --prompt "pick up the apple and place it on the plate" \\
      --policy-host 127.0.0.1 --policy-port 2000

Backward-compatible script name (same module)::

    python scripts/run_gr00t_stylish_diner_inference.py --scene table_pnp --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
import sys
from typing import Any, TextIO

# ``g1_minimal_sim`` root so ``python scripts/<this>.py`` works without PYTHONPATH.
_G1_SIM_ROOT = Path(__file__).resolve().parent.parent
_root_s = str(_G1_SIM_ROOT)
if _root_s not in sys.path:
    sys.path.insert(0, _root_s)

import gymnasium as gym
import numpy as np

from g1_gear_wbc_env import ENV_ID, register_g1_gear_wbc_env
from gr00t_observation_builder import Gr00tObservationBuilder
from policy_ab_dump import dump_policy_roundtrip_pack
from hand_gripper import (
    GR00T_POLICY_HAND_PD_SOURCE_KEY,
    SL_LEFT_ARM,
    SL_LEFT_HAND,
    SL_RIGHT_ARM,
    SL_RIGHT_HAND,
    hand_perm_policy_to_mj,
)
import oracle_action_chunk_io


def _ensure_isaac_gr00t_importable() -> None:
    root = Path(__file__).resolve().parents[2] / "Isaac-GR00T"
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


# Default plan rate for both training (Gr00tTeleopEpisodeLogger.sample_hz) and
# inference (here). Keep these aligned.
DEFAULT_PLAN_HZ = 50.0

# Scenes supported by ``G1GearWBCEnv`` / ``resolve_gear_wbc_config`` (see ``g1_gear_wbc_env``).
SCENE_CHOICES: tuple[str, ...] = ("stylish_diner", "table_pnp", "table_pnp_apple", "floor")

# When ``--prompt`` / ``--task`` are omitted, pick a reasonable default per scene.
# Per-scene default ``annotation.human.task_description`` strings.
#
# For apple-to-plate (``table_pnp`` / ``table_pnp_apple``) the string is copied
# **verbatim** from upstream ``LMPnPAppleToPlate._get_instruction`` in
# ``Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/
# gr00trobocasa/robocasa/environments/locomanipulation/locomanip_basic.py:681``.
# Vision-language policies fail catastrophically on subtle prompt drift (e.g.
# dropping the "walk left and" directive turned a CloudWalk PnP checkpoint from
# Tier-A success → Tier-B "hands lift toward face" — see ``activeContext.md``
# May 2026 visual-alignment writeup). Keep these strings byte-exact with
# upstream; only override at the CLI via ``--prompt``.
DEFAULT_TASK_BY_SCENE: dict[str, str] = {
    "stylish_diner": "pick up the blue block",
    "table_pnp": "pick up the apple, walk left and place the apple on the plate.",
    "table_pnp_apple": "pick up the apple, walk left and place the apple on the plate.",
    "floor": "walk and balance",
}


def resolve_task_description(scene: str, task_description: str | None) -> str:
    """Return language instruction for the policy; apply per-scene default if unset."""
    if task_description is not None and str(task_description).strip():
        return str(task_description)
    if scene not in DEFAULT_TASK_BY_SCENE:
        raise ValueError(f"unknown scene {scene!r}; expected one of {SCENE_CHOICES}")
    return DEFAULT_TASK_BY_SCENE[scene]


# CLI sentinel + valid resolved values for --arm-action-mode.
ARM_ACTION_MODE_AUTO = "auto"
ARM_ACTION_MODE_RELATIVE = "relative"
ARM_ACTION_MODE_ABSOLUTE = "absolute"
ARM_ACTION_MODE_CHOICES: tuple[str, ...] = (
    ARM_ACTION_MODE_AUTO,
    ARM_ACTION_MODE_RELATIVE,
    ARM_ACTION_MODE_ABSOLUTE,
)
ARM_ACTION_MODE_RESOLVED: tuple[str, ...] = (
    ARM_ACTION_MODE_RELATIVE,
    ARM_ACTION_MODE_ABSOLUTE,
)

# Printed once per process if hands MJCF expects policy fingers but chunk omits keys.
_warned_policy_hand_keys_missing = False

# Printed once per process if the policy chunk omits ``action.waist`` while a
# ``robot_model`` is wired into the live branch. Mirrors the hand-keys warning so
# operators notice a regression instead of silently re-introducing bug #5
# (dropped waist -> arms swing toward face). See ``memory-bank/activeContext.md``.
_warned_policy_waist_key_missing = False


def resolve_arm_action_mode(mode: str, *, branch: str) -> str:
    """Resolve ``--arm-action-mode`` for the current branch.

    The default ``auto`` differs by branch because the on-the-wire action
    representation differs (see the module docstring for the derivation):

    - ``live``: policy server's ``unapply_action`` already converts RELATIVE
      arm keys to absolute via ``_convert_to_absolute_action``, so the
      script must NOT add ``state_now`` again. Resolves to ``absolute``.
    - ``replay``: NPZ stores ``arm_target_q − state`` deltas (logger
      default ``relative_arm_actions=True``); the script must add
      ``state_now`` back. Resolves to ``relative``.
    - ``oracle_replay``: replay ``replan_*.npz`` chunks captured from Tier-A
      ``rollout_policy.py``; tensors are already absolute joint targets (same as
      live server output). Resolves to ``absolute``.

    Explicit ``relative`` / ``absolute`` overrides the branch default and
    is returned as-is. Any other value raises ``ValueError``.
    """
    if mode not in ARM_ACTION_MODE_CHOICES:
        raise ValueError(
            f"arm_action_mode must be one of {ARM_ACTION_MODE_CHOICES}, got {mode!r}"
        )
    if branch not in ("live", "replay", "oracle_replay"):
        raise ValueError(
            f"branch must be 'live', 'replay', or 'oracle_replay', got {branch!r}"
        )
    if mode != ARM_ACTION_MODE_AUTO:
        return mode
    if branch == "replay":
        return ARM_ACTION_MODE_RELATIVE
    # ``live`` and ``oracle_replay``: server / recorded oracle chunks are absolute targets.
    return ARM_ACTION_MODE_ABSOLUTE


# ----------------------------------------------------------------------------
# action.waist -> rt.control_dict["rpy_cmd"] via Pinocchio FK (bug #5 fix).
# ----------------------------------------------------------------------------
# Upstream ``G1DecoupledWholeBodyPolicy.get_action`` (in
# ``Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/
# policy/g1_decoupled_whole_body_policy.py``) translates the GR00T policy's
# ``action.waist`` joint targets into the *lower-body* policy's
# ``torso_orientation_rpy`` command by:
#
#   1. ``q = np.zeros(num_dofs); q[upper_body_indices] = target_upper_body_pose``.
#   2. Pinocchio FK on that ``q``; read ``torso_link`` rotation and ``pelvis``
#      rotation in world.
#   3. Strip non-yaw components from the pelvis frame, then express the torso
#      rotation in that yaw-only pelvis frame.
#   4. Convert that to (roll, pitch, yaw) via ``pinocchio.rpy.matrixToRpy``.
#   5. Hand the result to ``g1_gear_wbc_policy.G1GearWbcPolicy.get_action`` as
#      ``torso_orientation_rpy``, which writes it into the policy obs as
#      ``roll_cmd / pitch_cmd / yaw_cmd`` (single_obs[4:7]).
#
# In ``g1_minimal_sim`` the same ONNX leg+waist policy is fed by
# ``gear_wbc_obs.compute_single_obs``, which reads ``control_dict["rpy_cmd"]``
# at the identical layout (``command[4:7]`` of the 86-D obs). So the bridge
# mirrors the upstream FK and writes the result directly into ``rpy_cmd``.
#
# Without this translation, ``rpy_cmd`` stays at the yaml default ``[0, 0, 0]``,
# the leg policy keeps the torso upright, the head-mounted egoview never looks
# down at the table, and the GR00T policy (which expects waist-driven torso
# pitch) compensates by sending arm joint targets that swing the palms toward
# face level. That was the May 2026 "hands lift toward the face" A/B failure
# on the CloudWalk PnP checkpoint when ``rollout_policy.py`` succeeded.


def _waist_action_to_rpy_cmd(robot_model: Any, waist_action: np.ndarray) -> np.ndarray:
    """Translate a single-step ``action.waist`` into ``rpy_cmd`` (roll, pitch, yaw).

    Mirrors ``G1DecoupledWholeBodyPolicy.get_action`` exactly, so any future
    upstream change to that FK should be reflected here in one place. The
    returned vector matches the layout ``g1_minimal_sim`` already uses for
    ``control_dict["rpy_cmd"]`` (``rpy[0]=roll``, ``rpy[1]=pitch``,
    ``rpy[2]=yaw``).

    Parameters
    ----------
    robot_model : RobotModel
        Same Pinocchio model used by ``Gr00tObservationBuilder`` (i.e.
        ``get_robot_type_and_model("G1", enable_waist_ik=True)``).
    waist_action : np.ndarray
        Length-N waist joint targets in the policy's emission order
        (sorted DoF index order — matches ``get_joint_group_indices("waist")``).

    Returns
    -------
    np.ndarray
        Length-3 (roll, pitch, yaw) in radians, ``dtype=float32`` to match
        ``rt.control_dict["rpy_cmd"]``.
    """
    from pinocchio import rpy as pin_rpy

    waist_indices = list(robot_model.get_joint_group_indices("waist"))
    w = np.asarray(waist_action, dtype=np.float64).reshape(-1)
    if w.size != len(waist_indices):
        raise ValueError(
            f"waist_action size {w.size} != robot_model 'waist' DoF count "
            f"{len(waist_indices)}; check MODALITY_CONFIGS / supplemental_info"
        )
    q_full = np.zeros(int(robot_model.num_dofs), dtype=np.float64)
    q_full[waist_indices] = w
    robot_model.cache_forward_kinematics(q_full, auto_clip=False)
    R_torso = np.asarray(
        robot_model.frame_placement("torso_link").rotation, dtype=np.float64
    )
    R_pelv = np.asarray(
        robot_model.frame_placement("pelvis").rotation, dtype=np.float64
    )
    pelvis_yaw = float(np.arctan2(R_pelv[1, 0], R_pelv[0, 0]))
    R_z_pelv = np.asarray(pin_rpy.rpyToMatrix(0.0, 0.0, pelvis_yaw), dtype=np.float64)
    R_torso_in_yaw_pelv = R_z_pelv.T @ R_torso
    rpy_vec = np.asarray(pin_rpy.matrixToRpy(R_torso_in_yaw_pelv), dtype=np.float64)
    return rpy_vec.astype(np.float32, copy=False)


def build_hand_permutations(
    robot_model: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (left_perm, right_perm) for ``action.{side}_hand`` -> MuJoCo qpos order.

    ``perm[i] = j`` means MuJoCo slot ``i`` of ``arm_target_q[SL_{side}_HAND]``
    is filled from policy slot ``j`` of ``action.{side}_hand``.

    The policy emits hands in ``robot_model.get_joint_group_indices("{side}_hand")``
    order (sorted DoF index), which on the G1 puts index → middle → thumb.
    Our MuJoCo qpos tree puts thumb → middle → index (see
    ``hand_gripper.LEFT_HAND_JOINT_NAMES``). Without this permutation,
    ``arm_target_q[SL_LEFT_HAND] = action.left_hand`` routes index commands
    onto thumb joints (bug #6, May 2026).
    """
    left_idx = list(robot_model.get_joint_group_indices("left_hand"))
    right_idx = list(robot_model.get_joint_group_indices("right_hand"))
    left_names = [str(robot_model.joint_names[i]) for i in left_idx]
    right_names = [str(robot_model.joint_names[i]) for i in right_idx]
    return (
        hand_perm_policy_to_mj("left", left_names),
        hand_perm_policy_to_mj("right", right_names),
    )


def assert_arm_joint_order_matches(robot_model: Any) -> None:
    """Fail fast if ``robot_model`` and our MuJoCo ``arm_target_q`` disagree on arm joint order.

    ``arm_target_q[SL_LEFT_ARM] = action.left_arm`` is a *direct* write that
    only works because ``robot_model.get_joint_group_indices("left_arm")``
    happens to enumerate the same 7 joints in the same order as MuJoCo's
    kinematic-tree qpos slice (see ``gear_wbc_config.POLICY_JOINT_NAMES``).
    If upstream ever reorders the supplemental info, hands-toward-face will
    return in a new flavour; we want to crash on startup instead.
    """
    expected_left = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    expected_right = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    for group, expected in (
        ("left_arm", expected_left),
        ("right_arm", expected_right),
    ):
        idx = list(robot_model.get_joint_group_indices(group))
        names = [str(robot_model.joint_names[i]) for i in idx]
        if names != expected:
            raise RuntimeError(
                f"robot_model joint group {group!r} order changed; "
                f"got {names!r}, expected {expected!r}. arm_target_q[SL_*_ARM] "
                "is in MuJoCo qpos kinematic order and requires this match. "
                "Update either hand_gripper SL_*_ARM mapping or this assertion."
            )


def _warn_if_waist_action_missing(action_plan: dict[str, np.ndarray]) -> None:
    """One-time warning when ``action.waist`` is absent on the live branch."""
    global _warned_policy_waist_key_missing
    if _warned_policy_waist_key_missing:
        return
    if "action.waist" in action_plan:
        return
    _warned_policy_waist_key_missing = True
    print(
        "WARNING: live policy chunk omits action.waist — torso ``rpy_cmd`` "
        "will stay at the yaml default. With a checkpoint that expects "
        "waist-driven torso pitch (e.g. CloudWalk PnP), arms typically swing "
        "toward face level to compensate. Keys present: "
        f"{sorted(action_plan.keys())!r}."
    )


def resolve_plan_substeps(sim_dt: float, plan_hz: float) -> int:
    """Substeps to advance per plan step so ``n * sim_dt ≈ 1 / plan_hz``.

    Concretely with ``sim_dt=0.005`` and ``plan_hz=50.0`` this is 4. The
    minimum is 1 (degenerate, replicates the historical 4x-too-fast loop)
    so callers can still force the old behaviour by passing
    ``plan_hz = 1 / sim_dt`` (e.g. 200 Hz).
    """
    if not (sim_dt > 0.0):
        raise ValueError(f"sim_dt must be > 0, got {sim_dt!r}")
    if not (plan_hz > 0.0):
        raise ValueError(f"plan_hz must be > 0, got {plan_hz!r}")
    n = int(round(1.0 / (plan_hz * float(sim_dt))))
    return max(1, n)


def resolve_video_fps(video_fps: int | None, plan_hz: float) -> int:
    """Encode FPS for ``--save-video`` MP4s.

    The inference loops append **one RGB frame per plan step** (each frame
    spans ``n_substeps * sim_dt ≈ 1/plan_hz`` simulated seconds). Players
    show ``fps`` frames per wall-clock second, so **playback matches
    simulation tempo** when ``fps ≈ plan_hz`` — the same cadence as
    ``Gr00tTeleopEpisodeLogger.sample_hz`` / training clips sampled once per
    plan period. If ``fps << plan_hz``, motion looks slow-motion; if
    ``fps >> plan_hz``, it looks sped up.

    When ``video_fps`` is ``None`` (CLI default), we use ``round(plan_hz)``
    so out-of-the-box recordings match training-time visually.
    """
    if video_fps is not None:
        if int(video_fps) < 1:
            raise ValueError(f"video_fps must be >= 1, got {video_fps!r}")
        return int(video_fps)
    if not (plan_hz > 0.0):
        raise ValueError(f"plan_hz must be > 0, got {plan_hz!r}")
    return max(1, int(round(float(plan_hz))))


def _stack_history(
    hist: deque[np.ndarray],
    horizon: int,
    dtype: np.dtype,
) -> np.ndarray:
    if len(hist) == 0:
        raise ValueError("empty history buffer")
    xs = list(hist)
    if len(xs) < horizon:
        pad = [xs[0]] * (horizon - len(xs))
        xs = pad + xs
    else:
        xs = xs[-horizon:]
    return np.stack(xs, axis=0).astype(dtype, copy=False)


def _resolve_video_source_key(video_key: str, obs_flat: dict[str, Any]) -> str:
    direct = f"video.{video_key}"
    if direct in obs_flat:
        return direct
    # Common alias when policy expects oak egoview naming.
    if "video.ego_view" in obs_flat:
        return "video.ego_view"
    cand = [k for k in obs_flat.keys() if k.startswith("video.")]
    if len(cand) == 1:
        return cand[0]
    raise KeyError(f"cannot resolve video source for policy key {video_key!r}; candidates={cand!r}")


def _select_plan_step(action_plan: dict[str, np.ndarray], idx: int) -> dict[str, np.ndarray]:
    step: dict[str, np.ndarray] = {}
    for k, v in action_plan.items():
        if v.ndim != 2:
            raise ValueError(f"expected action plan tensor (T,D) for {k}, got {v.shape}")
        t = min(idx, v.shape[0] - 1)
        step[k] = v[t]
    return step


def _clip_qpos_segment_to_joint_limits(
    model: Any,
    *,
    qpos_slice_start: int,
    segment: np.ndarray,
) -> np.ndarray:
    """Clip each scalar in ``segment`` to the MuJoCo joint limit owning ``qpos[i]``.

    ``qpos_slice_start`` is the index of ``segment[0]`` in ``model.qpos`` (e.g.
    ``7 + num_act`` for the Policy leg/arm qpos block). Unlimited joints are
    skipped.     Keeps crazy policy outputs inside the MJCF envelope so PD does not
    fight impossible targets (reduces pretzel-like contortions).
    """
    m = model
    out = np.asarray(segment, dtype=np.float64).reshape(-1).copy()
    base = int(qpos_slice_start)
    for i in range(out.size):
        iq = base + i
        jid = -1
        for j in range(int(m.njnt)):
            ja = int(m.jnt_qposadr[j])
            ja_next = int(m.jnt_qposadr[j + 1]) if j + 1 < int(m.njnt) else int(m.nq)
            if ja <= iq < ja_next:
                jid = j
                break
        if jid < 0:
            continue
        if not bool(m.jnt_limited[jid]):
            continue
        lo, hi = float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1])
        if lo > hi:
            lo, hi = hi, lo
        if lo < hi:
            out[i] = float(np.clip(out[i], lo, hi))
    return out.astype(np.float32, copy=False)


def _maybe_clip_arm_hand_vectors(
    model: Any | None,
    *,
    qpos_arm_start: int | None,
    clip: bool,
    la: np.ndarray | None,
    ra: np.ndarray | None,
    lh: np.ndarray | None,
    rh: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not clip or model is None or qpos_arm_start is None:
        return la, ra, lh, rh
    base = int(qpos_arm_start)
    if la is not None:
        la = _clip_qpos_segment_to_joint_limits(
            model, qpos_slice_start=base + SL_LEFT_ARM.start, segment=la
        )
    if ra is not None:
        ra = _clip_qpos_segment_to_joint_limits(
            model, qpos_slice_start=base + SL_RIGHT_ARM.start, segment=ra
        )
    if lh is not None:
        lh = _clip_qpos_segment_to_joint_limits(
            model, qpos_slice_start=base + SL_LEFT_HAND.start, segment=lh
        )
    if rh is not None:
        rh = _clip_qpos_segment_to_joint_limits(
            model, qpos_slice_start=base + SL_RIGHT_HAND.start, segment=rh
        )
    return la, ra, lh, rh


def _gather_arm_hand_targets_from_action_step(
    action_step: dict[str, np.ndarray],
    state_now: dict[str, np.ndarray],
    *,
    arm_action_mode: str,
    hand_permutation: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Recover pre-clip arm/hand targets from a single plan row (same rules as ``_apply_action_step``)."""

    def _arm_target(which: str) -> np.ndarray | None:
        k = f"action.{which}_arm"
        if k not in action_step:
            return None
        a = np.asarray(action_step[k], dtype=np.float32).reshape(7)
        if arm_action_mode == "relative":
            a = a + np.asarray(state_now[f"state.{which}_arm"], dtype=np.float32).reshape(7)
        return a

    la = _arm_target("left")
    ra = _arm_target("right")

    def _hand_target(which: str, perm: np.ndarray | None) -> np.ndarray | None:
        k = f"action.{which}_hand"
        if k not in action_step:
            return None
        v = np.asarray(action_step[k], dtype=np.float32).reshape(-1)
        if v.size != 7:
            raise ValueError(f"action.{which}_hand expected 7 DoF, got {v.shape}")
        if perm is None:
            return v.copy()
        if perm.shape != (7,):
            raise ValueError(
                f"hand permutation for {which} must be shape (7,), got {perm.shape}"
            )
        return v[perm].astype(np.float32, copy=False)

    left_perm = hand_permutation[0] if hand_permutation is not None else None
    right_perm = hand_permutation[1] if hand_permutation is not None else None
    lh = _hand_target("left", left_perm)
    rh = _hand_target("right", right_perm)
    return la, ra, lh, rh


def _apply_action_step(
    rt: Any,
    action_step: dict[str, np.ndarray],
    state_now: dict[str, np.ndarray],
    *,
    arm_action_mode: str,
    clip_arm_targets_to_limits: bool = False,
    apply_waist_fn: Any = None,
    hand_permutation: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Write a single decoded policy step into ``rt.control_dict``.

    Returns a small dict of *applied* tensors (post arm-mode conversion) so
    the caller can log them or compare against the runtime's PD targets.

    Parameters
    ----------
    apply_waist_fn : Callable[[np.ndarray], np.ndarray] | None
        When provided and ``action.waist`` is present in the chunk, called as
        ``rpy = apply_waist_fn(waist_action)`` and the length-3 ``rpy``
        (roll, pitch, yaw) result is written into ``rt.control_dict["rpy_cmd"]``.
        Mirrors upstream ``G1DecoupledWholeBodyPolicy.get_action`` so the leg
        ONNX policy receives the GR00T-commanded torso orientation; without
        this wiring the arms typically swing toward face level to compensate.
        Pass ``None`` to leave ``rpy_cmd`` untouched (legacy / opt-out path).
    hand_permutation : (np.ndarray, np.ndarray) | None
        Optional ``(left_perm, right_perm)`` index arrays converting the
        policy's ``action.{side}_hand`` order (``robot_model`` sorted DoF idx,
        i.e. index → middle → thumb) into MuJoCo's qpos kinematic-tree order
        (``hand_gripper.LEFT_HAND_JOINT_NAMES``, i.e. thumb → middle → index).
        Pass ``None`` to write the action through unchanged (legacy behaviour;
        produces wrong fingers actuating).

    When ``action.left_hand`` and/or ``action.right_hand`` are applied on a
    28-DoF ``arm_target_q`` runtime, sets ``control_dict[GR00T_POLICY_HAND_PD_SOURCE_KEY]``
    so :class:`GearWBCRuntime` PD-tracks those slices instead of overwriting them
    from ``gripper_*`` each substep.
    """
    applied: dict[str, np.ndarray] = {}
    with rt.cmd_lock:
        if "action.navigate_command" in action_step:
            nav = np.asarray(action_step["action.navigate_command"], dtype=np.float32).reshape(-1)
            rt.control_dict["loco_cmd"][:] = nav[:3]
            applied["loco_cmd"] = nav[:3].copy()
        if "action.base_height_command" in action_step:
            h = float(np.asarray(action_step["action.base_height_command"], dtype=np.float32).reshape(-1)[0])
            rt.control_dict["height_cmd"] = h
            applied["height_cmd"] = np.array([h], dtype=np.float32)
        if apply_waist_fn is not None and "action.waist" in action_step:
            w = np.asarray(action_step["action.waist"], dtype=np.float64).reshape(-1)
            rpy = np.asarray(apply_waist_fn(w), dtype=np.float32).reshape(-1)
            if rpy.size != 3:
                raise ValueError(f"apply_waist_fn must return length-3 rpy, got {rpy.shape}")
            rpy_buf = rt.control_dict.get("rpy_cmd")
            if rpy_buf is None:
                rt.control_dict["rpy_cmd"] = rpy.copy()
            else:
                np.asarray(rpy_buf).reshape(-1)[:3] = rpy
            applied["rpy_cmd"] = rpy.copy()

        if "arm_target_q" not in rt.control_dict:
            return applied
        # IMPORTANT: ``GearWBCRuntime._init_arm_state`` initializes ``arm_target_q`` as
        # ``self.data.qpos[...].copy()``, i.e. an **fp64** ndarray (qpos is fp64 in MuJoCo).
        # Calling ``np.asarray(..., dtype=np.float32)`` here would force a fresh fp32 copy,
        # silently dropping any writes we make to the slice. Take the raw buffer instead and
        # let assignment broadcast/cast to its dtype. This bug was the second half of the
        # "robot doesn't move" symptom (the first was the relative/absolute arm convention).
        atq = rt.control_dict["arm_target_q"]
        if not isinstance(atq, np.ndarray):
            atq = np.asarray(atq)
            rt.control_dict["arm_target_q"] = atq
        atq = atq.reshape(-1)
        n_arm = int(atq.shape[0])
        if getattr(rt, "has_hands", False) and n_arm >= 28:
            rt.control_dict[GR00T_POLICY_HAND_PD_SOURCE_KEY] = False

        la, ra, lh, rh = _gather_arm_hand_targets_from_action_step(
            action_step,
            state_now,
            arm_action_mode=arm_action_mode,
            hand_permutation=hand_permutation,
        )

        qpos_arm_start: int | None = None
        if clip_arm_targets_to_limits:
            na = getattr(rt, "num_act", None)
            if na is not None:
                qpos_arm_start = 7 + int(na)
        la, ra, lh, rh = _maybe_clip_arm_hand_vectors(
            getattr(rt, "model", None),
            qpos_arm_start=qpos_arm_start,
            clip=bool(clip_arm_targets_to_limits and qpos_arm_start is not None),
            la=la,
            ra=ra,
            lh=lh,
            rh=rh,
        )

        if rt.has_hands and n_arm >= 28:
            if la is not None:
                atq[SL_LEFT_ARM] = la
                applied["arm_target_q.left_arm"] = la.copy()
            if ra is not None:
                atq[SL_RIGHT_ARM] = ra
                applied["arm_target_q.right_arm"] = ra.copy()
            if lh is not None:
                atq[SL_LEFT_HAND] = lh
                applied["arm_target_q.left_hand"] = lh.copy()
            if rh is not None:
                atq[SL_RIGHT_HAND] = rh
                applied["arm_target_q.right_hand"] = rh.copy()
            if lh is not None or rh is not None:
                rt.control_dict[GR00T_POLICY_HAND_PD_SOURCE_KEY] = True
        elif n_arm >= 14:
            if la is not None:
                atq[SL_LEFT_ARM] = la
                applied["arm_target_q.left_arm"] = la.copy()
            if ra is not None:
                atq[SL_RIGHT_ARM] = ra
                applied["arm_target_q.right_arm"] = ra.copy()
            # No articulated hands in this mode.
    return applied


def _json_safe_float(x: float | None) -> float | None:
    if x is None:
        return None
    xf = float(x)
    if not math.isfinite(xf):
        return None
    return xf


def _max_abs_vec_diff(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        return None
    return float(np.max(np.abs(aa - bb)))


def _inference_trace_row(
    rt: Any,
    applied: dict[str, np.ndarray],
    action_step: dict[str, np.ndarray],
    state_now: dict[str, np.ndarray],
    *,
    arm_action_mode: str,
    hand_permutation: tuple[np.ndarray, np.ndarray] | None,
    clip_arm_targets_to_limits: bool,
    step_i: int,
    chunk_row: int | None,
    n_replans: int | None,
    replanned: bool,
    branch: str,
    n_substeps: int,
) -> dict[str, Any]:
    """One JSON-serializable record: commands, apply fidelity, post-step tracking error."""

    na = getattr(rt, "num_act", None)
    qpos_arm_start: int | None = None
    if na is not None:
        qpos_arm_start = 7 + int(na)

    la_e, ra_e, lh_e, rh_e = _gather_arm_hand_targets_from_action_step(
        action_step,
        state_now,
        arm_action_mode=arm_action_mode,
        hand_permutation=hand_permutation,
    )
    la_e, ra_e, lh_e, rh_e = _maybe_clip_arm_hand_vectors(
        getattr(rt, "model", None),
        qpos_arm_start=qpos_arm_start,
        clip=bool(clip_arm_targets_to_limits and qpos_arm_start is not None),
        la=la_e,
        ra=ra_e,
        lh=lh_e,
        rh=rh_e,
    )

    apply_la = _max_abs_vec_diff(
        la_e, applied.get("arm_target_q.left_arm")  # type: ignore[arg-type]
    )
    apply_ra = _max_abs_vec_diff(
        ra_e, applied.get("arm_target_q.right_arm")  # type: ignore[arg-type]
    )
    apply_lh = _max_abs_vec_diff(
        lh_e, applied.get("arm_target_q.left_hand")  # type: ignore[arg-type]
    )
    apply_rh = _max_abs_vec_diff(
        rh_e, applied.get("arm_target_q.right_hand")  # type: ignore[arg-type]
    )

    nav_diff: float | None = None
    if "action.navigate_command" in action_step and "loco_cmd" in applied:
        nav = np.asarray(action_step["action.navigate_command"], dtype=np.float64).reshape(-1)[:3]
        lc_ap = np.asarray(applied["loco_cmd"], dtype=np.float64).reshape(-1)[:3]
        nav_diff = float(np.max(np.abs(nav - lc_ap)))

    row: dict[str, Any] = {
        "kind": "inference_trace",
        "branch": branch,
        "step": int(step_i),
        "wall_time": time.time(),
        "sim_time_s": _json_safe_float(
            float(step_i) * float(n_substeps) * float(rt.model.opt.timestep)
        ),
        "n_substeps": int(n_substeps),
        "chunk_row": chunk_row,
        "n_replans": n_replans,
        "replanned": bool(replanned),
        "apply_max_abs_la": _json_safe_float(apply_la),
        "apply_max_abs_ra": _json_safe_float(apply_ra),
        "apply_max_abs_lh": _json_safe_float(apply_lh),
        "apply_max_abs_rh": _json_safe_float(apply_rh),
        "nav_policy_vs_loco_max_abs": _json_safe_float(nav_diff),
    }

    with rt.cmd_lock:
        lc = np.asarray(rt.control_dict["loco_cmd"], dtype=np.float32).reshape(-1)[:3].copy()
        hc = float(rt.control_dict["height_cmd"])
        if "rpy_cmd" in rt.control_dict:
            rpy = np.asarray(rt.control_dict["rpy_cmd"], dtype=np.float64).reshape(-1)[:3].copy()
        else:
            rpy = np.zeros(3, dtype=np.float64)
        pelvis_z = float(rt.data.qpos[2])
        if "arm_target_q" not in rt.control_dict or na is None:
            row["loco_cmd"] = [float(lc[i]) for i in range(3)]
            row["height_cmd"] = _json_safe_float(hc)
            row["rpy_cmd"] = [float(rpy[i]) for i in range(3)]
            row["pelvis_z"] = _json_safe_float(pelvis_z)
            return row

        atq_now = np.asarray(rt.control_dict["arm_target_q"], dtype=np.float64).reshape(-1)
        base = int(qpos_arm_start)
        n_arm = int(atq_now.shape[0])

        def _track(sl: slice) -> float | None:
            if atq_now.size < sl.stop:
                return None
            tgt = np.asarray(atq_now[sl], dtype=np.float64).reshape(-1)
            meas = np.asarray(
                rt.data.qpos[base + sl.start : base + sl.stop],
                dtype=np.float64,
            ).reshape(-1)
            return float(np.max(np.abs(tgt - meas)))

        row["track_max_abs_la"] = _json_safe_float(_track(SL_LEFT_ARM))
        row["track_max_abs_ra"] = _json_safe_float(_track(SL_RIGHT_ARM))
        if bool(getattr(rt, "has_hands", False)) and n_arm >= 28:
            row["track_max_abs_lh"] = _json_safe_float(_track(SL_LEFT_HAND))
            row["track_max_abs_rh"] = _json_safe_float(_track(SL_RIGHT_HAND))

        row["loco_cmd"] = [float(lc[i]) for i in range(3)]
        row["height_cmd"] = _json_safe_float(hc)
        row["rpy_cmd"] = [float(rpy[i]) for i in range(3)]
        row["pelvis_z"] = _json_safe_float(pelvis_z)

    return row


def _write_inference_trace_row(
    fh: TextIO,
    row: dict[str, Any],
    *,
    warn_apply_rad: float,
    warn_track_rad: float,
    stderr_warn: bool,
) -> None:
    """Append one JSON line; optional stderr when thresholds exceeded."""

    def _hot(name: str, th: float) -> bool:
        if th <= 0.0:
            return False
        v = row.get(name)
        if v is None:
            return False
        try:
            return float(v) > float(th)
        except (TypeError, ValueError):
            return False

    hot: list[str] = []
    for name in (
        "apply_max_abs_la",
        "apply_max_abs_ra",
        "apply_max_abs_lh",
        "apply_max_abs_rh",
    ):
        if _hot(name, warn_apply_rad):
            hot.append(name)
    for name in (
        "track_max_abs_la",
        "track_max_abs_ra",
        "track_max_abs_lh",
        "track_max_abs_rh",
    ):
        if _hot(name, warn_track_rad):
            hot.append(name)
    if _hot("nav_policy_vs_loco_max_abs", warn_apply_rad):
        hot.append("nav_policy_vs_loco_max_abs")

    row["warn_thresholds"] = {
        "apply_rad": float(warn_apply_rad),
        "track_rad": float(warn_track_rad),
    }
    if hot:
        row["warn_metrics"] = hot

    fh.write(json.dumps(row, allow_nan=False) + "\n")
    fh.flush()

    if stderr_warn and hot:
        print(
            f"WARNING[inference_trace] step={row.get('step')} exceeded thresholds: {hot}",
            file=sys.stderr,
        )


# ----------------------------------------------------------------------------
# Diagnostics helpers
# ----------------------------------------------------------------------------


def _vec_summary(name: str, x: np.ndarray) -> str:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return (
        f"{name}: shape={tuple(np.asarray(x).shape)} dtype={np.asarray(x).dtype} "
        f"min={a.min():+.4f} max={a.max():+.4f} mean={a.mean():+.4f} std={a.std():+.4f} "
        f"|x|={np.linalg.norm(a):.4f} first8={np.array2string(a[:8], precision=4, suppress_small=True)}"
    )


def _print_action_plan(plan: dict[str, np.ndarray], step_idx: int) -> None:
    print(f"-- policy action plan @ replan step={step_idx} --")
    for k in sorted(plan.keys()):
        v = np.asarray(plan[k])
        print(
            f"  {k}: T={v.shape[0]} D={v.shape[1] if v.ndim > 1 else 1} "
            f"first[:1]={np.array2string(v[0], precision=4, suppress_small=True)}"
        )


def _print_hand_action_plan_summary(plan: dict[str, np.ndarray]) -> None:
    """Compact stats for ``action.{left,right}_hand`` rows (server sends ABSOLUTE joints)."""
    for key, tag in (("action.left_hand", "L"), ("action.right_hand", "R")):
        if key not in plan:
            print(
                f"  hand {tag}: MISSING — no finger targets from policy for this replan; "
                "``arm_target_q`` hand slice stays at prior/init pose"
            )
            continue
        v = np.asarray(plan[key], dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 7:
            print(f"  hand {tag}: unexpected shape {v.shape} (want T×7)")
            continue
        row_norms = np.linalg.norm(v, axis=1)
        t_std_mean = float(np.mean(np.std(v, axis=0)))
        print(
            f"  hand {tag}: T={v.shape[0]} "
            f"|row| min/max/mean={row_norms.min():.3f}/{row_norms.max():.3f}/{row_norms.mean():.3f} "
            f"mean joint σ across time={t_std_mean:.5f} "
            f"row0={np.array2string(v[0], precision=3, suppress_small=True)}"
        )


def _warn_if_hands_mjcf_missing_hand_policy_keys(rt: Any, plan: dict[str, np.ndarray]) -> None:
    global _warned_policy_hand_keys_missing
    if _warned_policy_hand_keys_missing:
        return
    if not bool(getattr(rt, "has_hands", False)):
        return
    if int(getattr(rt, "n_arm", 0)) < 28:
        return
    if "action.left_hand" in plan and "action.right_hand" in plan:
        return
    _warned_policy_hand_keys_missing = True
    print(
        "WARNING: hands MJCF (28-D arm_target_q) but policy chunk omits "
        "action.left_hand and/or action.right_hand — "
        f"keys present: {sorted(plan.keys())!r}. "
        "Fingers are not driven by the policy until both keys are returned."
    )


def _periodic_line_hand_suffix(atq_now: np.ndarray, rt: Any) -> str:
    """Append ``|atq.lh|``, ``|qpos.lh|``, etc. when ``arm_target_q`` includes hands."""
    if atq_now.size < 28:
        return ""
    base = 7 + int(rt.num_act)
    lh_atq = np.asarray(atq_now[SL_LEFT_HAND], dtype=np.float64).reshape(-1)
    rh_atq = np.asarray(atq_now[SL_RIGHT_HAND], dtype=np.float64).reshape(-1)
    qh = rt.data.qpos
    lh_q = np.asarray(qh[base + SL_LEFT_HAND.start : base + SL_LEFT_HAND.stop], dtype=np.float64)
    rh_q = np.asarray(qh[base + SL_RIGHT_HAND.start : base + SL_RIGHT_HAND.stop], dtype=np.float64)
    return (
        f" |atq.lh|={float(np.linalg.norm(lh_atq)):.3f} |atq.rh|={float(np.linalg.norm(rh_atq)):.3f}"
        f" |qpos.lh|={float(np.linalg.norm(lh_q)):.3f} |qpos.rh|={float(np.linalg.norm(rh_q)):.3f}"
    )


def _print_obs_summary(obs_flat: dict[str, Any], state_keys: list[str]) -> None:
    print("-- observation snapshot --")
    for k in state_keys:
        sk = f"state.{k}"
        if sk in obs_flat:
            print("  " + _vec_summary(sk, np.asarray(obs_flat[sk])))
    for vk in [k for k in obs_flat.keys() if k.startswith("video.")]:
        v = np.asarray(obs_flat[vk])
        print(
            f"  {vk}: shape={tuple(v.shape)} dtype={v.dtype} "
            f"mean_rgb=({v[..., 0].mean():.1f},{v[..., 1].mean():.1f},{v[..., 2].mean():.1f})"
        )


def _save_frame_png(path: Path, rgb: np.ndarray) -> None:
    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), np.asarray(rgb, dtype=np.uint8))
        print(f"saved frame0 image: {path}")
    except Exception as exc:  # pragma: no cover - diagnostic best-effort
        print(f"WARNING: could not save frame0 png ({exc})")


# ----------------------------------------------------------------------------
# NPZ replay (sanity check the obs/action -> runtime bridge without any model)
# ----------------------------------------------------------------------------


def _load_npz_actions(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        # NPZ logger replaces '.' with '_' in saved keys (see Gr00tTeleopEpisodeLogger.save_npz).
        keys = list(z.files)
        out: dict[str, np.ndarray] = {}
        for k in keys:
            if k.startswith("action_"):
                out[f"action.{k[len('action_'):]}"] = np.asarray(z[k]).astype(np.float32)
        if not out:
            raise ValueError(f"no action_* arrays found in {npz_path}")
    return out


def _replay_step_from_npz(
    actions: dict[str, np.ndarray],
    t: int,
) -> dict[str, np.ndarray]:
    step: dict[str, np.ndarray] = {}
    for k, v in actions.items():
        i = min(t, v.shape[0] - 1)
        step[k] = np.asarray(v[i], dtype=np.float32)
    return step


def build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI parser. Factored out of ``main`` so tests can pin defaults."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--policy-host", type=str, default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=2000)
    p.add_argument(
        "--prompt",
        "--task",
        dest="task_description",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "Language instruction (e.g. annotation.human.task_description). "
            "``--task`` is a backward-compatible alias. "
            "If omitted, a default is chosen from ``--scene`` "
            "(diner: blue block; table_pnp_apple / default apple string: apple+plate; "
            "table_pnp: same prompt but legacy blue box MJCF; floor: walk/balance)."
        ),
    )
    p.add_argument(
        "--scene",
        choices=SCENE_CHOICES,
        default="stylish_diner",
        help=(
            "MuJoCo scene for G1GearWBC-v0: stylish_diner, table_pnp (vendored blue box), "
            "table_pnp_apple (B1 apple+plate MJCF under scenes/table_pnp_apple/), or floor."
        ),
    )
    p.add_argument(
        "--camera-name",
        type=str,
        default="oak_egoview",
        help=(
            "Observation camera name passed to Gr00tObservationBuilder. "
            "Default 'oak_egoview' matches the RoboCasa training-time logger "
            "(pos=(0.102,-0.009,0.424), fovy=79.5; see scenes/table_pnp_apple/"
            "g1_gear_wbc_hands_table_pnp_apple.xml). Pass '--camera-name head_pov' "
            "to reproduce the older mid-torso view for A/B; if the named camera "
            "is missing from the MJCF, the builder silently falls back to head_pov."
        ),
    )
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument(
        "--arm-action-mode",
        choices=ARM_ACTION_MODE_CHOICES,
        default=ARM_ACTION_MODE_AUTO,
        help=(
            "How to interpret action.left_arm / action.right_arm. The default "
            "'auto' picks per branch: 'absolute' for live PolicyClient (server "
            "unapply_action already absolutized when use_relative_action=True) "
            "and 'relative' for --replay-npz (NPZ stores arm_target_q - state "
            "deltas by logger default). Override only if you changed the "
            "logger / exporter / launcher representation so disk or server "
            "output no longer matches. Picking the wrong mode for live policy "
            "(the historical 'relative' default) double-adds state and "
            "compounds joint targets to ~11 rad within ~50 plan steps."
        ),
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=30,
        help=(
            "How many policy-plan steps to execute before requesting a new plan. "
            "Default 30 matches ``unitree_g1`` action delta_indices length in "
            "Isaac-GR00T ``embodiment_configs``. Smaller = more frequent replans "
            "(can look jittery); use 1 only for diagnostics."
        ),
    )
    p.add_argument(
        "--no-clip-arm-targets",
        action="store_true",
        help=(
            "Do not clip arm/hand joint targets to MuJoCo joint limits before PD. "
            "Default is to clip, which caps impossible targets when the policy drifts "
            "or OOD and reduces violent self-intersecting poses."
        ),
    )
    p.add_argument(
        "--no-apply-waist",
        action="store_true",
        help=(
            "Skip applying ``action.waist`` -> ``rpy_cmd`` via Pinocchio FK. "
            "Default is to apply, mirroring upstream "
            "``G1DecoupledWholeBodyPolicy.get_action``. Passing this flag "
            "restores the pre-fix behaviour where the ONNX leg policy keeps "
            "torso upright regardless of the GR00T waist command — useful as "
            "an A/B knob to reproduce the May 2026 ``hands lift toward face`` "
            "symptom on PnP checkpoints."
        ),
    )
    p.add_argument(
        "--no-permute-hands",
        action="store_true",
        help=(
            "Skip the policy -> MuJoCo qpos permutation for ``action.{left,right}_hand``. "
            "Default is to permute so policy slot 0 (index_0) maps to MuJoCo slot 5 etc. "
            "Pass this flag to write the policy hand vector through unchanged "
            "(reproduces the May 2026 bug #6 where index commands actuate thumb joints)."
        ),
    )
    p.add_argument(
        "--plan-hz",
        type=float,
        default=DEFAULT_PLAN_HZ,
        help=(
            "Plan rate (Hz) used to derive the number of mj_step substeps per "
            "policy plan step: n_sub = round(1 / (plan_hz * sim_dt)). The "
            "default 50.0 matches Gr00tTeleopEpisodeLogger.sample_hz and the "
            "upstream SyncEnv control_freq=50; keep this aligned with the "
            "training-time sample_hz or per-step targets are applied at the "
            "wrong wall-clock rate (the symptom is exploding arm targets / "
            "'pretzel' contortion shortly after policy start)."
        ),
    )
    p.add_argument("--no-hands", action="store_true")
    p.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Optional output MP4 path (ego camera).",
    )
    p.add_argument(
        "--video-fps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "MP4 frame rate. Default: match ``--plan-hz`` (one frame is written per "
            "plan step, so this keeps playback speed aligned with training footage). "
            "Pass an explicit value only for editing / upload constraints."
        ),
    )
    p.add_argument(
        "--policy-ab-dump",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "After the first successful PolicyClient.get_action, write obs tensors, "
            "returned action tensors, manifest.json, and obs/video_ego_view_t0.png under "
            "DIR for numerical A/B vs Isaac-GR00T rollout_policy.py (--policy-ab-dump there)."
        ),
    )
    p.add_argument(
        "--debug-policy-prints",
        action="store_true",
        help=(
            "Print first-N policy actions (incl. hand chunk summary: MISSING vs norm stats), "
            "observation summaries, periodic lines with |atq.lh|/|qpos.lh| when hands enabled, "
            "and save a frame0 PNG next to --save-video."
        ),
    )
    p.add_argument(
        "--debug-print-every",
        type=int,
        default=50,
        help="With --debug-policy-prints, also print a periodic step summary.",
    )
    p.add_argument(
        "--debug-print-first-n",
        type=int,
        default=3,
        help="With --debug-policy-prints, dump full action stats for the first N replans.",
    )
    p.add_argument(
        "--replay-oracle-action-chunks",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Variant A: replay ``replan_*.npz`` action chunks recorded by Isaac-GR00T "
            "``rollout_policy.py --record-oracle-action-chunks DIR`` (same checkpoint / "
            "``--n-action-steps`` as capture). Skips PolicyClient; uses the same "
            "``_apply_action_step`` + physics cadence as live inference. Mutually "
            "exclusive with ``--replay-npz``. Default ``--arm-action-mode auto`` "
            "resolves to ``absolute`` (oracle chunks are post-server absolutized)."
        ),
    )
    p.add_argument(
        "--replay-npz",
        type=Path,
        default=None,
        help=(
            "Sanity-check mode: bypass the policy server and feed action.* from a "
            "recorded NPZ (produced by gr00t_teleop_logger.save_npz) directly into "
            "_apply_action_step using --arm-action-mode. Use this to confirm the "
            "obs/action -> runtime bridge is wired correctly: replaying a logged "
            "episode should reproduce visible arm motion / lift behaviour. "
            "If replay also stays idle, the issue is in the bridge, not in the policy."
        ),
    )
    p.add_argument(
        "--inference-trace-jsonl",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Append one JSON object per plan step (after physics) with loco/height/rpy, "
            "pelvis_z, apply-fidelity (policy row vs arm_target_q after clip), and "
            "tracking error (arm_target_q vs measured qpos). Use with jq/plotting; "
            "pair with --inference-trace-every to subsample."
        ),
    )
    p.add_argument(
        "--inference-trace-every",
        type=int,
        default=1,
        metavar="N",
        help="Write a trace row every N plan steps (default 1 = every step).",
    )
    p.add_argument(
        "--inference-trace-warn-apply-rad",
        type=float,
        default=0.05,
        metavar="RAD",
        help=(
            "With --inference-trace-warn-stderr, emit WARNING if any apply_* or "
            "nav_policy_vs_loco_max_abs exceeds this many radians (or dimensionless for nav)."
        ),
    )
    p.add_argument(
        "--inference-trace-warn-track-rad",
        type=float,
        default=0.45,
        metavar="RAD",
        help=(
            "With --inference-trace-warn-stderr, emit WARNING if any track_max_abs_* "
            "exceeds this many radians (PD lag / contacts can be large; tune as needed)."
        ),
    )
    p.add_argument(
        "--inference-trace-warn-stderr",
        action="store_true",
        help=(
            "Print a one-line WARNING to stderr when --inference-trace-jsonl thresholds "
            "are exceeded (see --inference-trace-warn-apply-rad / warn-track-rad)."
        ),
    )
    return p


def main() -> int:
    p = build_argparser()
    args = p.parse_args()
    ro_chunks = getattr(args, "replay_oracle_action_chunks", None)
    if ro_chunks is not None and args.replay_npz is not None:
        print(
            "error: use only one of --replay-oracle-action-chunks or --replay-npz",
            file=sys.stderr,
        )
        return 2
    args.task_description = resolve_task_description(
        str(args.scene), getattr(args, "task_description", None)
    )

    _ensure_isaac_gr00t_importable()

    register_g1_gear_wbc_env()
    env = gym.make(
        ENV_ID,
        scene=args.scene,
        use_hands=not args.no_hands,
    )
    rt = env.unwrapped._rt
    env.reset()

    obs_builder = Gr00tObservationBuilder.for_locomanip_default(
        camera_name=str(args.camera_name)
    )
    robot_model = obs_builder.robot_model

    # Fail-fast guard: if the supplemental joint group order ever drifts away
    # from MuJoCo qpos kinematic order, ``arm_target_q[SL_*_ARM]`` writes will
    # silently route shoulder targets to elbow joints etc. (bug #5 / #6 family).
    assert_arm_joint_order_matches(robot_model)

    apply_waist_fn = None
    if not bool(getattr(args, "no_apply_waist", False)):
        def apply_waist_fn(w: np.ndarray) -> np.ndarray:  # noqa: E306
            return _waist_action_to_rpy_cmd(robot_model, w)

    hand_perm: tuple[np.ndarray, np.ndarray] | None = None
    if not args.no_hands and not bool(getattr(args, "no_permute_hands", False)):
        hand_perm = build_hand_permutations(robot_model)

    sim_dt = float(rt.model.opt.timestep)
    n_substeps = resolve_plan_substeps(sim_dt=sim_dt, plan_hz=float(args.plan_hz))
    if args.debug_policy_prints:
        print(
            "=== plan cadence ===\n"
            f"  sim_dt={sim_dt:.6f}s  plan_hz={float(args.plan_hz):.3f}Hz  "
            f"n_substeps_per_plan_step={n_substeps}  "
            f"effective_plan_period={n_substeps * sim_dt * 1000:.2f}ms"
        )
        print(
            "=== bridge wiring ===\n"
            f"  apply_waist: {apply_waist_fn is not None} "
            f"(--no-apply-waist={bool(getattr(args, 'no_apply_waist', False))})\n"
            f"  permute_hands: {hand_perm is not None} "
            f"(--no-permute-hands={bool(getattr(args, 'no_permute_hands', False))}, "
            f"no_hands={bool(args.no_hands)})"
        )
        if hand_perm is not None:
            print(
                f"  hand perm (policy->mj): left={hand_perm[0].tolist()} "
                f"right={hand_perm[1].tolist()}"
            )

    # ------------------------------------------------------------------
    # Branch A0: oracle Tier-A action-chunk replay (Variant A, no server).
    # ------------------------------------------------------------------
    if getattr(args, "replay_oracle_action_chunks", None) is not None:
        return _run_oracle_action_chunk_replay_loop(
            args=args,
            env=env,
            rt=rt,
            obs_builder=obs_builder,
            n_substeps=n_substeps,
            apply_waist_fn=apply_waist_fn,
            hand_permutation=hand_perm,
        )

    # ------------------------------------------------------------------
    # Branch A: NPZ replay (skips PolicyClient entirely).
    # ------------------------------------------------------------------
    if args.replay_npz is not None:
        return _run_replay_loop(
            args=args,
            env=env,
            rt=rt,
            obs_builder=obs_builder,
            n_substeps=n_substeps,
            apply_waist_fn=apply_waist_fn,
            hand_permutation=hand_perm,
        )

    # ------------------------------------------------------------------
    # Branch B: live PolicyClient inference.
    # ------------------------------------------------------------------
    return _run_policy_loop(
        args=args,
        env=env,
        rt=rt,
        obs_builder=obs_builder,
        n_substeps=n_substeps,
        apply_waist_fn=apply_waist_fn,
        hand_permutation=hand_perm,
    )


def _advance_physics_one_plan_step(env: gym.Env, n_substeps: int) -> None:
    """Advance ``n_substeps`` mj_step calls; PD targets in ``rt.control_dict``
    stay fixed throughout, so this holds a plan step for one plan period."""
    if n_substeps < 1:
        raise ValueError(f"n_substeps must be >= 1, got {n_substeps!r}")
    for _ in range(n_substeps):
        env.step(np.zeros(1, dtype=np.float32))


def _open_video_writer(args: argparse.Namespace):
    if args.save_video is None:
        return None, None
    import imageio.v2 as imageio

    out = args.save_video.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = resolve_video_fps(args.video_fps, float(args.plan_hz))
    writer = imageio.get_writer(str(out), fps=int(fps))
    print(f"video writer: {out}  fps={fps}")
    return writer, out


def _run_policy_loop(
    *,
    args: argparse.Namespace,
    env: gym.Env,
    rt: Any,
    obs_builder: Gr00tObservationBuilder,
    n_substeps: int,
    apply_waist_fn: Any = None,
    hand_permutation: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    from gr00t.policy.server_client import PolicyClient

    arm_mode_resolved = resolve_arm_action_mode(
        str(args.arm_action_mode), branch="live"
    )

    client = PolicyClient(host=args.policy_host, port=args.policy_port, strict=False)
    if not client.ping():
        raise RuntimeError(
            f"cannot reach policy server at {args.policy_host}:{args.policy_port}; "
            "start run_gr00t_server.py first"
        )
    client.reset()
    modality = client.get_modality_config()
    video_keys = list(modality["video"].modality_keys)
    state_keys = list(modality["state"].modality_keys)
    language_keys = list(modality["language"].modality_keys)
    video_h = int(len(modality["video"].delta_indices))
    state_h = int(len(modality["state"].delta_indices))
    if video_h <= 0 or state_h <= 0:
        raise ValueError(f"invalid modality horizons: video={video_h}, state={state_h}")

    if args.debug_policy_prints:
        print("=== policy modality config ===")
        print(f"  video keys: {video_keys}  horizon={video_h}")
        print(f"  state keys: {state_keys}  horizon={state_h}")
        print(f"  language keys: {language_keys}")
        print(
            f"  arm_action_mode: {args.arm_action_mode} -> {arm_mode_resolved} (branch=live)  "
            f"n_action_steps: {args.n_action_steps}  "
            f"plan_hz: {float(args.plan_hz):.3f}  n_substeps: {n_substeps}"
        )
        print(f"  task_description: {args.task_description!r}")
        print(f"  has_hands: {rt.has_hands}  n_arm: {int(rt.n_arm)}")

    state_hist: dict[str, deque[np.ndarray]] = {
        k: deque(maxlen=state_h) for k in state_keys
    }
    video_hist: dict[str, deque[np.ndarray]] = {
        k: deque(maxlen=video_h) for k in video_keys
    }
    video_source_for_key: dict[str, str] = {}

    writer, out_path = _open_video_writer(args)

    trace_f: TextIO | None = None
    trace_path = getattr(args, "inference_trace_jsonl", None)
    if trace_path is not None:
        tp = Path(trace_path).expanduser().resolve()
        tp.parent.mkdir(parents=True, exist_ok=True)
        trace_f = tp.open("a", encoding="utf-8")
        print(f"inference trace JSONL -> {tp.resolve()}")

    action_plan: dict[str, np.ndarray] = {}
    action_plan_idx = 0
    n_replans = 0
    saved_frame0 = False
    policy_ab_dumped = False

    try:
        for step_i in range(int(args.max_steps)):
            replanned_this_step = False
            obs_flat = obs_builder.build(
                rt.model,
                rt.data,
                task_description=str(args.task_description),
                include_video=True,
            )

            if (
                args.debug_policy_prints
                and not saved_frame0
                and out_path is not None
                and "ego_view_image" in obs_flat
            ):
                _save_frame_png(
                    out_path.with_name(out_path.stem + "_frame0.png"),
                    np.asarray(obs_flat["ego_view_image"], dtype=np.uint8),
                )
                _print_obs_summary(obs_flat, state_keys)
                saved_frame0 = True

            for k in state_keys:
                sk = f"state.{k}"
                if sk not in obs_flat:
                    raise KeyError(f"missing state key in observation: {sk}")
                state_hist[k].append(np.asarray(obs_flat[sk], dtype=np.float32).reshape(-1))

            for k in video_keys:
                src = video_source_for_key.get(k)
                if src is None:
                    src = _resolve_video_source_key(k, obs_flat)
                    video_source_for_key[k] = src
                    print(f"video key mapping: policy video.{k} <- {src}")
                frame = np.asarray(obs_flat[src], dtype=np.uint8)
                video_hist[k].append(frame)

            if (not action_plan) or (action_plan_idx >= int(args.n_action_steps)):
                replanned_this_step = True
                policy_obs: dict[str, Any] = {}
                for k in video_keys:
                    x = _stack_history(video_hist[k], video_h, np.uint8)
                    policy_obs[f"video.{k}"] = x[None, ...]  # (B=1,T,H,W,C)
                for k in state_keys:
                    x = _stack_history(state_hist[k], state_h, np.float32)
                    policy_obs[f"state.{k}"] = x[None, ...]  # (B=1,T,D)
                for lk in language_keys:
                    policy_obs[lk] = (str(args.task_description),)

                action_raw, _info = client.get_action(policy_obs)
                action_plan = {}
                for k, v in action_raw.items():
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.ndim != 3 or arr.shape[0] != 1:
                        raise ValueError(f"unexpected action tensor for {k}: {arr.shape}")
                    action_plan[k] = arr[0]  # (T,D)
                action_plan_idx = 0
                n_replans += 1

                if args.policy_ab_dump is not None and not policy_ab_dumped:
                    dump_policy_roundtrip_pack(
                        args.policy_ab_dump,
                        side="minimal_sim",
                        policy_obs=dict(policy_obs),
                        actions=dict(action_raw),
                        extra={
                            "scene": str(args.scene),
                            "plan_step": int(step_i),
                            "n_replans": int(n_replans),
                            "arm_action_mode": str(args.arm_action_mode),
                            "arm_action_mode_resolved": str(arm_mode_resolved),
                            "plan_hz": float(args.plan_hz),
                            "n_action_steps": int(args.n_action_steps),
                            "task_description": str(args.task_description),
                        },
                    )
                    print(f"policy A/B dump (minimal_sim) -> {args.policy_ab_dump.resolve()}")
                    policy_ab_dumped = True

                _warn_if_hands_mjcf_missing_hand_policy_keys(rt, action_plan)
                if apply_waist_fn is not None:
                    _warn_if_waist_action_missing(action_plan)
                if args.debug_policy_prints and n_replans <= int(args.debug_print_first_n):
                    _print_action_plan(action_plan, step_idx=step_i)
                    _print_hand_action_plan_summary(action_plan)

            chunk_row = int(action_plan_idx)
            action_step = _select_plan_step(action_plan, action_plan_idx)
            action_plan_idx += 1
            state_now = {f"state.{k}": np.asarray(state_hist[k][-1], dtype=np.float32) for k in state_keys}
            applied = _apply_action_step(
                rt,
                action_step,
                state_now,
                arm_action_mode=arm_mode_resolved,
                clip_arm_targets_to_limits=not bool(args.no_clip_arm_targets),
                apply_waist_fn=apply_waist_fn,
                hand_permutation=hand_permutation,
            )

            if (
                args.debug_policy_prints
                and step_i < int(args.debug_print_first_n)
            ):
                print(f"-- applied step {step_i} (arm_mode={arm_mode_resolved}) --")
                for k in sorted(applied.keys()):
                    print("  " + _vec_summary(k, applied[k]))

            _advance_physics_one_plan_step(env, n_substeps)

            if writer is not None:
                writer.append_data(np.asarray(obs_flat["video.ego_view"], dtype=np.uint8))

            trace_every = max(1, int(getattr(args, "inference_trace_every", 1)))
            if trace_f is not None and step_i % trace_every == 0:
                row = _inference_trace_row(
                    rt,
                    applied,
                    action_step,
                    state_now,
                    arm_action_mode=arm_mode_resolved,
                    hand_permutation=hand_permutation,
                    clip_arm_targets_to_limits=not bool(args.no_clip_arm_targets),
                    step_i=step_i,
                    chunk_row=chunk_row,
                    n_replans=int(n_replans),
                    replanned=bool(replanned_this_step),
                    branch="live",
                    n_substeps=n_substeps,
                )
                _write_inference_trace_row(
                    trace_f,
                    row,
                    warn_apply_rad=float(
                        getattr(args, "inference_trace_warn_apply_rad", 0.05)
                    ),
                    warn_track_rad=float(
                        getattr(args, "inference_trace_warn_track_rad", 0.45)
                    ),
                    stderr_warn=bool(
                        getattr(args, "inference_trace_warn_stderr", False)
                    ),
                )

            if step_i % max(1, int(args.debug_print_every)) == 0:
                with rt.cmd_lock:
                    lc = np.asarray(rt.control_dict["loco_cmd"], dtype=np.float32).copy()
                    hc = float(rt.control_dict["height_cmd"])
                    rpy = (
                        np.asarray(rt.control_dict["rpy_cmd"], dtype=np.float64).copy()
                        if "rpy_cmd" in rt.control_dict
                        else np.zeros(3, dtype=np.float64)
                    )
                    atq_now = (
                        np.asarray(rt.control_dict["arm_target_q"], dtype=np.float64).copy()
                        if "arm_target_q" in rt.control_dict
                        else np.zeros(0, dtype=np.float64)
                    )
                la_atq = atq_now[SL_LEFT_ARM] if atq_now.size >= 7 else atq_now
                ra_atq = atq_now[SL_RIGHT_ARM] if atq_now.size >= 21 else np.zeros(0)
                meas_la = np.asarray(
                    rt.data.qpos[7 + rt.num_act + SL_LEFT_ARM.start : 7 + rt.num_act + SL_LEFT_ARM.stop],
                    dtype=np.float64,
                )
                msg = (
                    f"step={step_i:05d} loco=({lc[0]:+.3f},{lc[1]:+.3f},{lc[2]:+.3f}) "
                    f"rpy=({rpy[0]:+.3f},{rpy[1]:+.3f},{rpy[2]:+.3f}) "
                    f"height={hc:+.3f} pelvis_z={float(rt.data.qpos[2]):.3f} "
                    f"|atq.la|={float(np.linalg.norm(la_atq)):.3f} "
                    f"|atq.ra|={float(np.linalg.norm(ra_atq)):.3f} "
                    f"|qpos.la|={float(np.linalg.norm(meas_la)):.3f}"
                )
                msg += _periodic_line_hand_suffix(atq_now, rt)
                if args.debug_policy_prints and atq_now.size >= 7:
                    msg += (
                        f" atq.la={np.array2string(la_atq, precision=3, suppress_small=True)}"
                        f" qpos.la={np.array2string(meas_la, precision=3, suppress_small=True)}"
                    )
                print(msg)
    finally:
        if trace_f is not None:
            trace_f.close()
        if writer is not None:
            writer.close()
        obs_builder.close()
        env.close()

    print("done")
    return 0


def _run_oracle_action_chunk_replay_loop(
    *,
    args: argparse.Namespace,
    env: gym.Env,
    rt: Any,
    obs_builder: Gr00tObservationBuilder,
    n_substeps: int,
    apply_waist_fn: Any = None,
    hand_permutation: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    """Replay Tier-A ``replan_*.npz`` chunks (Variant A) without ``PolicyClient``."""

    arm_mode_resolved = resolve_arm_action_mode(
        str(args.arm_action_mode), branch="oracle_replay"
    )
    root = Path(args.replay_oracle_action_chunks).expanduser().resolve()
    chunk_files = oracle_action_chunk_io.sorted_replan_paths(root)
    if not chunk_files:
        raise RuntimeError(f"no replan_*.npz under {root}")

    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and args.debug_policy_prints:
        print(f"oracle replay: manifest {manifest_path}")
    print(
        f"oracle replay: {len(chunk_files)} chunk file(s) under {root}  "
        f"n_substeps={n_substeps}  arm_action_mode={args.arm_action_mode} -> {arm_mode_resolved}"
    )

    writer, out_path = _open_video_writer(args)
    trace_f: TextIO | None = None
    trace_path = getattr(args, "inference_trace_jsonl", None)
    if trace_path is not None:
        tp = Path(trace_path).expanduser().resolve()
        tp.parent.mkdir(parents=True, exist_ok=True)
        trace_f = tp.open("a", encoding="utf-8")
        print(f"inference trace JSONL -> {tp.resolve()}")

    state_keys_seen: list[str] = [
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    ]
    saved_frame0 = False

    action_plan: dict[str, np.ndarray] = {}
    plan_rows = 0
    action_plan_idx = 0
    file_idx = 0
    n_chunks_consumed = 0

    try:
        for step_i in range(int(args.max_steps)):
            if not action_plan or action_plan_idx >= plan_rows:
                if file_idx >= len(chunk_files):
                    print(f"oracle replay: exhausted chunks at plan step={step_i}")
                    break
                action_plan = oracle_action_chunk_io.load_replan_chunk(chunk_files[file_idx])
                file_idx += 1
                n_chunks_consumed += 1
                plan_rows = int(next(iter(action_plan.values())).shape[0])
                if plan_rows != int(args.n_action_steps):
                    print(
                        f"WARNING: oracle chunk T={plan_rows} != --n-action-steps="
                        f"{int(args.n_action_steps)} (file {chunk_files[file_idx - 1].name})"
                    )
                action_plan_idx = 0
                if args.debug_policy_prints:
                    print(
                        f"oracle replay: loaded chunk {n_chunks_consumed}/"
                        f"{len(chunk_files)} rows={plan_rows} keys={sorted(action_plan.keys())}"
                    )

            chunk_row = int(action_plan_idx)
            action_step = _select_plan_step(action_plan, action_plan_idx)
            action_plan_idx += 1

            obs_flat = obs_builder.build(
                rt.model,
                rt.data,
                task_description=str(args.task_description),
                include_video=True,
            )
            if not saved_frame0 and out_path is not None and "ego_view_image" in obs_flat:
                _save_frame_png(
                    out_path.with_name(out_path.stem + "_oracle_chunk_frame0.png"),
                    np.asarray(obs_flat["ego_view_image"], dtype=np.uint8),
                )
                if args.debug_policy_prints:
                    _print_obs_summary(obs_flat, state_keys_seen)
                saved_frame0 = True

            state_now = {
                f"state.{k}": np.asarray(obs_flat.get(f"state.{k}"), dtype=np.float32).reshape(-1)
                for k in state_keys_seen
                if f"state.{k}" in obs_flat
            }
            applied = _apply_action_step(
                rt,
                action_step,
                state_now,
                arm_action_mode=arm_mode_resolved,
                clip_arm_targets_to_limits=not bool(args.no_clip_arm_targets),
                apply_waist_fn=apply_waist_fn,
                hand_permutation=hand_permutation,
            )
            _advance_physics_one_plan_step(env, n_substeps)
            if writer is not None:
                writer.append_data(np.asarray(obs_flat["video.ego_view"], dtype=np.uint8))

            trace_every = max(1, int(getattr(args, "inference_trace_every", 1)))
            if trace_f is not None and step_i % trace_every == 0:
                row = _inference_trace_row(
                    rt,
                    applied,
                    action_step,
                    state_now,
                    arm_action_mode=arm_mode_resolved,
                    hand_permutation=hand_permutation,
                    clip_arm_targets_to_limits=not bool(args.no_clip_arm_targets),
                    step_i=step_i,
                    chunk_row=chunk_row,
                    n_replans=int(n_chunks_consumed),
                    replanned=chunk_row == 0,
                    branch="oracle_replay",
                    n_substeps=n_substeps,
                )
                _write_inference_trace_row(
                    trace_f,
                    row,
                    warn_apply_rad=float(
                        getattr(args, "inference_trace_warn_apply_rad", 0.05)
                    ),
                    warn_track_rad=float(
                        getattr(args, "inference_trace_warn_track_rad", 0.45)
                    ),
                    stderr_warn=bool(
                        getattr(args, "inference_trace_warn_stderr", False)
                    ),
                )

            if step_i % max(1, int(args.debug_print_every)) == 0:
                with rt.cmd_lock:
                    lc = np.asarray(rt.control_dict["loco_cmd"], dtype=np.float32).copy()
                    hc = float(rt.control_dict["height_cmd"])
                    rpy = (
                        np.asarray(rt.control_dict["rpy_cmd"], dtype=np.float64).copy()
                        if "rpy_cmd" in rt.control_dict
                        else np.zeros(3, dtype=np.float64)
                    )
                    atq = (
                        np.asarray(rt.control_dict["arm_target_q"], dtype=np.float64).copy()
                        if "arm_target_q" in rt.control_dict
                        else np.zeros(0, dtype=np.float64)
                    )
                la_atq = atq[SL_LEFT_ARM] if atq.size >= 7 else atq
                ra_atq = atq[SL_RIGHT_ARM] if atq.size >= 21 else np.zeros(0)
                meas_la = np.asarray(
                    rt.data.qpos[7 + rt.num_act + SL_LEFT_ARM.start : 7 + rt.num_act + SL_LEFT_ARM.stop],
                    dtype=np.float64,
                )
                omsg = (
                    f"oracle_replay step={step_i:05d} chunk={n_chunks_consumed} row={chunk_row} "
                    f"loco=({lc[0]:+.3f},{lc[1]:+.3f},{lc[2]:+.3f}) "
                    f"pelvis_z={float(rt.data.qpos[2]):.3f} "
                    f"|atq.la|={float(np.linalg.norm(la_atq)):.3f} "
                    f"|qpos.la|={float(np.linalg.norm(meas_la)):.3f}"
                )
                omsg += _periodic_line_hand_suffix(atq, rt)
                print(omsg)
    finally:
        if trace_f is not None:
            trace_f.close()
        if writer is not None:
            writer.close()
        obs_builder.close()
        env.close()

    print("oracle chunk replay done")
    return 0


def _run_replay_loop(
    *,
    args: argparse.Namespace,
    env: gym.Env,
    rt: Any,
    obs_builder: Gr00tObservationBuilder,
    n_substeps: int,
    apply_waist_fn: Any = None,
    hand_permutation: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    """Replay a recorded NPZ episode through ``_apply_action_step``.

    Used as a sanity check: if the bridge from action.* -> arm_target_q is
    correct and PD is healthy, replaying a logged demo (with the matching
    --arm-action-mode) should reproduce roughly the original arm/hand motion.
    """
    arm_mode_resolved = resolve_arm_action_mode(
        str(args.arm_action_mode), branch="replay"
    )

    npz_path = Path(args.replay_npz).expanduser().resolve()
    actions = _load_npz_actions(npz_path)
    T = next(iter(actions.values())).shape[0]
    print(
        f"replay: loaded {len(actions)} action keys from {npz_path} "
        f"(T={T} plan steps; n_substeps={n_substeps}; "
        f"arm_action_mode={args.arm_action_mode} -> {arm_mode_resolved})"
    )
    if args.debug_policy_prints:
        print("replay action keys + first row stats:")
        for k in sorted(actions.keys()):
            v = actions[k]
            print(f"  {k}: shape={v.shape} dtype={v.dtype} first[0]={v[0]}")

    if bool(getattr(rt, "has_hands", False)) and int(getattr(rt, "n_arm", 0)) >= 28:
        for hk in ("action.left_hand", "action.right_hand"):
            if hk not in actions:
                print(
                    f"WARNING: replay NPZ missing {hk!r} — hand slice of ``arm_target_q`` "
                    "will not be updated from the recording."
                )

    writer, out_path = _open_video_writer(args)
    trace_f: TextIO | None = None
    trace_path = getattr(args, "inference_trace_jsonl", None)
    if trace_path is not None:
        tp = Path(trace_path).expanduser().resolve()
        tp.parent.mkdir(parents=True, exist_ok=True)
        trace_f = tp.open("a", encoding="utf-8")
        print(f"inference trace JSONL -> {tp.resolve()}")

    saved_frame0 = False
    n_steps = min(int(args.max_steps), T)

    state_keys_seen: list[str] = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]

    try:
        for step_i in range(n_steps):
            obs_flat = obs_builder.build(
                rt.model,
                rt.data,
                task_description=str(args.task_description),
                include_video=True,
            )
            if not saved_frame0 and out_path is not None and "ego_view_image" in obs_flat:
                _save_frame_png(
                    out_path.with_name(out_path.stem + "_replay_frame0.png"),
                    np.asarray(obs_flat["ego_view_image"], dtype=np.uint8),
                )
                if args.debug_policy_prints:
                    _print_obs_summary(obs_flat, state_keys_seen)
                saved_frame0 = True

            state_now = {
                f"state.{k}": np.asarray(obs_flat.get(f"state.{k}"), dtype=np.float32).reshape(-1)
                for k in state_keys_seen
                if f"state.{k}" in obs_flat
            }
            action_step = _replay_step_from_npz(actions, step_i)
            applied = _apply_action_step(
                rt,
                action_step,
                state_now,
                arm_action_mode=arm_mode_resolved,
                clip_arm_targets_to_limits=not bool(args.no_clip_arm_targets),
                apply_waist_fn=apply_waist_fn,
                hand_permutation=hand_permutation,
            )
            _advance_physics_one_plan_step(env, n_substeps)
            if writer is not None:
                writer.append_data(np.asarray(obs_flat["video.ego_view"], dtype=np.uint8))

            trace_every = max(1, int(getattr(args, "inference_trace_every", 1)))
            if trace_f is not None and step_i % trace_every == 0:
                row = _inference_trace_row(
                    rt,
                    applied,
                    action_step,
                    state_now,
                    arm_action_mode=arm_mode_resolved,
                    hand_permutation=hand_permutation,
                    clip_arm_targets_to_limits=not bool(args.no_clip_arm_targets),
                    step_i=step_i,
                    chunk_row=int(step_i),
                    n_replans=None,
                    replanned=False,
                    branch="replay",
                    n_substeps=n_substeps,
                )
                _write_inference_trace_row(
                    trace_f,
                    row,
                    warn_apply_rad=float(
                        getattr(args, "inference_trace_warn_apply_rad", 0.05)
                    ),
                    warn_track_rad=float(
                        getattr(args, "inference_trace_warn_track_rad", 0.45)
                    ),
                    stderr_warn=bool(
                        getattr(args, "inference_trace_warn_stderr", False)
                    ),
                )

            if step_i % max(1, int(args.debug_print_every)) == 0:
                with rt.cmd_lock:
                    lc = np.asarray(rt.control_dict["loco_cmd"], dtype=np.float32).copy()
                    hc = float(rt.control_dict["height_cmd"])
                    rpy = (
                        np.asarray(rt.control_dict["rpy_cmd"], dtype=np.float64).copy()
                        if "rpy_cmd" in rt.control_dict
                        else np.zeros(3, dtype=np.float64)
                    )
                    atq = (
                        np.asarray(rt.control_dict["arm_target_q"], dtype=np.float64).copy()
                        if "arm_target_q" in rt.control_dict
                        else np.zeros(0, dtype=np.float64)
                    )
                la_atq = atq[SL_LEFT_ARM] if atq.size >= 7 else atq
                ra_atq = atq[SL_RIGHT_ARM] if atq.size >= 21 else np.zeros(0)
                meas_la = np.asarray(
                    rt.data.qpos[7 + rt.num_act + SL_LEFT_ARM.start : 7 + rt.num_act + SL_LEFT_ARM.stop],
                    dtype=np.float64,
                )
                rmsg = (
                    f"replay step={step_i:05d} loco=({lc[0]:+.3f},{lc[1]:+.3f},{lc[2]:+.3f}) "
                    f"rpy=({rpy[0]:+.3f},{rpy[1]:+.3f},{rpy[2]:+.3f}) "
                    f"height={hc:+.3f} pelvis_z={float(rt.data.qpos[2]):.3f} "
                    f"|atq.la|={float(np.linalg.norm(la_atq)):.3f} "
                    f"|atq.ra|={float(np.linalg.norm(ra_atq)):.3f} "
                    f"|qpos.la|={float(np.linalg.norm(meas_la)):.3f}"
                )
                rmsg += _periodic_line_hand_suffix(atq, rt)
                print(rmsg)
    finally:
        if trace_f is not None:
            trace_f.close()
        if writer is not None:
            writer.close()
        obs_builder.close()
        env.close()

    print("replay done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
