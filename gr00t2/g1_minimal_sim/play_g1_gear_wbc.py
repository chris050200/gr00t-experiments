#!/usr/bin/env python3
"""Play G1 Gear WBC through Gymnasium (Phase 2a): same physics as ``spawn_g1_floor.py --stand``.

Examples (from ``g1_minimal_sim/``, use your Isaac-GR00T venv python):

    python play_g1_gear_wbc.py --headless --max-steps 2000
    python play_g1_gear_wbc.py --teleop
    python play_g1_gear_wbc.py --scene floor --teleop
    python play_g1_gear_wbc.py --vr-teleop
    python play_g1_gear_wbc.py --vr-teleop --debug-vr-stream
    python play_g1_gear_wbc.py --scene table_pnp --headless --max-steps 2000
    python play_g1_gear_wbc.py --headless --hands --max-steps 800 --headless-scripted-demo wiggle \\
        --gr00t-record-dir /tmp/g1_syn --gr00t-task "synthetic wiggle (pipeline test)"
    python play_g1_gear_wbc.py --vr-teleop --debug-vr-stream --log /tmp/g1_vr.jsonl

``--log`` truncates that file each run; use ``--log-append`` to continue an existing log.

Default ``--scene`` is **stylish_diner**; ``--scene table_pnp`` is the vendored table + blue ``table_box``;
``--scene table_pnp_apple`` loads the B1 apple+plate MJCF under ``scenes/table_pnp_apple/``.

Optional ``--keyboard-grasp-primitives`` (with ``--teleop``): symmetric palm spacing (**x**/**c**),
forward/back (**Home** / **End**), and raise/lower (**Page Up** / **Page Down**); see
``keyboard_grasp_primitives.py``. Default **off**.

Optional ``--lock-ee-orient`` (with ``--teleop``): each step resets ``ee_{left,right}_quat`` to the
spawn home in torso frame (position-only arms). Ignored when ``--vr-ik`` is on (stream owns orientation).

**P2 GR00T logging:** ``--gr00t-record-dir DIR`` saves ``episode_000.npz`` (+ metadata) each run — paired with WholeBodyControl venv for ``gr00t_wbc``. Optional ``--gr00t-task "..."``, ``--gr00t-record-video`` (needs GL).

``--teleop`` needs ``pynput``, DISPLAY, and **terminal focus** for keys (see ``spawn_g1_floor.py``).
``--vr-teleop`` listens for ``vr_teleop.openvr_udp_sender`` (default bind ``0.0.0.0:5006``); sticks → ``loco_cmd``.
``--debug-vr-stream`` mounts stream poses on ``torso_link`` (see
``memory-bank/vr_stream_debug_overlays.md``).

Requires: ``pip install gymnasium`` (in addition to mujoco / onnxruntime / pyyaml).
"""

from __future__ import annotations

import argparse
import math
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gear_wbc_pd import (
    ARM_PD_KD_DEFAULT_PER_JOINT,
    ARM_PD_KD_GRASP_PER_JOINT,
    ARM_PD_KP_DEFAULT_PER_JOINT,
    ARM_PD_KP_GRASP_PER_JOINT,
    ARM_TAU_CLIP_GRASP_NM,
    ARM_TAU_CLIP_NM,
)
from hand_gripper import GRIP_KEY_STEP_DEFAULT
from vr_stream_ik_mode import normalize_vr_ik_anchor_mode, normalize_vr_ik_mode
from vr_teleop.vr_stream_target_viz import VR_STREAM_VIZ_Z_OFFSET_M
from task_scene import available_tasks, resolve_task_scene


def _resolve_arm_pd_overrides(args: argparse.Namespace) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    float | None,
]:
    """Resolve ``--arm-gain-profile`` + per-joint overrides into runtime kwargs.

    Returns ``(kp_per_joint, kd_per_joint, tau_clip_nm)``; each is ``None`` when nothing was
    overridden so the runtime keeps its compiled-in defaults (no behaviour drift).
    """
    profile = str(args.arm_gain_profile)
    if profile == "default":
        kp = None
        kd = None
        clip = None
    else:  # grasp
        kp = ARM_PD_KP_GRASP_PER_JOINT.copy()
        kd = ARM_PD_KD_GRASP_PER_JOINT.copy()
        clip = float(ARM_TAU_CLIP_GRASP_NM)

    def _seed_kp() -> np.ndarray:
        nonlocal kp
        if kp is None:
            kp = ARM_PD_KP_DEFAULT_PER_JOINT.copy()
        return kp

    def _seed_kd() -> np.ndarray:
        nonlocal kd
        if kd is None:
            kd = ARM_PD_KD_DEFAULT_PER_JOINT.copy()
        return kd

    if args.arm_kp_shoulder is not None:
        v = float(args.arm_kp_shoulder)
        a = _seed_kp()
        a[0] = a[1] = a[2] = v
    if args.arm_kd_shoulder is not None:
        v = float(args.arm_kd_shoulder)
        a = _seed_kd()
        a[0] = a[1] = a[2] = v
    if args.arm_kp_elbow is not None:
        a = _seed_kp()
        a[3] = float(args.arm_kp_elbow)
    if args.arm_kd_elbow is not None:
        a = _seed_kd()
        a[3] = float(args.arm_kd_elbow)
    if args.arm_kp_wrist_roll is not None:
        a = _seed_kp()
        a[4] = float(args.arm_kp_wrist_roll)
    if args.arm_kd_wrist_roll is not None:
        a = _seed_kd()
        a[4] = float(args.arm_kd_wrist_roll)
    if args.arm_kp_wrist_pitch is not None:
        a = _seed_kp()
        a[5] = float(args.arm_kp_wrist_pitch)
    if args.arm_kd_wrist_pitch is not None:
        a = _seed_kd()
        a[5] = float(args.arm_kd_wrist_pitch)
    if args.arm_kp_wrist_yaw is not None:
        a = _seed_kp()
        a[6] = float(args.arm_kp_wrist_yaw)
    if args.arm_kd_wrist_yaw is not None:
        a = _seed_kd()
        a[6] = float(args.arm_kd_wrist_yaw)
    if args.arm_tau_clip_nm is not None:
        clip = float(args.arm_tau_clip_nm)
    return kp, kd, clip


@dataclass(frozen=True)
class WiggleSchedule:
    """Precomputed wiggle parameters + optional per-step joint noise (built once before the loop)."""

    period: float
    phase0: float
    amp_left: float
    amp_right: float
    height_amp: float
    grip_amp: float
    loco_scale: float
    noise_atq: np.ndarray | None
    noise_grip: np.ndarray | None
    is_legacy: bool

    @staticmethod
    def legacy() -> WiggleSchedule:
        """Original fixed-sine wiggle (tests + default CLI with no wiggle flags)."""
        return WiggleSchedule(
            period=400.0,
            phase0=0.0,
            amp_left=1.0,
            amp_right=1.0,
            height_amp=0.012,
            grip_amp=0.18,
            loco_scale=1.0,
            noise_atq=None,
            noise_grip=None,
            is_legacy=True,
        )


def build_wiggle_schedule(max_steps: int, args: argparse.Namespace) -> WiggleSchedule:
    """Build schedule once per headless run. Noise tables are shape (max_steps, …), O(1) per step."""
    use_legacy = (
        args.wiggle_seed is None
        and not args.wiggle_randomize
        and math.isclose(float(args.wiggle_arm_noise_rad), 0.0, abs_tol=0.0, rel_tol=0.0)
        and math.isclose(float(args.wiggle_loco_scale), 1.0, rel_tol=0.0, abs_tol=1e-9)
    )
    if use_legacy:
        return WiggleSchedule.legacy()

    if args.wiggle_seed is not None:
        seed = int(args.wiggle_seed) & 0xFFFFFFFFFFFFFFFF
    else:
        seed = int.from_bytes(secrets.token_bytes(8), "little", signed=False) & 0xFFFFFFFFFFFFFFFF

    rng = np.random.default_rng(seed)
    period = float(rng.uniform(320.0, 480.0))
    phase0 = float(rng.uniform(0.0, 2.0 * math.pi))
    amp_left = float(rng.uniform(0.85, 1.15))
    amp_right = float(rng.uniform(0.85, 1.15))
    height_amp = float(rng.uniform(0.008, 0.016))
    grip_amp = float(rng.uniform(0.14, 0.22))
    sigma = float(args.wiggle_arm_noise_rad)
    noise_atq = noise_grip = None
    if sigma > 0.0 and max_steps > 0:
        noise_atq = rng.normal(0.0, sigma, size=(max_steps, 28)).astype(np.float32)
        np.clip(noise_atq, -3.0 * sigma, 3.0 * sigma, out=noise_atq)
        gs = 0.04 * min(1.0, sigma / 0.03) if sigma > 1e-9 else 0.04
        noise_grip = rng.normal(0.0, gs, size=(max_steps, 2)).astype(np.float32)
        np.clip(noise_grip, -0.12, 0.12, out=noise_grip)

    return WiggleSchedule(
        period=period,
        phase0=phase0,
        amp_left=amp_left,
        amp_right=amp_right,
        height_amp=height_amp,
        grip_amp=grip_amp,
        loco_scale=float(args.wiggle_loco_scale),
        noise_atq=noise_atq,
        noise_grip=noise_grip,
        is_legacy=False,
    )


def _apply_headless_scripted_demo(
    rt: Any,
    step_n: int,
    mode: str,
    base_arm_tgt: np.ndarray,
    base_height: float,
    base_loco: np.ndarray,
    wiggle_sched: WiggleSchedule | None,
) -> None:
    """Write scripted motion into ``control_dict`` before ``step_physics``.

    When no teleop / VR IK sources are active, :class:`GearWBCRuntime` does **not** run arm IK;
    it PD-tracks ``arm_target_q`` directly, so oscillating joint targets produces visible motion and
    non-trivial logged actions for P2 / export smoke tests.

    ``wiggle_sched`` is built once (optionally with precomputed per-step noise); the inner loop
    only indexes arrays and evaluates sines here — no RNG on the hot path.
    """
    if mode == "none":
        return
    if mode != "wiggle":
        raise ValueError(f"unknown headless scripted demo mode: {mode!r}")

    sched = wiggle_sched if wiggle_sched is not None else WiggleSchedule.legacy()
    ph = 2.0 * math.pi * float(step_n) / sched.period + sched.phase0
    aL, aR = sched.amp_left, sched.amp_right
    with rt.cmd_lock:
        atq = np.asarray(rt.control_dict["arm_target_q"], dtype=np.float32).reshape(-1)
        n = int(atq.shape[0])
        atq[:] = np.asarray(base_arm_tgt, dtype=np.float32).reshape(-1)[:n]
        # ``arm_target_q`` layout: 14 = left_arm(7) + right_arm(7); 28 = + hands between arms.
        if n >= 7:
            atq[3] += np.float32(0.18 * aL * math.sin(ph))
            atq[5] += np.float32(0.10 * aL * math.sin(2.0 * ph))
        if n >= 28:
            atq[17] += np.float32(0.12 * aR * math.sin(ph + 0.9))
            atq[19] += np.float32(0.08 * aR * math.sin(2.0 * ph + 1.0))
        elif n >= 14:
            atq[10] += np.float32(0.12 * aR * math.sin(ph + 0.9))
            atq[12] += np.float32(0.08 * aR * math.sin(2.0 * ph + 1.0))
        if sched.noise_atq is not None and step_n < int(sched.noise_atq.shape[0]):
            m = min(n, int(sched.noise_atq.shape[1]))
            atq[:m] += sched.noise_atq[int(step_n), :m]
        if rt.has_hands and n >= 28 and "gripper_left" in rt.control_dict:
            g0 = float(0.82 + sched.grip_amp * math.sin(ph * 1.7))
            g1 = float(0.82 + sched.grip_amp * math.cos(ph * 1.7))
            if sched.noise_grip is not None and step_n < int(sched.noise_grip.shape[0]):
                g0 += float(sched.noise_grip[int(step_n), 0])
                g1 += float(sched.noise_grip[int(step_n), 1])
            rt.control_dict["gripper_left"] = g0
            rt.control_dict["gripper_right"] = g1
        rt.control_dict["height_cmd"] = float(base_height + sched.height_amp * math.sin(ph * 0.4))
        lc = np.asarray(base_loco, dtype=np.float32).reshape(3).copy()
        s = sched.loco_scale
        lc[0] += np.float32(0.06 * s * math.sin(ph * 0.22))
        lc[2] += np.float32(0.04 * s * math.sin(ph * 0.18))
        rt.control_dict["loco_cmd"][:] = lc


def _parse_block_friction(s: str | None) -> tuple[float, float, float] | None:
    if s is None:
        return None
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--diner-block-friction expects 'TAN SLIP SPIN' (3 floats); got {s!r}"
        )
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--diner-block-friction parse error: {e}") from e


def _make_gr00t_p2_logger(unwrapped: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    """Return ``(Gr00tTeleopEpisodeLogger | None, Gr00tObservationBuilder | None)``."""
    if args.gr00t_record_dir is None:
        return None, None
    from gr00t_observation_builder import Gr00tObservationBuilder
    from gr00t_teleop_logger import Gr00tTeleopEpisodeLogger

    obs_b = Gr00tObservationBuilder.for_locomanip_default()
    log = Gr00tTeleopEpisodeLogger(
        unwrapped._rt,
        obs_builder=obs_b,
        task_description=str(args.gr00t_task or ""),
        include_video=bool(args.gr00t_record_video),
    )
    return log, obs_b


def _finalize_gr00t_p2_logger(
    log: Any,
    obs_b: Any,
    args: argparse.Namespace,
) -> None:
    if log is None:
        return
    try:
        if log.frames:
            out = Path(args.gr00t_record_dir).resolve() / "episode_000.npz"
            log.save_npz(out)
            print(
                f"GR00T P2 log saved: {out} ({len(log.frames)} steps, "
                f"video={'on' if args.gr00t_record_video else 'off'})"
            )
        elif args.gr00t_record_dir is not None:
            print("GR00T P2: no frames recorded (0 steps).")
    finally:
        if obs_b is not None:
            obs_b.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No MuJoCo viewer; step env for --max-steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Steps after reset (headless) or viewer loop limit when set with viewer.",
    )
    parser.add_argument(
        "--headless-scripted-demo",
        type=str,
        choices=("none", "wiggle"),
        default="none",
        help="With --headless only: deterministic control_dict motion for synthetic P2 logs / pipeline tests.",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Keyboard teleop (viewer only; same keymap as spawn_g1_floor --stand --teleop).",
    )
    parser.add_argument(
        "--keyboard-grasp-primitives",
        action="store_true",
        help="With --teleop only: optional symmetric palm spacing (x/c), forward/back (Home/End), "
        "and raise/lower (Page Up/Down). Default off.",
    )
    parser.add_argument(
        "--lock-ee-orient",
        action="store_true",
        help="With --teleop: each step set palm quats to init (torso frame); position-only arm control. "
        "No effect with --vr-ik on (VR stream owns orientation).",
    )
    parser.add_argument(
        "--openvr-teleop",
        action="store_true",
        help="OpenVR controllers → IK (viewer only; pip install openvr; SteamVR).",
    )
    parser.add_argument(
        "--udp-teleop",
        type=str,
        default=None,
        metavar="HOST:PORT",
        help="Listen for UDP JSON from legacy_vr_code.openvr_teleop_client on the VR PC (e.g. 0.0.0.0:5005). "
        "Allowed with --headless. Not combinable with --vr-teleop.",
    )
    parser.add_argument(
        "--vr-teleop",
        nargs="?",
        const="0.0.0.0:5006",
        default=None,
        metavar="HOST:PORT",
        help="Listen for UDP from vr_teleop.openvr_udp_sender (Quest/SteamVR host). "
        "Default bind 0.0.0.0:5006 if flag given alone. Sticks → locomotion only (Phase A). "
        "Allowed with --headless. Not combinable with --udp-teleop.",
    )
    parser.add_argument(
        "--debug-ee-targets",
        action="store_true",
        help="Viewer only: draw world-space IK palm targets (green=left, orange=right).",
    )
    parser.add_argument(
        "--debug-vr-stream",
        action="store_true",
        help="Viewer only: draw streamed HMD + controllers as arrows (blue/green/orange). "
        "Requires --vr-teleop.",
    )
    parser.add_argument(
        "--debug-vr-stream-torso-z",
        choices=("full", "fixed_world", "frozen_calib"),
        default="full",
        help="With --debug-vr-stream: full=rigid Z; fixed_world=z=Z-offset value; "
        "frozen_calib=lock each device's world Z at first composed sample.",
    )
    parser.add_argument(
        "--debug-vr-stream-z-offset",
        type=float,
        default=VR_STREAM_VIZ_Z_OFFSET_M,
        metavar="M",
        help="world frame: +Z added to raw stream positions. torso+fixed_world: absolute world Z (m) "
        f"for all three arrows (default {VR_STREAM_VIZ_Z_OFFSET_M}).",
    )
    parser.add_argument(
        "--debug-vr-stream-calib-orient",
        choices=("full", "yaw_only"),
        default="yaw_only",
        help="Calibration orientation mode for HMD0 in torso compose: "
        "full uses full HMD quaternion, yaw_only drops startup pitch/roll coupling.",
    )
    parser.add_argument(
        "--vr-receiver-yaw-deg",
        type=float,
        default=-90.0,
        metavar="DEG",
        help="Receiver-side yaw correction (degrees) for VR torso compose. "
        "Applied on sim side to the tracked torso orientation basis only.",
    )
    parser.add_argument(
        "--vr-ik",
        type=normalize_vr_ik_mode,
        default="off",
        metavar="MODE",
        help="With --vr-teleop: stream controllers → torso-frame ee_* IK targets. off | on.",
    )
    parser.add_argument(
        "--vr-ik-anchor",
        type=normalize_vr_ik_anchor_mode,
        default="legacy",
        metavar="MODE",
        help="With --vr-teleop and --vr-ik on: how stream HMD/controllers map to sim world palms. "
        "legacy=frozen HMD0 (old); torso=current headset-relative on torso_link; "
        "pelvis=same relative chain mounted from pelvis (constant torso–pelvis offset at calib).",
    )
    parser.add_argument(
        "--vr-head-orient",
        choices=("full", "yaw_only"),
        default="yaw_only",
        help="With --vr-teleop and non-legacy anchors: orientation used for current HMD in "
        "inv(T_hmd) compose. yaw_only keeps world Z shared and drops HMD pitch/roll coupling.",
    )
    parser.add_argument(
        "--vr-ik-z-offset-m",
        type=float,
        default=0.0,
        metavar="M",
        help="With --vr-teleop and --vr-ik on: add world +Z offset (m) to composed VR IK palm targets "
        "before conversion to torso-frame ee_*.",
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=None,
        help="Directory with g1_gear_wbc.yaml (default: vendored g1 bundle next to Isaac-GR00T).",
    )
    parser.add_argument(
        "--max-viewer-steps",
        type=int,
        default=None,
        help="If set, quit viewer after this many steps (default: run until window closed).",
    )
    parser.add_argument(
        "--hands",
        action="store_true",
        help="Use g1_gear_wbc_hands.xml (grip: 9/0 left, =/- right, h hard-close both, y open both).",
    )
    parser.add_argument(
        "--scene",
        choices=("stylish_diner", "floor", "table_pnp", "table_pnp_apple"),
        default="stylish_diner",
        help="MJCF preset: stylish diner (default), empty floor, or table + apple + plate.",
    )
    parser.add_argument(
        "--task",
        choices=available_tasks(),
        default=None,
        help="Task preset from scenes/<task>/task_spec.yaml. Overrides --scene and --config-yaml.",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="Override Gear WBC yaml (basename vs --resources-dir, or absolute path).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSONL trace path (VR / teleop IK). Overwrites each run unless --log-append.",
    )
    parser.add_argument(
        "--log-append",
        action="store_true",
        help="Append to --log instead of truncating (multiple sessions in one file).",
    )
    parser.add_argument(
        "--log-hz",
        type=float,
        default=30.0,
        help="Max trace rows per second (default: 30). Use 0 to log every sim step.",
    )
    parser.add_argument(
        "--forensic-burst",
        action="store_true",
        help="Enable tier-B forensic burst logging around trigger episodes (same JSONL file).",
    )
    parser.add_argument(
        "--forensic-pre-steps",
        type=int,
        default=180,
        help="Forensic burst: keep this many pre-trigger rows in ring buffer.",
    )
    parser.add_argument(
        "--forensic-post-steps",
        type=int,
        default=240,
        help="Forensic burst: emit this many rows after trigger.",
    )
    parser.add_argument(
        "--forensic-trigger-pos-err-m",
        type=float,
        default=0.15,
        help="Forensic burst trigger threshold: max pos error (m).",
    )
    parser.add_argument(
        "--forensic-trigger-quat-err-deg",
        type=float,
        default=45.0,
        help="Forensic burst trigger threshold: max quat error (deg).",
    )
    parser.add_argument(
        "--forensic-trigger-raw-step-l2",
        type=float,
        default=1.0,
        help="Forensic burst trigger threshold: IK raw step L2.",
    )
    parser.add_argument(
        "--forensic-trigger-step-scale-max",
        type=float,
        default=0.40,
        help="Forensic burst trigger threshold: IK step-scale <= this value.",
    )
    parser.add_argument(
        "--forensic-cooldown-steps",
        type=int,
        default=240,
        help="Forensic burst cooldown between episodes.",
    )
    parser.add_argument(
        "--viewer-camera",
        choices=("default", "head_pov"),
        default="default",
        help="Viewer only: default orbit camera; head_pov uses MJCF camera ``head_pov`` on torso.",
    )
    parser.add_argument(
        "--arm-gain-profile",
        choices=("default", "grasp"),
        default="default",
        help=(
            "Arm PD profile. 'default' (kp=45 uniform, kd=1.2, clip=30 Nm) preserves the "
            "historical free-air IK tracking baseline. 'grasp' stiffens elbow + wrists "
            f"(kp={list(ARM_PD_KP_GRASP_PER_JOINT)}, "
            f"kd={list(ARM_PD_KD_GRASP_PER_JOINT)}, "
            f"clip={ARM_TAU_CLIP_GRASP_NM:.0f} Nm) so palm-pinch can sustain contact load."
        ),
    )
    parser.add_argument("--arm-kp-shoulder", type=float, default=None,
                        help="Per-axis kp for shoulder pitch/roll/yaw (overrides profile).")
    parser.add_argument("--arm-kd-shoulder", type=float, default=None,
                        help="Per-axis kd for shoulder pitch/roll/yaw (overrides profile).")
    parser.add_argument("--arm-kp-elbow", type=float, default=None,
                        help="kp for elbow (overrides profile).")
    parser.add_argument("--arm-kd-elbow", type=float, default=None,
                        help="kd for elbow (overrides profile).")
    parser.add_argument("--arm-kp-wrist-roll", type=float, default=None,
                        help="kp for wrist_roll (overrides profile).")
    parser.add_argument("--arm-kd-wrist-roll", type=float, default=None,
                        help="kd for wrist_roll (overrides profile).")
    parser.add_argument("--arm-kp-wrist-pitch", type=float, default=None,
                        help="kp for wrist_pitch (most contact-loaded; overrides profile).")
    parser.add_argument("--arm-kd-wrist-pitch", type=float, default=None,
                        help="kd for wrist_pitch (overrides profile).")
    parser.add_argument("--arm-kp-wrist-yaw", type=float, default=None,
                        help="kp for wrist_yaw (overrides profile).")
    parser.add_argument("--arm-kd-wrist-yaw", type=float, default=None,
                        help="kd for wrist_yaw (overrides profile).")
    parser.add_argument(
        "--arm-tau-clip-nm",
        type=float,
        default=None,
        help=f"Per-actuator arm torque clip (Nm). Default {ARM_TAU_CLIP_NM:.0f}; grasp profile {ARM_TAU_CLIP_GRASP_NM:.0f}.",
    )
    parser.add_argument(
        "--diner-block-density",
        type=float,
        default=None,
        metavar="KG_PER_M3",
        help="Override target_block density at load time (no XML edit). XML default 5 kg/m^3 "
        "(very light, ~50-110 g depending on cube size) — treat as a starting point for "
        "prototyping the grasp pipeline, not as realistic manipulation mass. Raise to "
        "100-1000 to stress-test grasps with realistic objects. "
        "See memory-bank/INDEX.md (MuJoCo grasp / contact) and activeContext.md for realism bands.",
    )
    parser.add_argument(
        "--diner-block-friction",
        type=_parse_block_friction,
        default=None,
        metavar="\"TAN SLIP SPIN\"",
        help="Override target_block friction (3 floats). Diner default '3.0 0.1 0.005'.",
    )
    parser.add_argument(
        "--print-arm-tau",
        action="store_true",
        help="With --teleop: append arm-tau peak summary (post-clip wrist_pitch L/R + clip_frac) "
        "to the periodic teleop status print. Useful when sweeping --arm-* gains.",
    )
    parser.add_argument(
        "--grip-key-step",
        type=float,
        default=None,
        metavar="STEP",
        help="With --hands: per-tap delta for 9/0 and =/- on the gripper open scalar [0,1]. "
        f"Default {GRIP_KEY_STEP_DEFAULT} (fine steps). Must be in (0, 1].",
    )
    parser.add_argument(
        "--print-grip-force",
        action="store_true",
        help="With --teleop --hands: append block normal-force peak summary (MuJoCo |Fn|) to the "
        "periodic status print for tuning pinch vs slip.",
    )
    parser.add_argument(
        "--gr00t-record-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="P2: after run, save GR00T-shaped obs+action to DIR/episode_000.npz (requires "
        "Isaac-GR00T WholeBodyControl venv for gr00t_wbc). Use with --headless or viewer.",
    )
    parser.add_argument(
        "--gr00t-task",
        type=str,
        default="",
        metavar="TEXT",
        help="Task string for annotation.human.task_description in recorded obs (and metadata).",
    )
    parser.add_argument(
        "--gr00t-record-video",
        action="store_true",
        help="With --gr00t-record-dir: include ego_view frames (needs OpenGL; may require DISPLAY or MUJOCO_GL).",
    )
    wiggle = parser.add_argument_group(
        "headless wiggle demo (only with --headless --headless-scripted-demo wiggle)"
    )
    wiggle.add_argument(
        "--wiggle-seed",
        type=int,
        default=None,
        metavar="N",
        help="RNG seed for episode-specific period/phase/amplitudes and optional noise table. "
        "Omit with default noise/loco flags for the original fixed-sine wiggle.",
    )
    wiggle.add_argument(
        "--wiggle-randomize",
        action="store_true",
        help="Randomize episode motion params; uses --wiggle-seed if set, else an OS random seed.",
    )
    wiggle.add_argument(
        "--wiggle-arm-noise-rad",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Std-dev (rad) of per-step Gaussian noise on first 28 arm_target_q entries; precomputed once.",
    )
    wiggle.add_argument(
        "--wiggle-loco-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Scales the sinusoidal forward/yaw loco_cmd offsets (0 = stand still aside from height oscillation).",
    )
    args = parser.parse_args()

    if args.grip_key_step is not None:
        if not (0.0 < float(args.grip_key_step) <= 1.0):
            parser.error("--grip-key-step must be in (0, 1]")
        if not args.hands:
            parser.error("--grip-key-step requires --hands")
    if args.print_grip_force and not args.hands:
        parser.error("--print-grip-force requires --hands")
    if args.print_grip_force and not args.teleop:
        parser.error("--print-grip-force requires --teleop")
    if args.keyboard_grasp_primitives and not args.teleop:
        parser.error("--keyboard-grasp-primitives requires --teleop")
    if args.lock_ee_orient and not args.teleop:
        parser.error("--lock-ee-orient requires --teleop")
    if args.teleop and args.headless:
        parser.error("--teleop cannot be used with --headless")
    if args.openvr_teleop and args.headless:
        parser.error("--openvr-teleop cannot be used with --headless")
    if args.debug_ee_targets and args.headless:
        parser.error("--debug-ee-targets requires viewer (no --headless)")
    if args.debug_vr_stream and args.headless:
        parser.error("--debug-vr-stream requires viewer (no --headless)")
    if args.debug_vr_stream and args.vr_teleop is None:
        parser.error("--debug-vr-stream requires --vr-teleop")
    if args.vr_teleop is not None and args.udp_teleop is not None:
        parser.error("--vr-teleop and --udp-teleop use different protocols; use only one")
    if args.vr_ik != "off" and args.vr_teleop is None:
        parser.error("--vr-ik requires --vr-teleop")
    if args.vr_ik_anchor != "legacy" and args.vr_teleop is None:
        parser.error("--vr-ik-anchor requires --vr-teleop")
    if args.vr_ik_anchor != "legacy" and args.vr_ik == "off":
        parser.error("--vr-ik-anchor is only used with --vr-ik on")
    if args.headless and args.viewer_camera != "default":
        parser.error("--viewer-camera requires a viewer (no --headless)")
    if args.headless_scripted_demo != "none" and not args.headless:
        parser.error("--headless-scripted-demo requires --headless")
    _wiggle_opts = (
        args.wiggle_seed is not None
        or bool(args.wiggle_randomize)
        or not math.isclose(float(args.wiggle_arm_noise_rad), 0.0, abs_tol=0.0, rel_tol=0.0)
        or not math.isclose(float(args.wiggle_loco_scale), 1.0, rel_tol=0.0, abs_tol=1e-9)
    )
    if _wiggle_opts and str(args.headless_scripted_demo) != "wiggle":
        parser.error("wiggle options require --headless-scripted-demo wiggle")

    import gymnasium as gym

    from g1_gear_wbc_env import ENV_ID, ENV_ID_TABLE_PNP, register_g1_gear_wbc_env

    register_g1_gear_wbc_env()

    scene = str(args.scene)
    config_yaml = args.config_yaml
    task_meta = None
    if args.task is not None:
        scene, cfg_from_task, task_meta = resolve_task_scene(args.task)
        config_yaml = Path(cfg_from_task) if cfg_from_task is not None else None
        print(f"Task preset: {args.task} -> scene={scene}")
        if task_meta is not None:
            print(f"Task name: {task_meta.get('task_name', args.task)}")

    env_id = ENV_ID_TABLE_PNP if scene in ("table_pnp", "table_pnp_apple") else ENV_ID
    print(f"play_g1_gear_wbc: scene={scene!r} -> env_id={env_id!r} (omit --scene for default stylish_diner)")
    arm_kp_per_joint, arm_kd_per_joint, arm_tau_clip_nm = _resolve_arm_pd_overrides(args)
    use_hands = bool(args.hands)
    kwargs: dict = {
        "scene": scene,
        "keyboard_grasp_primitives": bool(args.keyboard_grasp_primitives),
        "lock_ee_orient": bool(args.lock_ee_orient),
        "enable_teleop": args.teleop,
        "openvr_teleop": args.openvr_teleop,
        "udp_teleop_bind": args.udp_teleop,
        "vr_teleop_bind": args.vr_teleop,
        "vr_receiver_yaw_deg": args.vr_receiver_yaw_deg,
        "vr_ik_mode": args.vr_ik,
        "vr_debug_stream": args.debug_vr_stream,
        "vr_debug_torso_z_mode": args.debug_vr_stream_torso_z,
        "vr_debug_z_offset_m": args.debug_vr_stream_z_offset,
        "vr_stream_calib_orient": args.debug_vr_stream_calib_orient,
        "vr_ik_anchor_mode": args.vr_ik_anchor,
        "vr_head_orient_mode": args.vr_head_orient,
        "vr_ik_z_offset_m": args.vr_ik_z_offset_m,
        "use_hands": use_hands,
        "debug_ee_targets": args.debug_ee_targets,
        "log_path": str(args.log) if args.log is not None else None,
        "log_hz": float(args.log_hz),
        "log_append": bool(args.log_append),
        "forensic_burst": bool(args.forensic_burst),
        "forensic_pre_steps": int(args.forensic_pre_steps),
        "forensic_post_steps": int(args.forensic_post_steps),
        "forensic_trigger_pos_err_m": float(args.forensic_trigger_pos_err_m),
        "forensic_trigger_quat_err_deg": float(args.forensic_trigger_quat_err_deg),
        "forensic_trigger_raw_step_l2": float(args.forensic_trigger_raw_step_l2),
        "forensic_trigger_step_scale_max": float(args.forensic_trigger_step_scale_max),
        "forensic_cooldown_steps": int(args.forensic_cooldown_steps),
        "arm_pd_kp_per_joint": arm_kp_per_joint,
        "arm_pd_kd_per_joint": arm_kd_per_joint,
        "arm_tau_clip_nm": arm_tau_clip_nm,
        "block_density": args.diner_block_density,
        "block_friction": args.diner_block_friction,
        "print_arm_tau": bool(args.print_arm_tau),
        "grip_key_step": args.grip_key_step,
        "print_grip_force": bool(args.print_grip_force),
    }
    if args.resources_dir is not None:
        kwargs["resources_dir"] = args.resources_dir
    if config_yaml is not None:
        kwargs["config_yaml"] = config_yaml

    env = gym.make(env_id, **kwargs)
    unwrapped = env.unwrapped

    if args.headless:
        gr00t_log, gr00t_obs_b = _make_gr00t_p2_logger(unwrapped, args)
        env.reset()
        demo = str(args.headless_scripted_demo)
        base_arm = base_h = base_loco = None
        wiggle_sched: WiggleSchedule | None = None
        if demo != "none":
            rt0 = unwrapped._rt
            with rt0.cmd_lock:
                base_arm = np.asarray(rt0.control_dict["arm_target_q"], dtype=np.float32).copy()
                base_h = float(rt0.control_dict["height_cmd"])
                base_loco = np.asarray(rt0.control_dict["loco_cmd"], dtype=np.float32).copy()
            if demo == "wiggle":
                wiggle_sched = build_wiggle_schedule(int(args.max_steps), args)
                if wiggle_sched.is_legacy:
                    print("Headless scripted demo: wiggle (legacy fixed sine; pipeline smoke).")
                else:
                    print(
                        "Headless scripted demo: wiggle (episode schedule + optional precomputed noise)."
                    )
            else:
                print(f"Headless scripted demo: {demo!r}")
        try:
            for i in range(args.max_steps):
                if demo != "none" and base_arm is not None and base_loco is not None:
                    _apply_headless_scripted_demo(
                        unwrapped._rt, i, demo, base_arm, float(base_h), base_loco, wiggle_sched
                    )
                env.step(np.zeros(1, dtype=np.float32))
                if gr00t_log is not None:
                    gr00t_log.record_step()
        finally:
            _finalize_gr00t_p2_logger(gr00t_log, gr00t_obs_b, args)
        z = float(unwrapped.data.qpos[2])
        print(f"Headless done ({args.max_steps} steps). pelvis z={z:.3f} m")
        env.close()
        return

    from mujoco import viewer as mj_viewer

    if args.openvr_teleop:
        print(
            "OpenVR: SteamVR + headset (e.g. Quest Link). Controllers drive arms; "
            "triggers grip when --hands. Combine with --teleop for keyboard legs / z reset."
        )
    if args.udp_teleop:
        print(
            f"UDP teleop on {args.udp_teleop!r} — run python -m legacy_vr_code.openvr_teleop_client on the VR PC "
            "(see legacy_vr_code/openvr_teleop_client.py)."
        )
    if args.vr_teleop is not None:
        print(
            f"VR teleop on {args.vr_teleop!r} — VR PC: python -m vr_teleop.openvr_udp_sender --target <sim-ip>:PORT "
            "(same port as bind). Sticks → loco_cmd; combine --teleop for keyboard height/rpy/z."
        )
        if args.vr_ik != "off":
            print(
                f"VR IK mode: {args.vr_ik}; anchor: {args.vr_ik_anchor} "
                f"(torso-frame ee_* from stream; head orient={args.vr_head_orient}; "
                f"z-offset={args.vr_ik_z_offset_m:.4f} m; "
                "use --debug-vr-stream to verify frames)."
            )
    if args.teleop:
        if unwrapped._rt.n_arm > 0:
            gh = ""
            if unwrapped._rt.has_hands:
                gh = "Hands: 9/0 left; =/- right; h hard-close both; y open both.\n"
            kp_eff = np.asarray(unwrapped._rt.arm_pd_kp_per_joint, dtype=np.float32)
            kd_eff = np.asarray(unwrapped._rt.arm_pd_kd_per_joint, dtype=np.float32)
            print(
                "Teleop: **hold** w/s forward-back, a/d strafe, q/e yaw (release all loco keys to stand); "
                "z full reset. 1/2 height; 3–8 rpy; m/n freq.\n"
                "Arms (torso-frame IK): left i/k j/l u/p; right r/t f/g v/b; "
                ", . ; ' [ ] orient.\n"
                f"{gh}"
                f"Arm PD (per-arm joint, [pitch,roll,yaw,elbow,wR,wP,wY]): "
                f"kp={np.round(kp_eff, 1).tolist()} kd={np.round(kd_eff, 2).tolist()} "
                f"clip={unwrapped._rt.arm_tau_clip_nm:.1f} Nm "
                f"(profile={args.arm_gain_profile})\n"
                f"Focus **terminal**. Close viewer to quit. ({env_id})"
            )
            if args.keyboard_grasp_primitives:
                print(
                    "Grasp primitives: **x** widen / **c** narrow (torso Y); "
                    "**Home** / **End** forward/back both (torso X); "
                    "**Page Up** / **Page Down** raise/lower both (torso Z). Quats & grippers unchanged."
                )
            if args.print_grip_force:
                print(
                    "Grip force: periodic status includes |Fn| peaks on target_block (L/R/other/total). "
                    "Use with --print-arm-tau to correlate squeeze vs wrist torque."
                )
            if args.lock_ee_orient:
                print(
                    "Lock EE orient: palm quats reset to spawn each step (torso frame); "
                    "orientation keys have no lasting effect unless VR IK is on."
                )
        else:
            print(
                "Teleop: **hold** w/s forward-back, a/d strafe, q/e yaw; z reset; 1–8 rpy/height; m/n freq. "
                "Focus terminal. Close viewer to quit."
            )
    else:
        print(f"Gymnasium {env_id} — close viewer to quit.")

    env.reset()
    gr00t_log, gr00t_obs_b = _make_gr00t_p2_logger(unwrapped, args)
    max_steps = args.max_viewer_steps if args.max_viewer_steps is not None else None
    debug_overlay = unwrapped.debug_ee_targets or args.debug_vr_stream
    if unwrapped.debug_ee_targets:
        print(
            "Debug: green / orange spheres + large arrows = world IK palm targets + target orientation "
            "(torso-frame ee_* → world)."
        )
    if args.debug_vr_stream:
        print(
            "Debug VR stream (torso frame): arrows follow torso_link; "
            f"Z mode={args.debug_vr_stream_torso_z} "
            f"(fixed_world uses Z={args.debug_vr_stream_z_offset:.2f} m). "
            f"receiver yaw={args.vr_receiver_yaw_deg:.1f} deg. "
            f"calib orient={args.debug_vr_stream_calib_orient}. "
            "Recalibrate: restart play or gym reset()."
        )
    if unwrapped.debug_ee_targets:
        from teleop_target_viz import fill_ee_target_geoms
    if args.debug_vr_stream:
        from vr_teleop.vr_stream_target_viz import fill_vr_stream_arrow_geoms

    try:
        with mj_viewer.launch_passive(unwrapped.model, unwrapped.data) as viewer:
            if args.viewer_camera == "head_pov":
                from viewer_camera import viewer_use_fixed_camera

                viewer_use_fixed_camera(unwrapped.model, viewer.cam, "head_pov")
            n = 0
            while viewer.is_running():
                env.step(np.zeros(1, dtype=np.float32))
                if gr00t_log is not None:
                    gr00t_log.record_step()
                if debug_overlay:
                    with viewer.lock():
                        geoms = viewer.user_scn.geoms
                        i = 0
                        if unwrapped.debug_ee_targets:
                            rt_dbg = unwrapped._rt
                            ti_dbg = int(rt_dbg.torso_index)
                            xm_dbg = np.asarray(
                                unwrapped.data.xmat[ti_dbg], dtype=np.float64
                            ).reshape(3, 3)
                            xq_dbg = np.asarray(
                                unwrapped.data.xquat[ti_dbg], dtype=np.float64
                            ).reshape(4)
                            i = fill_ee_target_geoms(
                                geoms,
                                i,
                                unwrapped.model,
                                unwrapped.data,
                                rt_dbg.control_dict,
                                rt_dbg.torso_index,
                                enabled=True,
                                n_arm=rt_dbg.n_arm,
                                ref_quat_wxyz=rt_dbg._ee_decode_quat(xm_dbg, xq_dbg),
                                ref_rotmat=rt_dbg._ee_ref_rotmat(xm_dbg, xq_dbg),
                            )
                        if args.debug_vr_stream:
                            recv = unwrapped._rt._vr_teleop_receiver
                            st_vr, _t = (
                                recv.snapshot()
                                if recv is not None
                                else (None, 0.0)
                            )
                            ti = unwrapped._rt.torso_index
                            d = unwrapped.data
                            rt0 = unwrapped._rt
                            pi = int(rt0.pelvis_index)
                            if pi >= 0:
                                ppos = np.asarray(d.xpos[pi], dtype=np.float64).reshape(3)
                                pxm = np.asarray(d.xmat[pi], dtype=np.float64).reshape(3, 3)
                            else:
                                ppos, pxm = None, None
                            i = fill_vr_stream_arrow_geoms(
                                geoms,
                                i,
                                st_vr,
                                enabled=True,
                                z_offset_m=args.debug_vr_stream_z_offset,
                                torso_xpos=d.xpos[ti],
                                torso_xmat=d.xmat[ti].reshape(3, 3),
                                calib=rt0._vr_stream_calib,
                                torso_z_mode=args.debug_vr_stream_torso_z,
                                torso_yaw_offset_deg=rt0.vr_receiver_yaw_deg,
                                calib_orient_mode=args.debug_vr_stream_calib_orient,
                                vr_ik_anchor_mode=str(getattr(rt0, "vr_ik_anchor_mode", "legacy")),
                                pelvis_xpos=ppos,
                                pelvis_xmat=pxm,
                                mount_T=getattr(rt0, "_vr_ik_mount_mat4", None),
                                head_orient_mode=str(getattr(rt0, "vr_head_orient_mode", "yaw_only")),
                            )
                        viewer.user_scn.ngeom = i
                viewer.sync()
                n += 1
                if max_steps is not None and n >= max_steps:
                    break
    finally:
        _finalize_gr00t_p2_logger(gr00t_log, gr00t_obs_b, args)
    env.close()


if __name__ == "__main__":
    main()
