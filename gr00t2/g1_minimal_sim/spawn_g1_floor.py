#!/usr/bin/env python3
"""Minimal MuJoCo scene: Unitree G1 on a flat floor.

Uses the same `g1_gear_wbc.xml` asset bundle as GR00T-WholeBodyControl (robot + ground plane).

**Passive physics** (robot collapses — no controller):

    python spawn_g1_floor.py
    python spawn_g1_floor.py --headless

**Stand in place** — **Gear WBC ONNX** (same leg controller as GR00T G1 sim), not the VLA.
Needs ONNX files under ``Isaac-GR00T/.../resources/robots/g1/policy/``: either ``ft92.onnx`` /
``ft109.onnx`` **or** ``GR00T-WholeBodyControl-Balance.onnx`` / ``GR00T-WholeBodyControl-Walk.onnx``
(after ``git lfs pull`` or the setup script below).

    python spawn_g1_floor.py --stand
    python spawn_g1_floor.py --stand --headless --max-steps 20000

**Arms:** without teleop, PD holds ``arm_target_q`` (MJCF default pose). With
``--stand --teleop`` or ``--stand --openvr-teleop``, arms use **IK** each step toward palm
**position + quaternion** (wxyz) stored in **torso_link** frame and converted to world for IK;
see ``arm_ik.py`` / ``openvr_teleop.py`` and the teleop banner.

**Teleop** (viewer + keyboard). Legs: same idea as ``run_mujoco_gear_wbc.py``. Needs
``pip install pynput``. Focus the **terminal** while the viewer runs.

    python spawn_g1_floor.py --stand --teleop

**OpenVR** (SteamVR + Quest Link, viewer only): ``pip install openvr`` then
``python spawn_g1_floor.py --stand --openvr-teleop`` (combine with ``--teleop`` for keyboard
legs / ``z`` reset; ``--hands`` for analog triggers on grippers).

**Split-machine VR (UDP v1):** on the sim PC, ``--stand --udp-teleop 0.0.0.0:5005`` (optional
``--teleop`` for keyboard, ``--hands`` for grippers). On the VR PC, use
``legacy_vr_code/openvr_teleop_client.py`` (or copy the whole ``legacy_vr_code/`` package),
``pip install openvr numpy``, then from ``g1_minimal_sim``::

    python -m legacy_vr_code.openvr_teleop_client --host <sim_ip> --port 5005

**Split-machine VR (``vr_teleop`` stream):** ``--stand --vr-teleop`` (default bind ``0.0.0.0:5006``;
headless OK). VR PC: ``python -m vr_teleop.openvr_udp_sender --target <sim_ip>:5006``. Sticks drive
``loco_cmd`` (not the same protocol as ``--udp-teleop``).

**Python:** from this directory (``g1_minimal_sim``), use the **full** path below — do not type
literal ``...`` in the shell. If the venv is missing, from **Isaac-GR00T** repo root run
``bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh``, or use any env with
``pip install mujoco onnxruntime pyyaml``:

    ../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --stand
    ../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --stand --teleop

**Gymnasium (same Gear WBC stack):** ``play_g1_gear_wbc.py`` (``--headless`` / ``--teleop``); env id ``G1GearWBC-v0`` defaults to **stylish diner**; ``G1GearWBC-TablePnP-v0`` is table + apple + plate. Use ``--scene floor`` for the empty checker floor. Needs ``pip install gymnasium``.

**Scenes (``--stand``):** default ``--scene stylish_diner`` (merged MJCF under ``scenes/stylish_diner_1/``); ``--scene table_pnp`` loads vendored ``g1_gear_wbc_table.yaml`` (blue ``table_box``); ``--scene table_pnp_apple`` loads ``scenes/table_pnp_apple/`` (B1 apple+plate). Override yaml with ``--config-yaml path/to/foo.yaml`` (absolute path sets resources dir to that file's parent).

Optional PNG (passive mode only; needs ``MUJOCO_GL=egl`` or osmesa):

    MUJOCO_GL=egl python spawn_g1_floor.py --headless --save-frame /tmp/g1.png

Install MuJoCo in your own env: ``pip install mujoco`` (not ``conda install mujoco`` on defaults).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vr_stream_ik_mode import normalize_vr_ik_anchor_mode, normalize_vr_ik_mode
from task_scene import available_tasks, resolve_task_scene


def default_g1_xml_path() -> Path:
    """Resolve path to vendored G1 MJCF next to this repo layout."""
    here = Path(__file__).resolve().parent
    isaac_groot = here.parent / "Isaac-GR00T"
    return (
        isaac_groot
        / "external_dependencies"
        / "GR00T-WholeBodyControl"
        / "gr00t_wbc"
        / "sim2mujoco"
        / "resources"
        / "robots"
        / "g1"
        / "g1_gear_wbc.xml"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Override path to G1 MJCF (default: vendored g1_gear_wbc.xml)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No GUI: run physics steps only (for machines without DISPLAY).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="With --headless (incl. --stand --headless), number of mj_step iterations (default: 5000).",
    )
    parser.add_argument(
        "--save-frame",
        type=Path,
        default=None,
        help="With --headless (no --stand), render one RGB frame (needs MUJOCO_GL=egl or osmesa).",
    )
    parser.add_argument(
        "--stand",
        action="store_true",
        help="Run Gear WBC ONNX (stand/walk) so the robot balances; needs policy/*.onnx by the yaml.",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="With --stand (viewer only): keyboard teleop via pynput (pip install pynput).",
    )
    parser.add_argument(
        "--keyboard-grasp-primitives",
        action="store_true",
        help="With --stand --teleop: optional symmetric palm EE nudges "
        "(x/c, Home/End, Page Up/Down). Default off.",
    )
    parser.add_argument(
        "--lock-ee-orient",
        action="store_true",
        help="With --stand --teleop: each step reset palm quats to init (torso frame). No effect with --vr-ik on.",
    )
    parser.add_argument(
        "--openvr-teleop",
        action="store_true",
        help="With --stand (viewer only): OpenVR controllers → IK + triggers (pip install openvr; SteamVR).",
    )
    parser.add_argument(
        "--udp-teleop",
        type=str,
        default=None,
        metavar="HOST:PORT",
        help="With --stand: listen for UDP JSON from legacy_vr_code.openvr_teleop_client (e.g. 0.0.0.0:5005). "
        "Works with --headless. Combine with --teleop for keyboard. Not with --vr-teleop.",
    )
    parser.add_argument(
        "--vr-teleop",
        nargs="?",
        const="0.0.0.0:5006",
        default=None,
        metavar="HOST:PORT",
        help="With --stand: listen for vr_teleop.openvr_udp_sender (default 0.0.0.0:5006). "
        "Sticks → loco_cmd. Works with --headless. Not with --udp-teleop.",
    )
    parser.add_argument(
        "--vr-receiver-yaw-deg",
        type=float,
        default=-90.0,
        metavar="DEG",
        help="Receiver-side yaw correction (degrees) for VR torso compose "
        "(sim PC only, no sender changes).",
    )
    parser.add_argument(
        "--vr-ik",
        type=normalize_vr_ik_mode,
        default="off",
        metavar="MODE",
        help="With --stand --vr-teleop: stream controllers → torso-frame ee_* IK targets. off | on.",
    )
    parser.add_argument(
        "--vr-ik-anchor",
        type=normalize_vr_ik_anchor_mode,
        default="legacy",
        metavar="MODE",
        help="With --stand --vr-teleop and --vr-ik on: legacy | torso (head-relative) | pelvis.",
    )
    parser.add_argument(
        "--vr-head-orient",
        choices=("full", "yaw_only"),
        default="yaw_only",
        help="With --stand --vr-teleop and non-legacy anchors: orientation used for current HMD "
        "in inv(T_hmd) compose. yaw_only keeps world Z shared.",
    )
    parser.add_argument(
        "--vr-ik-z-offset-m",
        type=float,
        default=0.0,
        metavar="M",
        help="With --stand --vr-teleop and --vr-ik on: add world +Z offset (m) to composed VR IK "
        "palm targets before conversion to torso-frame ee_*.",
    )
    parser.add_argument(
        "--debug-vr-stream",
        action="store_true",
        help="With --stand (viewer only): draw streamed HMD + controllers as arrows. Requires --vr-teleop.",
    )
    parser.add_argument(
        "--debug-vr-stream-torso-z",
        choices=("full", "fixed_world", "frozen_calib"),
        default="full",
        help="With --debug-vr-stream: torso Z policy (see vr_stream_target_viz).",
    )
    parser.add_argument(
        "--debug-vr-stream-z-offset",
        type=float,
        default=None,
        metavar="M",
        help="With --debug-vr-stream: Z offset / fixed_world Z (m). Default: VR_STREAM_VIZ_Z_OFFSET_M.",
    )
    parser.add_argument(
        "--debug-vr-stream-calib-orient",
        choices=("full", "yaw_only"),
        default="yaw_only",
        help="With --debug-vr-stream: HMD0 calibration orientation mode.",
    )
    parser.add_argument(
        "--debug-ee-targets",
        action="store_true",
        help="With --stand (viewer only): draw world-space IK palm targets (green=left, orange=right).",
    )
    parser.add_argument(
        "--hands",
        action="store_true",
        help="With --stand: load g1_gear_wbc_hands.xml "
        "(grip: 9/0 left, =/- right, h hard-close both, y open both).",
    )
    parser.add_argument(
        "--scene",
        choices=("stylish_diner", "floor", "table_pnp", "table_pnp_apple"),
        default="stylish_diner",
        help="With --stand: MJCF preset — stylish diner (default), floor, table_pnp, or table_pnp_apple.",
    )
    parser.add_argument(
        "--task",
        choices=available_tasks(),
        default=None,
        help="With --stand: task preset from scenes/<task>/task_spec.yaml. Overrides --scene and --config-yaml.",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="With --stand: Gear WBC yaml (basename relative to MJCF dir, or absolute path). Overrides --scene.",
    )
    parser.add_argument(
        "--viewer-camera",
        choices=("default", "head_pov"),
        default="default",
        help="With --stand (viewer only): default orbit camera; head_pov uses MJCF camera ``head_pov``.",
    )
    parser.add_argument(
        "--arm-gain-profile",
        choices=("default", "grasp"),
        default="default",
        help="Arm PD profile (with --stand). 'default' preserves historical kp=45/kd=1.2/clip=30; "
        "'grasp' stiffens elbow + wrists for friction-grasp contact load. See play_g1_gear_wbc.py for details.",
    )
    parser.add_argument("--arm-kp-shoulder", type=float, default=None)
    parser.add_argument("--arm-kd-shoulder", type=float, default=None)
    parser.add_argument("--arm-kp-elbow", type=float, default=None)
    parser.add_argument("--arm-kd-elbow", type=float, default=None)
    parser.add_argument("--arm-kp-wrist-roll", type=float, default=None)
    parser.add_argument("--arm-kd-wrist-roll", type=float, default=None)
    parser.add_argument("--arm-kp-wrist-pitch", type=float, default=None)
    parser.add_argument("--arm-kd-wrist-pitch", type=float, default=None)
    parser.add_argument("--arm-kp-wrist-yaw", type=float, default=None)
    parser.add_argument("--arm-kd-wrist-yaw", type=float, default=None)
    parser.add_argument(
        "--arm-tau-clip-nm",
        type=float,
        default=None,
        help="Per-actuator arm torque clip (Nm). Default 30; grasp profile 40.",
    )
    parser.add_argument(
        "--diner-block-density",
        type=float,
        default=None,
        metavar="KG_PER_M3",
        help="Override target_block density at load time (no XML edit).",
    )
    parser.add_argument(
        "--diner-block-friction",
        type=str,
        default=None,
        metavar="\"TAN SLIP SPIN\"",
        help="Override target_block friction (3 floats, space-separated).",
    )
    parser.add_argument(
        "--print-arm-tau",
        action="store_true",
        help="With --stand --teleop: append arm-tau peak summary to status print.",
    )
    parser.add_argument(
        "--grip-key-step",
        type=float,
        default=None,
        metavar="STEP",
        help="With --stand --hands: per-tap grip increment for 9/0 and =/-. Default 0.018; must be in (0, 1].",
    )
    parser.add_argument(
        "--print-grip-force",
        action="store_true",
        help="With --stand --teleop --hands: append target_block |Fn| peak summary to status print.",
    )
    args = parser.parse_args()

    if args.debug_vr_stream and args.headless:
        parser.error("--debug-vr-stream cannot be used with --headless")
    if args.debug_vr_stream and args.vr_teleop is None:
        parser.error("--debug-vr-stream requires --vr-teleop")
    if args.keyboard_grasp_primitives and not args.teleop:
        parser.error("--keyboard-grasp-primitives requires --teleop")
    if args.lock_ee_orient and not args.teleop:
        parser.error("--lock-ee-orient requires --teleop")
    if args.teleop and not args.stand:
        parser.error("--teleop requires --stand")
    if args.openvr_teleop and not args.stand:
        parser.error("--openvr-teleop requires --stand")
    if args.udp_teleop and not args.stand:
        parser.error("--udp-teleop requires --stand")
    if args.vr_teleop is not None and not args.stand:
        parser.error("--vr-teleop requires --stand")
    if args.vr_teleop is not None and args.udp_teleop is not None:
        parser.error("--vr-teleop and --udp-teleop are mutually exclusive")
    if args.vr_ik != "off" and args.vr_teleop is None:
        parser.error("--vr-ik requires --vr-teleop")
    if args.vr_ik_anchor != "legacy" and args.vr_teleop is None:
        parser.error("--vr-ik-anchor requires --vr-teleop")
    if args.vr_ik_anchor != "legacy" and args.vr_ik == "off":
        parser.error("--vr-ik-anchor is only used with --vr-ik on")
    if args.teleop and args.headless:
        parser.error("--teleop cannot be used with --headless")
    if args.openvr_teleop and args.headless:
        parser.error("--openvr-teleop cannot be used with --headless")
    if args.debug_ee_targets and args.headless:
        parser.error("--debug-ee-targets requires a viewer (no --headless)")
    if args.task is not None and not args.stand:
        parser.error("--task requires --stand")
    if args.headless and args.viewer_camera != "default":
        parser.error("--viewer-camera requires a viewer (no --headless)")
    if args.grip_key_step is not None:
        if not (0.0 < float(args.grip_key_step) <= 1.0):
            parser.error("--grip-key-step must be in (0, 1]")
        if not args.hands:
            parser.error("--grip-key-step requires --hands")
    if args.print_grip_force and not args.hands:
        parser.error("--print-grip-force requires --hands")
    if args.print_grip_force and not args.teleop:
        parser.error("--print-grip-force requires --teleop")

    if args.debug_vr_stream_z_offset is None:
        from vr_teleop.vr_stream_target_viz import VR_STREAM_VIZ_Z_OFFSET_M

        z_off = float(VR_STREAM_VIZ_Z_OFFSET_M)
    else:
        z_off = float(args.debug_vr_stream_z_offset)

    xml_path = (args.xml or default_g1_xml_path()).resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(
            f"G1 MJCF not found: {xml_path}\n"
            "Expected Isaac-GR00T next to g1_minimal_sim/, or pass --xml explicitly."
        )

    if args.stand:
        import numpy as np  # local: no numpy import at module top; needed only for --stand path

        from gear_wbc_pd import (
            ARM_PD_KD_DEFAULT_PER_JOINT,
            ARM_PD_KD_GRASP_PER_JOINT,
            ARM_PD_KP_DEFAULT_PER_JOINT,
            ARM_PD_KP_GRASP_PER_JOINT,
            ARM_TAU_CLIP_GRASP_NM,
        )
        from gear_wbc_stand import resolve_gear_wbc_config, run_gear_wbc

        scene = str(args.scene)
        config_yaml = args.config_yaml
        if args.task is not None:
            scene, cfg_from_task, task_meta = resolve_task_scene(args.task)
            config_yaml = Path(cfg_from_task) if cfg_from_task is not None else None
            print(f"Task preset: {args.task} -> scene={scene}")
            print(f"Task name: {task_meta.get('task_name', args.task)}")

        # Resolve --arm-gain-profile + per-joint overrides.
        if args.arm_gain_profile == "grasp":
            kp_pj: np.ndarray | None = ARM_PD_KP_GRASP_PER_JOINT.copy()
            kd_pj: np.ndarray | None = ARM_PD_KD_GRASP_PER_JOINT.copy()
            tau_clip: float | None = float(ARM_TAU_CLIP_GRASP_NM)
        else:
            kp_pj = None
            kd_pj = None
            tau_clip = None
        for name, idx in (
            ("arm_kp_shoulder", (0, 1, 2)),
            ("arm_kp_elbow", (3,)),
            ("arm_kp_wrist_roll", (4,)),
            ("arm_kp_wrist_pitch", (5,)),
            ("arm_kp_wrist_yaw", (6,)),
        ):
            v = getattr(args, name)
            if v is not None:
                if kp_pj is None:
                    kp_pj = ARM_PD_KP_DEFAULT_PER_JOINT.copy()
                for i in idx:
                    kp_pj[i] = float(v)
        for name, idx in (
            ("arm_kd_shoulder", (0, 1, 2)),
            ("arm_kd_elbow", (3,)),
            ("arm_kd_wrist_roll", (4,)),
            ("arm_kd_wrist_pitch", (5,)),
            ("arm_kd_wrist_yaw", (6,)),
        ):
            v = getattr(args, name)
            if v is not None:
                if kd_pj is None:
                    kd_pj = ARM_PD_KD_DEFAULT_PER_JOINT.copy()
                for i in idx:
                    kd_pj[i] = float(v)
        if args.arm_tau_clip_nm is not None:
            tau_clip = float(args.arm_tau_clip_nm)

        block_friction: tuple[float, float, float] | None = None
        if args.diner_block_friction is not None:
            parts = args.diner_block_friction.replace(",", " ").split()
            if len(parts) != 3:
                parser.error("--diner-block-friction expects 'TAN SLIP SPIN' (3 floats)")
            try:
                block_friction = (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError as e:
                parser.error(f"--diner-block-friction parse error: {e}")

        resources_dir, cy = resolve_gear_wbc_config(
            xml_path.parent,
            use_hands=args.hands,
            scene=scene,
            config_yaml=config_yaml,
        )
        run_gear_wbc(
            resources_dir,
            headless=args.headless,
            max_steps=args.max_steps if args.headless else None,
            teleop=args.teleop,
            openvr_teleop=args.openvr_teleop,
            udp_teleop_bind=args.udp_teleop,
            vr_teleop_bind=args.vr_teleop,
            vr_receiver_yaw_deg=args.vr_receiver_yaw_deg,
            vr_ik_mode=args.vr_ik,
            vr_debug_stream=args.debug_vr_stream,
            vr_debug_torso_z_mode=args.debug_vr_stream_torso_z,
            vr_debug_z_offset_m=z_off,
            vr_stream_calib_orient=args.debug_vr_stream_calib_orient,
            vr_ik_anchor_mode=args.vr_ik_anchor,
            vr_head_orient_mode=args.vr_head_orient,
            vr_ik_z_offset_m=args.vr_ik_z_offset_m,
            config_yaml=cy,
            debug_ee_targets=args.debug_ee_targets,
            viewer_camera=args.viewer_camera,
            keyboard_grasp_primitives=args.keyboard_grasp_primitives,
            lock_ee_orient=args.lock_ee_orient,
            arm_pd_kp_per_joint=kp_pj,
            arm_pd_kd_per_joint=kd_pj,
            arm_tau_clip_nm=tau_clip,
            block_density=args.diner_block_density,
            block_friction=block_friction,
            print_arm_tau=bool(args.print_arm_tau),
            grip_key_step=args.grip_key_step,
            print_grip_force=bool(args.print_grip_force),
        )
        return

    if args.save_frame is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print(f"Loaded {xml_path.name}  (nq={model.nq}, nu={model.nu})")

    if args.headless:
        for _ in range(args.max_steps):
            mujoco.mj_step(model, data)
        if args.save_frame is not None:
            try:
                from PIL import Image
            except ImportError as e:
                raise ImportError(
                    "Saving frames needs Pillow: pip install pillow"
                ) from e
            renderer = mujoco.Renderer(model, height=480, width=640)
            renderer.update_scene(data)
            pixels = renderer.render()
            Image.fromarray(pixels).save(args.save_frame)
            print(f"Wrote frame to {args.save_frame}")
        print(f"Headless run finished ({args.max_steps} steps).")
        return

    import mujoco.viewer

    print("MuJoCo passive viewer — close window to quit.")
    if not os.environ.get("DISPLAY"):
        print(
            "\nNote: DISPLAY is unset; GLFW will fail on this machine.\n"
            "Use:  python spawn_g1_floor.py --headless\n"
            "Or set DISPLAY (e.g. local desktop :0, or ssh -X).\n"
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
