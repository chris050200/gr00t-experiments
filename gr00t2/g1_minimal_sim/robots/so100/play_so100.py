#!/usr/bin/env python3
"""Play / preview SO-100 on a checkerboard floor (MuJoCo passive viewer).

Loads ``so100_plane_world.xml`` next to this script. The arm base is moved to the origin in
xy and lifted in z so the lowest non-floor geometry sits slightly above z=0.

Use ``--teleop`` for a Tk panel: **world-frame** desired pose for ``ee_site`` (position + wxyz
quaternion) plus **Jaw** gripper. Damped IK chases the target (may not fully reach). A **red
arrow** in the viewer shows the commanded EE pose.

Use ``--vr-teleop [HOST:PORT]`` (same UDP protocol as ``vr_teleop/openvr_udp_sender.py`` / G1
``--vr-teleop``) to drive the EE target from the **right controller** ``pos_mj`` / ``quat_mj``.
When packets are fresh, VR **overwrites** the world target each frame (Tk nudges apply only
when VR is stale or alongside if you use both). Optional: right **trigger** → Jaw when
``--vr-grip-trigger`` (default on).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_G1_MINIMAL_SIM = _HERE.parent.parent
if str(_G1_MINIMAL_SIM) not in sys.path:
    sys.path.insert(0, str(_G1_MINIMAL_SIM))

from ik_site import fk_site_pose_world, ik_step_site, mat_from_quat_wxyz, quat_wxyz_normalize  # noqa: E402
from ik_target_viz import fill_ik_target_arrow  # noqa: E402
from teleop_gui import TeleopState, run_teleop_window  # noqa: E402
from vr_teleop.openvr_stream import start_openvr_stream_receiver  # noqa: E402

_DEFAULT_XML = _HERE / "so100_plane_world.xml"

_EE_SITE = "ee_site"
_JOINT_JAW = "Jaw"
_ARM_NV = 5


def _align_base_above_floor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    body_name: str = "Base",
    floor_geom_name: str = "floor",
    margin: float = 0.003,
) -> None:
    """Shift ``body_name`` along z so non-floor geoms clear the z=0 plane by ``margin``."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"body not found: {body_name!r}")
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
    if floor_gid < 0:
        raise ValueError(f"geom not found: {floor_geom_name!r}")

    model.body_pos[bid][0] = 0.0
    model.body_pos[bid][1] = 0.0
    model.body_pos[bid][2] = 0.0
    mujoco.mj_forward(model, data)

    z_min = float("inf")
    for gid in range(model.ngeom):
        if gid == floor_gid:
            continue
        z = float(data.geom_xpos[gid, 2] - model.geom_rbound[gid])
        z_min = min(z_min, z)

    if z_min < float("inf"):
        dz = margin - z_min
        model.body_pos[bid, 2] += dz
    mujoco.mj_forward(model, data)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--xml",
        type=Path,
        default=_DEFAULT_XML,
        help="MJCF path (default: so100_plane_world.xml beside this script)",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.003,
        help="Clearance above z=0 for lowest non-floor geom (meters).",
    )
    p.add_argument(
        "--teleop",
        action="store_true",
        help="Open Tk teleop (world-frame EE target + gripper + red arrow).",
    )
    p.add_argument(
        "--no-debug-arrow",
        action="store_true",
        help="With --teleop / --vr-teleop: hide the red IK target arrow.",
    )
    p.add_argument(
        "--vr-teleop",
        nargs="?",
        const="0.0.0.0:5006",
        default=None,
        metavar="HOST:PORT",
        help="UDP bind for vr_teleop stream (default 0.0.0.0:5006). Right controller → EE target.",
    )
    p.add_argument(
        "--vr-stale-s",
        type=float,
        default=0.5,
        metavar="S",
        help="Ignore VR packets older than this many seconds (wall clock since recv).",
    )
    p.add_argument(
        "--vr-ox",
        type=float,
        default=0.0,
        help="Added to streamed right-controller position X (m, world).",
    )
    p.add_argument(
        "--vr-oy",
        type=float,
        default=0.0,
        help="Added to streamed right-controller position Y (m, world).",
    )
    p.add_argument(
        "--vr-oz",
        type=float,
        default=0.0,
        help="Added to streamed right-controller position Z (m, world).",
    )
    p.add_argument(
        "--vr-grip-trigger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map right trigger to Jaw (open at 0, closed at 1). Use --no-vr-grip-trigger to keep Tk-only grip.",
    )
    p.add_argument(
        "--trans-step",
        type=float,
        default=0.01,
        help="Translation button step in **world** axes (meters).",
    )
    p.add_argument(
        "--rot-step",
        type=float,
        default=0.06,
        help="Roll/pitch/yaw button step = rotation about world X/Y/Z (radians).",
    )
    p.add_argument(
        "--ik-iters",
        type=int,
        default=8,
        help="Inner IK substeps per viewer frame when teleop / VR is active.",
    )
    p.add_argument(
        "--hold-repeat-ms",
        type=int,
        default=70,
        metavar="MS",
        help="With --teleop: repeat interval while holding translate/rotate/grip buttons (ms).",
    )
    args = p.parse_args()
    xml_path = args.xml.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    _align_base_above_floor(model, data, margin=args.margin)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = data.qpos[: model.nu]
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _EE_SITE)
    jaw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _JOINT_JAW)
    if site_id < 0 or jaw_jid < 0:
        raise RuntimeError("expected ee_site and Jaw joint in model")

    jaw_lo, jaw_hi = (float(model.jnt_range[jaw_jid, 0]), float(model.jnt_range[jaw_jid, 1]))

    teleop_state: TeleopState | None = None
    vr_recv = None
    use_teleop = bool(args.teleop) or (args.vr_teleop is not None)
    show_arrow = False
    vr_off = np.asarray([args.vr_ox, args.vr_oy, args.vr_oz], dtype=np.float64)

    if use_teleop:
        teleop_state = TeleopState()
        p0, q0 = fk_site_pose_world(model, data, site_id)
        jaw_adr = int(model.jnt_qposadr[jaw_jid])
        with teleop_state.lock:
            teleop_state.p_world[:] = p0
            teleop_state.q_world[:] = q0
            teleop_state.gripper = float(data.qpos[jaw_adr])
        if args.teleop:
            gui_thread = threading.Thread(
                target=lambda: run_teleop_window(
                    teleop_state,
                    trans_step=args.trans_step,
                    rot_step=args.rot_step,
                    gripper_lo=jaw_lo,
                    gripper_hi=jaw_hi,
                    hold_repeat_ms=args.hold_repeat_ms,
                ),
                daemon=True,
            )
            gui_thread.start()
        if args.vr_teleop is not None:
            vr_recv = start_openvr_stream_receiver(str(args.vr_teleop))
        show_arrow = not args.no_debug_arrow

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if teleop_state is not None:
                    if teleop_state.take_fk_sync_request():
                        p_s, q_s = fk_site_pose_world(model, data, site_id)
                        with teleop_state.lock:
                            teleop_state.p_world[:] = p_s
                            teleop_state.q_world[:] = q_s

                    if vr_recv is not None:
                        st, t_rx = vr_recv.snapshot()
                        if (
                            st is not None
                            and st.right.valid
                            and t_rx > 0.0
                            and (time.monotonic() - t_rx) < float(args.vr_stale_s)
                        ):
                            with teleop_state.lock:
                                teleop_state.p_world[:] = st.right.pos + vr_off
                                teleop_state.q_world[:] = quat_wxyz_normalize(st.right.quat)
                                if args.vr_grip_trigger and st.right_input_valid:
                                    t = float(max(0.0, min(1.0, st.right_trigger)))
                                    teleop_state.gripper = jaw_lo + (jaw_hi - jaw_lo) * (1.0 - t)

                    p_w, q_w, g_cmd = teleop_state.snapshot_targets()
                    R_w = mat_from_quat_wxyz(q_w)
                    for _ in range(max(1, args.ik_iters)):
                        ik_step_site(
                            model,
                            data,
                            site_id=site_id,
                            p_des_world=p_w,
                            R_des_world=R_w,
                            arm_nv=_ARM_NV,
                        )
                    data.ctrl[:_ARM_NV] = data.qpos[:_ARM_NV]
                    data.ctrl[_ARM_NV] = float(g_cmd)
                    data.qvel[:] *= 0.98
                    mujoco.mj_step(model, data)

                    with viewer.lock():
                        n = fill_ik_target_arrow(
                            viewer.user_scn.geoms,
                            0,
                            p_w,
                            q_w,
                            enabled=show_arrow,
                        )
                        viewer.user_scn.ngeom = n
                else:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                    data.qvel[:] = 0.0
                    data.ctrl[:] = data.qpos[: model.nu]
                    mujoco.mj_forward(model, data)
                viewer.sync()
    finally:
        if vr_recv is not None:
            vr_recv.stop()


if __name__ == "__main__":
    main()
