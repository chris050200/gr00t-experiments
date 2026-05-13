#!/usr/bin/env python3
"""UDP pose stream ↔ MuJoCo debug viewer (prove SteamVR→MJ mapping on the sim machine).

**Receive (default):** bind UDP, decode JSON **v2** packets, draw world triad + HMD + controllers
like the standalone OpenVR debug script (arrows / sphere), without OpenVR on this host.

**Gamepad (receive, sim machine):** GLFW / VR sticks move a **planar root** (MJ **XY**, **+Z** up): **left Y** forward/back, **left X** yaw, **right X** strafe. The **RGB root triad** sits at the joystick pose. **HMD/controllers** default to **`--ride-joystick-root`**: body offsets are taken from the **first live UDP pose**, then redrawn each frame as ``t_root + Rz(yaw) @ p_body`` so the cluster **moves and yaws with the stick** without the old “orbit world origin” bug. Use **`--no-ride-joystick-root`** for raw **room** tracking (compose still fixes off-center yaw). **`--world-frame`** skips stripping entirely (pure UDP world).

**POV preview (receive):** mocap **``hmd_pov``** at **HMD world** pose + orientation (eye offset in HMD local). OpenCV + **`--pov-rotate`**, etc.

**Send (VR PC):** poll OpenVR, map with ``vr_steamvr_mujoco_mapping``, emit the same **v2** JSON.

JSON **v2** (UTF-8, one datagram per packet)::

    {
      "v": 2,
      "seq": <int>,
      "hmd": {"pos": [x,y,z], "quat": [w,x,y,z]},
      "left": {"pos": [...], "quat": [...]},
      "right": {"pos": [...], "quat": [...]},
      "left_stick": [x, y],
      "right_stick": [x, y]
    }

Omit pose blocks if tracking is invalid. **Sticks:** OpenVR ``rAxis`` (Quest / Touch thumbsticks on VR PC); always sent (zeros if unknown). **Sim** uses them for planar root when fresh; falls back to GLFW gamepad if disabled or stale.

Examples::

    # Sim / dev machine (MuJoCo viewer + listen), from g1_minimal_sim:
    python -m legacy_vr_code.vr_teleop_test_streaming receive --bind 0.0.0.0:5006

    # Same + floating HMD POV (opencv-python)
    python -m legacy_vr_code.vr_teleop_test_streaming receive --bind 0.0.0.0:5006 --pov-preview

    # VR PC (SteamVR + OpenVR)
    python -m legacy_vr_code.vr_teleop_test_streaming send --host <sim-ip> --port 5006 \\
        --tracking-universe seated
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from typing import Any

import numpy as np

import mujoco
import mujoco.viewer

from .openvr_device_indices import resolve_openvr_tracked_device_indices
from .pov_image import pov_rotate_bgr
from .teleop_udp_bind import parse_udp_bind
from .vr_steamvr_mujoco_mapping import openvr_pose_to_mujoco
from .vr_teleop_root_kinematics import (
    Rz,
    apply_axis_deadzone,
    integrate_root_planar,
    world_pose_to_root_planar,
)

PROTOCOL_V2 = 2


def _poll_glfw_gamepad_sticks(joystick_id: int = 0) -> tuple[float, float, float] | None:
    """Return ``(left_x, left_y, right_x)`` (GLFW gamepad axes 0,1,2) or ``None``."""
    try:
        import glfw
    except ImportError:
        return None
    jid = int(joystick_id)
    if not glfw.joystick_present(jid) or not glfw.joystick_is_gamepad(jid):
        return None
    # PyGLFW: ``GamepadState`` out-param; some builds return the state object.
    try:
        st = glfw.GamepadState()
        ok = glfw.get_gamepad_state(jid, st)
        if not ok:
            return None
        ax = st.axes
    except TypeError:
        st = glfw.get_gamepad_state(jid)
        if st is None:
            return None
        ax = st.axes
    if len(ax) < 3:
        return None
    return (float(ax[0]), float(ax[1]), float(ax[2]))


# --- Same layout as the clean OpenVR debug script (visuals only) -----------------


def arrow_rotation_matrix(direction: np.ndarray) -> np.ndarray:
    """3×3 rotation so arrow local +Z aligns with ``direction`` (world)."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.eye(3)
    d = d / n
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, d)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(d, a)
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        return np.eye(3)
    u = u / nu
    v = np.cross(u, d)
    return np.column_stack([u, v, d])


def _R3_from_mjmat(mat9: np.ndarray) -> np.ndarray:
    """3×3 rotation from ``mju_quat2Mat`` output (column-major 9)."""
    return np.asarray(mat9, dtype=np.float64).reshape(3, 3, order="F")


def _mjv_mat_from_R(R: np.ndarray) -> np.ndarray:
    """Column-major length-9 orientation for ``mjv_initGeom`` (matches MuJoCo)."""
    return np.asarray(R, dtype=np.float64).reshape(3, 3).reshape(9, order="F")


def minimal_xml(*, pov_camera: bool = False) -> str:
    pov_block = ""
    if pov_camera:
        # MuJoCo camera looks along local -Z with +Y up in the image. HMD/OpenVR pose
        # maps device frame into MJ; without this offset the viewport is often rolled 90°
        # (e.g. sky on the left). wxyz = +90° about camera +Z (roll around view axis).
        pov_block = """
    <body name="hmd_pov_mocap" mocap="true" pos="0 0 1.2">
      <camera name="hmd_pov" pos="0 0 0" quat="0.7071067811865476 0 0 0.7071067811865476" fovy="80"/>
    </body>"""
    return f"""<mujoco model="vr_udp_pose_debug">
  <option timestep="0.01" gravity="0 0 -0.01"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35"/>
  </visual>
  <worldbody>
    <light diffuse="0.9 0.9 0.9" pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="0 0 -1.83" size="10 10 0.01" rgba="0.22 0.25 0.28 1"/>{pov_block}
  </worldbody>
</mujoco>"""


def _pov_pose_for_mocap(
    hmd_pos: np.ndarray,
    hmd_q: np.ndarray,
    t_root: np.ndarray,
    yaw: float,
    world_frame: bool,
    eye_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """HMD (UDP world) → mocap (pos, quat wxyz); ``eye_offset`` in HMD local frame.

    If ``world_frame`` is False, strip with ``world_pose_to_root_planar`` so the POV
    matches the root-relative debug geoms (same ``t_root`` / ``yaw`` as the viewer).
    """
    Rw_h = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(Rw_h, np.asarray(hmd_q, dtype=np.float64).reshape(4))
    Rwh = _R3_from_mjmat(Rw_h)
    p = np.asarray(hmd_pos, dtype=np.float64).reshape(3)
    off = np.asarray(eye_offset, dtype=np.float64).reshape(3)
    if world_frame:
        p_base, R_base = p, Rwh
    else:
        p_base, R_base = world_pose_to_root_planar(p, Rwh, t_root, yaw)
    p_out = p_base + R_base @ off
    Rf = np.reshape(R_base, 9, order="F")
    q_out = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q_out, Rf)
    return p_out, q_out


def _openvr_thumbsticks(
    system: Any,
    left_idx: int | None,
    right_idx: int | None,
    stick_axis: int,
) -> tuple[float, float, float, float]:
    """Return ``(lx, ly, rx, ry)`` from OpenVR controller ``rAxis[stick_axis]`` (x,y each)."""
    lx = ly = rx = ry = 0.0
    ax_i = int(stick_axis)
    if left_idx is not None:
        try:
            ok, st = system.getControllerState(left_idx)
            if ok and st is not None and ax_i < len(st.rAxis):
                a = st.rAxis[ax_i]
                lx, ly = float(a.x), float(a.y)
        except Exception:
            pass
    if right_idx is not None:
        try:
            ok, st = system.getControllerState(right_idx)
            if ok and st is not None and ax_i < len(st.rAxis):
                a = st.rAxis[ax_i]
                rx, ry = float(a.x), float(a.y)
        except Exception:
            pass
    return lx, ly, rx, ry


def _parse_pose_block(block: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(block, dict):
        return None
    if "pos" not in block or "quat" not in block:
        return None
    p = np.asarray(block["pos"], dtype=np.float64).reshape(3)
    q = np.asarray(block["quat"], dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-9:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q = q / n
    return p, q


def _latest_udp_json(sock: socket.socket, buf: bytearray) -> dict[str, Any] | None:
    """Drain queue; return last valid JSON object or None."""
    last: dict[str, Any] | None = None
    while True:
        try:
            n, _ = sock.recvfrom_into(buf)
        except BlockingIOError:
            break
        if n <= 0:
            break
        try:
            obj = json.loads(buf[:n].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict):
            last = obj
    return last


def cmd_receive(args: argparse.Namespace) -> None:
    host, port = parse_udp_bind(args.bind)
    vr_off = np.asarray(args.vr_offset, dtype=np.float64).reshape(3)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.setblocking(False)
    buf = bytearray(65536)

    lock = threading.Lock()
    state: dict[str, Any] = {
        "hmd_p": np.array([0.0, 0.0, 1.2]) + vr_off,
        "hmd_q": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "left_p": np.array([0.4, 0.4, 1.2]) + vr_off,
        "right_p": np.array([-0.4, 0.4, 1.2]) + vr_off,
        "left_mat": np.eye(3, dtype=np.float64).flatten(),
        "right_mat": np.eye(3, dtype=np.float64).flatten(),
        "vr_sticks": None,  # (lx, ly, rx) or None
        "vr_sticks_time": 0.0,
        "got_udp_hmd": False,
    }

    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            pkt = _latest_udp_json(sock, buf)
            if pkt is None:
                time.sleep(0.002)
                continue
            if int(pkt.get("v", -1)) != PROTOCOL_V2:
                continue
            with lock:
                hb = _parse_pose_block(pkt.get("hmd"))
                if hb is not None:
                    state["hmd_p"] = hb[0] + vr_off
                    state["hmd_q"] = hb[1].copy()
                    state["got_udp_hmd"] = True
                lb = _parse_pose_block(pkt.get("left"))
                if lb is not None:
                    state["left_p"] = lb[0] + vr_off
                    lm = np.zeros(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(lm, lb[1])
                    state["left_mat"] = lm
                rb = _parse_pose_block(pkt.get("right"))
                if rb is not None:
                    state["right_p"] = rb[0] + vr_off
                    rm = np.zeros(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(rm, rb[1])
                    state["right_mat"] = rm
                if "left_stick" in pkt and "right_stick" in pkt:
                    try:
                        ls = pkt["left_stick"]
                        rs = pkt["right_stick"]
                        state["vr_sticks"] = (
                            float(ls[0]),
                            float(ls[1]),
                            float(rs[0]),
                        )
                        state["vr_sticks_time"] = time.monotonic()
                    except (TypeError, IndexError, ValueError, KeyError):
                        pass

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    pov_preview = bool(args.pov_preview)
    model = mujoco.MjModel.from_xml_string(minimal_xml(pov_camera=pov_preview))
    data = mujoco.MjData(model)
    origin = np.array([0.0, 0.0, 0.05], dtype=np.float64)
    arrow_size = np.array([0.012, 0.028, 0.65], dtype=np.float64)
    root_arrow = np.array([0.008, 0.018, 0.35], dtype=np.float64)
    root_z = float(args.root_z)
    dz = float(args.joy_deadzone)
    joy_id = int(args.joystick_id)
    use_gamepad = not bool(args.no_gamepad)
    use_vr_sticks = not bool(args.no_vr_sticks)
    vr_stick_timeout = float(args.vr_stick_timeout_s)
    world_frame = bool(args.world_frame)
    pov_mocap_id: int | None = None
    pov_renderer: mujoco.Renderer | None = None
    pov_cam_name = "hmd_pov"
    pov_frame_i = 0
    if pov_preview:
        try:
            import cv2
        except ImportError as e:
            raise SystemExit(
                "--pov-preview requires OpenCV: pip install opencv-python"
            ) from e
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hmd_pov_mocap")
        if bid < 0:
            raise SystemExit("internal: hmd_pov_mocap missing from model")
        pov_mocap_id = int(model.body_mocapid[bid])
        if pov_mocap_id < 0:
            raise SystemExit("internal: hmd_pov_mocap is not mocap")
        w = max(64, int(args.pov_width))
        h = max(64, int(args.pov_height))
        pov_renderer = mujoco.Renderer(model, height=h, width=w)
        # OpenCV/Qt: create the window on the same thread as imshow (inside the viewer loop),
        # not here on the main thread — avoids QObject::moveToThread warnings / crashes.
    eye_off = np.asarray(args.pov_eye_offset, dtype=np.float64).reshape(3)
    pov_every = max(1, int(args.pov_every))
    pov_rotate = str(args.pov_rotate).strip().lower()

    root_x = root_y = yaw = 0.0
    t_prev = time.perf_counter()
    warned_no_gp = False

    print(f"UDP pose viewer listening on {host}:{port} (protocol v{PROTOCOL_V2})", flush=True)
    if use_vr_sticks:
        print(
            f"Root sticks: prefer VR UDP (left_stick/right_stick) when age < {vr_stick_timeout:.2f}s; "
            "else GLFW if enabled.",
            flush=True,
        )
    elif use_gamepad:
        print(
            "Gamepad: GLFW joystick "
            f"{joy_id} — left Y forward/back, left X + right X yaw; "
            f"{'world' if world_frame else 'root-relative'} device frame.",
            flush=True,
        )
    if pov_preview:
        print(
            f"POV preview: camera `{pov_cam_name}` @ {args.pov_width}x{args.pov_height}, "
            f"every {pov_every} viewer frame(s); eye offset {eye_off.tolist()}; "
            f"display rotate={pov_rotate}",
            flush=True,
        )
    pov_cv_window_ready = False
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            ngeom = 11
            viewer.user_scn.ngeom = ngeom
            g = viewer.user_scn.geoms

            mat_x = _mjv_mat_from_R(arrow_rotation_matrix([1, 0, 0]))
            mujoco.mjv_initGeom(
                g[0],
                mujoco.mjtGeom.mjGEOM_ARROW,
                size=arrow_size,
                pos=origin,
                mat=mat_x,
                rgba=[0.95, 0.18, 0.18, 0.78],
            )
            mat_y = _mjv_mat_from_R(arrow_rotation_matrix([0, 1, 0]))
            mujoco.mjv_initGeom(
                g[1],
                mujoco.mjtGeom.mjGEOM_ARROW,
                size=arrow_size,
                pos=origin,
                mat=mat_y,
                rgba=[0.18, 0.85, 0.18, 0.78],
            )
            mat_z = _mjv_mat_from_R(arrow_rotation_matrix([0, 0, 1]))
            mujoco.mjv_initGeom(
                g[2],
                mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.014, 0.032, 0.82], dtype=np.float64),
                pos=origin,
                mat=mat_z,
                rgba=[0.20, 0.45, 0.98, 0.78],
            )

            while viewer.is_running():
                now = time.perf_counter()
                dt = float(np.clip(now - t_prev, 0.0, 0.1))
                t_prev = now

                lx = ly = rx = 0.0
                used_vr = False
                with lock:
                    vs = state["vr_sticks"]
                    vs_t = float(state["vr_sticks_time"])
                if (
                    use_vr_sticks
                    and vs is not None
                    and (now - vs_t) <= vr_stick_timeout
                ):
                    used_vr = True
                    lx = apply_axis_deadzone(vs[0], dz)
                    ly = apply_axis_deadzone(vs[1], dz)
                    rx = apply_axis_deadzone(vs[2], dz)
                elif use_gamepad:
                    sticks = _poll_glfw_gamepad_sticks(joy_id)
                    if sticks is None:
                        if not warned_no_gp and not used_vr:
                            print(
                                "Gamepad: GLFW reports no gamepad on this joystick id — "
                                "root motion needs VR sticks in UDP or a pad.",
                                flush=True,
                            )
                            warned_no_gp = True
                    else:
                        warned_no_gp = False
                        lx = apply_axis_deadzone(sticks[0], dz)
                        ly = apply_axis_deadzone(sticks[1], dz)
                        rx = apply_axis_deadzone(sticks[2], dz)

                root_x, root_y, yaw = integrate_root_planar(
                    root_x,
                    root_y,
                    yaw,
                    lx,
                    ly,
                    rx,
                    dt,
                    move_scale=float(args.joy_move_scale),
                    yaw_left_scale=float(args.joy_yaw_left),
                    yaw_right_scale=float(args.joy_yaw_right),
                    invert_forward=bool(args.invert_forward),
                )
                t_root = np.array([root_x, root_y, root_z], dtype=np.float64)
                R_root = Rz(yaw)
                root_origin = t_root + np.array([0.0, 0.0, 0.02], dtype=np.float64)

                with lock:
                    hmd_pos = state["hmd_p"].copy()
                    left_pos = state["left_p"].copy()
                    right_pos = state["right_p"].copy()
                    left_mat = state["left_mat"].copy()
                    right_mat = state["right_mat"].copy()
                    hmd_q = state["hmd_q"].copy()

                Rw_h = np.zeros(9, dtype=np.float64)
                mujoco.mju_quat2Mat(Rw_h, hmd_q)
                Rwh = _R3_from_mjmat(Rw_h)

                if world_frame:
                    hmd_d, lh_d, rh_d = hmd_pos, left_pos, right_pos
                    hmd_m = _mjv_mat_from_R(np.eye(3))
                    lm_m, rm_m = left_mat, right_mat
                else:
                    hmd_d, Rhd = world_pose_to_root_planar(hmd_pos, Rwh, t_root, yaw)
                    lh_d, Rld = world_pose_to_root_planar(
                        left_pos,
                        _R3_from_mjmat(left_mat),
                        t_root,
                        yaw,
                    )
                    rh_d, Rrd = world_pose_to_root_planar(
                        right_pos,
                        _R3_from_mjmat(right_mat),
                        t_root,
                        yaw,
                    )
                    hmd_m = _mjv_mat_from_R(Rhd)
                    lm_m, rm_m = _mjv_mat_from_R(Rld), _mjv_mat_from_R(Rrd)

                with viewer.lock():
                    mujoco.mjv_initGeom(
                        g[3],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.085, 0, 0],
                        pos=hmd_d,
                        mat=_mjv_mat_from_R(np.eye(3)),
                        rgba=[0.15, 0.75, 0.95, 0.88],
                    )
                    mujoco.mjv_initGeom(
                        g[4],
                        mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[0.008, 0.016, 0.17],
                        pos=hmd_d,
                        mat=hmd_m,
                        rgba=[0.25, 0.82, 1.0, 0.85],
                    )
                    mujoco.mjv_initGeom(
                        g[5],
                        mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[0.011, 0.024, 0.24],
                        pos=lh_d,
                        mat=lm_m,
                        rgba=[0.05, 0.95, 0.25, 0.94],
                    )
                    mujoco.mjv_initGeom(
                        g[6],
                        mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[0.011, 0.024, 0.24],
                        pos=rh_d,
                        mat=rm_m,
                        rgba=[0.95, 0.25, 0.08, 0.94],
                    )
                    # Root triad (planar yaw): at world t_root, axes = columns of R_root
                    for i, (axis, rgba) in enumerate(
                        (
                            (np.array([1.0, 0.0, 0.0]), [0.92, 0.55, 0.12, 0.85]),
                            (np.array([0.0, 1.0, 0.0]), [0.55, 0.92, 0.15, 0.85]),
                            (np.array([0.0, 0.0, 1.0]), [0.35, 0.55, 0.95, 0.85]),
                        )
                    ):
                        d = R_root @ axis
                        mat_a = _mjv_mat_from_R(arrow_rotation_matrix(d))
                        sz = root_arrow.copy()
                        if i == 2:
                            sz[2] = 0.45
                        mujoco.mjv_initGeom(
                            g[7 + i],
                            mujoco.mjtGeom.mjGEOM_ARROW,
                            size=sz,
                            pos=root_origin,
                            mat=mat_a,
                            rgba=rgba,
                        )
                    viewer.user_scn.ngeom = ngeom

                if pov_preview and pov_mocap_id is not None and pov_renderer is not None:
                    import cv2

                    if not pov_cv_window_ready:
                        cv2.namedWindow("vr_hmd_pov_mujoco", cv2.WINDOW_NORMAL)
                        pov_cv_window_ready = True
                    pp, qq = _pov_pose_for_mocap(
                        hmd_pos,
                        hmd_q,
                        t_root,
                        yaw,
                        world_frame,
                        eye_off,
                    )
                    data.mocap_pos[pov_mocap_id] = pp
                    data.mocap_quat[pov_mocap_id] = qq
                    mujoco.mj_forward(model, data)
                    pov_frame_i += 1
                    if pov_frame_i % pov_every == 0:
                        pov_renderer.update_scene(data, camera=pov_cam_name)
                        rgb = pov_renderer.render()
                        bgr = rgb[..., ::-1].copy()
                        bgr = pov_rotate_bgr(bgr, pov_rotate)
                        cv2.resizeWindow("vr_hmd_pov_mujoco", int(bgr.shape[1]), int(bgr.shape[0]))
                        cv2.imshow("vr_hmd_pov_mujoco", bgr)
                        cv2.waitKey(1)

                mujoco.mj_step(model, data)
                viewer.sync()
    finally:
        if pov_preview:
            try:
                import cv2

                cv2.destroyWindow("vr_hmd_pov_mujoco")
                cv2.waitKey(1)
            except Exception:
                pass
        stop.set()
        th.join(timeout=1.0)
        sock.close()


def cmd_send(args: argparse.Namespace) -> None:
    try:
        import openvr
    except ImportError as e:
        print("Install OpenVR on the VR PC: pip install openvr", file=sys.stderr)
        raise SystemExit(1) from e

    openvr.init(openvr.VRApplication_Scene)
    system = openvr.VRSystem()
    if system is None:
        openvr.shutdown()
        raise SystemExit("OpenVR.VRSystem() is None — is SteamVR running?")

    try:
        compositor = openvr.VRCompositor()
    except Exception:
        compositor = None

    seated = args.tracking_universe.strip().lower() == "seated"
    universe = (
        openvr.TrackingUniverseSeated if seated else openvr.TrackingUniverseStanding
    )
    if compositor is not None:
        try:
            compositor.setTrackingSpace(
                openvr.TrackingUniverseSeated
                if seated
                else openvr.TrackingUniverseStanding
            )
        except Exception:
            pass

    poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
    render_poses = poses_t()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / max(float(args.hz), 1.0)
    t_next = time.perf_counter()
    seq = 0
    target = (args.host.strip(), int(args.port))

    print(
        f"Streaming OpenVR poses → UDP v{PROTOCOL_V2} to {target[0]}:{target[1]} at {args.hz} Hz "
        f"({args.tracking_universe}); thumbsticks rAxis[{args.stick_axis}] in left_stick/right_stick. "
        "Ctrl+C to stop.",
        flush=True,
    )

    try:
        while True:
            if compositor is not None:
                compositor.waitGetPoses(render_poses, None)
            else:
                system.getDeviceToAbsoluteTrackingPose(universe, 0.0, render_poses)

            hmd_idx, left_idx, right_idx = resolve_openvr_tracked_device_indices(
                system, render_poses, openvr
            )

            payload: dict[str, Any] = {"v": PROTOCOL_V2, "seq": seq}
            seq += 1

            if hmd_idx is not None and render_poses[hmd_idx].bPoseIsValid:
                p, q = openvr_pose_to_mujoco(
                    render_poses[hmd_idx].mDeviceToAbsoluteTracking
                )
                payload["hmd"] = {"pos": p.tolist(), "quat": q.tolist()}

            if left_idx is not None and render_poses[left_idx].bPoseIsValid:
                p, q = openvr_pose_to_mujoco(
                    render_poses[left_idx].mDeviceToAbsoluteTracking
                )
                payload["left"] = {"pos": p.tolist(), "quat": q.tolist()}

            if right_idx is not None and render_poses[right_idx].bPoseIsValid:
                p, q = openvr_pose_to_mujoco(
                    render_poses[right_idx].mDeviceToAbsoluteTracking
                )
                payload["right"] = {"pos": p.tolist(), "quat": q.tolist()}

            slx, sly, srx, sry = _openvr_thumbsticks(
                system, left_idx, right_idx, int(args.stick_axis)
            )
            payload["left_stick"] = [slx, sly]
            payload["right_stick"] = [srx, sry]

            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            sock.sendto(data, target)

            t_next += period
            sleep = t_next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                t_next = time.perf_counter()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        sock.close()
        openvr.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_recv = sub.add_parser("receive", help="Listen for UDP v2 pose JSON; MuJoCo viewer.")
    p_recv.add_argument(
        "--bind",
        default="0.0.0.0:5006",
        help="UDP bind HOST:PORT (default: 0.0.0.0:5006).",
    )
    p_recv.add_argument(
        "--vr-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.4],
        metavar=("X", "Y", "Z"),
        help="Added to streamed positions for display (default: 0 0 0.4).",
    )
    p_recv.add_argument(
        "--no-gamepad",
        action="store_true",
        help="Disable planar root motion (no GLFW gamepad).",
    )
    p_recv.add_argument(
        "--world-frame",
        action="store_true",
        help="Draw HMD/controllers in world frame; default strips planar root yaw/translation.",
    )
    p_recv.add_argument(
        "--joy-deadzone",
        type=float,
        default=0.15,
        help="Analog stick deadzone [0,1) for gamepad axes (default: 0.15).",
    )
    p_recv.add_argument(
        "--joy-move-scale",
        type=float,
        default=1.2,
        help="Forward/back scale (m/s) from left Y after deadzone (default: 1.2).",
    )
    p_recv.add_argument(
        "--joy-yaw-left",
        type=float,
        default=2.0,
        help="Yaw rate scale (rad/s) from left stick X (default: 2.0).",
    )
    p_recv.add_argument(
        "--joy-yaw-right",
        type=float,
        default=2.0,
        help="Yaw rate scale (rad/s) from right stick X (default: 2.0).",
    )
    p_recv.add_argument(
        "--root-z",
        type=float,
        default=0.05,
        help="World Z height (m) for root triad base (default: 0.05).",
    )
    p_recv.add_argument(
        "--joystick-id",
        type=int,
        default=0,
        help="GLFW joystick id (first gamepad is usually 0; default: 0).",
    )
    p_recv.add_argument(
        "--invert-forward",
        action="store_true",
        help="Invert left-stick forward/back (if stick up moves backward).",
    )
    p_recv.add_argument(
        "--no-vr-sticks",
        action="store_true",
        help="Ignore left_stick/right_stick in UDP; use GLFW only (or no root input).",
    )
    p_recv.add_argument(
        "--vr-stick-timeout-s",
        type=float,
        default=0.35,
        help="Drop VR stick values older than this (seconds); then use GLFW if enabled (default: 0.35).",
    )
    p_recv.add_argument(
        "--pov-preview",
        action="store_true",
        help="OpenCV window: render MuJoCo camera hmd_pov (mocap) locked to HMD pose + orientation.",
    )
    p_recv.add_argument(
        "--pov-width",
        type=int,
        default=640,
        help="POV render width (default: 640).",
    )
    p_recv.add_argument(
        "--pov-height",
        type=int,
        default=480,
        help="POV render height (default: 480).",
    )
    p_recv.add_argument(
        "--pov-every",
        type=int,
        default=1,
        help="Render POV every N viewer frames (default: 1).",
    )
    p_recv.add_argument(
        "--pov-eye-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Extra translation in HMD local frame before mocap (default: 0 0 0).",
    )
    p_recv.add_argument(
        "--pov-rotate",
        choices=("none", "ccw90", "cw90", "180", "transpose"),
        default="none",
        help="Extra pixel transform after render (default: none; hmd_pov MJCF already rolls +90° about view axis).",
    )
    p_recv.set_defaults(func=cmd_receive)

    p_send = sub.add_parser("send", help="OpenVR → UDP v2 (run on VR PC).")
    p_send.add_argument("--host", required=True, help="Sim machine address.")
    p_send.add_argument("--port", type=int, default=5006, help="UDP port (default: 5006).")
    p_send.add_argument("--hz", type=float, default=60.0, help="Send rate (default: 60).")
    p_send.add_argument(
        "--tracking-universe",
        choices=("seated", "standing"),
        default="seated",
        help="OpenVR tracking universe (default: seated, matches typical debug script).",
    )
    p_send.add_argument(
        "--stick-axis",
        type=int,
        default=0,
        help="OpenVR rAxis index for Quest/Touch thumbsticks (default: 0, same as teleop client).",
    )
    p_send.set_defaults(func=cmd_send)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
