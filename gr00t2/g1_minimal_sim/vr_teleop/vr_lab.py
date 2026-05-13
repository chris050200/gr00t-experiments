"""Blank MuJoCo VR lab viewer for streamed headset/controller poses.

Runs on the sim machine (no headset required). It listens for UDP packets from
``vr_teleop.openvr_udp_sender`` and renders:
- world axis gizmo at origin (X red, Y green, Z blue),
- headset arrow (blue),
- left controller arrow (green),
- right controller arrow (orange).

This module is a frame-composition lab: the sender publishes mapped poses/inputs,
and this viewer applies a virtual-base SE(2) transform (XY + yaw) to validate
that controller/headset geometry stays coherent while "walking/turning".

Important: this pivot logic is for bring-up visualization. Robot teleop should
compose torso/head-relative intent with the robot's live base pose each tick.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from .openvr_stream import (
    PoseState,
    StreamState,
    default_stream_state,
    parse_bind,
    stream_state_from_packet,
)
from .viewer_arrows import RGBA_HMD, RGBA_LEFT, RGBA_RIGHT, VR_DEVICE_ARROW_SIZE, init_pose_arrow_geom


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _rz(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _draw_axis_connector(
    geom: Any,
    axis_dir: np.ndarray,
    axis_len: float,
    rgba: np.ndarray,
    *,
    width: float = 0.01,
) -> None:
    """Draw one world axis arrow from origin to ``axis_len * axis_dir``."""
    d = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    end = axis_len * d
    connector_fn = getattr(mujoco, "mjv_makeConnector", None)
    if connector_fn is None:
        connector_fn = getattr(mujoco, "mjv_connector")
        connector_fn(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            float(width),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.asarray(end, dtype=np.float64).reshape(3),
        )
    else:
        connector_fn(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            float(width),
            0.0,
            0.0,
            0.0,
            float(end[0]),
            float(end[1]),
            float(end[2]),
        )
    geom.rgba[:] = np.asarray(rgba, dtype=np.float32)


def _make_world_xml() -> str:
    return """<mujoco model="vr_lab">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25"/>
  </visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="8 8 0.01" rgba="0.18 0.20 0.24 1"/>
  </worldbody>
</mujoco>
"""


def _recv_latest(sock: socket.socket, buf: bytearray) -> StreamState | None:
    latest: StreamState | None = None
    while True:
        try:
            n, _ = sock.recvfrom_into(buf)
        except BlockingIOError:
            break
        if n <= 0:
            break
        try:
            pkt = json.loads(buf[:n].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(pkt, dict):
            continue
        st = stream_state_from_packet(pkt)
        if st is not None:
            latest = st
    return latest


def _pose_in_virtual_base(
    pose: PoseState,
    *,
    pivot_stream_pos: np.ndarray,
    pivot_world_pos: np.ndarray,
    base_xy: np.ndarray,
    base_yaw: float,
) -> PoseState:
    """Apply virtual-base SE(2) around a stream-space pivot (typically current HMD).

    Why this shape:
    - rotating absolute world positions causes "carousel around origin" artifacts,
    - rotating a local offset around a moving pivot keeps headset/controllers coherent.
    """
    R = _rz(base_yaw)
    q_base = _quat_from_yaw(base_yaw)
    out = PoseState(valid=pose.valid, pos=pose.pos.copy(), quat=pose.quat.copy())
    p_local = out.pos - np.asarray(pivot_stream_pos, dtype=np.float64).reshape(3)
    out.pos = np.asarray(pivot_world_pos, dtype=np.float64).reshape(3) + (R @ p_local.reshape(3)).reshape(3)
    out.pos[0] += float(base_xy[0])
    out.pos[1] += float(base_xy[1])
    out.quat = _quat_mul_wxyz(q_base, out.quat)
    qn = np.linalg.norm(out.quat)
    if qn > 1e-12:
        out.quat /= qn
    else:
        out.quat[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return out


def run_vr_lab(bind: str) -> None:
    host, port = parse_bind(bind)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.setblocking(False)
    buf = bytearray(65536)

    model = mujoco.MjModel.from_xml_string(_make_world_xml())
    data = mujoco.MjData(model)
    state = default_stream_state()
    print(f"[vr_lab] listening on udp://{host}:{port}")

    axis_len = 0.25
    dev_size = VR_DEVICE_ARROW_SIZE
    base_xy = np.zeros(2, dtype=np.float64)
    base_yaw = 0.0
    hmd_pivot_stream = np.zeros(3, dtype=np.float64)
    t_prev = time.monotonic()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_now = time.monotonic()
            dt = max(0.0, min(0.1, t_now - t_prev))
            t_prev = t_now
            newest = _recv_latest(sock, buf)
            if newest is not None:
                state = newest
            if state.hmd.valid:
                # Use current HMD stream position as yaw pivot so turn-in-place stays local.
                hmd_pivot_stream = np.asarray(state.hmd.pos, dtype=np.float64).reshape(3).copy()
            # Virtual-base commands from sticks:
            #   forward/back  <- left stick Y
            #   strafe L/R    <- right stick X
            #   yaw rate      <- left stick X
            yaw_rate_cmd = float(state.left_stick_xy[0])
            base_yaw += yaw_rate_cmd * dt
            base_yaw = float(np.arctan2(np.sin(base_yaw), np.cos(base_yaw)))
            cmd_body = np.array(
                [float(state.left_stick_xy[1]), float(state.right_stick_xy[0]), 0.0],
                dtype=np.float64,
            )
            dxy_world = (_rz(base_yaw) @ cmd_body)[:2] * dt
            base_xy += dxy_world

            with viewer.lock():
                g = viewer.user_scn.geoms

                # World axis gizmo at origin.
                _draw_axis_connector(
                    g[0],
                    np.array([1.0, 0.0, 0.0], dtype=np.float64),
                    axis_len,
                    np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
                )
                _draw_axis_connector(
                    g[1],
                    np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    axis_len,
                    np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32),
                )
                _draw_axis_connector(
                    g[2],
                    np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    axis_len,
                    np.array([0.1, 0.35, 1.0, 1.0], dtype=np.float32),
                )

                # Device markers: apply the same virtual-base transform to HMD and both
                # controllers so their relative geometry stays consistent while translating/yawing.
                hmd_pivot_world = hmd_pivot_stream.copy()
                hmd_draw = _pose_in_virtual_base(
                    state.hmd,
                    pivot_stream_pos=hmd_pivot_stream,
                    pivot_world_pos=hmd_pivot_world,
                    base_xy=base_xy,
                    base_yaw=base_yaw,
                )
                left_draw = _pose_in_virtual_base(
                    state.left,
                    pivot_stream_pos=hmd_pivot_stream,
                    pivot_world_pos=hmd_pivot_world,
                    base_xy=base_xy,
                    base_yaw=base_yaw,
                )
                right_draw = _pose_in_virtual_base(
                    state.right,
                    pivot_stream_pos=hmd_pivot_stream,
                    pivot_world_pos=hmd_pivot_world,
                    base_xy=base_xy,
                    base_yaw=base_yaw,
                )
                init_pose_arrow_geom(g[3], hmd_draw, RGBA_HMD, size=dev_size)
                init_pose_arrow_geom(g[4], left_draw, RGBA_LEFT, size=dev_size)
                init_pose_arrow_geom(g[5], right_draw, RGBA_RIGHT, size=dev_size)

                viewer.user_scn.ngeom = 6

            mujoco.mj_step(model, data)
            viewer.sync()
            age_ms = (
                max(0.0, (time.monotonic() - state.stamp) * 1000.0)
                if state.stamp > 0.0
                else -1.0
            )
            print(
                f"\r[vr_lab] seq={state.seq:6d} age_ms={age_ms:7.1f}"
                f" hmd={'Y' if state.hmd.valid else 'N'}"
                f" l={'Y' if state.left.valid else 'N'}"
                f" r={'Y' if state.right.valid else 'N'}"
                f" | li={'Y' if state.left_input_valid else 'N'}"
                f" ls=({state.left_stick_xy[0]: .2f},{state.left_stick_xy[1]: .2f})"
                f" lt={state.left_trigger: .2f}"
                f" | ri={'Y' if state.right_input_valid else 'N'}"
                f" rs=({state.right_stick_xy[0]: .2f},{state.right_stick_xy[1]: .2f})"
                f" rt={state.right_trigger: .2f}"
                f" | cmd_fwd={float(state.left_stick_xy[1]): .2f}"
                f" cmd_strafe={float(state.right_stick_xy[0]): .2f}"
                f" cmd_yaw={yaw_rate_cmd: .2f}",
                end="",
                flush=True,
            )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal VR pose lab viewer (UDP -> MuJoCo arrows).")
    p.add_argument(
        "--bind",
        default="0.0.0.0:5006",
        help="UDP bind address for incoming packets (HOST:PORT).",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    run_vr_lab(args.bind)


if __name__ == "__main__":
    main()
