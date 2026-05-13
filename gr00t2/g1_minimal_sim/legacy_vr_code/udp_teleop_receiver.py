"""UDP JSON receiver for split-machine VR / joystick teleop (sim side).

Pairs with ``openvr_teleop_client.py`` on the VR PC. Palm poses use JSON ``palm_frame``:

- ``world`` (default if key omitted): ``pos`` / ``quat`` are MuJoCo **world**; mapped with
  ``ee_pose_ref_from_world`` → ``torso_link``-frame ``ee_*``.
- ``hmd_relative``: palms vs headset in MJ axes; **mounted on** ``torso_link`` via
  ``ee_pose_world_from_ref`` then ``ee_pose_ref_from_world`` (same math as
  ``tests/test_hmd_relative_compose.py``).

Protocol ``v``: 1 — see ``openvr_teleop_client.py``.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

import mujoco
import numpy as np

from arm_ik import ArmSideIK
from ee_frame import (
    ee_pose_ref_from_world,
    ee_pose_world_from_ref,
    normalize_quat,
    quat_wxyz_to_rotmat,
)
from .teleop_udp_bind import parse_udp_bind


def _as_float_vec(x: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} must have length {n}, got {a.size}")
    return a


def _as_quat_wxyz(x: Any, name: str) -> np.ndarray:
    q = _as_float_vec(x, 4, name)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def apply_udp_packet_to_control_dict(
    packet: dict[str, Any],
    *,
    data: mujoco.MjData,
    control_dict: dict[str, Any],
    torso_index: int,
    left_ik: ArmSideIK | None,
    right_ik: ArmSideIK | None,
    has_hands: bool,
) -> None:
    """Apply one decoded JSON object to ``control_dict`` (call from sim thread only).

    Reads ``data.xpos/xmat/xquat`` for ``torso_index``; ensure ``mj_forward`` has been run.
    """
    if int(packet.get("v", 1)) != 1:
        return

    if "loco_cmd" in packet:
        lc = np.asarray(packet["loco_cmd"], dtype=np.float32).reshape(-1)
        if lc.size >= 3:
            control_dict["loco_cmd"][:] = lc[:3]

    if "height_cmd" in packet:
        control_dict["height_cmd"] = float(packet["height_cmd"])
    if "rpy_cmd" in packet:
        rpy = np.asarray(packet["rpy_cmd"], dtype=np.float32).reshape(-1)
        if rpy.size >= 3:
            control_dict["rpy_cmd"][:] = rpy[:3]
    if "freq_cmd" in packet:
        control_dict["freq_cmd"] = float(packet["freq_cmd"])

    if has_hands:
        if "gripper_left" in packet:
            g = float(np.clip(float(packet["gripper_left"]), 0.0, 1.0))
            control_dict["gripper_left"] = g
        if "gripper_right" in packet:
            g = float(np.clip(float(packet["gripper_right"]), 0.0, 1.0))
            control_dict["gripper_right"] = g

    xpos = data.xpos[torso_index]
    qb = np.asarray(data.xquat[torso_index], dtype=np.float64)
    R_ref = quat_wxyz_to_rotmat(normalize_quat(qb))

    palm_frame = str(packet.get("palm_frame", "world")).strip().lower()
    use_hmd_rel = palm_frame == "hmd_relative"

    for side, ik, key_p, key_q in (
        ("left", left_ik, "left_palm", "ee_left_pos"),
        ("right", right_ik, "right_palm", "ee_right_pos"),
    ):
        block = packet.get(key_p)
        if block is None or ik is None:
            continue
        if not isinstance(block, dict):
            continue
        if "pos" not in block or "quat" not in block:
            continue
        p_pkt = _as_float_vec(block["pos"], 3, f"{key_p}.pos")
        q_pkt = _as_quat_wxyz(block["quat"], f"{key_p}.quat")
        if use_hmd_rel:
            p_w, q_w = ee_pose_world_from_ref(xpos, R_ref, qb, p_pkt, q_pkt)
            p_b, q_b = ee_pose_ref_from_world(xpos, R_ref, qb, p_w, q_w)
        else:
            p_b, q_b = ee_pose_ref_from_world(xpos, R_ref, qb, p_pkt, q_pkt)
        qkey = "ee_left_quat" if side == "left" else "ee_right_quat"
        control_dict[key_q] = p_b.astype(np.float32)
        control_dict[qkey] = q_b.astype(np.float32)


class UdpTeleopReceiver:
    """Background thread: recv UDP datagrams, keep latest JSON object for the sim thread."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest: dict[str, Any] | None = None

    def start(self, host: str, port: int) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.settimeout(0.25)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def consume_latest(self) -> dict[str, Any] | None:
        """Return and clear the latest packet (sim thread)."""
        with self._latest_lock:
            p = self._latest
            self._latest = None
            return p

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data_b, _addr = self._sock.recvfrom(65536)
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                continue
            try:
                text = data_b.decode("utf-8")
                obj = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            with self._latest_lock:
                self._latest = obj


def start_udp_receiver_for_runtime(runtime: Any, bind_spec: str) -> UdpTeleopReceiver:
    host, port = parse_udp_bind(bind_spec)
    recv = UdpTeleopReceiver(runtime)
    recv.start(host, port)
    return recv
