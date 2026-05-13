"""UDP JSON from ``vr_teleop.openvr_udp_sender`` — parse poses + sticks, map sticks → ``loco_cmd``.

Protocol ``v``: 1 — same envelope as the VR-PC sender (``hmd`` / ``left`` / ``right`` + ``inputs``).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


def parse_bind(bind: str) -> tuple[str, int]:
    host, sep, port_s = bind.rpartition(":")
    if not sep:
        raise ValueError(f"bind must be HOST:PORT, got {bind!r}")
    return host, int(port_s)


@dataclass
class PoseState:
    valid: bool
    pos: np.ndarray
    quat: np.ndarray


@dataclass
class StreamState:
    seq: int
    stamp: float
    hmd: PoseState
    left: PoseState
    right: PoseState
    left_stick_xy: np.ndarray
    right_stick_xy: np.ndarray
    left_trigger: float
    right_trigger: float
    left_input_valid: bool
    right_input_valid: bool


def _default_pose(pos: list[float]) -> PoseState:
    return PoseState(
        valid=False,
        pos=np.asarray(pos, dtype=np.float64),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def default_stream_state() -> StreamState:
    return StreamState(
        seq=-1,
        stamp=0.0,
        hmd=_default_pose([0.0, 0.0, 1.4]),
        left=_default_pose([0.0, 0.25, 1.2]),
        right=_default_pose([0.0, -0.25, 1.2]),
        left_stick_xy=np.zeros(2, dtype=np.float64),
        right_stick_xy=np.zeros(2, dtype=np.float64),
        left_trigger=0.0,
        right_trigger=0.0,
        left_input_valid=False,
        right_input_valid=False,
    )


def parse_pose_block(block: Any) -> PoseState:
    if not isinstance(block, dict):
        return _default_pose([0.0, 0.0, 1.0])
    valid = bool(block.get("valid", False))
    pos = np.asarray(block.get("pos_mj", [0.0, 0.0, 1.0]), dtype=np.float64).reshape(3)
    quat = np.asarray(
        block.get("quat_mj", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64
    ).reshape(4)
    qn = np.linalg.norm(quat)
    if qn < 1e-9:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quat = quat / qn
    return PoseState(valid=valid, pos=pos, quat=quat)


def parse_input_side(block: Any) -> tuple[np.ndarray, float, bool]:
    if not isinstance(block, dict):
        return np.zeros(2, dtype=np.float64), 0.0, False
    valid = bool(block.get("valid", False))
    stick_xy = np.asarray(block.get("stick_xy", [0.0, 0.0]), dtype=np.float64).reshape(2)
    trig = float(block.get("trigger", 0.0))
    return stick_xy, trig, valid


def stream_state_from_packet(pkt: dict[str, Any]) -> StreamState | None:
    if pkt.get("v") != 1:
        return None
    inputs = pkt.get("inputs", {})
    left_block = inputs.get("left", {}) if isinstance(inputs, dict) else {}
    right_block = inputs.get("right", {}) if isinstance(inputs, dict) else {}
    left_stick_xy, left_trigger, left_input_valid = parse_input_side(left_block)
    right_stick_xy, right_trigger, right_input_valid = parse_input_side(right_block)
    return StreamState(
        seq=int(pkt.get("seq", -1)),
        stamp=float(pkt.get("ts_mono_s", 0.0)),
        hmd=parse_pose_block(pkt.get("hmd")),
        left=parse_pose_block(pkt.get("left")),
        right=parse_pose_block(pkt.get("right")),
        left_stick_xy=left_stick_xy,
        right_stick_xy=right_stick_xy,
        left_trigger=left_trigger,
        right_trigger=right_trigger,
        left_input_valid=left_input_valid,
        right_input_valid=right_input_valid,
    )


def _apply_deadzone_scalar(x: float, dz: float) -> float:
    ax = abs(float(x))
    if ax < dz:
        return 0.0
    # Remap [dz, 1] → (0, 1] for |x| ≤ 1
    edge = (ax - dz) / max(1e-9, 1.0 - dz)
    return float(np.sign(x) * min(1.0, edge))


def sticks_to_loco_cmd(
    left_stick_xy: np.ndarray,
    right_stick_xy: np.ndarray,
    *,
    deadzone: float = 0.12,
    fwd_scale: float = 0.75,
    strafe_scale: float = 0.75,
    yaw_scale: float = 0.55,
) -> np.ndarray:
    """Map sticks to ``loco_cmd`` [forward, strafe, yaw] (same convention as ``vr_lab``).

    Forward/back ← left stick Y; strafe ← right stick X; yaw ← left stick X.
    Scales match ~keyboard-saturated ``loco_cmd`` (0.1 per key × several steps).
    """
    lx = _apply_deadzone_scalar(float(left_stick_xy[0]), deadzone)
    ly = _apply_deadzone_scalar(float(left_stick_xy[1]), deadzone)
    rx = _apply_deadzone_scalar(float(right_stick_xy[0]), deadzone)
    return np.array(
        [fwd_scale * ly, strafe_scale * rx, yaw_scale * lx],
        dtype=np.float32,
    )


class OpenvrStreamUdpReceiver:
    """Background thread: recv UDP datagrams; keep latest parsed :class:`StreamState`."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: StreamState | None = None
        self._recv_mono: float = 0.0

    def start(self, bind_spec: str) -> None:
        if self._thread is not None:
            return
        host, port = parse_bind(bind_spec)
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

    def snapshot(self) -> tuple[StreamState | None, float]:
        """Latest state and ``time.monotonic()`` when it was received (0 if never)."""
        with self._lock:
            return self._latest, float(self._recv_mono)

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
            st = stream_state_from_packet(obj)
            if st is None:
                continue
            with self._lock:
                self._latest = st
                self._recv_mono = time.monotonic()


def start_openvr_stream_receiver(bind_spec: str) -> OpenvrStreamUdpReceiver:
    recv = OpenvrStreamUdpReceiver()
    recv.start(bind_spec)
    return recv
