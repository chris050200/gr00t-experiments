#!/usr/bin/env python3
"""Standalone OpenVR → UDP JSON sender for split-machine G1 teleop (VR PC only).

Deploy **`legacy_vr_code/`** on the machine that runs **SteamVR + Quest Link / Air Link** (or any OpenVR
headset). Install:

    pip install openvr numpy

Run from ``g1_minimal_sim`` (sim machine IP/port must match ``--udp-teleop``)::

    python -m legacy_vr_code.openvr_teleop_client --host 192.168.1.50 --port 5005

Keep SteamVR in **Standing** or **Room-scale** tracking. Tune ``--axis-map`` if arms point
the wrong way relative to the robot (OpenVR Y-up vs MuJoCo Z-up).

JSON protocol ``v`` 1 (one UDP datagram per line / per packet, UTF-8)::

    {
      "v": 1,
      "seq": <int>,
      "palm_frame": "hmd_relative",
      "loco_cmd": [vx, vy, vyaw],
      "gripper_left": <0..1>,
      "gripper_right": <0..1>,
      "left_palm": {"pos": [x,y,z], "quat": [w,x,y,z]},
      "right_palm": {"pos": [x,y,z], "quat": [w,x,y,z]}
    }

- ``loco_cmd``: same units as keyboard teleop (scaled by sim yaml ``cmd_scale``); stick
  deflects set these directly each frame.
- ``gripper_*``: **1 = open, 0 = closed** (matches ``hand_gripper``); triggers map to
  ``1.0 - trigger_pull``.
- ``*_palm``: position (m) and orientation **wxyz** after OpenVR→MuJoCo frame fix; sim
  converts to ``torso_link`` frame for IK.

- ``palm_frame``: ``"hmd_relative"`` (default) or ``"world"``. **HMD-relative:** controller
  pose **vs headset** in MJ-oriented axes; sim composes with **``torso_link``** so playspace
  origin no longer maps 1:1 into sim world. **World:** legacy room-absolute palms.

**Tier A (split-machine comfort):** until the first **recenter chord** (both grips
pressed together on the **rising edge**), palm keys are omitted so the sim keeps
its current arm targets. After a chord, palms are sent as **anchor + (raw − origin)**
position and **anchor * inv(origin) * raw** orientation (per hand), then **smoothed**
(low-pass position, slerp quaternion). Chord again anytime to re-zero. Use
``--no-require-recenter`` for legacy absolute room poses.

This file does **not** import MuJoCo. Deploy the whole ``legacy_vr_code/`` package on the VR PC
(or run from a checkout with ``g1_minimal_sim`` on ``PYTHONPATH``).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from typing import Any

import numpy as np

from .openvr_device_indices import resolve_openvr_tracked_device_indices
from .vr_steamvr_mujoco_mapping import R_STEAMVR_LINEAR_TO_MJ, openvr_pose_to_mujoco_custom

# ---------------------------------------------------------------------------
# Math (self-contained)
# ---------------------------------------------------------------------------


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    return _quat_normalize(np.array([w, -x, -y, -z], dtype=np.float64))


def _quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation (wxyz), t in [0, 1]."""
    a = _quat_normalize(q0)
    b = _quat_normalize(q1)
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return _quat_normalize(out)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_0
    s1 = np.sin(theta) / sin_0
    return _quat_normalize(s0 * a + s1 * b)


def _R_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) from unit quaternion wxyz (body→fixed, same as MuJoCo body quat)."""
    w, x, y, z = _quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) → unit quaternion wxyz."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(R)
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return _quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def _parse_axis_map(s: str) -> np.ndarray:
    """Nine floats row-major 3×3 mapping OpenVR linear coords → MuJoCo world linear coords."""
    parts = [float(x) for x in s.replace(",", " ").split()]
    if len(parts) != 9:
        raise ValueError("axis-map needs 9 floats (3×3 row-major)")
    return np.array(parts, dtype=np.float64).reshape(3, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Sim machine address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="UDP port the sim binds with --udp-teleop (default: 5005).",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=60.0,
        help="Send rate (default: 60).",
    )
    parser.add_argument(
        "--axis-map",
        type=str,
        default="",
        help="Optional 9 floats: 3×3 R maps OpenVR position to MuJoCo world (row-major). "
        "Empty = built-in Y-up→Z-up preset.",
    )
    parser.add_argument(
        "--loco-scale",
        type=float,
        default=0.45,
        help="Max |stick| → |loco_cmd| component scale (default: 0.45).",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.12,
        help="Joystick deadzone (default: 0.12).",
    )
    parser.add_argument(
        "--stick-axis",
        type=int,
        default=0,
        help="OpenVR rAxis index for 2D locomotion stick (default: 0).",
    )
    parser.add_argument(
        "--trigger-axis",
        type=int,
        default=1,
        help="OpenVR rAxis index whose .x is analog trigger (default: 1).",
    )
    parser.add_argument(
        "--palm-alpha",
        type=float,
        default=0.35,
        help="Low-pass on palm pos + slerp on quat each step after mapping (0..1], default 0.35).",
    )
    parser.add_argument(
        "--no-require-recenter",
        action="store_true",
        help="Send absolute room palms immediately (legacy; can be unstable). Default is chord+gated.",
    )
    parser.add_argument(
        "--palm-frame",
        choices=("hmd_relative", "world"),
        default="hmd_relative",
        help="hmd_relative: palms vs headset, sim mounts on torso_link (default). world: legacy room-absolute.",
    )
    parser.add_argument(
        "--tracking-universe",
        choices=("seated", "standing"),
        default="standing",
        help="OpenVR tracking universe (default: standing, matches prior hardcoded behavior).",
    )
    args = parser.parse_args()

    try:
        import openvr
    except ImportError as e:
        print("Install OpenVR: pip install openvr", file=sys.stderr)
        raise SystemExit(1) from e

    R_fix = (
        _parse_axis_map(args.axis_map)
        if args.axis_map
        else np.asarray(R_STEAMVR_LINEAR_TO_MJ, dtype=np.float64).copy()
    )

    openvr.init(openvr.VRApplication_Scene)
    vrsys = openvr.VRSystem()
    if vrsys is None:
        raise SystemExit("OpenVR.VRSystem() is None — is SteamVR running?")

    try:
        compositor = openvr.VRCompositor()
    except Exception:
        compositor = None

    seated = str(args.tracking_universe).strip().lower() == "seated"
    universe = (
        openvr.TrackingUniverseSeated if seated else openvr.TrackingUniverseStanding
    )
    if compositor is not None:
        try:
            compositor.setTrackingSpace(universe)
        except Exception:
            pass

    grip_btn = int(getattr(openvr, "k_EButton_Grip", 2))

    def _grip_pressed(ul: int) -> bool:
        return (int(ul) & (1 << grip_btn)) != 0

    poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
    poses = poses_t()
    period = 1.0 / max(args.hz, 1.0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    t_next = time.perf_counter()
    palm_alpha = float(np.clip(float(args.palm_alpha), 1e-6, 1.0))
    require_recenter = not bool(args.no_require_recenter)
    recentered = not require_recenter
    prev_chord = False

    # Per-hand: after at least one chord including this hand, origin_* / anchor_* set; filt_* smoothed out.
    def _empty_hand() -> dict[str, Any]:
        return {
            "origin_p": None,
            "origin_q": None,
            "anchor_p": None,
            "anchor_q": None,
            "filt_p": None,
            "filt_q": None,
        }

    hand_L = _empty_hand()
    hand_R = _empty_hand()

    print(
        f"Sending UDP v1 JSON to {args.host}:{args.port} at {args.hz} Hz. Ctrl+C to stop.",
        flush=True,
    )
    if require_recenter:
        print(
            "Palm stream armed: squeeze **both grips** once (rising edge) to recenter / start arms.",
            flush=True,
        )
    print(
        f"Palm frame: {args.palm_frame} — "
        + (
            "controllers relative to HMD (needs headset tracking)."
            if args.palm_frame == "hmd_relative"
            else "room-absolute MJ-oriented poses."
        ),
        flush=True,
    )
    print(
        f"Tracking universe: {args.tracking_universe} "
        f"({'compositor waitGetPoses' if compositor is not None else 'getDeviceToAbsoluteTrackingPose'}).",
        flush=True,
    )

    use_hmd_rel = args.palm_frame == "hmd_relative"

    def raw_palm_mj(idx: int | None) -> tuple[np.ndarray, np.ndarray] | None:
        if idx is None:
            return None
        p = poses[idx]
        if not p.bPoseIsValid:
            return None
        t_mj, q_mj = openvr_pose_to_mujoco_custom(
            p.mDeviceToAbsoluteTracking, R_fix
        )
        return t_mj.astype(np.float64), q_mj.astype(np.float64)

    def rel_palm_vs_hmd(
        h_tq: tuple[np.ndarray, np.ndarray] | None,
        c_tq: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Controller pose in HMD frame (MJ-oriented translation + wxyz)."""
        if h_tq is None or c_tq is None:
            return None
        th, qh = h_tq
        tc, qc = c_tq
        Rh = _R_from_quat_wxyz(qh)
        Rc = _R_from_quat_wxyz(qc)
        p_rel = Rh.T @ (tc - th)
        Rrel = Rh.T @ Rc
        q_rel = _quat_normalize(_quat_from_matrix(Rrel))
        return p_rel.astype(np.float64), q_rel.astype(np.float64)

    def mapped_pq(
        h: dict[str, Any],
        raw_p: np.ndarray,
        raw_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if h["origin_p"] is None:
            return raw_p, raw_q
        mp = h["anchor_p"] + (raw_p - h["origin_p"])
        mq = _quat_mul_wxyz(
            h["anchor_q"],
            _quat_mul_wxyz(_quat_inv_wxyz(h["origin_q"]), raw_q),
        )
        return mp, _quat_normalize(mq)

    def smooth_update(
        h: dict[str, Any],
        mp: np.ndarray,
        mq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        fp, fq = h["filt_p"], h["filt_q"]
        if fp is None or fq is None:
            h["filt_p"], h["filt_q"] = mp.copy(), mq.copy()
            return h["filt_p"], h["filt_q"]
        h["filt_p"] = (1.0 - palm_alpha) * fp + palm_alpha * mp
        h["filt_q"] = _quat_slerp_wxyz(fq, mq, palm_alpha)
        return h["filt_p"], h["filt_q"]

    def palm_json(h: dict[str, Any], raw: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any] | None:
        if raw is None:
            return None
        if require_recenter and h["origin_p"] is None:
            return None
        mp, mq = mapped_pq(h, raw[0], raw[1])
        sp, sq = smooth_update(h, mp, mq)
        return {"pos": sp.tolist(), "quat": sq.tolist()}

    try:
        while True:
            if compositor is not None:
                compositor.waitGetPoses(poses, None)
            else:
                vrsys.getDeviceToAbsoluteTrackingPose(universe, 0.0, poses)

            _hmd_i, li, ri = resolve_openvr_tracked_device_indices(vrsys, poses, openvr)

            if use_hmd_rel:
                h_raw = raw_palm_mj(_hmd_i)
                rawL = rel_palm_vs_hmd(h_raw, raw_palm_mj(li))
                rawR = rel_palm_vs_hmd(h_raw, raw_palm_mj(ri))
            else:
                rawL = raw_palm_mj(li)
                rawR = raw_palm_mj(ri)

            lx, ly, l_trig = 0.0, 0.0, 0.0
            rx, r_trig = 0.0, 0.0
            ok_l = ok_r = False
            st_l = st_r = None
            if li is not None:
                ok_l, st_l = vrsys.getControllerState(li)
                if ok_l and args.stick_axis < len(st_l.rAxis):
                    ax = st_l.rAxis[args.stick_axis]
                    lx, ly = float(ax.x), float(ax.y)
                if ok_l and args.trigger_axis < len(st_l.rAxis):
                    l_trig = float(st_l.rAxis[args.trigger_axis].x)
            if ri is not None:
                ok_r, st_r = vrsys.getControllerState(ri)
                if ok_r and args.stick_axis < len(st_r.rAxis):
                    ax = st_r.rAxis[args.stick_axis]
                    rx = float(ax.x)
                if ok_r and args.trigger_axis < len(st_r.rAxis):
                    r_trig = float(st_r.rAxis[args.trigger_axis].x)

            chord = (
                ok_l
                and ok_r
                and st_l is not None
                and st_r is not None
                and _grip_pressed(st_l.ulButtonPressed)
                and _grip_pressed(st_r.ulButtonPressed)
            )
            chord_rising = chord and not prev_chord
            prev_chord = bool(chord)

            if chord_rising:
                if rawL is not None:
                    hl = hand_L
                    hl["origin_p"] = rawL[0].copy()
                    hl["origin_q"] = rawL[1].copy()
                    if hl["filt_p"] is None:
                        hl["anchor_p"] = rawL[0].copy()
                        hl["anchor_q"] = rawL[1].copy()
                    else:
                        hl["anchor_p"] = hl["filt_p"].copy()
                        hl["anchor_q"] = hl["filt_q"].copy()
                    hl["filt_p"] = hl["anchor_p"].copy()
                    hl["filt_q"] = hl["anchor_q"].copy()
                if rawR is not None:
                    hr = hand_R
                    hr["origin_p"] = rawR[0].copy()
                    hr["origin_q"] = rawR[1].copy()
                    if hr["filt_p"] is None:
                        hr["anchor_p"] = rawR[0].copy()
                        hr["anchor_q"] = rawR[1].copy()
                    else:
                        hr["anchor_p"] = hr["filt_p"].copy()
                        hr["anchor_q"] = hr["filt_q"].copy()
                    hr["filt_p"] = hr["anchor_p"].copy()
                    hr["filt_q"] = hr["anchor_q"].copy()
                if rawL is not None or rawR is not None:
                    recentered = True
                    print(
                        "Recenter chord — Δ palm + smoothing (repeat chord anytime).",
                        flush=True,
                    )

            # Locomotion: left stick move/strafe, right stick x = yaw (common game layout).
            def apply_deadzone(x: float, y: float, dz: float) -> tuple[float, float]:
                n = np.hypot(x, y)
                if n < dz:
                    return 0.0, 0.0
                s = (n - dz) / max(n * (1.0 - dz), 1e-6)
                return x * s, y * s

            def scalar_deadzone(v: float, dz: float) -> float:
                av = abs(v)
                if av < dz:
                    return 0.0
                return float(np.copysign((av - dz) / max(1.0 - dz, 1e-6), v))

            f = float(args.loco_scale)
            dz = float(args.deadzone)
            mx, my = apply_deadzone(lx, ly, dz)
            # Left stick: forward/strafe (WASD-like); right stick X: yaw (q/e-like).
            yaw_cmd = scalar_deadzone(rx, dz) * f
            loco_cmd = [float(my * f), float(mx * f), yaw_cmd]

            payload: dict[str, Any] = {
                "v": 1,
                "seq": seq,
                "loco_cmd": loco_cmd,
            }
            seq += 1

            # Grippers: 1 open, 0 closed
            payload["gripper_left"] = float(np.clip(1.0 - l_trig, 0.0, 1.0))
            payload["gripper_right"] = float(np.clip(1.0 - r_trig, 0.0, 1.0))

            send_palms = (not require_recenter) or recentered
            if send_palms:
                lp = palm_json(hand_L, rawL)
                if lp is not None:
                    payload["left_palm"] = lp
                rp = palm_json(hand_R, rawR)
                if rp is not None:
                    payload["right_palm"] = rp
                if lp is not None or rp is not None:
                    payload["palm_frame"] = args.palm_frame

            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            sock.sendto(data, (args.host, args.port))

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


if __name__ == "__main__":
    main()
