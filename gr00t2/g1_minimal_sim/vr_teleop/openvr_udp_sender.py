"""OpenVR pose sender for split-machine VR debug.

Run this on the VR PC (Quest2/SteamVR host). It streams HMD and controller poses
over UDP to ``vr_teleop.vr_lab`` running on another machine.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from typing import Any

import openvr

from vr_teleop.mapping import (
    openvr_mat34_to_Rt,
    openvr_to_mujoco_pos,
    openvr_rotmat_to_mujoco_quat_wxyz,
    quat_from_matrix_wxyz,
)


@dataclass(frozen=True)
class DeviceIndices:
    hmd: int | None
    left: int | None
    right: int | None


def _controller_input_block(
    vr_system: Any,
    dev_idx: int | None,
    *,
    stick_axis: int,
    trigger_axis: int,
) -> dict[str, Any]:
    """Read one controller's analog inputs from OpenVR controller state."""
    out: dict[str, Any] = {
        "valid": False,
        "stick_xy": [0.0, 0.0],
        "trigger": 0.0,
    }
    if dev_idx is None:
        return out
    ok, st = vr_system.getControllerState(dev_idx)
    if not bool(ok):
        return out
    out["valid"] = True
    if 0 <= int(stick_axis) < len(st.rAxis):
        ax = st.rAxis[int(stick_axis)]
        out["stick_xy"] = [float(ax.x), float(ax.y)]
    if 0 <= int(trigger_axis) < len(st.rAxis):
        out["trigger"] = float(st.rAxis[int(trigger_axis)].x)
    return out


def _controller_state_debug(
    vr_system: Any,
    dev_idx: int | None,
    *,
    stick_axis: int,
    trigger_axis: int,
) -> tuple[bool, float, float, float]:
    """Return (ok, sx, sy, trig) for periodic sender-side debugging."""
    if dev_idx is None:
        return False, 0.0, 0.0, 0.0
    ok, st = vr_system.getControllerState(dev_idx)
    if not bool(ok):
        return False, 0.0, 0.0, 0.0
    sx = sy = trig = 0.0
    if 0 <= int(stick_axis) < len(st.rAxis):
        ax = st.rAxis[int(stick_axis)]
        sx, sy = float(ax.x), float(ax.y)
    if 0 <= int(trigger_axis) < len(st.rAxis):
        trig = float(st.rAxis[int(trigger_axis)].x)
    return True, sx, sy, trig


def parse_target(target: str) -> tuple[str, int]:
    host, sep, port_s = target.rpartition(":")
    if not sep:
        raise ValueError(f"target must be HOST:PORT, got {target!r}")
    return host, int(port_s)


def resolve_device_indices(vr_system: Any) -> DeviceIndices:
    hmd = left = right = None
    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        dclass = vr_system.getTrackedDeviceClass(idx)
        if dclass == openvr.TrackedDeviceClass_HMD:
            hmd = idx
        elif dclass == openvr.TrackedDeviceClass_Controller:
            role = vr_system.getControllerRoleForTrackedDeviceIndex(idx)
            if role == openvr.TrackedControllerRole_LeftHand:
                left = idx
            elif role == openvr.TrackedControllerRole_RightHand:
                right = idx
    return DeviceIndices(hmd=hmd, left=left, right=right)


def _pose_block_from_tracked_pose(
    pose: Any | None,
) -> dict[str, Any]:
    if pose is None:
        return {"valid": False}
    out: dict[str, Any] = {"valid": bool(pose.bPoseIsValid)}
    if not bool(pose.bPoseIsValid):
        return out

    R_vr, t_vr = openvr_mat34_to_Rt(pose.mDeviceToAbsoluteTracking)
    q_vr = quat_from_matrix_wxyz(R_vr)
    p_mj = openvr_to_mujoco_pos(t_vr)
    q_mj = openvr_rotmat_to_mujoco_quat_wxyz(R_vr)

    out.update(
        {
            "pos_vr": t_vr.tolist(),
            "quat_vr": q_vr.tolist(),
            "pos_mj": p_mj.tolist(),
            "quat_mj": q_mj.tolist(),
        }
    )
    return out


def run_sender(
    target: str,
    *,
    hz: float,
    tracking_space: str,
    stick_axis: int,
    trigger_axis: int,
) -> None:
    host, port = parse_target(target)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    openvr.init(openvr.VRApplication_Scene)
    vr_system = openvr.VRSystem()
    compositor = openvr.VRCompositor()

    if tracking_space == "standing":
        compositor.setTrackingSpace(openvr.TrackingUniverseStanding)
    else:
        compositor.setTrackingSpace(openvr.TrackingUniverseSeated)

    poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
    render_poses = poses_t()
    game_poses = poses_t()

    dev = resolve_device_indices(vr_system)
    print(f"[openvr_sender] devices hmd={dev.hmd} left={dev.left} right={dev.right}")

    seq = 0
    period = 1.0 / max(1e-6, hz)
    last_debug_t = 0.0

    print(f"[openvr_sender] streaming udp://{host}:{port} @ {hz:.1f}Hz ({tracking_space})")
    try:
        while True:
            t0 = time.monotonic()
            compositor.waitGetPoses(render_poses, game_poses)
            dev = resolve_device_indices(vr_system)

            pkt = {
                "v": 1,
                "seq": seq,
                "ts_mono_s": t0,
                "tracking_space": tracking_space,
                "hmd": _pose_block_from_tracked_pose(
                    render_poses[dev.hmd] if dev.hmd is not None else None,
                ),
                "left": _pose_block_from_tracked_pose(
                    render_poses[dev.left] if dev.left is not None else None,
                ),
                "right": _pose_block_from_tracked_pose(
                    render_poses[dev.right] if dev.right is not None else None,
                ),
                "inputs": {
                    "left": _controller_input_block(
                        vr_system,
                        dev.left,
                        stick_axis=stick_axis,
                        trigger_axis=trigger_axis,
                    ),
                    "right": _controller_input_block(
                        vr_system,
                        dev.right,
                        stick_axis=stick_axis,
                        trigger_axis=trigger_axis,
                    ),
                },
            }
            sock.sendto(json.dumps(pkt, separators=(",", ":")).encode("utf-8"), (host, port))
            seq += 1

            # Temporary low-rate debug to diagnose Quest2 controller-state issues.
            if t0 - last_debug_t >= 1.0:
                lok, lsx, lsy, ltr = _controller_state_debug(
                    vr_system,
                    dev.left,
                    stick_axis=stick_axis,
                    trigger_axis=trigger_axis,
                )
                rok, rsx, rsy, rtr = _controller_state_debug(
                    vr_system,
                    dev.right,
                    stick_axis=stick_axis,
                    trigger_axis=trigger_axis,
                )
                print(
                    "[openvr_sender] "
                    f"idx L={dev.left} R={dev.right} | "
                    f"L ok={int(lok)} stick=({lsx:+.2f},{lsy:+.2f}) trig={ltr:.2f} | "
                    f"R ok={int(rok)} stick=({rsx:+.2f},{rsy:+.2f}) trig={rtr:.2f}"
                )
                last_debug_t = t0

            dt = time.monotonic() - t0
            sleep_s = period - dt
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    finally:
        openvr.shutdown()
        sock.close()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenVR UDP pose sender for vr_teleop.vr_lab")
    p.add_argument("--target", default="127.0.0.1:5006", help="Destination HOST:PORT")
    p.add_argument("--hz", type=float, default=90.0, help="Send frequency in Hz")
    p.add_argument(
        "--tracking-space",
        choices=("seated", "standing"),
        default="seated",
        help="OpenVR tracking universe.",
    )
    p.add_argument(
        "--stick-axis",
        type=int,
        default=0,
        help="OpenVR rAxis index for joystick/thumbstick XY (default: 0).",
    )
    p.add_argument(
        "--trigger-axis",
        type=int,
        default=1,
        help="OpenVR rAxis index whose .x is analog trigger (default: 1).",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    run_sender(
        args.target,
        hz=args.hz,
        tracking_space=args.tracking_space,
        stick_axis=int(args.stick_axis),
        trigger_axis=int(args.trigger_axis),
    )


if __name__ == "__main__":
    main()

