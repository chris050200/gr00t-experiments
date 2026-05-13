"""Resolve OpenVR HMD / controller indices each frame (roles + sorted fallback).

SteamVR often reports **Invalid** controller roles until bindings settle; some runtimes
never set Left/Right. Shared by legacy teleop v1 client and pose-stream v2 send.
"""

from __future__ import annotations

from typing import Any


def resolve_openvr_tracked_device_indices(
    system: Any,
    render_poses: Any,
    openvr_mod: Any,
) -> tuple[int | None, int | None, int | None]:
    """Return ``(hmd_idx, left_idx, right_idx)`` for this poll.

    ``render_poses[i].bPoseIsValid`` is read for controller fallback ordering.
    """
    hmd_idx: int | None = None
    left_idx: int | None = None
    right_idx: int | None = None
    controller_idxs: list[int] = []

    for i in range(openvr_mod.k_unMaxTrackedDeviceCount):
        if not system.isTrackedDeviceConnected(i):
            continue
        cls = system.getTrackedDeviceClass(i)
        if cls == openvr_mod.TrackedDeviceClass_HMD:
            hmd_idx = i
        elif cls == openvr_mod.TrackedDeviceClass_Controller:
            controller_idxs.append(i)
            role = system.getControllerRoleForTrackedDeviceIndex(i)
            if role == openvr_mod.TrackedControllerRole_LeftHand:
                left_idx = i
            elif role == openvr_mod.TrackedControllerRole_RightHand:
                right_idx = i

    if hmd_idx is None:
        hmd_idx = int(getattr(openvr_mod, "k_unTrackedDeviceIndex_Hmd", 0))

    valid_ctrl = [i for i in controller_idxs if render_poses[i].bPoseIsValid]
    valid_ctrl.sort()
    if len(valid_ctrl) >= 2:
        if left_idx is None:
            left_idx = valid_ctrl[0]
        if right_idx is None:
            for j in valid_ctrl:
                if j != left_idx:
                    right_idx = j
                    break
    elif len(valid_ctrl) == 1:
        if left_idx is None and right_idx is None:
            left_idx = valid_ctrl[0]
        elif left_idx is None and right_idx is not None and valid_ctrl[0] != right_idx:
            left_idx = valid_ctrl[0]
        elif right_idx is None and left_idx is not None and valid_ctrl[0] != left_idx:
            right_idx = valid_ctrl[0]

    return hmd_idx, left_idx, right_idx
