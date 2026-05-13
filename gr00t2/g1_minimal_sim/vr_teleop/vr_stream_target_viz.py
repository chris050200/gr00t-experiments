"""OpenVR stream poses in the passive viewer: torso-mounted debug arrows."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from vr_stream_ik_mode import normalize_vr_ik_anchor_mode

from .openvr_stream import PoseState, StreamState
from .viewer_arrows import RGBA_HMD, RGBA_LEFT, RGBA_RIGHT, VR_DEVICE_ARROW_SIZE, init_pose_arrow_geom
from .vr_stream_torso_compose import (
    VrTorsoVizCalib,
    pose_for_overlay_device,
    torso_xmat_with_yaw_offset,
    world_pose_stream_device_for_anchor,
)

# SteamVR / ``openvr_to_mujoco_pos`` origin is often below the G1 sim floor (~0.3 m). Raise markers
# into humanoid scale (G1 standing ~1.7 m tall; pelvis z ~0.74 m in this stack).
VR_STREAM_VIZ_Z_OFFSET_M = 1.0

TorsoZMode = Literal["full", "fixed_world", "frozen_calib"]
CalibOrientMode = Literal["full", "yaw_only"]


def _pose_shift_z(pose: PoseState, dz: float) -> PoseState:
    p = np.asarray(pose.pos, dtype=np.float64).reshape(3).copy()
    p[2] += float(dz)
    q = np.asarray(pose.quat, dtype=np.float64).reshape(4).copy()
    return PoseState(valid=pose.valid, pos=p, quat=q)


def fill_vr_stream_arrow_geoms(
    geoms: Any,
    geom_start: int,
    state: StreamState | None,
    *,
    enabled: bool,
    z_offset_m: float = VR_STREAM_VIZ_Z_OFFSET_M,
    torso_xpos: np.ndarray | None = None,
    torso_xmat: np.ndarray | None = None,
    calib: VrTorsoVizCalib | None = None,
    torso_z_mode: TorsoZMode = "fixed_world",
    torso_yaw_offset_deg: float = 0.0,
    calib_orient_mode: CalibOrientMode = "full",
    vr_ik_anchor_mode: str = "legacy",
    pelvis_xpos: np.ndarray | None = None,
    pelvis_xmat: np.ndarray | None = None,
    mount_T: np.ndarray | None = None,
    head_orient_mode: str = "yaw_only",
) -> int:
    """Draw HMD + left + right arrows into ``geoms[geom_start : geom_start + 3]``.

    Torso-mounted mode: default ``legacy`` uses ``W_T = W_T_torso * inv(S_T_hmd0) * S_T_dev``
    after first valid HMD calibrates ``S_T_hmd0`` on ``calib``. Until then, behaves like stream +
    ``z_offset_m``. If ``vr_ik_anchor_mode`` is ``torso`` or ``pelvis``, uses the same compose as
    VR IK (see ``world_pose_stream_device_for_anchor``), including ``mount_T`` for pelvis.

    ``torso_z_mode``: ``full`` rigid Z; ``fixed_world`` sets ``z = z_offset_m``; ``frozen_calib``
    locks each device's world Z at first composed sample.

    Returns the next free geom index (``geom_start`` if disabled or ``state is None``).
    """
    if not enabled or state is None:
        return geom_start
    dz = float(z_offset_m)
    devices: tuple[PoseState, ...] = (state.hmd, state.left, state.right)
    rgba = (RGBA_HMD, RGBA_LEFT, RGBA_RIGHT)

    use_torso = calib is not None and torso_xpos is not None and torso_xmat is not None
    if use_torso:
        torso_xmat_vr = torso_xmat_with_yaw_offset(torso_xmat, torso_yaw_offset_deg)
        calib.maybe_calibrate(state, orient_mode=calib_orient_mode)
        drawn: list[PoseState] = []
        anchor = normalize_vr_ik_anchor_mode(vr_ik_anchor_mode)
        for i, dev in enumerate(devices):
            if calib.is_ready():
                if anchor == "legacy":
                    po = pose_for_overlay_device(
                        dev,
                        torso_xpos=torso_xpos,
                        torso_xmat=torso_xmat_vr,
                        calib=calib,
                        device_index=i,
                        z_mode=torso_z_mode,
                        z_fixed_world_m=dz,
                    )
                else:
                    po0 = world_pose_stream_device_for_anchor(
                        i,
                        dev,
                        stream=state,
                        anchor_mode=anchor,
                        torso_xpos=torso_xpos,
                        torso_R_vr=torso_xmat_vr,
                        calib=calib,
                        yaw_offset_deg=float(torso_yaw_offset_deg),
                        pelvis_xpos=pelvis_xpos,
                        pelvis_xmat=pelvis_xmat,
                        mount_T=mount_T,
                        head_orient_mode=head_orient_mode,
                    )
                    if po0.valid:
                        pz = calib.apply_z_mode(i, po0.pos, torso_z_mode, dz)
                        po = PoseState(valid=True, pos=pz, quat=po0.quat)
                    else:
                        po = PoseState(
                            valid=False,
                            pos=np.asarray(dev.pos, dtype=np.float64).reshape(3).copy(),
                            quat=np.asarray(dev.quat, dtype=np.float64).reshape(4).copy(),
                        )
            else:
                po = PoseState(
                    valid=dev.valid,
                    pos=np.asarray(dev.pos, dtype=np.float64).reshape(3).copy(),
                    quat=np.asarray(dev.quat, dtype=np.float64).reshape(4).copy(),
                )
                if po.valid:
                    po = _pose_shift_z(po, dz)
            drawn.append(po)
    else:
        drawn = [_pose_shift_z(d, dz) for d in devices]

    for k in range(3):
        init_pose_arrow_geom(geoms[geom_start + k], drawn[k], rgba[k], size=VR_DEVICE_ARROW_SIZE)
    return geom_start + 3
