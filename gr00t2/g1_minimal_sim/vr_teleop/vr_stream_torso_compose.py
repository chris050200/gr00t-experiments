"""SE(3) compose: stream poses mounted on ``torso_link`` for VR debug overlays (numpy only).

World draw pose per device (legacy / overlay):

    W_T_draw = W_T_torso * inv(S_T_hmd0) * S_T_dev

Calibration stores the first valid streamed HMD pose ``S_T_hmd0`` after reset.

VR IK can instead use **current** headset-relative controllers (see
``compose_head_relative_controller_world_pose``) optionally mounted from ``pelvis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .openvr_stream import PoseState, StreamState


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (body axes as columns, body→world) → unit quaternion wxyz."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _mat4_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Rigid transform: p_parent = R @ p_local + t (stream / device frames)."""
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4).copy()
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_to_yaw_wxyz(q: np.ndarray) -> float:
    q = _quat_normalize_wxyz(q)
    w, x, y, z = q
    # Yaw around +Z from quaternion (wxyz), right-handed.
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(s, c))


def _yaw_only_quat_wxyz(q: np.ndarray) -> np.ndarray:
    yaw = _quat_to_yaw_wxyz(q)
    h = 0.5 * yaw
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def normalize_head_orient_mode(mode: str) -> str:
    """Normalize HMD orientation mode for head-relative compose."""
    m = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "yaw": "yaw_only",
        "yawonly": "yaw_only",
        "yaw_only": "yaw_only",
        "full": "full",
    }
    if m in aliases:
        return aliases[m]
    raise ValueError(f"unknown head_orient_mode {mode!r} (use full or yaw_only)")


def _mat4_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def _mat4_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _pos_quat_from_mat4(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = T[:3, 3].copy()
    q = _rotmat_to_quat_wxyz(T[:3, :3])
    return p, q


def torso_xmat_with_yaw_offset(torso_xmat: np.ndarray, yaw_offset_deg: float) -> np.ndarray:
    """Return torso orientation with extra world-Z yaw offset (degrees).

    This is receiver-side only and intentionally rotates orientation basis without
    changing torso world position, so walking away from origin does not induce
    artificial translation drift.
    """
    yaw = np.deg2rad(float(yaw_offset_deg))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    Rz = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    R_torso = np.asarray(torso_xmat, dtype=np.float64).reshape(3, 3)
    return Rz @ R_torso


def compose_head_relative_controller_world_pose(
    anchor_xpos: np.ndarray,
    anchor_R: np.ndarray,
    mount_T: np.ndarray | None,
    hmd_pos: np.ndarray,
    hmd_quat_wxyz: np.ndarray,
    dev_pos: np.ndarray,
    dev_quat_wxyz: np.ndarray,
    *,
    head_orient_mode: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """World palm/controller pose: head-relative stream, then mounted on sim anchor.

    ``W_T_out = W_T_anchor @ mount @ inv(S_T_hmd) @ S_T_dev``.

    - ``anchor_*``: sim body world pose (position + **columns** = world rotation).
    - ``mount``: optional fixed 4×4 (use ``None`` for identity). For pelvis mode,
      ``mount = inv(W_T_pelvis_calib) @ W_T_torso_calib`` so the chain matches legacy
      at the calibration instant.
    - Stream poses ``hmd_*``, ``dev_*`` are in the sender / SteamVR tracking frame ``S``
      (same numeric convention as legacy compose).
    """
    q_a = _rotmat_to_quat_wxyz(np.asarray(anchor_R, dtype=np.float64).reshape(3, 3))
    T_a = _mat4_from_pos_quat(anchor_xpos, q_a)
    hom = normalize_head_orient_mode(head_orient_mode)
    hq = np.asarray(hmd_quat_wxyz, dtype=np.float64).reshape(4)
    if hom == "yaw_only":
        hq = _yaw_only_quat_wxyz(hq)
    T_h = _mat4_from_pos_quat(hmd_pos, hq)
    T_d = _mat4_from_pos_quat(dev_pos, dev_quat_wxyz)
    T_m = np.eye(4, dtype=np.float64) if mount_T is None else np.asarray(mount_T, dtype=np.float64).reshape(4, 4)
    T_out = _mat4_mul(T_a, _mat4_mul(T_m, _mat4_mul(_mat4_inv(T_h), T_d)))
    return _pos_quat_from_mat4(T_out)


def vr_ik_pelvis_mount_matrix(
    pelvis_xpos: np.ndarray,
    pelvis_xmat: np.ndarray,
    torso_xpos: np.ndarray,
    torso_xmat: np.ndarray,
    yaw_offset_deg: float,
) -> np.ndarray:
    """Return ``M = inv(W_T_pelvis) @ W_T_torso`` using the same yaw offset as VR compose."""
    R_p = torso_xmat_with_yaw_offset(pelvis_xmat, yaw_offset_deg)
    R_t = torso_xmat_with_yaw_offset(torso_xmat, yaw_offset_deg)
    q_p = _rotmat_to_quat_wxyz(R_p)
    q_t = _rotmat_to_quat_wxyz(R_t)
    T_p = _mat4_from_pos_quat(pelvis_xpos, q_p)
    T_t = _mat4_from_pos_quat(torso_xpos, q_t)
    return _mat4_mul(_mat4_inv(T_p), T_t)


def compose_world_draw_pose(
    torso_xpos: np.ndarray,
    torso_xmat: np.ndarray,
    hmd0_pos: np.ndarray,
    hmd0_quat_wxyz: np.ndarray,
    dev_pos: np.ndarray,
    dev_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World position + wxyz quaternion for one streamed device (see module docstring)."""
    T_torso = np.eye(4, dtype=np.float64)
    T_torso[:3, :3] = np.asarray(torso_xmat, dtype=np.float64).reshape(3, 3)
    T_torso[:3, 3] = np.asarray(torso_xpos, dtype=np.float64).reshape(3)
    T_h0 = _mat4_from_pos_quat(hmd0_pos, hmd0_quat_wxyz)
    T_dev = _mat4_from_pos_quat(dev_pos, dev_quat_wxyz)
    T_draw = _mat4_mul(T_torso, _mat4_mul(_mat4_inv(T_h0), T_dev))
    return _pos_quat_from_mat4(T_draw)


def torso_xquat_with_receiver_yaw(
    torso_xmat: np.ndarray, torso_xquat_wxyz: np.ndarray, yaw_offset_deg: float
) -> np.ndarray:
    """Torso orientation wxyz matching ``torso_xmat_with_yaw_offset`` (post-multiply world yaw).

    Used with **the same** ``torso_xmat_with_yaw_offset`` matrix for position in
    ``ee_pose_ref_from_world`` / ``ee_pose_world_from_ref`` so ref-frame position and
    orientation stay consistent (avoids heading-dependent wrist/orientation drift when
    ``vr_receiver_yaw_deg`` is non-zero).
    """
    R_vr = torso_xmat_with_yaw_offset(torso_xmat, yaw_offset_deg)
    return _rotmat_to_quat_wxyz(R_vr)


def stream_device_world_pose(
    dev: PoseState,
    *,
    stream: StreamState,
    torso_xpos: np.ndarray,
    torso_xmat: np.ndarray,
    yaw_offset_deg: float,
    calib: VrTorsoVizCalib,
    calib_orient_mode: str,
) -> PoseState:
    """World pose for one device using the torso-mounted stream chain (no overlay Z clamp).

    Until HMD calibration is ready, returns the raw stream pose (same as pre-calibration overlay).
    """
    if not dev.valid:
        return PoseState(valid=False, pos=dev.pos.copy(), quat=dev.quat.copy())
    calib.maybe_calibrate(stream, orient_mode=calib_orient_mode)
    if not calib.is_ready() or calib.hmd0_pos is None or calib.hmd0_quat is None:
        return PoseState(
            valid=True,
            pos=np.asarray(dev.pos, dtype=np.float64).reshape(3).copy(),
            quat=np.asarray(dev.quat, dtype=np.float64).reshape(4).copy(),
        )
    R_vr = torso_xmat_with_yaw_offset(torso_xmat, yaw_offset_deg)
    p_w, q_w = compose_world_draw_pose(
        torso_xpos,
        R_vr,
        calib.hmd0_pos,
        calib.hmd0_quat,
        dev.pos,
        dev.quat,
    )
    return PoseState(valid=True, pos=p_w, quat=q_w)


@dataclass
class VrTorsoVizCalib:
    """HMD stream snapshot for ``inv(S_T_hmd0)``; optional per-device frozen world Z."""

    hmd0_pos: np.ndarray | None = None
    hmd0_quat: np.ndarray | None = None
    frozen_z: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float64)
    )

    def reset(self) -> None:
        self.hmd0_pos = None
        self.hmd0_quat = None
        self.frozen_z[:] = np.nan

    def is_ready(self) -> bool:
        return self.hmd0_pos is not None and self.hmd0_quat is not None

    def maybe_calibrate(self, state: StreamState, orient_mode: str = "full") -> None:
        if self.is_ready():
            return
        if state.hmd.valid:
            self.hmd0_pos = np.asarray(state.hmd.pos, dtype=np.float64).reshape(3).copy()
            q = np.asarray(state.hmd.quat, dtype=np.float64).reshape(4).copy()
            om = orient_mode.lower().replace("-", "_")
            if om == "full":
                self.hmd0_quat = _quat_normalize_wxyz(q)
            elif om == "yaw_only":
                self.hmd0_quat = _yaw_only_quat_wxyz(q)
            else:
                raise ValueError(
                    f"unknown orient_mode {orient_mode!r} (use full or yaw_only)"
                )

    def apply_z_mode(
        self,
        device_index: int,
        p: np.ndarray,
        z_mode: str,
        z_fixed_world_m: float,
    ) -> np.ndarray:
        """Return position with Z adjusted; ``device_index`` 0=HMD, 1=left, 2=right."""
        p = np.asarray(p, dtype=np.float64).reshape(3).copy()
        zm = z_mode.lower().replace("-", "_")
        if zm == "full":
            return p
        if zm == "fixed_world":
            p[2] = float(z_fixed_world_m)
            return p
        if zm == "frozen_calib":
            i = int(device_index)
            if np.isnan(self.frozen_z[i]):
                self.frozen_z[i] = float(p[2])
            p[2] = float(self.frozen_z[i])
            return p
        raise ValueError(f"unknown z_mode {z_mode!r} (use full, fixed_world, frozen_calib)")


def world_pose_stream_device_for_anchor(
    device_index: int,
    dev: PoseState,
    *,
    stream: StreamState,
    anchor_mode: str,
    torso_xpos: np.ndarray,
    torso_R_vr: np.ndarray,
    calib: VrTorsoVizCalib,
    yaw_offset_deg: float,
    pelvis_xpos: np.ndarray | None,
    pelvis_xmat: np.ndarray | None,
    mount_T: np.ndarray | None,
    head_orient_mode: str = "full",
) -> PoseState:
    """World pose for stream HMD (0) / left (1) / right (2) matching :func:`gear_wbc_vr_stream.apply_vr_stream_ik_targets`.

    Caller should run ``calib.maybe_calibrate(stream, orient_mode=...)`` before this when needed.
    Does **not** apply overlay Z policy — use :meth:`VrTorsoVizCalib.apply_z_mode` on ``pos``.

    ``torso_R_vr`` is the same rotated torso basis as ``torso_xmat_with_yaw_offset(...)``.
    """
    if not dev.valid:
        return PoseState(valid=False, pos=dev.pos.copy(), quat=dev.quat.copy())
    if not calib.is_ready() or calib.hmd0_pos is None or calib.hmd0_quat is None:
        return PoseState(valid=True, pos=dev.pos.copy(), quat=dev.quat.copy())

    am = str(anchor_mode).strip().lower()
    if am == "legacy":
        p, q = compose_world_draw_pose(
            torso_xpos,
            torso_R_vr,
            calib.hmd0_pos,
            calib.hmd0_quat,
            dev.pos,
            dev.quat,
        )
        return PoseState(valid=True, pos=p, quat=q)

    if not stream.hmd.valid:
        return PoseState(valid=False, pos=dev.pos.copy(), quat=dev.quat.copy())

    h = stream.hmd
    if am == "torso":
        if int(device_index) == 0:
            q_t = _rotmat_to_quat_wxyz(np.asarray(torso_R_vr, dtype=np.float64).reshape(3, 3))
            pj = np.asarray(torso_xpos, dtype=np.float64).reshape(3).copy()
            return PoseState(valid=True, pos=pj, quat=q_t)
        pj, qj = compose_head_relative_controller_world_pose(
            torso_xpos,
            torso_R_vr,
            None,
            h.pos,
            h.quat,
            dev.pos,
            dev.quat,
            head_orient_mode=head_orient_mode,
        )
        return PoseState(valid=True, pos=pj, quat=qj)

    if am == "pelvis":
        if pelvis_xpos is None or pelvis_xmat is None:
            p, q = compose_world_draw_pose(
                torso_xpos,
                torso_R_vr,
                calib.hmd0_pos,
                calib.hmd0_quat,
                dev.pos,
                dev.quat,
            )
            return PoseState(valid=True, pos=p, quat=q)
        pm = np.asarray(pelvis_xmat, dtype=np.float64).reshape(3, 3)
        R_p = torso_xmat_with_yaw_offset(pm, yaw_offset_deg)
        if mount_T is None:
            p, q = compose_world_draw_pose(
                torso_xpos,
                torso_R_vr,
                calib.hmd0_pos,
                calib.hmd0_quat,
                dev.pos,
                dev.quat,
            )
            return PoseState(valid=True, pos=p, quat=q)
        px = np.asarray(pelvis_xpos, dtype=np.float64).reshape(3)
        if int(device_index) == 0:
            T_a = _mat4_from_pos_quat(px, _rotmat_to_quat_wxyz(R_p))
            T_out = _mat4_mul(T_a, np.asarray(mount_T, dtype=np.float64).reshape(4, 4))
            pj, qj = _pos_quat_from_mat4(T_out)
            return PoseState(valid=True, pos=pj, quat=qj)
        pj, qj = compose_head_relative_controller_world_pose(
            px,
            R_p,
            np.asarray(mount_T, dtype=np.float64).reshape(4, 4),
            h.pos,
            h.quat,
            dev.pos,
            dev.quat,
            head_orient_mode=head_orient_mode,
        )
        return PoseState(valid=True, pos=pj, quat=qj)

    raise ValueError(f"unknown anchor_mode {anchor_mode!r}")


def pose_for_overlay_device(
    dev: PoseState,
    *,
    torso_xpos: np.ndarray,
    torso_xmat: np.ndarray,
    calib: VrTorsoVizCalib,
    device_index: int,
    z_mode: str,
    z_fixed_world_m: float,
) -> PoseState:
    """Torso-mounted world pose for one device, with Z policy; invalid → dimmed raw pose."""
    if not dev.valid:
        return PoseState(valid=False, pos=dev.pos.copy(), quat=dev.quat.copy())
    if not calib.is_ready():
        return PoseState(valid=True, pos=dev.pos.copy(), quat=dev.quat.copy())
    assert calib.hmd0_pos is not None and calib.hmd0_quat is not None
    p, q = compose_world_draw_pose(
        torso_xpos,
        torso_xmat,
        calib.hmd0_pos,
        calib.hmd0_quat,
        dev.pos,
        dev.quat,
    )
    p = calib.apply_z_mode(device_index, p, z_mode, z_fixed_world_m)
    return PoseState(valid=True, pos=p, quat=q)
