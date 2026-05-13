"""Normalize ``vr_ik_mode`` / ``--vr-ik`` strings (no MuJoCo or ONNX imports).

Canonical values: ``"off"`` | ``"on"``. Legacy ``pos`` / ``pose`` map to ``on`` with a
deprecation warning.
"""

from __future__ import annotations

import warnings


def normalize_vr_ik_mode(mode: str) -> str:
    """Return ``\"off\"`` or ``\"on\"`` for :class:`GearWBCRuntime`."""
    v = str(mode).strip().lower().replace("-", "_")
    if v in ("pos", "pose", "on", "yes", "true", "1"):
        if v in ("pos", "pose"):
            warnings.warn(
                f"vr_ik_mode {mode!r} is deprecated; use 'on' (same behavior).",
                DeprecationWarning,
                stacklevel=2,
            )
        return "on"
    if v in ("off", "no", "false", "0"):
        return "off"
    raise ValueError(
        f"vr_ik_mode must be 'off' or 'on' (legacy pos/pose accepted as on); got {mode!r}"
    )


def normalize_vr_ik_anchor_mode(mode: str) -> str:
    """Return ``legacy`` | ``torso`` | ``pelvis`` for VR stream → IK world palm compose.

    - ``legacy``: ``W_T = W_T_torso * inv(S_T_hmd0) * S_T_ctrl`` (frozen calib HMD).
    - ``torso``: ``W_T = W_T_torso * inv(S_T_hmd(t)) * S_T_ctrl`` (current headset-relative).
    - ``pelvis``: ``W_T = W_T_pelvis * M * inv(S_T_hmd(t)) * S_T_ctrl`` with
      ``M = inv(W_T_pelvis) @ W_T_torso`` at first post-calib sample (constant).
    """
    v = str(mode).strip().lower().replace("-", "_")
    if v in ("legacy", "frozen_hmd", "hmd0"):
        return "legacy"
    if v in ("torso", "torso_head", "head_relative_torso"):
        return "torso"
    if v in ("pelvis", "pelvis_head", "head_relative_pelvis"):
        return "pelvis"
    raise ValueError(
        "vr_ik_anchor_mode must be 'legacy', 'torso', or 'pelvis' "
        f"(aliases: frozen_hmd, head_relative_torso, …); got {mode!r}"
    )
