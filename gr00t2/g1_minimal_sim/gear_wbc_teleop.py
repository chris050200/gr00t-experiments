"""Keyboard teleop: locomotion (hold-to-move), height / rpy keys (pynput listener)."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np

from arm_ik import apply_ik_arm_teleop_key, reset_ee_to_home
from gear_wbc_vr_stream import VR_STREAM_STALE_S
from hand_gripper import apply_gripper_teleop_key, reset_gripper_control_dict
from keyboard_grasp_primitives import (
    apply_keyboard_grasp_primitive,
    primitive_key_token,
)

# Locomotion keys: hold sets ``loco_cmd`` each physics step (same [fwd, strafe, yaw] as VR sticks).
_LOCO_CHARS: frozenset[str] = frozenset("wsadqe")
_loco_pressed: set[str] = set()
_teleop_print_lock = threading.Lock()
_teleop_last_print_mono: float = 0.0
_TELEOP_PRINT_INTERVAL_S = 0.35

# Same-token grasp primitives only: suppresses stacked moves from OS key-repeat on PageUp/Home/x/etc.
_GRASP_PRIMITIVE_DEBOUNCE_S = 0.15
_grasp_primitive_last_mono: dict[str, float] = {}


def _press_char(key: object) -> str | None:
    """Lowercase letter for KeyCode; ``None`` for modifiers / special keys."""
    ch = getattr(key, "char", None)
    if ch is None or len(ch) != 1:
        return None
    return ch.lower()


def _vr_stream_fresh(rt: Any, vr_t_rx: float) -> bool:
    if rt._vr_teleop_receiver is None or vr_t_rx <= 0.0:
        return False
    return (time.monotonic() - vr_t_rx) < VR_STREAM_STALE_S


def sync_keyboard_loco_hold(rt: Any, vr_snap: Any, vr_t_rx: float) -> None:
    """Set ``loco_cmd`` from held W/S/A/D/Q/E after VR stream; call each ``step_physics``.

    - **Hold** W/S → forward/back, A/D → strafe, Q/E → yaw (matches ``sticks_to_loco_cmd``).
    - Release all locomotion keys → ``cmd_init`` (stand branch).
    - If VR stream is **fresh** and **no** locomotion keys are held, VR keeps ``loco_cmd``.
    - If VR is fresh and user **holds** locomotion keys, keyboard overrides VR for locomotion.

    Caller must not hold ``rt.cmd_lock`` (same lock as the pynput listener).
    """
    if not rt.teleop:
        return
    with rt.cmd_lock:
        keys_active = bool(_loco_pressed.intersection(_LOCO_CHARS))
    if _vr_stream_fresh(rt, vr_t_rx) and not keys_active:
        return
    cmd_init = np.asarray(rt.config["cmd_init"], dtype=np.float64).reshape(3)
    fwd_scale, strafe_scale, yaw_scale = 0.75, 0.75, 0.55
    with rt.cmd_lock:
        if not _loco_pressed.intersection(_LOCO_CHARS):
            rt.control_dict["loco_cmd"][:] = cmd_init
            return
        fy = float(("w" in _loco_pressed) - ("s" in _loco_pressed))
        fx = float(("a" in _loco_pressed) - ("d" in _loco_pressed))
        lyaw = float(("q" in _loco_pressed) - ("e" in _loco_pressed))
        rt.control_dict["loco_cmd"][:] = np.array(
            [fwd_scale * fy, strafe_scale * fx, yaw_scale * lyaw],
            dtype=np.float64,
        )


def apply_mujoco_gear_wbc_key(
    control_dict: dict[str, Any], config: dict[str, Any], k: str
) -> None:
    """Apply one keypress for **non-locomotion** body keys (height, rpy, freq, reset).

    W/S/A/D/Q/E locomotion is handled by :func:`sync_keyboard_loco_hold` from held keys.

    **Locking:** when ``k == "z"``, clears ``_loco_pressed``; caller must hold ``cmd_lock``.
    """
    if k == "z":
        control_dict["loco_cmd"][:] = config["cmd_init"]
        control_dict["height_cmd"] = float(config["height_cmd"])
        control_dict["rpy_cmd"][:] = config["rpy_cmd"]
        control_dict["freq_cmd"] = float(config.get("freq_cmd", 0.75))
        _loco_pressed.clear()
        if "arm_target_q" in control_dict and "_arm_target_q_home" in control_dict:
            control_dict["arm_target_q"][:] = control_dict["_arm_target_q_home"]
        reset_ee_to_home(control_dict)
        reset_gripper_control_dict(control_dict)
    elif k == "1":
        control_dict["height_cmd"] += 0.05
    elif k == "2":
        control_dict["height_cmd"] -= 0.05
    elif k == "3":
        control_dict["rpy_cmd"][0] += 0.2
    elif k == "4":
        control_dict["rpy_cmd"][0] -= 0.2
    elif k == "5":
        control_dict["rpy_cmd"][1] += 0.2
    elif k == "6":
        control_dict["rpy_cmd"][1] -= 0.2
    elif k == "7":
        control_dict["rpy_cmd"][2] += 0.2
    elif k == "8":
        control_dict["rpy_cmd"][2] -= 0.2
    elif k == "m":
        control_dict["freq_cmd"] += 0.1
    elif k == "n":
        control_dict["freq_cmd"] -= 0.1


def _maybe_print_teleop_status(
    control_dict: dict[str, Any],
    *,
    arm_tau_provider: Callable[[], str | None] | None = None,
    grip_force_provider: Callable[[], str | None] | None = None,
) -> None:
    global _teleop_last_print_mono
    now = time.monotonic()
    with _teleop_print_lock:
        if now - _teleop_last_print_mono < _TELEOP_PRINT_INTERVAL_S:
            return
        _teleop_last_print_mono = now
    if "ee_left_pos" in control_dict:
        elp = np.asarray(control_dict["ee_left_pos"])
        erp = np.asarray(control_dict["ee_right_pos"])
        elq = np.asarray(control_dict["ee_left_quat"])
        erq = np.asarray(control_dict["ee_right_quat"])
        gbit = ""
        if "gripper_left" in control_dict:
            gbit = (
                f" | grip L={float(control_dict['gripper_left']):.2f} "
                f"R={float(control_dict['gripper_right']):.2f}"
            )
        msg = (
            "Commands: "
            f"loco_cmd={control_dict['loco_cmd']}, height={control_dict['height_cmd']:.2f}, "
            f"rpy_cmd={control_dict['rpy_cmd']}, freq={control_dict['freq_cmd']:.2f} | "
            f"ee_L(torso) pos={np.round(elp, 3)} quat_wxyz={np.round(elq, 3)} | "
            f"ee_R(torso) pos={np.round(erp, 3)} quat_wxyz={np.round(erq, 3)}"
            f"{gbit}"
        )
    else:
        msg = (
            "Current Commands: "
            f"loco_cmd = {control_dict['loco_cmd']}, "
            f"height_cmd = {control_dict['height_cmd']}, "
            f"rpy_cmd = {control_dict['rpy_cmd']}, "
            f"freq_cmd = {control_dict['freq_cmd']}"
        )
        if "arm_target_q" in control_dict:
            msg += f", arm_target_q = {control_dict['arm_target_q']}"
    if arm_tau_provider is not None:
        try:
            tau_msg = arm_tau_provider()
        except Exception:  # noqa: BLE001 — never let status print kill the listener
            tau_msg = None
        if tau_msg:
            msg = f"{msg}\n  {tau_msg}"
    if grip_force_provider is not None:
        try:
            grip_msg = grip_force_provider()
        except Exception:  # noqa: BLE001
            grip_msg = None
        if grip_msg:
            msg = f"{msg}\n  {grip_msg}"
    print(msg)


def start_gear_wbc_teleop_listener(
    control_dict: dict[str, Any],
    config: dict[str, Any],
    cmd_lock: threading.Lock,
    *,
    keyboard_grasp_primitives: bool = False,
    on_teleop_reset: Callable[[], None] | None = None,
    arm_tau_provider: Callable[[], str | None] | None = None,
    grip_force_provider: Callable[[], str | None] | None = None,
) -> None:
    try:
        import pynput.keyboard as pkb
    except ImportError as e:
        raise ImportError("Teleop requires pynput: pip install pynput") from e

    with cmd_lock:
        _loco_pressed.clear()
    _grasp_primitive_last_mono.clear()

    def on_press(key: object) -> None:
        k = _press_char(key)
        with cmd_lock:
            if k is not None and k in _LOCO_CHARS:
                _loco_pressed.add(k)
                return
        if keyboard_grasp_primitives:
            tok = primitive_key_token(key)
            if tok is not None:
                now = time.monotonic()
                t_prev = _grasp_primitive_last_mono.get(tok, -1e9)
                if now - t_prev < _GRASP_PRIMITIVE_DEBOUNCE_S:
                    return
                _grasp_primitive_last_mono[tok] = now
                with cmd_lock:
                    applied = apply_keyboard_grasp_primitive(control_dict, tok)
                if applied:
                    _maybe_print_teleop_status(
                        control_dict,
                        arm_tau_provider=arm_tau_provider,
                        grip_force_provider=grip_force_provider,
                    )
                    return
        if k is None:
            return
        with cmd_lock:
            apply_mujoco_gear_wbc_key(control_dict, config, k)
            apply_ik_arm_teleop_key(control_dict, k)
            apply_gripper_teleop_key(control_dict, k)
        if k == "z" and on_teleop_reset is not None:
            on_teleop_reset()
        _maybe_print_teleop_status(
            control_dict,
            arm_tau_provider=arm_tau_provider,
            grip_force_provider=grip_force_provider,
        )

    def on_release(key: object) -> None:
        k = _press_char(key)
        if k is None or k not in _LOCO_CHARS:
            return
        with cmd_lock:
            _loco_pressed.discard(k)

    listener = pkb.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
