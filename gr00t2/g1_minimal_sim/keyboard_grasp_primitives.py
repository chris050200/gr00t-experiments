"""Optional keyboard-only symmetric EE nudges (torso frame) for box-style grasping.

Enabled only when :func:`gear_wbc_teleop.start_gear_wbc_teleop_listener` is started with
``keyboard_grasp_primitives=True`` (CLI: ``--keyboard-grasp-primitives`` with ``--teleop``).

Keys (do not change ``ee_*_quat`` or grippers — only ``ee_left_pos`` / ``ee_right_pos``):

- **x** — widen: palms move apart along torso **+Y** (left +step, right −step).
- **c** — narrow: opposite of **x**.
- **Home** / **End** — both palms move **+X** / **−X** in torso frame (forward / backward).
- **Page Up** / **Page Down** — both palms move **+Z** / **−Z** in torso frame (raise / lower).

Step sizes are intentionally small so quick taps are usable while carrying a box. Positions are
clamped to ``±max_offset`` from ``_ee_*_pos_home`` per axis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Per-tap increments (torso frame). Kept small so single taps + OS key-repeat stay usable.
_STEP_FORWARD_M = 0.02
_STEP_LATERAL_M = 0.02
_STEP_VERTICAL_M = 0.0125
_MAX_OFFSET_FROM_HOME_M = 0.35


def primitive_key_token(key: object) -> str | None:
    """Map pynput key to a primitive token, or ``None``."""
    try:
        from pynput.keyboard import Key
    except ImportError:
        return None
    if key == Key.page_up:
        return "page_up"
    if key == Key.page_down:
        return "page_down"
    if key == Key.home:
        return "home"
    if key == Key.end:
        return "end"
    ch = getattr(key, "char", None)
    if ch is None or len(ch) != 1:
        return None
    c = ch.lower()
    if c in ("x", "c"):
        return c
    return None


def _clip_pair_to_home(control_dict: dict[str, Any]) -> None:
    for side in ("left", "right"):
        pk = f"ee_{side}_pos"
        hk = f"_ee_{side}_pos_home"
        p = np.asarray(control_dict[pk], dtype=np.float64).reshape(3)
        h = np.asarray(control_dict[hk], dtype=np.float64).reshape(3)
        lo = h - _MAX_OFFSET_FROM_HOME_M
        hi = h + _MAX_OFFSET_FROM_HOME_M
        control_dict[pk][:] = np.clip(p, lo, hi)


def apply_keyboard_grasp_primitive(control_dict: dict[str, Any], token: str) -> bool:
    """Apply one primitive if ``token`` is known and EE homes exist. Returns **True** if applied."""
    if token not in ("x", "c", "home", "end", "page_up", "page_down"):
        return False
    if "ee_left_pos" not in control_dict or "_ee_left_pos_home" not in control_dict:
        return False
    if "ee_right_pos" not in control_dict or "_ee_right_pos_home" not in control_dict:
        return False

    el = np.asarray(control_dict["ee_left_pos"], dtype=np.float64).reshape(3)
    er = np.asarray(control_dict["ee_right_pos"], dtype=np.float64).reshape(3)

    if token == "x":
        el[1] += _STEP_LATERAL_M
        er[1] -= _STEP_LATERAL_M
    elif token == "c":
        el[1] -= _STEP_LATERAL_M
        er[1] += _STEP_LATERAL_M
    elif token == "home":
        el[0] += _STEP_FORWARD_M
        er[0] += _STEP_FORWARD_M
    elif token == "end":
        el[0] -= _STEP_FORWARD_M
        er[0] -= _STEP_FORWARD_M
    elif token == "page_up":
        el[2] += _STEP_VERTICAL_M
        er[2] += _STEP_VERTICAL_M
    else:  # page_down
        el[2] -= _STEP_VERTICAL_M
        er[2] -= _STEP_VERTICAL_M

    control_dict["ee_left_pos"][:] = el
    control_dict["ee_right_pos"][:] = er
    _clip_pair_to_home(control_dict)
    return True
