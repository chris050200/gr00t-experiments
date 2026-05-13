from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hand_gripper  # noqa: E402
from hand_gripper import apply_gripper_teleop_key  # noqa: E402


def test_hard_close_and_open_both_grippers() -> None:
    control_dict = {"gripper_left": 0.7, "gripper_right": 0.4}

    apply_gripper_teleop_key(control_dict, "h")
    assert control_dict["gripper_left"] == 0.0
    assert control_dict["gripper_right"] == 0.0

    apply_gripper_teleop_key(control_dict, "y")
    assert control_dict["gripper_left"] == 1.0
    assert control_dict["gripper_right"] == 1.0


def test_hy_noop_without_articulated_grippers() -> None:
    """Without articulated grippers h/y must NOT trigger any cheat assist key."""
    hand_gripper._HY_NO_HANDS_WARNED = False
    control_dict: dict = {}

    apply_gripper_teleop_key(control_dict, "h")
    apply_gripper_teleop_key(control_dict, "y")

    # No keys should have been added — physics-only contract.
    assert control_dict == {}, f"unexpected control_dict mutations: {control_dict!r}"
