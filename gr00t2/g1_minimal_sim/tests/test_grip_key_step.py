from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hand_gripper import (  # noqa: E402
    GRIP_KEY_STEP_DEFAULT,
    apply_gripper_teleop_key,
    init_gripper_control_dict,
)


def test_init_gripper_sets_default_step() -> None:
    d: dict = {}
    init_gripper_control_dict(d)
    assert d["gripper_left"] == 1.0
    assert d["_grip_key_step"] == GRIP_KEY_STEP_DEFAULT


def test_incremental_close_uses_custom_step() -> None:
    d: dict = {}
    init_gripper_control_dict(d, grip_key_step=0.04)
    apply_gripper_teleop_key(d, "0")
    assert abs(float(d["gripper_left"]) - 0.96) < 1e-6
