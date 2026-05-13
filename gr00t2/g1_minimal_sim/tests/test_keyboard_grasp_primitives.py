"""Unit tests for optional keyboard symmetric EE primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from keyboard_grasp_primitives import (  # noqa: E402
    _MAX_OFFSET_FROM_HOME_M,
    _STEP_FORWARD_M,
    _STEP_LATERAL_M,
    _STEP_VERTICAL_M,
    apply_keyboard_grasp_primitive,
)
from gear_wbc_teleop import _LOCO_CHARS  # noqa: E402


def test_apply_x_widens_y_and_c_narrows() -> None:
    cd: dict = {
        "ee_left_pos": np.zeros(3, dtype=np.float64),
        "ee_right_pos": np.zeros(3, dtype=np.float64),
        "_ee_left_pos_home": np.zeros(3, dtype=np.float64),
        "_ee_right_pos_home": np.zeros(3, dtype=np.float64),
    }
    assert apply_keyboard_grasp_primitive(cd, "x") is True
    assert float(cd["ee_left_pos"][1]) == pytest.approx(_STEP_LATERAL_M)
    assert float(cd["ee_right_pos"][1]) == pytest.approx(-_STEP_LATERAL_M)
    assert apply_keyboard_grasp_primitive(cd, "c") is True
    assert float(cd["ee_left_pos"][1]) == pytest.approx(0.0)
    assert float(cd["ee_right_pos"][1]) == pytest.approx(0.0)


def test_close_key_is_not_locomotion() -> None:
    assert "c" not in _LOCO_CHARS
    assert set(_LOCO_CHARS) == set("wsadqe")


def test_clamps_to_max_offset_from_home() -> None:
    home = np.array([0.0, 0.0, 0.3], dtype=np.float64)
    cd: dict = {
        "ee_left_pos": home.copy(),
        "ee_right_pos": home.copy(),
        "_ee_left_pos_home": home.copy(),
        "_ee_right_pos_home": home.copy(),
    }
    n = int(np.ceil(_MAX_OFFSET_FROM_HOME_M / _STEP_LATERAL_M)) + 4
    for _ in range(n):
        apply_keyboard_grasp_primitive(cd, "x")
    assert float(np.abs(cd["ee_left_pos"][1] - home[1])) <= _MAX_OFFSET_FROM_HOME_M + 1e-6
    assert float(np.abs(cd["ee_right_pos"][1] - home[1])) <= _MAX_OFFSET_FROM_HOME_M + 1e-6


def test_page_up_down_vertical() -> None:
    cd: dict = {
        "ee_left_pos": np.array([0.0, 0.0, 0.2], dtype=np.float64),
        "ee_right_pos": np.array([0.0, 0.0, 0.2], dtype=np.float64),
        "_ee_left_pos_home": np.array([0.0, 0.0, 0.2], dtype=np.float64),
        "_ee_right_pos_home": np.array([0.0, 0.0, 0.2], dtype=np.float64),
    }
    assert apply_keyboard_grasp_primitive(cd, "page_up") is True
    assert float(cd["ee_left_pos"][2]) == pytest.approx(0.2 + _STEP_VERTICAL_M)
    assert apply_keyboard_grasp_primitive(cd, "page_down") is True
    assert float(cd["ee_left_pos"][2]) == pytest.approx(0.2)


def test_home_end_forward_backward() -> None:
    cd: dict = {
        "ee_left_pos": np.array([0.1, 0.0, 0.2], dtype=np.float64),
        "ee_right_pos": np.array([0.1, 0.0, 0.2], dtype=np.float64),
        "_ee_left_pos_home": np.array([0.1, 0.0, 0.2], dtype=np.float64),
        "_ee_right_pos_home": np.array([0.1, 0.0, 0.2], dtype=np.float64),
    }
    assert apply_keyboard_grasp_primitive(cd, "home") is True
    assert float(cd["ee_left_pos"][0]) == pytest.approx(0.1 + _STEP_FORWARD_M)
    assert float(cd["ee_right_pos"][0]) == pytest.approx(0.1 + _STEP_FORWARD_M)
    assert apply_keyboard_grasp_primitive(cd, "end") is True
    assert float(cd["ee_left_pos"][0]) == pytest.approx(0.1)
    assert float(cd["ee_right_pos"][0]) == pytest.approx(0.1)


def test_returns_false_without_ee() -> None:
    assert apply_keyboard_grasp_primitive({"loco_cmd": [0, 0, 0]}, "x") is False
