"""Unit tests for ``vr_teleop.openvr_stream`` stick → ``loco_cmd`` mapping."""

from __future__ import annotations

import numpy as np

from vr_teleop.openvr_stream import stream_state_from_packet, sticks_to_loco_cmd


def test_stream_state_from_packet_minimal() -> None:
    pkt = {
        "v": 1,
        "seq": 3,
        "ts_mono_s": 1.0,
        "hmd": {"valid": False},
        "left": {"valid": False},
        "right": {"valid": False},
        "inputs": {
            "left": {"valid": True, "stick_xy": [0.5, -1.0], "trigger": 0.0},
            "right": {"valid": True, "stick_xy": [0.25, 0.0], "trigger": 0.0},
        },
    }
    st = stream_state_from_packet(pkt)
    assert st is not None
    assert st.seq == 3
    np.testing.assert_allclose(st.left_stick_xy, [0.5, -1.0])
    np.testing.assert_allclose(st.right_stick_xy, [0.25, 0.0])


def test_stream_state_wrong_version() -> None:
    assert stream_state_from_packet({"v": 2}) is None


def test_sticks_to_loco_cmd_forward_strafe_yaw() -> None:
    left = np.array([0.0, 1.0], dtype=np.float64)
    right = np.array([1.0, 0.0], dtype=np.float64)
    lc = sticks_to_loco_cmd(left, right, deadzone=0.0)
    assert lc.shape == (3,)
    np.testing.assert_allclose(
        lc,
        np.array([0.75, 0.75, 0.0], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_sticks_deadzone() -> None:
    left = np.array([0.05, 0.05], dtype=np.float64)
    right = np.array([0.0, 0.0], dtype=np.float64)
    lc = sticks_to_loco_cmd(left, right, deadzone=0.12)
    np.testing.assert_allclose(lc, np.zeros(3, dtype=np.float32), atol=1e-6)
