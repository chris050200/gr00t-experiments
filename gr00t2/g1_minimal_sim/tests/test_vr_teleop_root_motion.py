"""Pure-numpy helpers for receive-side root motion (no GLFW / MuJoCo)."""

from __future__ import annotations

import numpy as np

from legacy_vr_code.vr_teleop_root_kinematics import (
    Rz,
    apply_axis_deadzone,
    compose_root_planar_to_world,
    integrate_root_planar,
    world_pose_to_root_planar,
)


def test_Rz_90() -> None:
    R = Rz(np.pi / 2)
    v = np.array([1.0, 0.0, 0.0])
    assert np.allclose(R @ v, [0.0, 1.0, 0.0], atol=1e-9)


def test_deadzone() -> None:
    assert apply_axis_deadzone(0.0, 0.2) == 0.0
    # Remap outside deadzone to full range: (0.5 - 0.2) / (1 - 0.2) = 0.375
    assert abs(apply_axis_deadzone(0.5, 0.2) - 0.375) < 1e-9


def test_integrate_forward() -> None:
    x, y, yaw = 0.0, 0.0, 0.0
    x, y, yaw = integrate_root_planar(
        x,
        y,
        yaw,
        0.0,
        -1.0,
        0.0,
        dt=0.1,
        move_scale=2.0,
        yaw_left_scale=1.0,
        yaw_right_scale=1.0,
        invert_forward=False,
    )
    # GLFW left Y up = -1 → we treat as forward → +X at yaw=0
    assert abs(x - 0.2) < 1e-9 and abs(y) < 1e-9


def test_world_to_root_planar() -> None:
    t = np.array([1.0, 0.0, 0.0])
    yaw = 0.0
    p_w = np.array([2.0, 0.0, 1.0])
    p_d, _ = world_pose_to_root_planar(p_w, np.eye(3), t, yaw)
    assert np.allclose(p_d, [1.0, 0.0, 1.0])


def test_compose_roundtrip_with_world_to_root() -> None:
    t = np.array([0.5, -0.25, 0.05])
    yaw = 0.7
    p_w0 = np.array([1.2, -0.1, 1.35])
    R0 = Rz(0.3) @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    p_d, R_d = world_pose_to_root_planar(p_w0, R0, t, yaw)
    p_w1, R1 = compose_root_planar_to_world(p_d, R_d, t, yaw)
    assert np.allclose(p_w1, p_w0, atol=1e-9)
    assert np.allclose(R1, R0, atol=1e-9)
