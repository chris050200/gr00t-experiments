"""Headless Gymnasium env: Gear WBC stands (pelvis height band)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")
import mujoco  # noqa: E402

from gear_wbc_stand import default_g1_resources_dir  # noqa: E402

from g1_gear_wbc_env import G1GearWBCEnv  # noqa: E402


def test_g1_gear_wbc_stylish_diner_env_headless_smoke() -> None:
    """Default merged diner scene loads and stands without falling through floor."""
    env = G1GearWBCEnv(enable_teleop=False, scene="stylish_diner")
    try:
        env.reset()
        for _ in range(800):
            env.step(np.zeros(1, dtype=np.float32))
        z = float(env.data.qpos[2])
        assert 0.55 < z < 0.95, f"pelvis z={z} m outside stand band (stylish_diner)"
    finally:
        env.close()


def test_g1_gear_wbc_table_pnp_env_pelvis_z_after_steps() -> None:
    rd = default_g1_resources_dir()
    if not (rd / "g1_gear_wbc_table.yaml").is_file():
        pytest.skip(f"table scene yaml not found: {rd}")
    env = G1GearWBCEnv(resources_dir=rd, enable_teleop=False, scene="table_pnp")
    try:
        env.reset()
        for _ in range(2000):
            env.step(np.zeros(1, dtype=np.float32))
        z = float(env.data.qpos[2])
        assert 0.60 < z < 0.88, f"pelvis z={z} m outside stand band (table scene)"
    finally:
        env.close()


def test_g1_gear_wbc_table_pnp_apple_env_headless_smoke() -> None:
    """B1 apple/plate MJCF under scenes/table_pnp_apple loads and stands."""
    scene_dir = _ROOT / "scenes" / "table_pnp_apple"
    if not (scene_dir / "g1_gear_wbc_table_pnp_apple.yaml").is_file():
        pytest.skip(f"table_pnp_apple yaml not found: {scene_dir}")
    rd = default_g1_resources_dir()
    if not (rd / "policy" / "ft92.onnx").is_file() and not (
        rd / "policy" / "GR00T-WholeBodyControl-Balance.onnx"
    ).is_file():
        pytest.skip(f"ONNX policy not found under {rd / 'policy'}")
    env = G1GearWBCEnv(enable_teleop=False, scene="table_pnp_apple")
    try:
        m = env.model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        assert "manip_apple" in names
        assert "manip_plate" in names
        assert "dc_table_pick" in names
        assert "dc_table_target" in names
        env.reset()
        for _ in range(2000):
            env.step(np.zeros(1, dtype=np.float32))
        z = float(env.data.qpos[2])
        assert 0.60 < z < 0.88, f"pelvis z={z} m outside stand band (table_pnp_apple)"
    finally:
        env.close()


def test_g1_gear_wbc_env_pelvis_z_after_steps() -> None:
    rd = default_g1_resources_dir()
    if not (rd / "g1_gear_wbc.yaml").is_file():
        pytest.skip(f"G1 resources not found: {rd}")
    env = G1GearWBCEnv(resources_dir=rd, enable_teleop=False, scene="floor")
    try:
        env.reset()
        for _ in range(2000):
            env.step(np.zeros(1, dtype=np.float32))
        z = float(env.data.qpos[2])
        assert 0.65 < z < 0.85, f"pelvis z={z} m outside expected stand band"
    finally:
        env.close()
