"""Optional: pin RobotModel state slice sizes for UNITREE_G1 (needs WholeBodyControl on path)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("numpy")

_WBC = (
    _ROOT.parent
    / "Isaac-GR00T"
    / "external_dependencies"
    / "GR00T-WholeBodyControl"
).resolve()


def _load_validate_module():
    path = _ROOT / "scripts" / "validate_gr00t_unitree_g1_shapes.py"
    spec = importlib.util.spec_from_file_location(
        "validate_gr00t_unitree_g1_shapes", path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gr00t_unitree_g1_state_shapes_match_rollout_config() -> None:
    """Same RobotModel settings as ``rollout_policy.get_groot_locomanip_env_fn`` (enable_waist=True)."""
    if not _WBC.is_dir():
        pytest.skip(f"Missing GR00T-WholeBodyControl at {_WBC}")

    sys.path.insert(0, str(_WBC))
    pytest.importorskip("gr00t_wbc")

    mod = _load_validate_module()
    shapes = mod.validate_robot_model(enable_waist_ik=True)

    # Three-finger G1 bundle: legs 6+6, waist 3, arms 7+7, hands 7+7 (see g1_supplemental_info).
    assert shapes["state.left_leg"] == 6
    assert shapes["state.right_leg"] == 6
    assert shapes["state.waist"] == 3
    assert shapes["state.left_arm"] == 7
    assert shapes["state.right_arm"] == 7
    assert shapes["state.left_hand"] == 7
    assert shapes["state.right_hand"] == 7
