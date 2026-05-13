"""Smoke: GR00T-shaped **state** obs from stylish_diner hands MJCF (no OpenGL).

Ego RGB is implemented in :mod:`gr00t_observation_builder` via ``mujoco.Renderer``; verify
interactively on a machine with ``DISPLAY`` / EGL / OSMesa (headless CI skips rendering).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("mujoco")

_WBC = (
    _ROOT.parent
    / "Isaac-GR00T"
    / "external_dependencies"
    / "GR00T-WholeBodyControl"
).resolve()


def _load_hands_model_data():
    import mujoco

    xml_path = (
        _ROOT / "scenes" / "stylish_diner_1" / "g1_gear_wbc_stylish_diner_hands.xml"
    )
    if not xml_path.is_file():
        pytest.skip(f"Missing MJCF {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_gr00t_observation_builder_state_shapes_headless() -> None:
    """No ``mujoco.Renderer`` — safe on headless CI."""
    if not _WBC.is_dir():
        pytest.skip(f"Missing GR00T-WholeBodyControl at {_WBC}")

    sys.path.insert(0, str(_WBC))
    pytest.importorskip("gr00t_wbc")

    from gr00t_observation_builder import Gr00tObservationBuilder

    model, data = _load_hands_model_data()

    b = Gr00tObservationBuilder.for_locomanip_default()
    try:
        obs = b.build(
            model,
            data,
            task_description="test instruction",
            include_video=False,
        )
    finally:
        b.close()

    assert obs["state.left_leg"].shape == (6,)
    assert obs["state.right_leg"].shape == (6,)
    assert obs["state.waist"].shape == (3,)
    assert obs["state.left_arm"].shape == (7,)
    assert obs["state.right_arm"].shape == (7,)
    assert obs["state.left_hand"].shape == (7,)
    assert obs["state.right_hand"].shape == (7,)
    assert obs["annotation.human.task_description"] == "test instruction"
    assert "ego_view_image" not in obs

