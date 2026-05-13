"""Smoke-load standalone RoboCasa-derived ``lab_dc_world.xml`` (no G1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("mujoco")

import mujoco  # noqa: E402


def test_lab_dc_world_xml_loads() -> None:
    xml = _ROOT / "scenes" / "lab_dc_layout" / "lab_dc_world.xml"
    if not xml.is_file():
        pytest.skip(f"missing {xml} (run scenes/lab_dc_layout/gen_lab_dc_world_xml.py)")
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert m.nq >= 7
    assert m.nbody >= 9
    # Apple free body + plate + two tables + world + inner bodies
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    assert "dc_apple" in names
    assert "dc_plate" in names
    assert "dc_table_pick" in names
    assert "dc_table_target" in names
