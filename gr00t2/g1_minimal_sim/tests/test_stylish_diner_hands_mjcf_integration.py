"""MJCF + config integration for stylish_diner with articulated hands.

Checks that ``resolve_gear_wbc_config(..., scene='stylish_diner', use_hands=True)``
points at the merged ``g1_gear_wbc_stylish_diner_hands.{xml,yaml}``, that
``GearWBCRuntime`` loads, finger joints exist, kinematic grasp-assist state has
not reappeared, and ``target_block`` satisfies coarse structural invariants.

This is **not** a manipulation or grasp-quality test — teleop / VR sessions remain
the canonical way to validate contact behavior. For stand-in-clutter via the
Gymnasium env API, see ``tests/test_g1_gear_wbc_env_headless.py``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gear_wbc_config import resolve_gear_wbc_config  # noqa: E402
from gear_wbc_stand import GearWBCRuntime  # noqa: E402


def test_resolve_stylish_diner_hands_config_files_exist() -> None:
    rd, ycfg = resolve_gear_wbc_config(None, scene="stylish_diner", use_hands=True)
    assert ycfg == "g1_gear_wbc_stylish_diner_hands.yaml"
    assert (rd / ycfg).is_file()
    assert (rd / "g1_gear_wbc_stylish_diner_hands.xml").is_file()


def test_stylish_diner_hands_runtime_mjcf_invariants_and_steps() -> None:
    rd, ycfg = resolve_gear_wbc_config(None, scene="stylish_diner", use_hands=True)
    rt = GearWBCRuntime(
        rd,
        teleop=False,
        config_yaml=ycfg,
    )

    for jn in (
        "left_hand_thumb_2_joint",
        "right_hand_thumb_2_joint",
        "left_hand_index_1_joint",
        "right_hand_middle_1_joint",
    ):
        assert mujoco.mj_name2id(rt.model, mujoco.mjtObj.mjOBJ_JOINT, jn) >= 0, (
            f"diner+hands MJCF missing finger joint {jn}"
        )

    assert not hasattr(rt, "_grasp_assist_active")
    assert not hasattr(rt, "_grasp_assist_body_id")

    bid = mujoco.mj_name2id(rt.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    assert bid >= 0
    gid_start = int(rt.model.body_geomadr[bid])
    half = np.asarray(rt.model.geom_size[gid_start, :3], dtype=np.float64)
    assert np.allclose(half[0], half[1]) and np.allclose(half[0], half[2]), (
        f"target_block expected to be a cube; got half-sizes {half}"
    )
    assert 0.05 <= float(half[0]) <= 0.20, (
        f"target_block half-size {half[0]:.3f} m outside sane prototype band [0.05, 0.20]"
    )
    vol = float(np.prod(2.0 * half))
    mass = float(rt.model.body_mass[bid])
    assert mass > 0.0
    inferred_density = mass / vol
    assert 1.0 <= inferred_density <= 2000.0, (
        f"target_block inferred density {inferred_density:.2f} kg/m^3 outside sane band"
    )

    for _ in range(20):
        rt.step_physics()
    z = float(rt.data.xpos[bid, 2])
    assert 0.5 < z < 1.5, f"target_block z={z:.3f} outside expected resting range"
