#!/usr/bin/env python3
"""P0 — Print ``RobotModel`` joint-group sizes used by ``prepare_observation_for_eval``.

Matches the **UNITREE_G1** state slices in ``embodiment_configs.MODALITY_CONFIGS["unitree_g1"]``.
Use the same flags as ``rollout_policy.get_groot_locomanip_env_fn`` (**``enable_waist=True``** → pass ``--waist-ik``).

**Run** (needs ``gr00t_wbc`` on ``PYTHONPATH``; Isaac-GR00T WholeBodyControl venv is easiest)::

    ../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \\
      scripts/validate_gr00t_unitree_g1_shapes.py --waist-ik

From ``g1_minimal_sim/`` directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_STATE_GROUPS = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)


def _minimal_sim_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _wholebody_control_root() -> Path:
    return (
        _minimal_sim_root().parent
        / "Isaac-GR00T"
        / "external_dependencies"
        / "GR00T-WholeBodyControl"
    ).resolve()


def _ensure_gr00t_wbc_path() -> Path:
    root = _wholebody_control_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected GR00T-WholeBodyControl at {root} (sibling layout: gr00t2/Isaac-GR00T/...)."
        )
    sys.path.insert(0, str(root))
    return root


def validate_robot_model(*, enable_waist_ik: bool) -> dict[str, int]:
    """Return mapping state_key -> dim for ``prepare_observation_for_eval`` outputs."""
    _ensure_gr00t_wbc_path()

    from gr00t_wbc.control.robot_model.instantiation import get_robot_type_and_model
    from gr00t_wbc.control.utils.n1_utils import prepare_observation_for_eval

    _robot_type, robot_model = get_robot_type_and_model(
        "G1", enable_waist_ik=enable_waist_ik
    )
    q = np.zeros(robot_model.num_joints, dtype=np.float64)
    obs: dict = {"q": q}
    prepare_observation_for_eval(robot_model, obs)

    out: dict[str, int] = {}
    for g in _STATE_GROUPS:
        key = f"state.{g}"
        out[key] = int(np.asarray(obs[key]).shape[-1])
    out["num_joints_q"] = robot_model.num_joints
    out["num_dofs_pinocchio"] = robot_model.num_dofs
    return out


def _optional_mujoco_joint_count(use_hands: bool) -> tuple[int, int] | None:
    """Return ``(nq, nu)`` from merged stylish_diner MJCF if MuJoCo is available."""
    try:
        import mujoco
    except ImportError:
        return None

    sd = _minimal_sim_root() / "scenes" / "stylish_diner_1"
    xml_name = (
        "g1_gear_wbc_stylish_diner_hands.xml"
        if use_hands
        else "g1_gear_wbc_stylish_diner.xml"
    )
    xml_path = sd / xml_name
    if not xml_path.is_file():
        return None
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return int(model.nq), int(model.nu)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print UNITREE_G1 state slice dimensions from RobotModel + prepare_observation_for_eval."
    )
    parser.add_argument(
        "--waist-ik",
        action="store_true",
        help="Use enable_waist_ik=True (matches rollout_policy gym.make(..., enable_waist=True)).",
    )
    parser.add_argument(
        "--mujoco-nq",
        action="store_true",
        help="Also print MuJoCo nq/nu for stylish_diner (+ hands if --hands).",
    )
    parser.add_argument(
        "--hands",
        action="store_true",
        help="With --mujoco-nq, load g1_gear_wbc_stylish_diner_hands.xml.",
    )
    args = parser.parse_args()

    shapes = validate_robot_model(enable_waist_ik=args.waist_ik)
    tag = "enable_waist_ik=True" if args.waist_ik else "enable_waist_ik=False"
    print(f"RobotModel('G1', {tag}) — reference for GR00T unitree_g1 state keys")
    print(f"  num_joints (q vector for prepare_observation_for_eval): {shapes['num_joints_q']}")
    print(f"  num_dofs (pinocchio nq): {shapes['num_dofs_pinocchio']}")
    for g in _STATE_GROUPS:
        key = f"state.{g}"
        print(f"  {key}: {shapes[key]}")
    print("  action.navigate_command: 3")
    print("  action.base_height_command: 1")

    if args.mujoco_nq:
        mc = _optional_mujoco_joint_count(use_hands=args.hands)
        if mc is None:
            print("\nMuJoCo: skipped (mujoco not importable or config failed).")
        else:
            nq, nu = mc
            label = "stylish_diner_hands" if args.hands else "stylish_diner (welded)"
            print(f"\nMuJoCo {label}: nq={nq}, nu={nu} (full model incl. free joint + actuators)")

    print(
        "\nNext: map MuJoCo qpos → RobotModel q (num_joints) when wiring ObservationBuilder; "
        "nq includes floating base + joint coords — do not assume nq == num_joints."
    )


if __name__ == "__main__":
    main()
