"""Gear WBC yaml loading and MJCF/resources path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np


POLICY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _resolve_onnx_file(configured_abs: str, fallbacks: tuple[str, ...]) -> str:
    """Use yaml path if present; otherwise try alternate filenames in the same directory."""
    p = Path(configured_abs)
    if p.is_file():
        return str(p)
    for name in fallbacks:
        alt = p.parent / name
        if alt.is_file():
            return str(alt)
    return str(p)


def _policy_joint_addrs(
    model: mujoco.MjModel, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    qadr = []
    dadr = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Policy joint not in MJCF: {name}")
        qadr.append(int(model.jnt_qposadr[jid]))
        dadr.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qadr, dtype=np.int32), np.asarray(dadr, dtype=np.int32)


def load_gear_wbc_config(
    resources_dir: Path, *, yaml_name: str = "g1_gear_wbc.yaml"
) -> dict[str, Any]:
    resources_dir = resources_dir.resolve()
    yaml_path = resources_dir / yaml_name
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing {yaml_path}")

    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Optional: load ONNX from the default vendored G1 bundle while MJCF lives elsewhere
    # (e.g. ``scenes/table_pnp_apple/*.yaml`` sets ``policy_bundle_dir: "__DEFAULT_G1_RESOURCES__"``).
    bundle_key = raw.pop("policy_bundle_dir", None)
    if bundle_key is None:
        policy_root: Path = resources_dir
    elif bundle_key == "__DEFAULT_G1_RESOURCES__":
        policy_root = default_g1_resources_dir()
    else:
        bp = Path(str(bundle_key))
        policy_root = bp.resolve() if bp.is_absolute() else (resources_dir / bp).resolve()

    raw["policy_path"] = str((policy_root / Path(raw["policy_path"])).resolve())
    raw["walk_policy_path"] = str((policy_root / Path(raw["walk_policy_path"])).resolve())
    raw["xml_path"] = str((resources_dir / Path(raw["xml_path"])).resolve())

    raw["policy_path"] = _resolve_onnx_file(
        raw["policy_path"],
        ("ft92.onnx", "GR00T-WholeBodyControl-Balance.onnx"),
    )
    raw["walk_policy_path"] = _resolve_onnx_file(
        raw["walk_policy_path"],
        ("ft109.onnx", "GR00T-WholeBodyControl-Walk.onnx"),
    )

    for key in ("kps", "kds", "default_angles", "cmd_scale", "cmd_init", "rpy_cmd"):
        if key in raw:
            raw[key] = np.asarray(raw[key], dtype=np.float32)
    return raw


def default_g1_resources_dir() -> Path:
    """Parent directory of ``g1_gear_wbc.xml`` (contains yaml, policy, meshes)."""
    here = Path(__file__).resolve().parent
    isaac_groot = here.parent / "Isaac-GR00T"
    return (
        isaac_groot
        / "external_dependencies"
        / "GR00T-WholeBodyControl"
        / "gr00t_wbc"
        / "sim2mujoco"
        / "resources"
        / "robots"
        / "g1"
    ).resolve()


def resolve_gear_wbc_config(
    resources_dir: Path | str | None,
    *,
    use_hands: bool = False,
    scene: str = "stylish_diner",
    config_yaml: str | Path | None = None,
) -> tuple[Path, str]:
    """Pick ``(resources_dir, yaml_basename)`` for :class:`gear_wbc_stand.GearWBCRuntime`.

    - If ``config_yaml`` is an absolute path to a file, its parent is ``resources_dir``.
    - Otherwise ``resources_dir`` defaults to :func:`default_g1_resources_dir` or the
      given directory; ``config_yaml`` is a basename in that folder when set.
    - ``scene`` is ``"stylish_diner"`` (default, merged diner MJCF), ``"floor"``,
      ``"table_pnp"`` (vendored table + blue ``table_box``), or ``"table_pnp_apple"``
      (B1: in-repo MJCF with red apple + plate; see ``scenes/table_pnp_apple/``).
    """
    if config_yaml is not None:
        p = Path(config_yaml)
        if p.is_absolute():
            p = p.resolve()
            if not p.is_file():
                raise FileNotFoundError(f"config yaml not found: {p}")
            return p.parent, p.name
        yaml_name = p.name
        rd = (
            Path(resources_dir).resolve()
            if resources_dir is not None
            else default_g1_resources_dir()
        )
        return rd, yaml_name

    if scene == "stylish_diner":
        sd = Path(__file__).resolve().parent / "scenes" / "stylish_diner_1"
        if use_hands:
            yaml_name = "g1_gear_wbc_stylish_diner_hands.yaml"
            xml_name = "g1_gear_wbc_stylish_diner_hands.xml"
        else:
            yaml_name = "g1_gear_wbc_stylish_diner.yaml"
            xml_name = "g1_gear_wbc_stylish_diner.xml"
        ypath = sd / yaml_name
        if not ypath.is_file():
            raise FileNotFoundError(f"stylish_diner config not found: {ypath}")
        xpath = sd / xml_name
        if not xpath.is_file():
            raise FileNotFoundError(f"stylish_diner MJCF not found: {xpath}")
        return sd.resolve(), yaml_name

    rd = (
        Path(resources_dir).resolve()
        if resources_dir is not None
        else default_g1_resources_dir()
    )
    if scene == "table_pnp":
        y = "g1_gear_wbc_hands_table.yaml" if use_hands else "g1_gear_wbc_table.yaml"
    elif scene == "floor":
        y = "g1_gear_wbc_hands.yaml" if use_hands else "g1_gear_wbc.yaml"
    elif scene == "table_pnp_apple":
        here = Path(__file__).resolve().parent
        sd = here / "scenes" / "table_pnp_apple"
        yaml_name = (
            "g1_gear_wbc_hands_table_pnp_apple.yaml"
            if use_hands
            else "g1_gear_wbc_table_pnp_apple.yaml"
        )
        ypath = sd / yaml_name
        if not ypath.is_file():
            raise FileNotFoundError(f"table_pnp_apple config not found: {ypath}")
        return sd.resolve(), yaml_name
    else:
        raise ValueError(
            f"unknown scene {scene!r}; expected 'stylish_diner', 'floor', 'table_pnp', "
            "or 'table_pnp_apple'"
        )
    return rd, y
