"""Smoke tests for ``arm_ik_v2.solve_dual_arm_ik_v2`` on bundled G1 MJCF.

Regression goal (May 2026): pin the fix where ``mj_jac`` was being called with the
body-local palm offset as ``point`` (instead of palm world position). That bug made the
position Jacobian's lever arm grow with ``||data.xpos[wrist]||``, so a base translated
or yawed away from origin used a wildly wrong descent direction. Tests below exercise
both the home pose and a translated/yawed base so a regression cannot pass silently.
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

from arm_ik import body_palm_pose, build_g1_arm_ik_specs, ee_world_targets_for_ik  # noqa: E402
from arm_ik_v2 import solve_dual_arm_ik_v2  # noqa: E402


def _set_free_base(data, xy, yaw_rad: float, z: float = 0.74) -> None:
    """G1 free joint qpos[0:7] = (x, y, z, qw, qx, qy, qz)."""
    data.qpos[0] = float(xy[0])
    data.qpos[1] = float(xy[1])
    data.qpos[2] = float(z)
    data.qpos[3] = float(np.cos(yaw_rad * 0.5))
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = float(np.sin(yaw_rad * 0.5))


def _default_g1_resources_dir() -> Path:
    here = Path(__file__).resolve().parents[1]
    return (
        here.parent
        / "Isaac-GR00T"
        / "external_dependencies"
        / "GR00T-WholeBodyControl"
        / "gr00t_wbc"
        / "sim2mujoco"
        / "resources"
        / "robots"
        / "g1"
    ).resolve()


def test_arm_ik_v2_self_track_nominal_pose() -> None:
    import mujoco

    rd = _default_g1_resources_dir()
    xml = rd / "g1_gear_wbc.xml"
    if not xml.is_file():
        pytest.skip(f"G1 MJCF not found: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    left, right = build_g1_arm_ik_specs(model)
    arm_qpos_ids = np.concatenate([left.qpos_ids, right.qpos_ids]).astype(np.int32)
    pl, ql = body_palm_pose(data, left)
    pr, qr = body_palm_pose(data, right)
    q_sol = solve_dual_arm_ik_v2(
        model,
        data,
        left,
        right,
        pl.astype(np.float32),
        ql.astype(np.float32),
        pr.astype(np.float32),
        qr.astype(np.float32),
        arm_qpos_ids=arm_qpos_ids,
        outer_iters=12,
        damping=0.2,
    )
    q_cur = np.asarray(data.qpos[arm_qpos_ids], dtype=np.float64)
    err = float(np.linalg.norm(np.asarray(q_sol, dtype=np.float64) - q_cur))
    assert err < 0.12, f"IK self-track joint error too large: {err}"
    assert np.all(np.isfinite(q_sol))


def _reach_offset_target(model, data, left, right, offset_world: np.ndarray) -> tuple[float, float]:
    """Solve IK toward (current_palm + offset_world) on both sides, return final pos errors (L, R)."""
    import mujoco

    mujoco.mj_forward(model, data)
    pl, ql = body_palm_pose(data, left)
    pr, qr = body_palm_pose(data, right)
    tgt_l = pl + offset_world
    tgt_r = pr + offset_world
    arm_qpos_ids = np.concatenate([left.qpos_ids, right.qpos_ids]).astype(np.int32)
    q_sol = solve_dual_arm_ik_v2(
        model,
        data,
        left,
        right,
        tgt_l.astype(np.float32),
        ql.astype(np.float32),
        tgt_r.astype(np.float32),
        qr.astype(np.float32),
        arm_qpos_ids=arm_qpos_ids,
        outer_iters=20,
        damping=0.15,
    )
    q_save = data.qpos.copy()
    try:
        data.qpos[arm_qpos_ids] = np.asarray(q_sol, dtype=np.float64)
        mujoco.mj_forward(model, data)
        pl2, _ = body_palm_pose(data, left)
        pr2, _ = body_palm_pose(data, right)
        err_l = float(np.linalg.norm(pl2 - tgt_l))
        err_r = float(np.linalg.norm(pr2 - tgt_r))
    finally:
        data.qpos[:] = q_save
        mujoco.mj_forward(model, data)
    return err_l, err_r


@pytest.mark.parametrize(
    "base_xy,base_yaw_deg,label",
    [
        ((0.0, 0.0), 0.0, "origin"),
        ((2.0, 0.0), 0.0, "translated_+2x"),
        ((0.5, 0.5), 180.0, "yawed_180"),
        ((-1.5, 0.7), 90.0, "translated_yawed_90"),
    ],
)
def test_arm_ik_v2_reach_small_offset_under_base_translation(base_xy, base_yaw_deg, label) -> None:
    """IK must converge to a small reachable offset target regardless of base pose.

    Pre-fix this would diverge or oscillate when ``data.xpos[wrist]`` was far from origin
    because mj_jac was given body-local coords as the world ``point``. Threshold (5 mm)
    is loose enough for a single 20-iter solve from default arm pose with damping 0.15.
    """
    import mujoco

    rd = _default_g1_resources_dir()
    xml = rd / "g1_gear_wbc.xml"
    if not xml.is_file():
        pytest.skip(f"G1 MJCF not found: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    _set_free_base(data, base_xy, np.radians(base_yaw_deg))
    mujoco.mj_forward(model, data)
    err_l, err_r = _reach_offset_target(
        model, data, *build_g1_arm_ik_specs(model), offset_world=np.array([0.06, 0.0, 0.04])
    )
    assert err_l < 2e-3, f"[{label}] left palm pos error too large: {err_l:.4f} m"
    assert err_r < 2e-3, f"[{label}] right palm pos error too large: {err_r:.4f} m"


def test_ee_world_targets_ref_quat_matches_torso_decode() -> None:
    """When ref_quat == xquat, decode matches default path (regression)."""
    import mujoco

    rd = _default_g1_resources_dir()
    xml = rd / "g1_gear_wbc.xml"
    if not xml.is_file():
        pytest.skip(f"G1 MJCF not found: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ti = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    xq = np.asarray(data.xquat[ti], dtype=np.float64).reshape(4)
    tl_p = np.array([0.3, 0.1, 0.2], dtype=np.float32)
    tl_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    tr_p = np.array([0.3, -0.1, 0.2], dtype=np.float32)
    tr_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    a0, a1, a2, a3 = ee_world_targets_for_ik(
        data, ti, tl_p, tl_q, tr_p, tr_q, ref_quat_wxyz=None
    )
    b0, b1, b2, b3 = ee_world_targets_for_ik(
        data, ti, tl_p, tl_q, tr_p, tr_q, ref_quat_wxyz=xq.copy()
    )
    assert float(np.linalg.norm(np.asarray(a0) - np.asarray(b0))) < 1e-6
    assert float(np.linalg.norm(np.asarray(a2) - np.asarray(b2))) < 1e-6
