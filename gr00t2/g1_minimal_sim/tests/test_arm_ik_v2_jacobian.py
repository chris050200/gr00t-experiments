"""Pin the ``mj_jac`` point-frame contract used by ``arm_ik_v2``.

Background (May 2026 fix): ``mj_jac`` requires ``point`` in **world** coordinates.
Previously ``arm_ik_v2`` passed ``palm_local`` (body-frame), so the position-Jacobian's
implicit lever arm was ``palm_local - x_body`` instead of ``R_body @ palm_local``. The
absolute error scaled with how far the wrist body had translated in world (e.g. while
walking), explaining the "stable near home, spasms during/after walking" symptom.

This test verifies Jp @ qdot agrees with finite-difference of palm world position to
machine precision, at multiple non-trivial poses including a translated/yawed base.
A regression of the original bug would fail by ~1e0, not pass at 1e-7.
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

from arm_ik import body_palm_pose, build_g1_arm_ik_specs  # noqa: E402


def _default_g1_resources_dir() -> Path:
    return (
        _ROOT.parent
        / "Isaac-GR00T"
        / "external_dependencies"
        / "GR00T-WholeBodyControl"
        / "gr00t_wbc"
        / "sim2mujoco"
        / "resources"
        / "robots"
        / "g1"
    ).resolve()


def _set_free_base(data, xy, yaw_rad: float, z: float = 0.74) -> None:
    data.qpos[0] = float(xy[0])
    data.qpos[1] = float(xy[1])
    data.qpos[2] = float(z)
    data.qpos[3] = float(np.cos(yaw_rad * 0.5))
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = float(np.sin(yaw_rad * 0.5))


def _palm_world(data, body_id: int, palm_local: np.ndarray) -> np.ndarray:
    xmat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return np.asarray(data.xpos[body_id], dtype=np.float64).reshape(3) + xmat @ palm_local


def _build_model_data():
    import mujoco

    xml = _default_g1_resources_dir() / "g1_gear_wbc.xml"
    if not xml.is_file():
        pytest.skip(f"G1 MJCF not found: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    return mujoco, model, data


def _jacp_palm(mujoco, model, data, body_id: int, palm_local: np.ndarray) -> np.ndarray:
    """Position Jacobian at the palm world point (the contract arm_ik_v2 must obey)."""
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    palm_w = _palm_world(data, body_id, palm_local)
    mujoco.mj_jac(model, data, jacp, jacr, palm_w, body_id)
    return jacp


def _fd_palm_velocity(
    mujoco,
    model,
    data,
    arm_qpos_ids: np.ndarray,
    body_id: int,
    palm_local: np.ndarray,
    qdot_arm: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    q_save = data.qpos.copy()
    try:
        q0 = data.qpos[arm_qpos_ids].copy()
        data.qpos[arm_qpos_ids] = q0 + eps * qdot_arm
        mujoco.mj_forward(model, data)
        p_plus = _palm_world(data, body_id, palm_local)
        data.qpos[:] = q_save
        data.qpos[arm_qpos_ids] = q0 - eps * qdot_arm
        mujoco.mj_forward(model, data)
        p_minus = _palm_world(data, body_id, palm_local)
        return (p_plus - p_minus) / (2.0 * eps)
    finally:
        data.qpos[:] = q_save
        mujoco.mj_forward(model, data)


@pytest.mark.parametrize(
    "base_xy,base_yaw_deg,arm_perturb,label",
    [
        ((0.0, 0.0), 0.0, 0.0, "home_base_origin"),
        ((0.0, 0.0), 0.0, 0.4, "perturbed_base_origin"),
        ((2.0, 0.0), 0.0, 0.4, "perturbed_base_+2x"),
        ((0.5, 0.5), 180.0, 0.4, "perturbed_base_yaw180"),
        ((-1.5, 0.7), 90.0, 0.3, "perturbed_base_translated_yaw90"),
    ],
)
def test_palm_jacobian_matches_fd_under_base_pose(
    base_xy, base_yaw_deg, arm_perturb, label
) -> None:
    """``Jp[:, dof_ids] @ qdot`` must equal FD of palm world position regardless of base pose."""
    mujoco, model, data = _build_model_data()
    rng = np.random.default_rng(int(abs(hash(label)) % (2**31)))

    _set_free_base(data, base_xy, np.radians(base_yaw_deg))
    left, right = build_g1_arm_ik_specs(model)
    for spec in (left, right):
        if arm_perturb > 0.0:
            data.qpos[spec.qpos_ids] = rng.standard_normal(spec.qpos_ids.size) * arm_perturb
    mujoco.mj_forward(model, data)

    for spec, side in ((left, "L"), (right, "R")):
        body_id = int(spec.body_id)
        palm_local = np.asarray(spec.palm_local, dtype=np.float64)
        cols = spec.dof_ids.astype(np.int32)
        arm_qpos_ids = spec.qpos_ids.astype(np.int32)
        Jp = _jacp_palm(mujoco, model, data, body_id, palm_local)[:, cols]

        for _ in range(8):
            qdot = rng.standard_normal(arm_qpos_ids.size)
            qdot = qdot / max(1.0, float(np.linalg.norm(qdot)))
            v_jac = Jp @ qdot
            v_fd = _fd_palm_velocity(
                mujoco, model, data, arm_qpos_ids, body_id, palm_local, qdot
            )
            err = float(np.linalg.norm(v_jac - v_fd))
            assert err < 1e-6, (
                f"[{label}/{side}] |Jp@qdot - FD| = {err:.3e} (expected < 1e-6); "
                f"v_jac={v_jac}, v_fd={v_fd}"
            )


def test_arm_ik_v2_passes_world_palm_point_to_mj_jac(monkeypatch) -> None:
    """Pin the contract: the inner mj_jac call inside ``solve_dual_arm_ik_v2`` must
    use the palm WORLD position as ``point``, not the body-local ``palm_local`` offset.

    Regression of the May 2026 bug (where body-local was passed) would fail this with
    ``point`` numerically equal to ``palm_local`` (~4 cm) instead of palm_world (~80 cm
    or more, depending on base placement).
    """
    mujoco, model, data = _build_model_data()
    _set_free_base(data, (1.0, 0.5), np.radians(45.0))
    left, right = build_g1_arm_ik_specs(model)
    rng = np.random.default_rng(1)
    for spec in (left, right):
        data.qpos[spec.qpos_ids] = rng.standard_normal(spec.qpos_ids.size) * 0.3
    mujoco.mj_forward(model, data)

    palm_world_l, _ = body_palm_pose(data, left)
    palm_world_r, _ = body_palm_pose(data, right)
    expected = {int(left.body_id): palm_world_l, int(right.body_id): palm_world_r}

    import arm_ik_v2

    seen: list[tuple[int, np.ndarray]] = []
    real_mj_jac = arm_ik_v2.mujoco.mj_jac

    def _spy_mj_jac(m, d, jp, jr, point, body):
        pt = np.asarray(point, dtype=np.float64).reshape(-1)
        seen.append((int(body), pt.copy()))
        return real_mj_jac(m, d, jp, jr, point, body)

    monkeypatch.setattr(arm_ik_v2.mujoco, "mj_jac", _spy_mj_jac)

    arm_qpos_ids = np.concatenate([left.qpos_ids, right.qpos_ids]).astype(np.int32)
    pl, ql = body_palm_pose(data, left)
    pr, qr = body_palm_pose(data, right)
    arm_ik_v2.solve_dual_arm_ik_v2(
        model,
        data,
        left,
        right,
        pl.astype(np.float32),
        ql.astype(np.float32),
        pr.astype(np.float32),
        qr.astype(np.float32),
        arm_qpos_ids=arm_qpos_ids,
        outer_iters=1,
        damping=0.15,
    )

    assert seen, "arm_ik_v2 did not call mj_jac"
    for body_id, pt in seen:
        assert body_id in expected, f"unexpected body_id={body_id} passed to mj_jac"
        ref = expected[body_id]
        gap = float(np.linalg.norm(pt - ref))
        body_local = np.asarray(
            (left if body_id == int(left.body_id) else right).palm_local, dtype=np.float64
        )
        gap_to_local = float(np.linalg.norm(pt - body_local))
        assert gap < 5e-3, (
            f"mj_jac point for body {body_id} = {pt} is {gap:.4f} m from palm_world={ref}; "
            f"distance to palm_local={body_local} is {gap_to_local:.4f} m. "
            "If the latter is small, arm_ik_v2 has reverted to the pre-fix bug."
        )
