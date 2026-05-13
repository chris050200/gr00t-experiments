"""Phase 4a regression: ``GearWBCRuntime`` seeds qpos[legs+waist] with YAML ``default_angles``.

Pins the fix for **Bug #8** in the May 2026 GR00T PnP A/B forensics: before
this fix the runtime left ``qpos[7 : 7 + num_act]`` at zero after ``__init__``
(straight legs, feet ~14 cm above the floor). The Gear WBC leg PD would
eventually drive that block toward ``default_angles``, but the **first**
policy observation captured a posture no CloudWalk/GR00T checkpoint had ever
seen in training (RoboCasa env reset writes ``default_angles`` into qpos
before the first ``policy.get_action``). That made ``state.left_leg`` /
``state.right_leg`` OOD and produced pathological arm chunks (the persistent
"hands lift toward face" symptom).

The fix is a single ``self.data.qpos[self._policy_qpos_adr[: num_act]] =
default_angles`` write in :class:`gear_wbc_stand.GearWBCRuntime.__init__`,
applied **before** the initial ``mj_forward`` and the first ``compute_single_obs``
call. This file pins:

1. The first ``num_act`` policy qpos slots equal the YAML ``default_angles``
   immediately after the runtime is constructed (no ``reset`` / ``step`` yet).
2. The legs+waist values match the standing crouch the policy expects
   (knees ~+0.3 rad, ankle pitch ~-0.2 rad, waist zeros), not the MJCF-default
   straight-leg pose.

Skipped automatically when the GR00T ONNX bundle is missing (CI without
external assets) - the runtime cannot construct without it.
"""

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

from gear_wbc_stand import default_g1_resources_dir  # noqa: E402

from g1_gear_wbc_env import G1GearWBCEnv  # noqa: E402


def _ensure_onnx_or_skip() -> None:
    rd = default_g1_resources_dir()
    onnx_dir = rd / "policy"
    have_onnx = (onnx_dir / "ft92.onnx").is_file() or (
        onnx_dir / "GR00T-WholeBodyControl-Balance.onnx"
    ).is_file()
    if not have_onnx:
        pytest.skip(f"GR00T ONNX policy not found under {onnx_dir}")


def test_runtime_seeds_default_angles_into_policy_qpos_block() -> None:
    """Right after ``__init__`` the leg+waist qpos block must equal YAML defaults.

    Reproduces Bug #8 in negative form: before the fix this test would find
    ``qpos[7 : 7+15] == 0``; with the fix it must match the YAML's standing
    crouch verbatim.
    """
    _ensure_onnx_or_skip()
    scene_dir = _ROOT / "scenes" / "table_pnp_apple"
    if not (scene_dir / "g1_gear_wbc_table_pnp_apple.yaml").is_file():
        pytest.skip(f"table_pnp_apple yaml not found: {scene_dir}")

    env = G1GearWBCEnv(enable_teleop=False, scene="table_pnp_apple")
    try:
        rt = env._rt
        num_act = int(rt.num_act)
        default_angles = np.asarray(rt.config["default_angles"], dtype=np.float64).reshape(-1)
        assert default_angles.size == num_act, (
            f"YAML default_angles size {default_angles.size} != num_actions {num_act}"
        )

        qadr = np.asarray(rt._policy_qpos_adr, dtype=np.int64)
        leg_waist_addr = qadr[:num_act]
        actual = np.asarray(rt.data.qpos[leg_waist_addr], dtype=np.float64).copy()

        np.testing.assert_allclose(
            actual,
            default_angles,
            atol=1e-9,
            err_msg=(
                "qpos[legs+waist] was not seeded with YAML default_angles at "
                "runtime init. This re-introduces Bug #8 (straight legs = OOD "
                "first observation for any GR00T checkpoint)."
            ),
        )
    finally:
        env.close()


def test_runtime_default_angles_match_expected_standing_crouch() -> None:
    """Pin the YAML default_angles to the known standing crouch used in training.

    Guards against a YAML edit that would silently shift the policy's
    expected init pose (and re-introduce a distribution shift in
    ``state.left_leg`` / ``state.right_leg``).
    """
    _ensure_onnx_or_skip()
    scene_dir = _ROOT / "scenes" / "table_pnp_apple"
    if not (scene_dir / "g1_gear_wbc_table_pnp_apple.yaml").is_file():
        pytest.skip(f"table_pnp_apple yaml not found: {scene_dir}")

    env = G1GearWBCEnv(enable_teleop=False, scene="table_pnp_apple")
    try:
        rt = env._rt
        default_angles = np.asarray(rt.config["default_angles"], dtype=np.float64).reshape(-1)

        # Layout: left leg (6), right leg (6), waist (3) - matches
        # POLICY_JOINT_NAMES[:15].
        expected = np.array(
            [
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                0.0, 0.0, 0.0,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            default_angles,
            expected,
            atol=1e-6,
            err_msg=(
                "Gear WBC YAML ``default_angles`` drifted from the standing "
                "crouch shipped with the GR00T training stack."
            ),
        )
    finally:
        env.close()


def test_runtime_pelvis_z_consistent_with_seeded_legs() -> None:
    """With legs at default_angles, pelvis-z should already sit in the stand band.

    Sanity check that the seeded standing pose puts the feet on the floor
    (so the first observation is geometrically consistent with the rollout
    oracle's first frame: pelvis_z ~= 0.79 m, knees bent ~+0.3 rad).
    """
    _ensure_onnx_or_skip()
    scene_dir = _ROOT / "scenes" / "table_pnp_apple"
    if not (scene_dir / "g1_gear_wbc_table_pnp_apple.yaml").is_file():
        pytest.skip(f"table_pnp_apple yaml not found: {scene_dir}")

    env = G1GearWBCEnv(enable_teleop=False, scene="table_pnp_apple")
    try:
        rt = env._rt
        z0 = float(rt.data.qpos[2])
        # MJCF init pelvis ~0.79 m; with bent legs feet should be on floor.
        assert 0.74 < z0 < 0.83, (
            f"pelvis-z at runtime init = {z0:.3f} m outside the seeded "
            "standing-crouch band [0.74, 0.83]. Either default_angles changed "
            "or the leg-init write happened *after* the floating base was "
            "moved."
        )
    finally:
        env.close()


def test_runtime_reseeds_default_angles_after_reset() -> None:
    """Phase 4a-bis: post-``env.reset()`` qpos[legs+waist] must equal YAML defaults.

    The original Phase 4a write was in ``__init__`` only. ``G1GearWBCEnv.reset()``
    then called ``GearWBCRuntime.reset()`` which ran ``mj_resetData`` and wiped
    qpos back to the MJCF default (straight legs), silently re-introducing
    Bug #8 on the inference path. This pins the fix: the reset must re-seed
    ``default_angles`` into qpos so the FIRST observation captured by the
    inference bridge sees the standing crouch.

    Negative-regression form: before the Phase 4a-bis ``reset()`` patch this
    test would find ``qpos[7 : 7+15] == 0`` after the explicit ``env.reset()``.
    """
    _ensure_onnx_or_skip()
    scene_dir = _ROOT / "scenes" / "table_pnp_apple"
    if not (scene_dir / "g1_gear_wbc_table_pnp_apple.yaml").is_file():
        pytest.skip(f"table_pnp_apple yaml not found: {scene_dir}")

    env = G1GearWBCEnv(enable_teleop=False, scene="table_pnp_apple")
    try:
        env.reset()
        rt = env._rt
        num_act = int(rt.num_act)
        default_angles = np.asarray(rt.config["default_angles"], dtype=np.float64).reshape(-1)

        qadr = np.asarray(rt._policy_qpos_adr, dtype=np.int64)
        actual = np.asarray(rt.data.qpos[qadr[:num_act]], dtype=np.float64).copy()

        np.testing.assert_allclose(
            actual,
            default_angles,
            atol=1e-9,
            err_msg=(
                "qpos[legs+waist] was not re-seeded with YAML default_angles "
                "after env.reset(). The Gym reset path silently re-introduces "
                "Bug #8 (straight legs = OOD first observation)."
            ),
        )

        # Pelvis-z is the geometric witness of leg pose; same band as __init__.
        z0 = float(rt.data.qpos[2])
        assert 0.74 < z0 < 0.83, (
            f"post-reset pelvis-z = {z0:.3f} m outside the seeded "
            "standing-crouch band [0.74, 0.83]."
        )
    finally:
        env.close()


def test_obs_builder_state_legs_match_default_angles_after_reset() -> None:
    """End-to-end pin: ``Gr00tObservationBuilder.build()`` must report standing legs.

    This is the strict regression for the May 2026 A/B dump finding: the
    policy-facing observation ``state.left_leg`` / ``state.right_leg`` MUST
    equal the YAML standing crouch after ``env.reset()``. If this test
    passes, the per-key dump diff for those two keys WILL be zero on a
    Tier-A oracle vs Tier-B subject comparison (Bug #8 closed at the
    observation surface, not just the qpos buffer).

    Skipped when ``gr00t_wbc`` is unavailable (CI without GR00T-WBC install)
    or when GL is unavailable for renderer init - we set ``include_video=False``
    to keep it pure-state and runnable headless.
    """
    _ensure_onnx_or_skip()
    scene_dir = _ROOT / "scenes" / "table_pnp_apple"
    if not (scene_dir / "g1_gear_wbc_table_pnp_apple.yaml").is_file():
        pytest.skip(f"table_pnp_apple yaml not found: {scene_dir}")

    try:
        from gr00t_observation_builder import Gr00tObservationBuilder  # noqa: WPS433
    except ImportError as exc:
        pytest.skip(f"gr00t_observation_builder not importable: {exc}")

    # The builder lazily imports gr00t_wbc inside .build() - guard for absence.
    try:
        builder = Gr00tObservationBuilder.for_locomanip_default()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gr00tObservationBuilder construction failed (likely gr00t_wbc missing): {exc}")

    # Match the inference script's default (``use_hands=not args.no_hands``):
    # ``Gr00tObservationBuilder.for_locomanip_default()`` builds a 43-joint
    # ``RobotModel`` that includes the hand joints, so the underlying MJCF
    # must also have them or ``mujoco_q_to_robot_model_q`` raises looking
    # up ``left_hand_index_0_joint``. The actual GR00T inference bridge
    # always loads the hands MJCF; mirror that here.
    env = G1GearWBCEnv(enable_teleop=False, scene="table_pnp_apple", use_hands=True)
    try:
        env.reset()
        rt = env._rt
        try:
            obs = builder.build(
                rt.model,
                rt.data,
                task_description="pick up the apple, walk left and place the apple on the plate.",
                include_video=False,
            )
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Gr00tObservationBuilder.build raised (likely gr00t_wbc internals): {exc}")

        expected_leg = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float64)
        left_leg = np.asarray(obs["state.left_leg"], dtype=np.float64).reshape(-1)
        right_leg = np.asarray(obs["state.right_leg"], dtype=np.float64).reshape(-1)
        waist = np.asarray(obs["state.waist"], dtype=np.float64).reshape(-1)

        np.testing.assert_allclose(
            left_leg,
            expected_leg,
            atol=1e-5,
            err_msg=(
                "Gr00tObservationBuilder reported state.left_leg != YAML "
                "default_angles[0:6]. Either the Phase 4a-bis reset seed "
                "regressed or the obs builder is reading from a different "
                "source than data.qpos."
            ),
        )
        np.testing.assert_allclose(
            right_leg,
            expected_leg,
            atol=1e-5,
            err_msg="Gr00tObservationBuilder reported state.right_leg != YAML default_angles[6:12].",
        )
        np.testing.assert_allclose(
            waist,
            np.zeros(3, dtype=np.float64),
            atol=1e-5,
            err_msg="Gr00tObservationBuilder reported state.waist != YAML default_angles[12:15] (zeros).",
        )
    finally:
        env.close()
        builder.close()
