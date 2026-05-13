"""Pin the inference-side cadence + arm-action contracts for ``run_gr00t_stylish_diner_inference``.

This file covers two regression-prone invariants of the inference bridge:

1. **Cadence**: the training-time logger samples observations + actions at
   ``sample_hz`` (50 Hz by default in :mod:`gr00t_teleop_logger`). The
   inference bridge MUST consume each policy-plan step over the same
   wall-clock period. With ``simulation_dt = 0.005`` that means **4
   mj_step substeps per plan step**. Mismatching this contract was the
   first half of the May 2026 "live policy explodes / pretzels the arms
   within 50 steps" symptom (see ``activeContext.md``).
2. **Arm action representation**: the policy server's
   ``StateActionProcessor.unapply_action`` (with ``use_relative_action=True``)
   *already* converts RELATIVE arm keys back to absolute joint targets via
   ``JointActionChunk.to_absolute_chunking``. The inference script must
   NOT add ``state_now`` again on the live branch (that double-add was
   the second half of the May 2026 pretzel symptom). NPZ replay, by
   contrast, sees raw deltas from disk and *must* add state back. The
   resolver pins the ``--arm-action-mode auto`` defaults: ``live ->
   absolute``, ``replay -> relative``.

This file is intentionally numpy-only: it loads the inference script as a
plain module (without registering the gym env or starting MuJoCo) by stubbing
the script-time imports of ``g1_gear_wbc_env`` / ``gr00t_observation_builder``
/ ``hand_gripper`` with the minimum surface those modules expose at import
time. That keeps the test fast and DISPLAY-free for CI.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "run_gr00t_stylish_diner_inference.py"

pytest.importorskip("numpy")
import numpy as np  # noqa: E402  (after importorskip)


def _install_stub_modules() -> list[str]:
    """Replace the inference script's heavy imports with lightweight stubs.

    Only the symbols the script actually references at module load are
    provided. Functions inside the script (``_run_policy_loop`` etc.) still
    need a real runtime, but the cadence helpers (``resolve_plan_substeps``,
    ``_advance_physics_one_plan_step``, ``DEFAULT_PLAN_HZ``) do not.

    Returns the list of module names this call inserted into ``sys.modules``
    so the caller can remove them after loading the inference script, to
    avoid leaking stubs into unrelated tests in the same session.

    The stubs intentionally mirror the **real** ``hand_gripper`` slice layout
    (left_arm / left_hand / right_arm / right_hand at 0:7 / 7:14 / 14:21 /
    21:28) so the inference module ends up with identical ``SL_*`` constants
    regardless of which version it imports first. Any test that asserts on
    those constants stays correct in either ordering.
    """
    inserted: list[str] = []

    if "g1_gear_wbc_env" not in sys.modules:
        m = types.ModuleType("g1_gear_wbc_env")
        m.ENV_ID = "G1GearWBC-v0-stub"  # type: ignore[attr-defined]

        def _register() -> None:
            return None

        m.register_g1_gear_wbc_env = _register  # type: ignore[attr-defined]
        sys.modules["g1_gear_wbc_env"] = m
        inserted.append("g1_gear_wbc_env")

    if "gr00t_observation_builder" not in sys.modules:
        m = types.ModuleType("gr00t_observation_builder")

        class _StubBuilder:
            @classmethod
            def for_locomanip_default(cls, **_: Any) -> "_StubBuilder":
                return cls()

            def build(self, *_: Any, **__: Any) -> dict[str, Any]:
                return {}

            def close(self) -> None:
                return None

        m.Gr00tObservationBuilder = _StubBuilder  # type: ignore[attr-defined]
        sys.modules["gr00t_observation_builder"] = m
        inserted.append("gr00t_observation_builder")

    if "hand_gripper" not in sys.modules:
        m = types.ModuleType("hand_gripper")
        # Mirror the real layout (see g1_minimal_sim/hand_gripper.py).
        m.SL_LEFT_ARM = slice(0, 7)  # type: ignore[attr-defined]
        m.SL_LEFT_HAND = slice(7, 14)  # type: ignore[attr-defined]
        m.SL_RIGHT_ARM = slice(14, 21)  # type: ignore[attr-defined]
        m.SL_RIGHT_HAND = slice(21, 28)  # type: ignore[attr-defined]
        m.GR00T_POLICY_HAND_PD_SOURCE_KEY = "_gr00t_use_policy_hand_targets"  # type: ignore[attr-defined]

        # Stub ``hand_perm_policy_to_mj`` that recovers the MuJoCo-side
        # permutation by name lookup. Matches the real implementation's
        # contract closely enough for tests that exercise
        # ``_apply_action_step(hand_permutation=...)`` against synthetic
        # joint-name lists.
        _LEFT_HAND_JOINT_NAMES_MJ = (
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
        )
        _RIGHT_HAND_JOINT_NAMES_MJ = tuple(
            n.replace("left", "right") for n in _LEFT_HAND_JOINT_NAMES_MJ
        )

        def _hand_perm_policy_to_mj(side: str, policy_joint_names: list[str]):
            mj_names = (
                _LEFT_HAND_JOINT_NAMES_MJ
                if side == "left"
                else _RIGHT_HAND_JOINT_NAMES_MJ
            )
            name_to_idx = {str(n): i for i, n in enumerate(policy_joint_names)}
            return np.asarray(
                [name_to_idx[mj] for mj in mj_names], dtype=np.int64
            )

        m.LEFT_HAND_JOINT_NAMES = _LEFT_HAND_JOINT_NAMES_MJ  # type: ignore[attr-defined]
        m.RIGHT_HAND_JOINT_NAMES = _RIGHT_HAND_JOINT_NAMES_MJ  # type: ignore[attr-defined]
        m.hand_perm_policy_to_mj = _hand_perm_policy_to_mj  # type: ignore[attr-defined]
        sys.modules["hand_gripper"] = m
        inserted.append("hand_gripper")

    return inserted


def _load_inference_module():
    """Load the inference script as a module without leaking stubs.

    Any stub inserted by ``_install_stub_modules`` is removed from
    ``sys.modules`` after the script has resolved its imports — the loaded
    module keeps a private reference to the stub via its bound names, but
    other tests in the same session can still ``import gr00t_observation_builder``
    and get the real module on first access.
    """
    pytest.importorskip("gymnasium")
    inserted = _install_stub_modules()
    spec = importlib.util.spec_from_file_location(
        "_run_gr00t_stylish_diner_inference_for_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None, _SCRIPT_PATH
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        for name in inserted:
            sys.modules.pop(name, None)
    return mod


# ---------------------------------------------------------------------------
# resolve_plan_substeps: pure math contract
# ---------------------------------------------------------------------------


def test_resolve_plan_substeps_default_diner_setup() -> None:
    mod = _load_inference_module()
    assert mod.resolve_plan_substeps(0.005, 50.0) == 4


def test_resolve_plan_substeps_default_constant_matches_logger_default() -> None:
    """``DEFAULT_PLAN_HZ`` must equal the logger's ``sample_hz`` default."""
    mod = _load_inference_module()
    assert mod.DEFAULT_PLAN_HZ == 50.0
    try:
        from gr00t_teleop_logger import Gr00tTeleopEpisodeLogger
    except Exception as exc:  # pragma: no cover - skip if logger deps missing
        pytest.skip(f"logger module not importable in this env: {exc}")
    import inspect

    sig = inspect.signature(Gr00tTeleopEpisodeLogger.__init__)
    sample_hz_default = sig.parameters["sample_hz"].default
    assert sample_hz_default == mod.DEFAULT_PLAN_HZ


@pytest.mark.parametrize(
    ("sim_dt", "plan_hz", "expected"),
    [
        (0.005, 50.0, 4),  # canonical diner setup
        (0.005, 200.0, 1),  # degenerate: 1 substep replicates the pre-fix bug
        (0.001, 50.0, 20),  # finer physics dt, same plan hz
        (0.005, 25.0, 8),  # half plan rate, double substeps
        (0.0025, 50.0, 8),  # half sim dt
        (0.005, 60.0, 3),  # rounding: 3.33 -> 3
        (0.005, 40.0, 5),  # rounding: 5.0 exactly
    ],
)
def test_resolve_plan_substeps_table(sim_dt: float, plan_hz: float, expected: int) -> None:
    mod = _load_inference_module()
    assert mod.resolve_plan_substeps(sim_dt, plan_hz) == expected


def test_resolve_video_fps_matches_plan_hz_when_unset() -> None:
    mod = _load_inference_module()
    assert mod.resolve_video_fps(None, 50.0) == 50
    assert mod.resolve_video_fps(None, 25.0) == 25
    assert mod.resolve_video_fps(None, 50.4) == 50


def test_resolve_video_fps_explicit_override() -> None:
    mod = _load_inference_module()
    assert mod.resolve_video_fps(20, 50.0) == 20
    assert mod.resolve_video_fps(1, 99.0) == 1


def test_json_safe_float_filters_nonfinite() -> None:
    mod = _load_inference_module()
    assert mod._json_safe_float(1.25) == 1.25
    assert mod._json_safe_float(float("nan")) is None
    assert mod._json_safe_float(float("inf")) is None


def test_max_abs_vec_diff() -> None:
    mod = _load_inference_module()
    assert mod._max_abs_vec_diff(np.array([1.0, -2.0]), np.array([1.1, -2.0])) == pytest.approx(0.1)
    assert mod._max_abs_vec_diff(None, np.zeros(2)) is None
    assert mod._max_abs_vec_diff(np.zeros(2), np.zeros(3)) is None


def test_gather_arm_hand_targets_absolute_matches_action_vectors() -> None:
    mod = _load_inference_module()
    state_now = {
        "state.left_arm": np.zeros(7, dtype=np.float32),
        "state.right_arm": np.ones(7, dtype=np.float32) * 0.5,
    }
    action_step = {
        "action.left_arm": np.linspace(0, 0.06, 7, dtype=np.float32),
        "action.right_arm": np.linspace(1, 1.06, 7, dtype=np.float32),
    }
    la, ra, lh, rh = mod._gather_arm_hand_targets_from_action_step(
        action_step,
        state_now,
        arm_action_mode="absolute",
        hand_permutation=None,
    )
    assert la is not None and ra is not None
    assert lh is None and rh is None
    assert np.allclose(la, action_step["action.left_arm"])
    assert np.allclose(ra, action_step["action.right_arm"])


def test_gather_arm_hand_targets_relative_adds_state() -> None:
    mod = _load_inference_module()
    state_now = {
        "state.left_arm": np.ones(7, dtype=np.float32) * 0.1,
        "state.right_arm": np.zeros(7, dtype=np.float32),
    }
    action_step = {
        "action.left_arm": np.zeros(7, dtype=np.float32),
        "action.right_arm": np.ones(7, dtype=np.float32) * 0.2,
    }
    la, ra, _, _ = mod._gather_arm_hand_targets_from_action_step(
        action_step,
        state_now,
        arm_action_mode="relative",
        hand_permutation=None,
    )
    assert la is not None and ra is not None
    assert np.allclose(la, state_now["state.left_arm"])
    assert np.allclose(ra, np.ones(7, dtype=np.float32) * 0.2)


def test_resolve_video_fps_rejects_bad_values() -> None:
    mod = _load_inference_module()
    with pytest.raises(ValueError, match="video_fps"):
        mod.resolve_video_fps(0, 50.0)
    with pytest.raises(ValueError, match="video_fps"):
        mod.resolve_video_fps(-1, 50.0)
    with pytest.raises(ValueError, match="plan_hz"):
        mod.resolve_video_fps(None, 0.0)


def test_resolve_plan_substeps_floor_clamps_to_one() -> None:
    """Even at huge plan_hz × sim_dt the function must return at least 1.

    A return of 0 would silently freeze physics during inference; clamping to 1
    means the worst case is the historical 4×-too-fast loop, which is loud.
    """
    mod = _load_inference_module()
    assert mod.resolve_plan_substeps(0.005, 1_000_000.0) == 1


@pytest.mark.parametrize("bad_sim_dt", [0.0, -1.0, -1e-6])
def test_resolve_plan_substeps_rejects_nonpositive_sim_dt(bad_sim_dt: float) -> None:
    mod = _load_inference_module()
    with pytest.raises(ValueError, match="sim_dt"):
        mod.resolve_plan_substeps(bad_sim_dt, 50.0)


@pytest.mark.parametrize("bad_plan_hz", [0.0, -1.0, -1e-6])
def test_resolve_plan_substeps_rejects_nonpositive_plan_hz(bad_plan_hz: float) -> None:
    mod = _load_inference_module()
    with pytest.raises(ValueError, match="plan_hz"):
        mod.resolve_plan_substeps(0.005, bad_plan_hz)


# ---------------------------------------------------------------------------
# _advance_physics_one_plan_step: substep loop contract
# ---------------------------------------------------------------------------


class _FakeEnv:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def step(self, action: Any) -> tuple[None, float, bool, bool, dict[str, Any]]:
        self.calls.append(np.asarray(action).copy())
        return None, 0.0, False, False, {}


def test_advance_physics_calls_env_step_n_times() -> None:
    mod = _load_inference_module()
    env = _FakeEnv()
    mod._advance_physics_one_plan_step(env, 4)
    assert len(env.calls) == 4
    for a in env.calls:
        assert a.dtype == np.float32
        assert a.shape == (1,)
        assert float(a[0]) == 0.0


def test_advance_physics_one_substep() -> None:
    mod = _load_inference_module()
    env = _FakeEnv()
    mod._advance_physics_one_plan_step(env, 1)
    assert len(env.calls) == 1


def test_advance_physics_rejects_zero_substeps() -> None:
    mod = _load_inference_module()
    env = _FakeEnv()
    with pytest.raises(ValueError, match="n_substeps"):
        mod._advance_physics_one_plan_step(env, 0)
    assert env.calls == []


def test_advance_physics_rejects_negative_substeps() -> None:
    mod = _load_inference_module()
    env = _FakeEnv()
    with pytest.raises(ValueError, match="n_substeps"):
        mod._advance_physics_one_plan_step(env, -3)
    assert env.calls == []


# ---------------------------------------------------------------------------
# resolve_arm_action_mode: ``auto`` defaults + override contract
# ---------------------------------------------------------------------------


def test_arm_action_mode_choices_include_auto() -> None:
    """The CLI must expose ``auto`` plus both explicit overrides."""
    mod = _load_inference_module()
    assert set(mod.ARM_ACTION_MODE_CHOICES) == {"auto", "relative", "absolute"}
    assert set(mod.ARM_ACTION_MODE_RESOLVED) == {"relative", "absolute"}
    assert mod.ARM_ACTION_MODE_AUTO == "auto"


def test_arm_action_mode_auto_live_resolves_to_absolute() -> None:
    """Live policy: server already absolutized RELATIVE arms; must NOT add state again."""
    mod = _load_inference_module()
    assert mod.resolve_arm_action_mode("auto", branch="live") == "absolute"


def test_arm_action_mode_auto_replay_resolves_to_relative() -> None:
    """NPZ replay: on-disk action.{side}_arm is ``arm_target_q - state``; must add state back."""
    mod = _load_inference_module()
    assert mod.resolve_arm_action_mode("auto", branch="replay") == "relative"


def test_arm_action_mode_auto_oracle_replay_resolves_to_absolute() -> None:
    """Oracle chunk replay: Tier-A tensors match server-side absolutized arms."""
    mod = _load_inference_module()
    assert mod.resolve_arm_action_mode("auto", branch="oracle_replay") == "absolute"


@pytest.mark.parametrize("branch", ["live", "replay", "oracle_replay"])
@pytest.mark.parametrize("override", ["relative", "absolute"])
def test_arm_action_mode_explicit_override_returns_as_is(
    branch: str, override: str
) -> None:
    """An explicit ``--arm-action-mode`` overrides the branch default."""
    mod = _load_inference_module()
    assert mod.resolve_arm_action_mode(override, branch=branch) == override


def test_arm_action_mode_rejects_unknown_mode() -> None:
    mod = _load_inference_module()
    with pytest.raises(ValueError, match="arm_action_mode"):
        mod.resolve_arm_action_mode("delta", branch="live")


@pytest.mark.parametrize("bad_branch", ["", "infer", "training", "LIVE", None])
def test_arm_action_mode_rejects_unknown_branch(bad_branch: Any) -> None:
    mod = _load_inference_module()
    with pytest.raises(ValueError, match="branch"):
        mod.resolve_arm_action_mode("auto", branch=bad_branch)  # type: ignore[arg-type]


def _make_fake_rt(mod: Any) -> Any:
    """Build a minimal runtime double for ``_apply_action_step``.

    Sized to fit the real ``hand_gripper`` slice layout
    (right_arm at 14:21, right_hand at 21:28), and reports
    ``has_hands=True / n_arm=28`` so the script picks the
    "hands present" branch and writes through ``mod.SL_LEFT_ARM`` /
    ``mod.SL_RIGHT_ARM`` regardless of slice ordering.
    """
    from contextlib import nullcontext

    n_arm = max(
        int(mod.SL_LEFT_ARM.stop),
        int(mod.SL_RIGHT_ARM.stop),
        int(mod.SL_LEFT_HAND.stop),
        int(mod.SL_RIGHT_HAND.stop),
    )

    class _FakeRt:
        def __init__(self) -> None:
            self.cmd_lock = nullcontext()
            self.has_hands = True
            self.n_arm = n_arm
            self.control_dict = {
                "loco_cmd": np.zeros(3, dtype=np.float32),
                "height_cmd": 0.0,
                "rpy_cmd": np.zeros(3, dtype=np.float32),
                "gripper_left": 1.0,
                "gripper_right": 1.0,
                "arm_target_q": np.zeros(n_arm, dtype=np.float64),
            }

    return _FakeRt()


def test_arm_action_mode_apply_step_absolute_writes_action_as_is() -> None:
    """End-to-end pin: with ``arm_action_mode='absolute'`` the script must
    write the server's arm action straight to ``arm_target_q`` without adding
    ``state_now``. This is the live-policy contract.
    """
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)
    arm_target = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    action_step = {
        "action.left_arm": arm_target.copy(),
        "action.right_arm": -arm_target.copy(),
    }
    state_now = {
        "state.left_arm": np.full(7, 5.0, dtype=np.float32),
        "state.right_arm": np.full(7, -5.0, dtype=np.float32),
    }
    applied = mod._apply_action_step(
        rt, action_step, state_now, arm_action_mode="absolute"
    )
    np.testing.assert_allclose(
        rt.control_dict["arm_target_q"][mod.SL_LEFT_ARM], arm_target
    )
    np.testing.assert_allclose(
        rt.control_dict["arm_target_q"][mod.SL_RIGHT_ARM], -arm_target
    )
    np.testing.assert_allclose(applied["arm_target_q.left_arm"], arm_target)
    np.testing.assert_allclose(applied["arm_target_q.right_arm"], -arm_target)


def test_arm_action_mode_apply_step_relative_adds_state_now() -> None:
    """End-to-end pin: with ``arm_action_mode='relative'`` the script must
    add ``state_now`` to the action before writing to ``arm_target_q``.
    This is the NPZ-replay contract.
    """
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)
    delta = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    state_l = np.full(7, 5.0, dtype=np.float32)
    state_r = np.full(7, -5.0, dtype=np.float32)
    action_step = {
        "action.left_arm": delta.copy(),
        "action.right_arm": -delta.copy(),
    }
    state_now = {
        "state.left_arm": state_l,
        "state.right_arm": state_r,
    }
    applied = mod._apply_action_step(
        rt, action_step, state_now, arm_action_mode="relative"
    )
    np.testing.assert_allclose(
        rt.control_dict["arm_target_q"][mod.SL_LEFT_ARM], state_l + delta
    )
    np.testing.assert_allclose(
        rt.control_dict["arm_target_q"][mod.SL_RIGHT_ARM], state_r - delta
    )
    np.testing.assert_allclose(applied["arm_target_q.left_arm"], state_l + delta)
    np.testing.assert_allclose(applied["arm_target_q.right_arm"], state_r - delta)


def test_cli_default_arm_action_mode_is_auto() -> None:
    """Smoke check: a fresh argparse with no flags defaults to ``auto``.

    Locks the new May 2026 default and prevents accidental reverts to the
    historical ``relative`` default that caused the pretzel symptom on the
    live branch.
    """
    mod = _load_inference_module()
    parser = mod.build_argparser() if hasattr(mod, "build_argparser") else None
    if parser is None:
        pytest.skip("inference script does not expose a parser factory")
    ns = parser.parse_args([])
    assert ns.arm_action_mode == "auto"
    assert int(ns.n_action_steps) == 30
    assert ns.no_clip_arm_targets is False
    assert ns.video_fps is None
    assert ns.scene == "stylish_diner"
    assert ns.task_description is None


def test_resolve_task_description_scene_defaults() -> None:
    mod = _load_inference_module()
    assert mod.resolve_task_description("stylish_diner", None) == "pick up the blue block"
    # Byte-exact match with RoboCasa ``LMPnPAppleToPlate._get_instruction``.
    # CloudWalk PnP checkpoint requires the full prompt including the
    # "walk left and" directive; truncation produced the May 2026
    # "hands lift toward face" symptom on the Tier-B apple scene.
    canonical_apple = (
        "pick up the apple, walk left and place the apple on the plate."
    )
    assert mod.resolve_task_description("table_pnp", None) == canonical_apple
    assert mod.resolve_task_description("table_pnp_apple", None) == canonical_apple
    assert mod.resolve_task_description("floor", None) == "walk and balance"
    assert mod.resolve_task_description("stylish_diner", "pick up the blue cube") == (
        "pick up the blue cube"
    )


# ---------------------------------------------------------------------------
# Visual-alignment pins: oak_egoview camera in both table_pnp_apple MJCFs
# and the --camera-name CLI default. These guard against silent regressions
# where a future MJCF edit drops the camera and the observation builder
# silently falls back to ``head_pov`` (mounted ~10 cm too high, less pitched
# down, fovy 110° vs 79.5°) — exactly the May 2026 distribution-shift bug.
# ---------------------------------------------------------------------------


_TABLE_PNP_APPLE_DIR = _REPO_ROOT / "scenes" / "table_pnp_apple"
_TABLE_PNP_APPLE_MJCFS = (
    _TABLE_PNP_APPLE_DIR / "g1_gear_wbc_hands_table_pnp_apple.xml",
    _TABLE_PNP_APPLE_DIR / "g1_gear_wbc_table_pnp_apple.xml",
)


# Pose/quat/fovy copied verbatim from RoboCasa
# ``gr00t_wbc/dexmg/gr00trobocasa/robocasa/models/robots/manipulators/g1_robot.py``
# (``_cam_config["{prefix}oak_egoview"]``). Any drift here means the
# inference-time ego frame no longer matches training-time, which is
# exactly the distribution shift the May 2026 fix targets.
_OAK_EGOVIEW_POS = (0.10209156, -0.00937542, 0.42446595)
_OAK_EGOVIEW_QUAT = (0.64367383, 0.26523914, -0.27106013, -0.66472446)
_OAK_EGOVIEW_FOVY = 79.5


def _parse_oak_egoview(xml_path: Path) -> dict[str, tuple[float, ...] | float]:
    """Extract pos/quat/fovy for the ``oak_egoview`` camera from an MJCF."""
    pytest.importorskip("xml.etree.ElementTree")
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    for cam in tree.getroot().iter("camera"):
        if cam.get("name") == "oak_egoview":
            pos = tuple(float(x) for x in (cam.get("pos") or "").split())
            quat = tuple(float(x) for x in (cam.get("quat") or "").split())
            fovy = float(cam.get("fovy") or "0")
            return {"pos": pos, "quat": quat, "fovy": fovy}
    raise AssertionError(
        f"{xml_path} is missing <camera name='oak_egoview' ... /> "
        "— observation builder will silently fall back to head_pov "
        "and re-introduce the May 2026 distribution-shift bug."
    )


@pytest.mark.parametrize(
    "xml_path",
    _TABLE_PNP_APPLE_MJCFS,
    ids=lambda p: p.name,
)
def test_table_pnp_apple_mjcf_has_oak_egoview_camera(xml_path: Path) -> None:
    spec = _parse_oak_egoview(xml_path)
    np.testing.assert_allclose(spec["pos"], _OAK_EGOVIEW_POS, atol=1e-8)
    np.testing.assert_allclose(spec["quat"], _OAK_EGOVIEW_QUAT, atol=1e-8)
    assert float(spec["fovy"]) == _OAK_EGOVIEW_FOVY


def test_hands_mjcf_recolours_hand_meshes_to_near_black() -> None:
    """Every hand-mesh <geom> rgba must be near-black (0.1) to match the
    RoboCasa Tier-A robot appearance (grippers are attached at runtime with
    rgba ``0.1 0.1 0.1 1``). Light-grey hands (``0.7 0.7 0.7 1``) re-introduce
    the May 2026 visual mismatch and the apple-grasp policy mis-fires.
    """
    pytest.importorskip("xml.etree.ElementTree")
    import xml.etree.ElementTree as ET

    hands_xml = _TABLE_PNP_APPLE_DIR / "g1_gear_wbc_hands_table_pnp_apple.xml"
    tree = ET.parse(hands_xml)
    bad: list[str] = []
    for geom in tree.getroot().iter("geom"):
        mesh = geom.get("mesh") or ""
        if not ("left_hand_" in mesh or "right_hand_" in mesh):
            continue
        rgba = (geom.get("rgba") or "").split()
        if not rgba:
            continue
        try:
            r, g, b, _a = (float(x) for x in rgba)
        except ValueError:
            continue
        if not (r <= 0.2 and g <= 0.2 and b <= 0.2):
            bad.append(f"{mesh}: rgba={rgba}")
    assert not bad, (
        "Hand mesh geoms are not near-black; this brings back the May 2026 "
        f"Tier-B vs Tier-A appearance mismatch. Offending geoms:\n  "
        + "\n  ".join(bad)
    )


def test_cli_default_camera_name_is_oak_egoview() -> None:
    """The default ego camera at the CLI must match the training-time view.

    If a future edit reverts this default to ``head_pov`` (or drops the
    default entirely), the inference bridge will once again render from a
    pose the CloudWalk policy never saw at training time.
    """
    mod = _load_inference_module()
    ns = mod.build_argparser().parse_args([])
    assert ns.camera_name == "oak_egoview"


def test_periodic_line_hand_suffix_skips_when_fewer_than_28_targets() -> None:
    mod = _load_inference_module()
    from types import SimpleNamespace

    rt = SimpleNamespace(num_act=15, data=SimpleNamespace(qpos=np.zeros(60, dtype=np.float64)))
    assert mod._periodic_line_hand_suffix(np.zeros(14, dtype=np.float64), rt) == ""


def test_periodic_line_hand_suffix_includes_hand_norms_when_28dof() -> None:
    mod = _load_inference_module()
    from types import SimpleNamespace

    rt = SimpleNamespace(num_act=15, data=SimpleNamespace(qpos=np.zeros(60, dtype=np.float64)))
    atq = np.linspace(0.01, 0.28, 28).astype(np.float64)
    s = mod._periodic_line_hand_suffix(atq, rt)
    assert "|atq.lh|" in s
    assert "|atq.rh|" in s
    assert "|qpos.lh|" in s
    assert "|qpos.rh|" in s


# ---------------------------------------------------------------------------
# action.waist -> rpy_cmd plumbing (bug #5 fix).
#
# These cases pin the contract that ``_apply_action_step`` writes the result
# of ``apply_waist_fn(waist_action)`` into ``rt.control_dict["rpy_cmd"]``
# when (and only when) ``apply_waist_fn`` is provided AND ``action.waist`` is
# in the step. The historical pre-fix behaviour silently dropped the waist
# action and is reproduced by passing ``apply_waist_fn=None`` (legacy path).
# ---------------------------------------------------------------------------


def test_apply_action_step_writes_rpy_cmd_when_waist_fn_provided() -> None:
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)
    captured: dict[str, np.ndarray] = {}

    def fake_apply_waist(w: np.ndarray) -> np.ndarray:
        captured["w_in"] = np.asarray(w).copy()
        return np.array([0.11, 0.22, 0.33], dtype=np.float32)

    action_step = {"action.waist": np.array([0.5, 0.6, 0.7], dtype=np.float32)}
    applied = mod._apply_action_step(
        rt,
        action_step,
        state_now={},
        arm_action_mode="absolute",
        apply_waist_fn=fake_apply_waist,
    )

    # Hook receives the original waist action (cast to float64 internally is ok).
    np.testing.assert_allclose(captured["w_in"], [0.5, 0.6, 0.7])
    # rpy_cmd buffer in the control_dict was overwritten in place.
    np.testing.assert_allclose(
        np.asarray(rt.control_dict["rpy_cmd"]), [0.11, 0.22, 0.33], atol=1e-6
    )
    # Applied dict surfaces the rpy_cmd write for downstream logging.
    np.testing.assert_allclose(applied["rpy_cmd"], [0.11, 0.22, 0.33], atol=1e-6)


def test_apply_action_step_skips_rpy_cmd_when_waist_fn_none() -> None:
    """Legacy / opt-out path: no ``apply_waist_fn`` -> ``rpy_cmd`` untouched.

    Pins the ``--no-apply-waist`` semantics that reproduce the May 2026
    pre-fix bridge so an A/B comparison stays possible.
    """
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)
    rt.control_dict["rpy_cmd"][:] = [0.05, 0.07, -0.03]  # canary

    applied = mod._apply_action_step(
        rt,
        {"action.waist": np.array([1.0, 1.0, 1.0], dtype=np.float32)},
        state_now={},
        arm_action_mode="absolute",
        apply_waist_fn=None,
    )

    np.testing.assert_allclose(
        np.asarray(rt.control_dict["rpy_cmd"]), [0.05, 0.07, -0.03], atol=1e-9
    )
    assert "rpy_cmd" not in applied


def test_apply_action_step_skips_rpy_cmd_when_action_waist_missing() -> None:
    """Hook present but chunk omits ``action.waist``: ``rpy_cmd`` untouched."""
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)
    rt.control_dict["rpy_cmd"][:] = [-0.02, 0.0, 0.04]

    called = []

    def fake_apply_waist(_w: np.ndarray) -> np.ndarray:
        called.append(True)
        return np.zeros(3, dtype=np.float32)

    applied = mod._apply_action_step(
        rt,
        {"action.navigate_command": np.array([0.1, 0.0, 0.0], dtype=np.float32)},
        state_now={},
        arm_action_mode="absolute",
        apply_waist_fn=fake_apply_waist,
    )

    assert called == []
    np.testing.assert_allclose(
        np.asarray(rt.control_dict["rpy_cmd"]), [-0.02, 0.0, 0.04], atol=1e-9
    )
    assert "rpy_cmd" not in applied


def test_apply_action_step_rejects_non_length_3_rpy() -> None:
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)

    def bad_waist(_w: np.ndarray) -> np.ndarray:
        return np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="length-3 rpy"):
        mod._apply_action_step(
            rt,
            {"action.waist": np.zeros(3, dtype=np.float32)},
            state_now={},
            arm_action_mode="absolute",
            apply_waist_fn=bad_waist,
        )


# ---------------------------------------------------------------------------
# hand permutation (bug #6 fix): policy idx (sorted DoF, index→middle→thumb)
# -> MuJoCo qpos idx (kinematic-tree, thumb→middle→index).
# ---------------------------------------------------------------------------


def _canonical_left_hand_perm() -> np.ndarray:
    """Hand permutation derived from the canonical joint-name conventions.

    Policy slot order (matches ``robot_model.get_joint_group_indices("left_hand")``
    on the G1):
        0: left_hand_index_0_joint
        1: left_hand_index_1_joint
        2: left_hand_middle_0_joint
        3: left_hand_middle_1_joint
        4: left_hand_thumb_0_joint
        5: left_hand_thumb_1_joint
        6: left_hand_thumb_2_joint

    MuJoCo qpos slot order (matches ``hand_gripper.LEFT_HAND_JOINT_NAMES``):
        0: left_hand_thumb_0_joint   <- policy slot 4
        1: left_hand_thumb_1_joint   <- policy slot 5
        2: left_hand_thumb_2_joint   <- policy slot 6
        3: left_hand_middle_0_joint  <- policy slot 2
        4: left_hand_middle_1_joint  <- policy slot 3
        5: left_hand_index_0_joint   <- policy slot 0
        6: left_hand_index_1_joint   <- policy slot 1
    """
    return np.asarray([4, 5, 6, 2, 3, 0, 1], dtype=np.int64)


def test_apply_action_step_applies_hand_permutation_left_and_right() -> None:
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)
    perm = _canonical_left_hand_perm()
    hand_perm = (perm.copy(), perm.copy())

    # Unique value per policy slot so the permutation is unambiguous.
    lh_policy = np.array([0.10, 0.11, 0.20, 0.21, 0.30, 0.31, 0.32], dtype=np.float32)
    rh_policy = np.array([-0.10, -0.11, -0.20, -0.21, -0.30, -0.31, -0.32], dtype=np.float32)
    applied = mod._apply_action_step(
        rt,
        {
            "action.left_hand": lh_policy.copy(),
            "action.right_hand": rh_policy.copy(),
        },
        state_now={},
        arm_action_mode="absolute",
        hand_permutation=hand_perm,
    )

    expected_lh_mj = np.array(
        [0.30, 0.31, 0.32, 0.20, 0.21, 0.10, 0.11], dtype=np.float32
    )
    expected_rh_mj = np.array(
        [-0.30, -0.31, -0.32, -0.20, -0.21, -0.10, -0.11], dtype=np.float32
    )

    np.testing.assert_allclose(
        rt.control_dict["arm_target_q"][mod.SL_LEFT_HAND], expected_lh_mj
    )
    np.testing.assert_allclose(
        rt.control_dict["arm_target_q"][mod.SL_RIGHT_HAND], expected_rh_mj
    )
    np.testing.assert_allclose(applied["arm_target_q.left_hand"], expected_lh_mj)
    np.testing.assert_allclose(applied["arm_target_q.right_hand"], expected_rh_mj)


def test_apply_action_step_legacy_writes_hand_without_permutation_when_none() -> None:
    """``--no-permute-hands``: policy vector goes straight through."""
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)

    lh_policy = np.array([0.10, 0.11, 0.20, 0.21, 0.30, 0.31, 0.32], dtype=np.float32)
    mod._apply_action_step(
        rt,
        {"action.left_hand": lh_policy.copy()},
        state_now={},
        arm_action_mode="absolute",
        hand_permutation=None,
    )

    np.testing.assert_allclose(
        rt.control_dict["arm_target_q"][mod.SL_LEFT_HAND], lh_policy
    )


def test_apply_action_step_sets_policy_hand_pd_flag_when_hands_applied() -> None:
    """GR00T hand vectors should tell GearWBC to PD-track ``arm_target_q`` hands."""
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)
    key = mod.GR00T_POLICY_HAND_PD_SOURCE_KEY
    rt.control_dict[key] = False
    mod._apply_action_step(
        rt,
        {"action.right_hand": np.linspace(0.01, 0.07, 7, dtype=np.float32)},
        state_now={},
        arm_action_mode="absolute",
    )
    assert rt.control_dict[key] is True


def test_apply_action_step_clears_policy_hand_pd_flag_when_arms_only() -> None:
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)
    key = mod.GR00T_POLICY_HAND_PD_SOURCE_KEY
    rt.control_dict[key] = True
    mod._apply_action_step(
        rt,
        {
            "action.left_arm": np.linspace(0.1, 0.2, 7, dtype=np.float32),
            "action.right_arm": np.linspace(-0.1, -0.2, 7, dtype=np.float32),
        },
        state_now={},
        arm_action_mode="absolute",
    )
    assert rt.control_dict[key] is False


def test_apply_action_step_rejects_wrong_perm_shape() -> None:
    mod = _load_inference_module()
    rt = _make_fake_rt(mod)
    bad_perm = (np.arange(5, dtype=np.int64), np.arange(7, dtype=np.int64))
    with pytest.raises(ValueError, match="hand permutation"):
        mod._apply_action_step(
            rt,
            {"action.left_hand": np.zeros(7, dtype=np.float32)},
            state_now={},
            arm_action_mode="absolute",
            hand_permutation=bad_perm,
        )


# ---------------------------------------------------------------------------
# CLI defaults for the new flags.
# ---------------------------------------------------------------------------


def test_cli_default_no_apply_waist_false() -> None:
    """Default behaviour: apply ``action.waist`` -> ``rpy_cmd`` (Fix A on)."""
    mod = _load_inference_module()
    ns = mod.build_argparser().parse_args([])
    assert ns.no_apply_waist is False


def test_cli_default_no_permute_hands_false() -> None:
    """Default behaviour: permute ``action.{side}_hand`` to MuJoCo order (Fix B on)."""
    mod = _load_inference_module()
    ns = mod.build_argparser().parse_args([])
    assert ns.no_permute_hands is False


# ---------------------------------------------------------------------------
# Optional integration: Pinocchio FK round-trip for the waist helper.
# Skipped on environments without ``pinocchio`` / ``gr00t_wbc``.
# ---------------------------------------------------------------------------


def _maybe_real_robot_model():
    """Try to import the real ``RobotModel`` from gr00t_wbc, else skip."""
    pytest.importorskip("pinocchio")
    try:
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        wbc_root = (
            repo_root / "Isaac-GR00T" / "external_dependencies" / "GR00T-WholeBodyControl"
        )
        if wbc_root.is_dir() and str(wbc_root) not in sys.path:
            sys.path.insert(0, str(wbc_root))
        from gr00t_wbc.control.robot_model.instantiation import get_robot_type_and_model
    except Exception as exc:
        pytest.skip(f"gr00t_wbc not importable in this env: {exc}")
    _t, model = get_robot_type_and_model("G1", enable_waist_ik=True)
    return model


def _resolve_waist_axis(robot_model: Any, axis: str) -> int:
    """Return the slot in ``action.waist`` corresponding to the named axis."""
    indices = list(robot_model.get_joint_group_indices("waist"))
    names = [str(robot_model.joint_names[i]) for i in indices]
    target = f"waist_{axis}_joint"
    if target not in names:
        pytest.skip(f"robot_model waist group does not expose {target!r}: {names}")
    return names.index(target)


@pytest.mark.parametrize(
    "axis,rpy_slot",
    [("yaw", 2), ("roll", 0), ("pitch", 1)],
)
def test_waist_action_to_rpy_cmd_axis_aligned(axis: str, rpy_slot: int) -> None:
    """Single-axis waist target round-trips to the matching rpy slot.

    Mirrors the upstream ``G1DecoupledWholeBodyPolicy.get_action`` formula.
    Setting only ``waist_<axis>_joint = 0.3`` (others zero) must produce
    ``rpy_cmd`` with ``≈0.3`` in the matching slot and ≈0 elsewhere.
    """
    mod = _load_inference_module()
    robot_model = _maybe_real_robot_model()
    slot = _resolve_waist_axis(robot_model, axis)

    waist = np.zeros(3, dtype=np.float32)
    waist[slot] = 0.3
    rpy = np.asarray(mod._waist_action_to_rpy_cmd(robot_model, waist), dtype=np.float64)
    assert rpy.shape == (3,)
    assert abs(float(rpy[rpy_slot]) - 0.3) < 1e-3
    for j in (0, 1, 2):
        if j != rpy_slot:
            assert abs(float(rpy[j])) < 1e-3


def test_waist_action_to_rpy_cmd_zero_is_zero() -> None:
    """Identity check: zero waist target -> zero rpy_cmd."""
    mod = _load_inference_module()
    robot_model = _maybe_real_robot_model()
    rpy = mod._waist_action_to_rpy_cmd(
        robot_model, np.zeros(3, dtype=np.float32)
    )
    np.testing.assert_allclose(np.asarray(rpy), [0.0, 0.0, 0.0], atol=1e-6)


def test_waist_action_size_mismatch_raises() -> None:
    """Size guard: wrong-length waist vector fails fast."""
    mod = _load_inference_module()
    robot_model = _maybe_real_robot_model()
    with pytest.raises(ValueError, match="waist"):
        mod._waist_action_to_rpy_cmd(robot_model, np.zeros(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# Arm joint order assertion (Fix C).
# ---------------------------------------------------------------------------


def test_assert_arm_joint_order_matches_passes_for_canonical_g1() -> None:
    mod = _load_inference_module()
    robot_model = _maybe_real_robot_model()
    mod.assert_arm_joint_order_matches(robot_model)


def test_assert_arm_joint_order_matches_rejects_drift() -> None:
    """If a future supplemental_info reorders left_arm, we fail fast."""
    mod = _load_inference_module()

    class _StubModel:
        def __init__(self) -> None:
            # Pretend shoulder_pitch / shoulder_roll are swapped.
            self.joint_names = [
                "left_shoulder_roll_joint",
                "left_shoulder_pitch_joint",
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
            ]

        def get_joint_group_indices(self, group: str) -> list[int]:
            if group == "left_arm":
                return list(range(0, 7))
            if group == "right_arm":
                return list(range(7, 14))
            raise ValueError(group)

    with pytest.raises(RuntimeError, match="left_arm"):
        mod.assert_arm_joint_order_matches(_StubModel())


# ---------------------------------------------------------------------------
# build_hand_permutations: integration via name lookup.
# ---------------------------------------------------------------------------


def test_real_hand_perm_policy_to_mj_matches_canonical() -> None:
    """End-to-end check against the *real* ``hand_gripper.hand_perm_policy_to_mj``.

    The stub used during ``_load_inference_module`` mirrors the contract by
    name, but a direct test against the real module catches any drift in
    ``LEFT_HAND_JOINT_NAMES`` / ``RIGHT_HAND_JOINT_NAMES``.
    """
    pytest.importorskip("mujoco")
    # Make sure the test does NOT see the stub installed by other tests.
    sys.modules.pop("hand_gripper", None)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hand_gripper import hand_perm_policy_to_mj

    policy_left = [
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
    ]
    perm = hand_perm_policy_to_mj("left", policy_left)
    np.testing.assert_array_equal(perm, _canonical_left_hand_perm())


def test_real_hand_perm_policy_to_mj_rejects_missing_joint() -> None:
    pytest.importorskip("mujoco")
    sys.modules.pop("hand_gripper", None)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hand_gripper import hand_perm_policy_to_mj

    bad = [
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_extra_joint",
    ]
    with pytest.raises(ValueError, match="not in policy_joint_names"):
        hand_perm_policy_to_mj("left", bad)


def test_build_hand_permutations_against_canonical_naming() -> None:
    mod = _load_inference_module()

    class _StubModel:
        def __init__(self) -> None:
            # Policy joint name order (left then right) matches the
            # robot_model supplemental_info ordering on the G1.
            self.joint_names = [
                "left_hand_index_0_joint",
                "left_hand_index_1_joint",
                "left_hand_middle_0_joint",
                "left_hand_middle_1_joint",
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "right_hand_index_0_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint",
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
            ]

        def get_joint_group_indices(self, group: str) -> list[int]:
            if group == "left_hand":
                return list(range(0, 7))
            if group == "right_hand":
                return list(range(7, 14))
            raise ValueError(group)

    left_perm, right_perm = mod.build_hand_permutations(_StubModel())
    canonical = _canonical_left_hand_perm()
    np.testing.assert_array_equal(left_perm, canonical)
    np.testing.assert_array_equal(right_perm, canonical)
