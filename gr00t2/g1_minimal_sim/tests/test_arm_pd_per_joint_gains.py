"""Per-arm-joint PD slice builder (Phase 1 friction-grasp tuning).

Pure-numpy unit checks: ``build_arm_pd_per_joint`` lays out the kp/kd vectors in the same
``arm_target_q`` slice order (left arm, [left hand], right arm, [right hand]) the runtime PD
in ``gear_wbc_stand.step_physics`` consumes, applies the walk-kd multiplier only to the arm
slice, and rejects mismatched ``n_u``. No MuJoCo / ONNX dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gear_wbc_pd import (  # noqa: E402
    ARM_PD_KD_DEFAULT_PER_JOINT,
    ARM_PD_KD_GRASP_PER_JOINT,
    ARM_PD_KP_DEFAULT_PER_JOINT,
    ARM_PD_KP_GRASP_PER_JOINT,
    ARM_TAU_CLIP_GRASP_NM,
    ARM_TAU_CLIP_NM,
    HAND_PD_KD_DEFAULT,
    HAND_PD_KP_DEFAULT,
    build_arm_pd_per_joint,
    left_right_wrist_pitch_indices,
)


def test_default_per_joint_vectors_preserve_historical_uniform_gains() -> None:
    """``--arm-gain-profile default`` must equal the pre-Phase-1 uniform kp/kd to avoid drift."""
    assert ARM_PD_KP_DEFAULT_PER_JOINT.shape == (7,)
    assert ARM_PD_KD_DEFAULT_PER_JOINT.shape == (7,)
    assert np.allclose(ARM_PD_KP_DEFAULT_PER_JOINT, 45.0)
    assert np.allclose(ARM_PD_KD_DEFAULT_PER_JOINT, 1.2)


def test_grasp_profile_stiffens_elbow_and_wrists() -> None:
    """Grasp profile must keep shoulders at 45 (kp) but bump elbow + wrists."""
    assert ARM_PD_KP_GRASP_PER_JOINT.shape == (7,)
    # Shoulders kept at default — IK reach posture should not change.
    assert np.allclose(ARM_PD_KP_GRASP_PER_JOINT[0:3], 45.0)
    # Elbow + wrist trio strictly stiffer than default.
    assert np.all(ARM_PD_KP_GRASP_PER_JOINT[3:7] > ARM_PD_KP_DEFAULT_PER_JOINT[3:7])
    # wrist_pitch (index 5) is the most contact-loaded DoF — must be the highest of the trio.
    assert ARM_PD_KP_GRASP_PER_JOINT[5] >= ARM_PD_KP_GRASP_PER_JOINT[4]
    assert ARM_PD_KP_GRASP_PER_JOINT[5] >= ARM_PD_KP_GRASP_PER_JOINT[6]
    assert np.all(ARM_PD_KD_GRASP_PER_JOINT[3:7] > ARM_PD_KD_DEFAULT_PER_JOINT[3:7])


def test_clip_constants_are_sane() -> None:
    assert ARM_TAU_CLIP_NM == 30.0  # historical baseline preserved
    assert ARM_TAU_CLIP_GRASP_NM > ARM_TAU_CLIP_NM


def test_build_no_hands_no_walk_tiles_arm_pattern() -> None:
    kp = np.array([45, 45, 45, 60, 90, 120, 60], dtype=np.float32)
    kd = np.array([1.2, 1.2, 1.2, 1.5, 2.0, 2.5, 1.6], dtype=np.float32)
    out_kp, out_kd = build_arm_pd_per_joint(
        14,
        has_hands=False,
        kp_per_arm_joint=kp,
        kd_per_arm_joint=kd,
        kd_walk_scale=1.45,
        walking=False,
    )
    assert out_kp.shape == (14,) and out_kd.shape == (14,)
    np.testing.assert_array_equal(out_kp[0:7], kp)
    np.testing.assert_array_equal(out_kp[7:14], kp)
    np.testing.assert_array_equal(out_kd[0:7], kd)
    np.testing.assert_array_equal(out_kd[7:14], kd)


def test_build_no_hands_walking_scales_only_kd() -> None:
    kp = ARM_PD_KP_DEFAULT_PER_JOINT
    kd = ARM_PD_KD_DEFAULT_PER_JOINT
    out_kp, out_kd = build_arm_pd_per_joint(
        14,
        has_hands=False,
        kp_per_arm_joint=kp,
        kd_per_arm_joint=kd,
        kd_walk_scale=1.45,
        walking=True,
    )
    np.testing.assert_array_equal(out_kp[0:7], kp)
    np.testing.assert_array_equal(out_kp[7:14], kp)
    np.testing.assert_allclose(out_kd[0:7], kd * 1.45)
    np.testing.assert_allclose(out_kd[7:14], kd * 1.45)


def test_build_hands_layout_and_hand_kp_kd() -> None:
    """Hands layout: [LEFT_ARM, LEFT_HAND, RIGHT_ARM, RIGHT_HAND] — hand slots use hand_kp/kd."""
    kp = ARM_PD_KP_GRASP_PER_JOINT
    kd = ARM_PD_KD_GRASP_PER_JOINT
    out_kp, out_kd = build_arm_pd_per_joint(
        28,
        has_hands=True,
        kp_per_arm_joint=kp,
        kd_per_arm_joint=kd,
        kd_walk_scale=1.45,
        walking=False,
        hand_kp=42.0,
        hand_kd=0.31,
    )
    assert out_kp.shape == (28,) and out_kd.shape == (28,)
    np.testing.assert_array_equal(out_kp[0:7], kp)
    np.testing.assert_array_equal(out_kp[7:14], np.full(7, 42.0, dtype=np.float32))
    np.testing.assert_array_equal(out_kp[14:21], kp)
    np.testing.assert_array_equal(out_kp[21:28], np.full(7, 42.0, dtype=np.float32))
    np.testing.assert_array_equal(out_kd[0:7], kd)
    np.testing.assert_array_equal(out_kd[7:14], np.full(7, 0.31, dtype=np.float32))
    np.testing.assert_array_equal(out_kd[14:21], kd)
    np.testing.assert_array_equal(out_kd[21:28], np.full(7, 0.31, dtype=np.float32))


def test_build_hands_walking_scales_arm_only_not_hand() -> None:
    """When walking, only the 7 arm joints get kd_walk_scale; hand kd stays at hand_kd."""
    kd = ARM_PD_KD_GRASP_PER_JOINT
    _, out_kd = build_arm_pd_per_joint(
        28,
        has_hands=True,
        kp_per_arm_joint=ARM_PD_KP_GRASP_PER_JOINT,
        kd_per_arm_joint=kd,
        kd_walk_scale=2.0,
        walking=True,
        hand_kp=HAND_PD_KP_DEFAULT,
        hand_kd=HAND_PD_KD_DEFAULT,
    )
    np.testing.assert_allclose(out_kd[0:7], kd * 2.0)
    np.testing.assert_allclose(out_kd[14:21], kd * 2.0)
    # Hand kd unaffected by walk scale.
    np.testing.assert_allclose(out_kd[7:14], HAND_PD_KD_DEFAULT)
    np.testing.assert_allclose(out_kd[21:28], HAND_PD_KD_DEFAULT)


def test_build_rejects_mismatched_n_u() -> None:
    with pytest.raises(ValueError, match="has_hands=True expects n_u=28"):
        build_arm_pd_per_joint(
            14,
            has_hands=True,
            kp_per_arm_joint=ARM_PD_KP_DEFAULT_PER_JOINT,
            kd_per_arm_joint=ARM_PD_KD_DEFAULT_PER_JOINT,
            kd_walk_scale=1.0,
            walking=False,
        )
    with pytest.raises(ValueError, match="has_hands=False expects n_u=14"):
        build_arm_pd_per_joint(
            28,
            has_hands=False,
            kp_per_arm_joint=ARM_PD_KP_DEFAULT_PER_JOINT,
            kd_per_arm_joint=ARM_PD_KD_DEFAULT_PER_JOINT,
            kd_walk_scale=1.0,
            walking=False,
        )


def test_wrist_pitch_indices_match_layout() -> None:
    """``--print-arm-tau`` indexes wrist_pitch in the n_u-length slice; verify both modes."""
    # No hands: arm slice is [LEFT_ARM(7), RIGHT_ARM(7)] -> wrist_pitch at 5 and 7+5=12.
    l, r = left_right_wrist_pitch_indices(has_hands=False)
    assert (l, r) == (5, 12)
    # Hands: [LEFT_ARM(7), LEFT_HAND(7), RIGHT_ARM(7), RIGHT_HAND(7)] -> 5 and 14+5=19.
    l, r = left_right_wrist_pitch_indices(has_hands=True)
    assert (l, r) == (5, 19)
