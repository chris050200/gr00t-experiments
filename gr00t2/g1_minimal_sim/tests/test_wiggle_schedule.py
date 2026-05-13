"""Wiggle schedule: legacy path + precomputed noise tables (no MuJoCo)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from play_g1_gear_wbc import build_wiggle_schedule


def _ns(**kwargs: object) -> argparse.Namespace:
    base = dict(
        wiggle_seed=None,
        wiggle_randomize=False,
        wiggle_arm_noise_rad=0.0,
        wiggle_loco_scale=1.0,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_wiggle_legacy_matches_fixed_period() -> None:
    s = build_wiggle_schedule(1000, _ns())
    assert s.is_legacy
    assert s.period == 400.0
    assert s.phase0 == 0.0
    assert s.noise_atq is None
    ph = 2.0 * math.pi * 100.0 / 400.0
    assert math.isclose(ph, math.pi / 2.0)


def test_wiggle_seed_produces_noise_table_shape() -> None:
    s = build_wiggle_schedule(50, _ns(wiggle_seed=12345, wiggle_arm_noise_rad=0.02))
    assert not s.is_legacy
    assert s.noise_atq is not None
    assert s.noise_atq.shape == (50, 28)
    assert s.noise_grip is not None
    assert s.noise_grip.shape == (50, 2)


def test_wiggle_randomize_without_seed_is_non_legacy() -> None:
    s = build_wiggle_schedule(10, _ns(wiggle_randomize=True))
    assert not s.is_legacy
    assert s.period >= 320.0 and s.period <= 480.0


def test_wiggle_loco_scale_only_triggers_variant() -> None:
    s = build_wiggle_schedule(5, _ns(wiggle_loco_scale=0.0))
    assert not s.is_legacy
    assert s.loco_scale == 0.0
