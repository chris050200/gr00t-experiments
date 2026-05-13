"""Regression test: table_box scripted walk keeps Gear WBC walking + tracks arms reasonably."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")
pytest.importorskip("matplotlib")

from table_box_walk_regression import run_table_box_walk_regression  # noqa: E402


def test_table_box_walk_teleop_regression_outputs_and_walking(tmp_path: Path) -> None:
    out_dir = tmp_path / "table_box_walk_regression"
    summary, trace_path, plot_path = run_table_box_walk_regression(
        out_dir=out_dir, steps=2400, lock_ee_orient=False
    )

    # Artifacts requested by workflow: raw trace + graph + summary JSON.
    assert trace_path.is_file(), f"missing trace artifact: {trace_path}"
    assert plot_path.is_file(), f"missing PNG artifact: {plot_path}"
    assert (out_dir / "table_pnp_baseline_off_summary.json").is_file()

    # Ensure scripted locomotion actually engages Gear WBC walk branch regime.
    assert summary.walk_steps >= 1200, f"insufficient walk steps: {summary.walk_steps}"
    assert summary.walk_fraction >= 0.45, f"walk fraction too low: {summary.walk_fraction:.3f}"
    assert (
        summary.pelvis_xy_displacement_m >= 0.85
    ), f"pelvis did not move enough: {summary.pelvis_xy_displacement_m:.3f} m"

    # Baseline stability envelopes (tune after collecting more table_box baselines).
    assert summary.max_err_left_m < 0.55, f"left max pos err too high: {summary.max_err_left_m:.3f} m"
    assert summary.max_err_right_m < 0.55, f"right max pos err too high: {summary.max_err_right_m:.3f} m"
    assert summary.p95_err_left_m < 0.35, f"left p95 pos err too high: {summary.p95_err_left_m:.3f} m"
    assert summary.p95_err_right_m < 0.35, f"right p95 pos err too high: {summary.p95_err_right_m:.3f} m"
