"""Unit tests for ``arm_ik_posture`` (numpy only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arm_ik_posture import augment_task_with_posture, posture_residual_norm  # noqa: E402


def test_augment_noop_when_weight_zero() -> None:
    J = np.eye(6, 7)
    e = np.arange(6, dtype=np.float64)
    q = np.zeros(7)
    qn = np.ones(7)
    Ja, ea = augment_task_with_posture(J, e, q, qn, 0.0, None)
    assert Ja.shape == J.shape
    assert ea.shape == e.shape
    assert np.allclose(Ja, J)
    assert np.allclose(ea, e)


def test_augment_noop_when_neutral_none() -> None:
    J = np.random.RandomState(0).randn(12, 14)
    e = np.random.RandomState(1).randn(12)
    q = np.zeros(14)
    Ja, ea = augment_task_with_posture(J, e, q, None, 0.08, np.ones(14))
    assert np.allclose(Ja, J)
    assert np.allclose(ea, e)


def test_augment_shapes_and_formula() -> None:
    m, n = 12, 14
    J = np.random.RandomState(42).randn(m, n)
    e = np.random.RandomState(43).randn(m)
    q = np.linspace(0, 0.1, n)
    qn = np.linspace(0.2, 0.3, n)
    pw = 0.04
    w = np.arange(1, n + 1, dtype=np.float64)
    Ja, ea = augment_task_with_posture(J, e, q, qn, pw, w)
    assert Ja.shape == (m + n, n)
    assert ea.shape == (m + n,)
    sw = np.sqrt(pw) * w
    assert np.allclose(Ja[:m], J)
    assert np.allclose(ea[:m], e)
    assert np.allclose(Ja[m:], np.diag(sw))
    assert np.allclose(ea[m:], sw * (qn - q))


def test_posture_residual_norm() -> None:
    q = np.zeros(3)
    qn = np.ones(3)
    r = posture_residual_norm(q, qn, 4.0, np.array([1.0, 2.0, 3.0]))
    sw = np.sqrt(4.0) * np.array([1.0, 2.0, 3.0])
    assert abs(r - float(np.linalg.norm(sw * (qn - q)))) < 1e-9
