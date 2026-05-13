"""Joint-space posture regularization for damped least-squares IK (numpy only).

Stacks diagonal ``sqrt(posture_weight) * per_joint_weight`` rows on the task Jacobian
and matching residuals ``sw * (q_neutral - q_current)``. Used from ``solve_dual_arm_ik_v2``
when ``posture_weight > 0`` (live path sets weights in ``gear_wbc_stand`` / ``gear_wbc_arm_ik``).
"""

from __future__ import annotations

import numpy as np


def augment_task_with_posture(
    J_task: np.ndarray,
    e_task: np.ndarray,
    q_current: np.ndarray,
    q_neutral: np.ndarray | None,
    posture_weight: float,
    per_joint_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(J_aug, e_aug)`` with optional posture rows; no-op if disabled.

    ``J_task`` is ``(m, n)``, ``e_task`` is ``(m,)``. ``q_current`` and ``q_neutral``
    are length ``n``. Same stacking used when merging posture rows into a stacked arm IK
    normal equation (see ``arm_ik_v2`` history if re-wiring).
    """
    J = np.asarray(J_task, dtype=np.float64)
    e = np.asarray(e_task, dtype=np.float64).reshape(-1)
    n = int(J.shape[1])
    if q_neutral is None or float(posture_weight) <= 0.0:
        return J, e
    qn = np.asarray(q_neutral, dtype=np.float64).reshape(n)
    qc = np.asarray(q_current, dtype=np.float64).reshape(n)
    if per_joint_weight is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.clip(np.asarray(per_joint_weight, dtype=np.float64).reshape(n), 0.0, None)
    sw = np.sqrt(float(posture_weight)) * w
    J_aug = np.vstack([J, np.diag(sw)])
    e_aug = np.concatenate([e, sw * (qn - qc)], axis=0)
    return J_aug, e_aug


def posture_residual_norm(
    q_current: np.ndarray,
    q_neutral: np.ndarray,
    posture_weight: float,
    per_joint_weight: np.ndarray | None,
) -> float:
    """Scalar diagnostic: weighted RMS of ``(q_neutral - q_current)`` in posture space."""
    if q_neutral is None or float(posture_weight) <= 0.0:
        return 0.0
    n = int(np.asarray(q_current).size)
    qc = np.asarray(q_current, dtype=np.float64).reshape(n)
    qn = np.asarray(q_neutral, dtype=np.float64).reshape(n)
    if per_joint_weight is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.clip(np.asarray(per_joint_weight, dtype=np.float64).reshape(n), 0.0, None)
    sw = np.sqrt(float(posture_weight)) * w
    r = sw * (qn - qc)
    return float(np.linalg.norm(r))
