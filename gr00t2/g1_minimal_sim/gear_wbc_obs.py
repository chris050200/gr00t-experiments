"""Policy observation layout (86-D) matching upstream Gear WBC MuJoCo script."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    q_conj = np.array([w, -x, -y, -z], dtype=np.float64)
    return np.array(
        [
            v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
            + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
            v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
            + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
            v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
            + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
            + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
        ],
        dtype=np.float32,
    )


def _gravity_orientation(quat: np.ndarray) -> np.ndarray:
    g = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return _quat_rotate_inverse(quat, g)


def compute_single_obs(
    data: mujoco.MjData,
    config: dict[str, Any],
    action: np.ndarray,
    control_dict: dict[str, Any],
    policy_qpos_adr: np.ndarray,
    policy_dof_adr: np.ndarray,
    torso_index: int,
) -> np.ndarray:
    """Match ``run_mujoco_gear_wbc.py::compute_observation`` layout (86-D).

    Uses the 29 leg/waist/arm joints the ONNX was trained on, even when the MJCF adds
    articulated hands (extra qpos after the wrists).
    """
    n_policy = int(policy_qpos_adr.shape[0])
    command = np.zeros(7, dtype=np.float32)
    command[:3] = np.asarray(control_dict["loco_cmd"], dtype=np.float32)[:3] * config["cmd_scale"]
    command[3] = float(control_dict["height_cmd"])
    command[4:7] = np.asarray(control_dict["rpy_cmd"], dtype=np.float32)[:3]

    qj = data.qpos[policy_qpos_adr].astype(np.float32).copy()
    dqj = data.qvel[policy_dof_adr].astype(np.float32).copy()
    quat = data.qpos[3:7].copy()
    omega = data.qvel[3:6].copy()

    padded_defaults = np.zeros(n_policy, dtype=np.float32)
    L = min(len(config["default_angles"]), n_policy)
    padded_defaults[:L] = config["default_angles"][:L]

    qj_scaled = (qj - padded_defaults) * config["dof_pos_scale"]
    dqj_scaled = dqj * config["dof_vel_scale"]
    gravity_orientation = _gravity_orientation(quat)
    omega_scaled = omega * config["ang_vel_scale"]

    single_obs = np.zeros(86, dtype=np.float32)
    single_obs[0:7] = command[:7]
    single_obs[7:10] = omega_scaled
    single_obs[10:13] = gravity_orientation
    single_obs[13 : 13 + n_policy] = qj_scaled
    single_obs[13 + n_policy : 13 + 2 * n_policy] = dqj_scaled
    single_obs[13 + 2 * n_policy : 13 + 2 * n_policy + 15] = action
    return single_obs
