"""P2 — Teleop / runtime step logger: GR00T-shaped **obs** + **action** at fixed sample rate.

Actions mirror the policy-side layout used by ``WholeBodyControlWrapper`` / ``concat_action``:
``navigate_command``, ``base_height_command``, ``waist``, ``left_arm``, ``right_arm``,
``left_hand``, ``right_hand`` (keys prefixed with ``action.`` for parity with eval).

Episodes are saved as compressed NPZ + small ``metadata.json`` (not yet full LeRobot format — P3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gr00t_observation_builder import Gr00tObservationBuilder
from hand_gripper import SL_LEFT_ARM, SL_LEFT_HAND, SL_RIGHT_ARM, SL_RIGHT_HAND


def extract_gr00t_action_dict(
    rt: Any,
    *,
    state_obs: dict[str, Any] | None = None,
    relative_arm_actions: bool = True,
) -> dict[str, np.ndarray]:
    """Build ``action.*`` arrays from :class:`gear_wbc_stand.GearWBCRuntime` ``control_dict``.

    * ``navigate_command`` ← ``loco_cmd`` (3,)
    * ``base_height_command`` ← ``height_cmd`` (1,)
    * ``waist`` ← ``target_dof_pos[12:15]`` — ONNX leg policy waist targets (matches 12 leg + 3 waist layout in yaml)
    * Arm / hand slices from ``arm_target_q`` when ``n_arm`` > 0; otherwise zeros (7,) each.

    When ``relative_arm_actions`` is true (default), ``left_arm`` / ``right_arm`` are converted
    to deltas against ``state.left_arm`` / ``state.right_arm`` from ``state_obs`` so logs align with
    the ``unitree_g1`` modality config defaults.
    """
    with rt.cmd_lock:
        loco = np.asarray(rt.control_dict["loco_cmd"], dtype=np.float32).reshape(3).copy()
        hcmd = float(rt.control_dict["height_cmd"])
        tdp = np.asarray(rt.target_dof_pos, dtype=np.float32).reshape(-1)
        if rt.n_arm > 0:
            atq = np.asarray(rt.control_dict["arm_target_q"], dtype=np.float32).reshape(-1)
        else:
            atq = np.array([], dtype=np.float32)

    z7 = np.zeros(7, dtype=np.float32)

    if tdp.shape[0] != 15:
        raise ValueError(f"expected target_dof_pos length 15, got {tdp.shape[0]}")
    waist = tdp[12:15].astype(np.float32)

    out: dict[str, np.ndarray] = {
        "action.navigate_command": loco,
        "action.base_height_command": np.array([hcmd], dtype=np.float32),
        "action.waist": waist,
    }

    if rt.has_hands and rt.n_arm >= 28:
        out["action.left_arm"] = atq[SL_LEFT_ARM].astype(np.float32).copy()
        out["action.right_arm"] = atq[SL_RIGHT_ARM].astype(np.float32).copy()
        out["action.left_hand"] = atq[SL_LEFT_HAND].astype(np.float32).copy()
        out["action.right_hand"] = atq[SL_RIGHT_HAND].astype(np.float32).copy()
    elif rt.n_arm >= 14:
        out["action.left_arm"] = atq[SL_LEFT_ARM].astype(np.float32).copy()
        out["action.right_arm"] = atq[SL_RIGHT_ARM].astype(np.float32).copy()
        out["action.left_hand"] = z7.copy()
        out["action.right_hand"] = z7.copy()
    else:
        out["action.left_arm"] = z7.copy()
        out["action.right_arm"] = z7.copy()
        out["action.left_hand"] = z7.copy()
        out["action.right_hand"] = z7.copy()

    if relative_arm_actions:
        if state_obs is None:
            raise ValueError(
                "relative_arm_actions requires state_obs with state.left_arm/state.right_arm"
            )
        s_la = np.asarray(state_obs["state.left_arm"], dtype=np.float32).reshape(7)
        s_ra = np.asarray(state_obs["state.right_arm"], dtype=np.float32).reshape(7)
        out["action.left_arm"] = out["action.left_arm"] - s_la
        out["action.right_arm"] = out["action.right_arm"] - s_ra

    return out


def _strip_obs_for_storage(obs: dict[str, Any]) -> dict[str, Any]:
    """Drop redundant ``q``; keep ``state.*``, language, video keys."""
    o: dict[str, Any] = {}
    for k, v in obs.items():
        if k == "q":
            continue
        o[k] = v
    return o


@dataclass
class Gr00tStepFrame:
    """One timestep after ``step_physics`` / ``env.step``."""

    step_index: int
    wall_time_s: float
    obs: dict[str, Any]
    action: dict[str, np.ndarray]


class Gr00tTeleopEpisodeLogger:
    """Accumulates frames at fixed rate; call :meth:`record_step` after each ``env.step``."""

    def __init__(
        self,
        rt: Any,
        *,
        obs_builder: Gr00tObservationBuilder,
        task_description: str,
        include_video: bool = False,
        sample_hz: float = 50.0,
        relative_arm_actions: bool = True,
    ) -> None:
        self._rt = rt
        self._obs_builder = obs_builder
        self._task = task_description
        self._include_video = bool(include_video)
        self._sample_hz = float(sample_hz)
        if self._sample_hz <= 0.0:
            raise ValueError(f"sample_hz must be > 0, got {sample_hz}")
        self._relative_arm_actions = bool(relative_arm_actions)
        self._sim_dt = float(self._rt.model.opt.timestep)
        if self._sim_dt <= 0.0:
            raise ValueError(f"invalid model.opt.timestep={self._sim_dt}")
        self._sample_period_s = 1.0 / self._sample_hz
        self._next_sample_time_s = self._sample_period_s
        self.frames: list[Gr00tStepFrame] = []

    def record_step(self) -> Gr00tStepFrame | None:
        """Snapshot obs + action at fixed rate (call **after** physics step)."""
        rt = self._rt
        sim_time_s = float(int(rt.counter) * self._sim_dt)
        if sim_time_s + 1e-12 < self._next_sample_time_s:
            return None
        obs = self._obs_builder.build(
            rt.model,
            rt.data,
            task_description=self._task,
            include_video=self._include_video,
        )
        action = extract_gr00t_action_dict(
            rt,
            state_obs=obs,
            relative_arm_actions=self._relative_arm_actions,
        )
        frame = Gr00tStepFrame(
            step_index=int(rt.counter),
            wall_time_s=sim_time_s,
            obs=_strip_obs_for_storage(obs),
            action=action,
        )
        self.frames.append(frame)
        while self._next_sample_time_s <= sim_time_s + 1e-12:
            self._next_sample_time_s += self._sample_period_s
        return frame

    def save_npz(
        self,
        path: str | Path,
        *,
        episode_index: int = 0,
    ) -> Path:
        """Stack frames into one compressed ``.npz`` + ``metadata.json`` next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            raise ValueError("no frames to save")

        T = len(self.frames)
        meta_action_keys = sorted(self.frames[0].action.keys())
        meta_obs_state_keys = sorted(
            k for k in self.frames[0].obs.keys() if k.startswith("state.")
        )

        arrays: dict[str, Any] = {
            "T": np.int32(T),
            "episode_index": np.int32(episode_index),
            "step_index": np.array([f.step_index for f in self.frames], dtype=np.int32),
            "wall_time_s": np.array([f.wall_time_s for f in self.frames], dtype=np.float64),
        }
        for ak in meta_action_keys:
            stacked = np.stack([f.action[ak] for f in self.frames], axis=0)
            # e.g. action.navigate_command -> action_navigate_command
            arrays[ak.replace(".", "_")] = stacked

        for sk in meta_obs_state_keys:
            stacked = np.stack(
                [np.asarray(f.obs[sk], dtype=np.float64) for f in self.frames], axis=0
            )
            arrays[sk.replace(".", "_")] = stacked

        lang0 = self.frames[0].obs.get("annotation.human.task_description", "")
        if not all(
            f.obs.get("annotation.human.task_description", "") == lang0
            for f in self.frames
        ):
            raise ValueError("task description changed mid-episode")
        # UTF-8 bytes for np.savez (avoid numpy object string quirks)
        arrays["annotation_human_task_description_bytes"] = np.frombuffer(
            str(lang0).encode("utf-8"), dtype=np.uint8
        )

        if self._include_video and "ego_view_image" in self.frames[0].obs:
            vid = np.stack(
                [f.obs["ego_view_image"] for f in self.frames], axis=0
            )
            if vid.dtype != np.uint8:
                vid = vid.astype(np.uint8)
            arrays["ego_view_image"] = vid

        np.savez_compressed(path, **arrays)

        meta = {
            "schema": "g1_minimal_sim.gr00t_teleop_logger/v1",
            "episode_index": episode_index,
            "T": T,
            "task_description": str(lang0),
            "action_keys": meta_action_keys,
            "obs_state_keys": meta_obs_state_keys,
            "include_video": self._include_video,
            "sample_hz": self._sample_hz,
            "sim_dt": self._sim_dt,
            "relative_arm_actions": self._relative_arm_actions,
            "notes": (
                "P2 raw log; not LeRobot. Frames are sampled at fixed sim-time rate, and "
                "left/right arm actions are logged as relative deltas against state.*."
            ),
        }
        meta_path = path.parent / f"{path.stem}_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return path
