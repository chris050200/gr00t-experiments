"""Gymnasium environment wrapping Gear WBC + MuJoCo G1 (Phase 2a spine).

Use ``play_g1_gear_wbc.py`` or ``gymnasium.make`` after ``register_g1_gear_wbc_env()``.
Registered ids: ``G1GearWBC-v0`` (default: stylish diner), ``G1GearWBC-TablePnP-v0``
(table scenes). Pass ``scene="table_pnp"`` for the vendored table + blue ``table_box``,
``scene="table_pnp_apple"`` for the B1 apple + plate MJCF under ``scenes/table_pnp_apple/``,
``scene="floor"``, ``scene="stylish_diner"``, or ``config_yaml=...`` to :class:`G1GearWBCEnv`.

Teleop (keyboard, OpenVR, legacy UDP, and/or ``--vr-teleop`` stream from
``vr_teleop.openvr_udp_sender``) updates the same ``control_dict`` as
``spawn_g1_floor.py --stand`` with matching flags; each ``step`` advances one simulation timestep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gear_wbc_stand import GearWBCRuntime, resolve_gear_wbc_config

ENV_ID = "G1GearWBC-v0"
ENV_ID_TABLE_PNP = "G1GearWBC-TablePnP-v0"


def register_g1_gear_wbc_env() -> None:
    """Idempotent registration of default (stylish diner), floor, and table env ids."""
    for eid, kwargs in (
        (ENV_ID, {"scene": "stylish_diner"}),
        (ENV_ID_TABLE_PNP, {"scene": "table_pnp"}),
    ):
        try:
            gym.spec(eid)
            continue
        except gym.error.Error:
            pass
        gym.register(
            id=eid,
            entry_point=f"{__name__}:G1GearWBCEnv",
            kwargs=kwargs,
        )


class G1GearWBCEnv(gym.Env):
    """Minimal Box obs (qpos, qvel); dummy scalar action (teleop drives ``control_dict``)."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        resources_dir: Path | str | None = None,
        *,
        enable_teleop: bool = False,
        openvr_teleop: bool = False,
        udp_teleop_bind: str | None = None,
        vr_teleop_bind: str | None = None,
        vr_receiver_yaw_deg: float = 0.0,
        vr_ik_mode: str = "off",
        vr_debug_stream: bool = False,
        vr_debug_torso_z_mode: str = "full",
        vr_debug_z_offset_m: float = 1.0,
        vr_stream_calib_orient: str = "yaw_only",
        vr_ik_anchor_mode: str = "legacy",
        vr_head_orient_mode: str = "yaw_only",
        vr_ik_z_offset_m: float = 0.0,
        use_hands: bool = False,
        log_path: str | Path | None = None,
        log_hz: float = 30.0,
        log_append: bool = False,
        forensic_burst: bool = False,
        forensic_pre_steps: int = 180,
        forensic_post_steps: int = 240,
        forensic_trigger_pos_err_m: float = 0.15,
        forensic_trigger_quat_err_deg: float = 45.0,
        forensic_trigger_raw_step_l2: float = 1.0,
        forensic_trigger_step_scale_max: float = 0.40,
        forensic_cooldown_steps: int = 240,
        render_mode: str | None = None,
        scene: str = "stylish_diner",
        config_yaml: str | Path | None = None,
        debug_ee_targets: bool = False,
        keyboard_grasp_primitives: bool = False,
        lock_ee_orient: bool = False,
        arm_pd_kp_per_joint: np.ndarray | None = None,
        arm_pd_kd_per_joint: np.ndarray | None = None,
        arm_tau_clip_nm: float | None = None,
        block_density: float | None = None,
        block_friction: tuple[float, float, float] | None = None,
        print_arm_tau: bool = False,
        grip_key_step: float | None = None,
        print_grip_force: bool = False,
    ):
        super().__init__()
        if (
            enable_teleop or openvr_teleop or udp_teleop_bind or vr_teleop_bind
        ) and render_mode == "rgb_array":
            raise ValueError(
                "enable_teleop / openvr_teleop / udp_teleop_bind / vr_teleop_bind with rgb_array "
                "is unsupported; use human viewer"
            )
        self.render_mode = render_mode
        rd, cy = resolve_gear_wbc_config(
            resources_dir,
            use_hands=use_hands,
            scene=scene,
            config_yaml=config_yaml,
        )
        self._rt = GearWBCRuntime(
            rd,
            teleop=enable_teleop,
            openvr_teleop=openvr_teleop,
            udp_teleop_bind=udp_teleop_bind,
            vr_teleop_bind=vr_teleop_bind,
            vr_receiver_yaw_deg=vr_receiver_yaw_deg,
            vr_ik_mode=vr_ik_mode,
            vr_debug_stream=vr_debug_stream,
            vr_debug_torso_z_mode=vr_debug_torso_z_mode,
            vr_debug_z_offset_m=vr_debug_z_offset_m,
            vr_stream_calib_orient=vr_stream_calib_orient,
            vr_ik_anchor_mode=vr_ik_anchor_mode,
            vr_head_orient_mode=vr_head_orient_mode,
            vr_ik_z_offset_m=vr_ik_z_offset_m,
            log_path=log_path,
            log_hz=log_hz,
            log_append=log_append,
            forensic_burst=forensic_burst,
            forensic_pre_steps=forensic_pre_steps,
            forensic_post_steps=forensic_post_steps,
            forensic_trigger_pos_err_m=forensic_trigger_pos_err_m,
            forensic_trigger_quat_err_deg=forensic_trigger_quat_err_deg,
            forensic_trigger_raw_step_l2=forensic_trigger_raw_step_l2,
            forensic_trigger_step_scale_max=forensic_trigger_step_scale_max,
            forensic_cooldown_steps=forensic_cooldown_steps,
            config_yaml=cy,
            keyboard_grasp_primitives=keyboard_grasp_primitives,
            lock_ee_orient=lock_ee_orient,
            arm_pd_kp_per_joint=arm_pd_kp_per_joint,
            arm_pd_kd_per_joint=arm_pd_kd_per_joint,
            arm_tau_clip_nm=arm_tau_clip_nm,
            block_density=block_density,
            block_friction=block_friction,
            print_arm_tau=print_arm_tau,
            grip_key_step=grip_key_step,
            print_grip_force=print_grip_force,
        )
        if enable_teleop:
            self._rt.start_teleop_listener()
        if openvr_teleop:
            self._rt.start_openvr_poller()

        self.debug_ee_targets = bool(debug_ee_targets)

        nq, nv = self._rt.model.nq, self._rt.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(nq + nv,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._renderer: mujoco.Renderer | None = None

    @property
    def model(self) -> mujoco.MjModel:
        return self._rt.model

    @property
    def data(self) -> mujoco.MjData:
        return self._rt.data

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._rt.reset()
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray | SupportsFloat
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action  # teleop or future policy will use control_dict / extended API
        self._rt.step_physics()
        obs = self._get_obs()
        z = float(self._rt.data.qpos[2])
        info = {"pelvis_z": z}
        return obs, 0.0, False, False, info

    def _get_obs(self) -> np.ndarray:
        d = self._rt.data
        return np.concatenate([d.qpos, d.qvel]).astype(np.float64)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._rt.model, height=480, width=640)
            self._renderer.update_scene(self._rt.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        self._rt.stop_openvr_poller()
        self._rt.stop_udp_teleop()
        self._rt.stop_vr_teleop()
        self._rt.close_trace()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
