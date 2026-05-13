"""Gear WBC ONNX standing / locomotion in MuJoCo (same stack as GR00T G1 sim).

The GR00T **VLA** predicts high-level actions (joints + navigate_command); **legged balance
and standing** come from these pretrained **Gear WBC** ONNX policies, not from loading a VLA
checkpoint.

Policy files live under ``.../resources/robots/g1/policy/``. The yaml names ``ft92.onnx`` /
``ft109.onnx``; some checkouts ship ``GR00T-WholeBodyControl-Balance.onnx`` /
``GR00T-WholeBodyControl-Walk.onnx`` instead — both are accepted automatically.

Implementation is split across ``gear_wbc_config``, ``gear_wbc_obs``, ``gear_wbc_onnx``,
``gear_wbc_pd``, ``gear_wbc_teleop``, ``gear_wbc_vr_stream``, ``gear_wbc_vr_trace``,
``gear_wbc_arm_ik`` — this module keeps the public :class:`GearWBCRuntime` façade.
"""

from __future__ import annotations

import collections
import threading
import warnings
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from arm_ik import body_palm_pose, build_g1_arm_ik_specs, init_ee_control_dict
from ee_frame import normalize_quat, quat_mul, quat_wxyz_to_rotmat
from gear_wbc_arm_ik import apply_arm_ik_step
from gear_wbc_config import (
    POLICY_JOINT_NAMES,
    default_g1_resources_dir,
    load_gear_wbc_config,
    resolve_gear_wbc_config,
    _policy_joint_addrs,
)
from gear_wbc_onnx import make_onnx_runner
from gear_wbc_obs import compute_single_obs
from gear_wbc_pd import (
    ARM_PD_KD_DEFAULT_PER_JOINT,
    ARM_PD_KP_DEFAULT_PER_JOINT,
    ARM_TAU_CLIP_NM,
    arm_actuator_qpos_slice_indices,
    build_arm_pd_per_joint,
    left_right_wrist_pitch_indices,
    pd_control,
)
from gear_wbc_teleop import (
    apply_mujoco_gear_wbc_key,
    start_gear_wbc_teleop_listener,
    sync_keyboard_loco_hold,
)
from gear_wbc_vr_stream import (
    apply_vr_stream_ik_targets,
    apply_vr_stream_locomotion,
    snapshot_vr_receiver,
)
from gear_wbc_vr_trace import emit_teleop_ik_row, emit_vr_compare_row
from hand_gripper import (
    GR00T_POLICY_HAND_PD_SOURCE_KEY,
    GRIP_KEY_STEP_DEFAULT,
    SL_LEFT_HAND,
    SL_RIGHT_HAND,
    hand_target_q,
    init_gripper_control_dict,
    reset_gripper_control_dict,
)
from legacy_vr_code.udp_teleop_receiver import (
    apply_udp_packet_to_control_dict,
    start_udp_receiver_for_runtime,
)
from trace_log import JsonlTrace
from vr_stream_ik_mode import normalize_vr_ik_anchor_mode, normalize_vr_ik_mode
from vr_teleop.openvr_stream import start_openvr_stream_receiver
from vr_teleop.vr_stream_torso_compose import (
    VrTorsoVizCalib,
    normalize_head_orient_mode,
    torso_xmat_with_yaw_offset,
    torso_xquat_with_receiver_yaw,
)

# Stronger orientation tracking in dual-arm IK. The previous value (0.45)
# let wrist orientation drift accumulate during long locomotion.
_ARM_IK_ROT_WEIGHT = 2.5
# Bias stacked IK toward arm home pose and previous solution (reduces pretzel / flip).
# Per-joint shaping lives in ``gear_wbc_arm_ik`` (looser shoulder pitch/roll + wrists).
_ARM_IK_POSTURE_WEIGHT = 0.042
# Lower temporal pull so IK can recover instead of sticking in a bad branch
# during long locomotion segments.
_ARM_IK_TEMPORAL_WEIGHT = 0.02
# Per sim-step cap on arm joint change after IK (rad per joint), then optional L2 cap on the 14D delta.
_ARM_IK_FRAME_DQ_MAX = 0.16
_ARM_IK_FRAME_DQ_L2_CAP = 1.15
# Blend from current q toward clamped IK: 1.0 = full step to clamped target (default).
_ARM_IK_STEP_BLEND = 1.0
# Adaptive posture: multiply base posture weight when ||q - q_home|| is large (capped).
_ARM_IK_POSTURE_DEV_SCALE = 0.65
_ARM_IK_POSTURE_DEV_BOOST = 2.0
_ARM_IK_POSTURE_BOOST_CAP = 3.6
# Extra arm joint damping while walking (targets move in world; reduces oscillation vs IK).
_ARM_KD_BASE = 1.2
_ARM_KD_WALK_SCALE = 1.45


class GearWBCRuntime:
    """Mutable Gear WBC sim state: one ``step_physics()`` = one ``mj_step``."""

    def __init__(
        self,
        resources_dir: Path,
        *,
        teleop: bool = False,
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
        control_dict: dict[str, Any] | None = None,
        config_yaml: str = "g1_gear_wbc.yaml",
        keyboard_grasp_primitives: bool = False,
        lock_ee_orient: bool = False,
        freeze_arms_when_walk: bool = False,
        arm_pd_kp_per_joint: np.ndarray | None = None,
        arm_pd_kd_per_joint: np.ndarray | None = None,
        arm_tau_clip_nm: float | None = None,
        block_density: float | None = None,
        block_friction: tuple[float, float, float] | None = None,
        print_arm_tau: bool = False,
        grip_key_step: float | None = None,
        print_grip_force: bool = False,
    ) -> None:
        self.resources_dir = Path(resources_dir).resolve()
        self.teleop = teleop
        self.openvr_teleop = openvr_teleop
        self.udp_teleop = bool(udp_teleop_bind)
        self.vr_teleop = bool(vr_teleop_bind)
        self._udp_receiver = (
            start_udp_receiver_for_runtime(self, udp_teleop_bind)
            if udp_teleop_bind
            else None
        )
        self._vr_teleop_receiver = (
            start_openvr_stream_receiver(vr_teleop_bind) if vr_teleop_bind else None
        )
        self.vr_receiver_yaw_deg = float(vr_receiver_yaw_deg)
        self.vr_ik_mode = normalize_vr_ik_mode(vr_ik_mode)
        if self.vr_ik_mode != "off" and not self.vr_teleop:
            raise ValueError("vr_ik_mode requires vr_teleop_bind (UDP stream receiver)")
        self.vr_debug_stream = bool(vr_debug_stream)
        self.vr_debug_torso_z_mode = str(vr_debug_torso_z_mode)
        self.vr_debug_z_offset_m = float(vr_debug_z_offset_m)
        self.vr_stream_calib_orient = str(vr_stream_calib_orient)
        self.vr_ik_anchor_mode = normalize_vr_ik_anchor_mode(vr_ik_anchor_mode)
        self.vr_head_orient_mode = normalize_head_orient_mode(vr_head_orient_mode)
        self.vr_ik_z_offset_m = float(vr_ik_z_offset_m)
        self._vr_ik_mount_mat4: np.ndarray | None = None
        self._trace: JsonlTrace | None = (
            JsonlTrace(log_path, append=log_append) if log_path is not None else None
        )
        self._trace_hz = float(log_hz)
        self._trace_last_emit_mono: float = 0.0
        self._trace_prev: dict[str, Any] = {}
        self._trace_arm_tau_pre_clip: np.ndarray | None = None
        self._trace_arm_tau_post_clip: np.ndarray | None = None
        self._forensic_burst_enable = bool(forensic_burst) and self._trace is not None
        self._forensic_pre_steps = int(max(1, forensic_pre_steps))
        self._forensic_post_steps = int(max(1, forensic_post_steps))
        self._forensic_trigger_pos_err_m = float(forensic_trigger_pos_err_m)
        self._forensic_trigger_quat_err_deg = float(forensic_trigger_quat_err_deg)
        self._forensic_trigger_raw_step_l2 = float(forensic_trigger_raw_step_l2)
        self._forensic_trigger_step_scale_max = float(forensic_trigger_step_scale_max)
        self._forensic_cooldown_steps = int(max(0, forensic_cooldown_steps))
        self._forensic_ring: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=self._forensic_pre_steps
        )
        self._forensic_active = False
        self._forensic_post_left = 0
        self._forensic_cooldown_left = 0
        self._forensic_episode_id = 0
        self._forensic_row_index = 0
        self._vr_stream_calib: VrTorsoVizCalib | None = (
            VrTorsoVizCalib() if (self.vr_debug_stream or self.vr_ik_mode != "off") else None
        )
        self._openvr_poller_handle: Any = None
        self.config = load_gear_wbc_config(self.resources_dir, yaml_name=config_yaml)
        self.stand_net = make_onnx_runner(self.config["policy_path"])
        self.walk_net = make_onnx_runner(self.config["walk_policy_path"])

        self.model = mujoco.MjModel.from_xml_path(self.config["xml_path"])
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(self.config["simulation_dt"])

        self.n_joints = int(self.model.nu)
        self._policy_qpos_adr, self._policy_dof_adr = _policy_joint_addrs(
            self.model, POLICY_JOINT_NAMES
        )
        self.n_joints_policy = int(self._policy_qpos_adr.shape[0])
        self.has_hands = (
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hand_thumb_0_joint"
            )
            >= 0
        )
        self.num_act = int(self.config["num_actions"])
        self.decim = int(self.config["control_decimation"])
        self.obs_hist_len = int(self.config["obs_history_len"])
        self.num_obs = int(self.config["num_obs"])
        self.torso_index = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self.pelvis_index = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        )
        if self.vr_ik_anchor_mode == "pelvis" and self.pelvis_index < 0:
            raise ValueError(
                "vr_ik_anchor_mode='pelvis' requires a ``pelvis`` body in the MJCF"
            )

        if control_dict is None:
            self.control_dict: dict[str, Any] = {
                "loco_cmd": self.config["cmd_init"].copy(),
                "height_cmd": float(self.config["height_cmd"]),
                "rpy_cmd": self.config["rpy_cmd"].copy(),
                "freq_cmd": float(self.config.get("freq_cmd", 0.75)),
            }
        else:
            self.control_dict = control_dict

        self.n_arm = self.n_joints - self.num_act
        self._arm_ctrl_to_qslice = arm_actuator_qpos_slice_indices(
            self.model, self.num_act, self.n_joints
        )
        # Per-arm-joint PD storage (and ``target_block`` overrides) must be set BEFORE
        # ``_patch_target_block_physics()`` and the first ``mj_forward`` so the override is
        # in effect for the very first sim step.
        if arm_pd_kp_per_joint is None:
            self.arm_pd_kp_per_joint = ARM_PD_KP_DEFAULT_PER_JOINT.copy()
        else:
            self.arm_pd_kp_per_joint = (
                np.asarray(arm_pd_kp_per_joint, dtype=np.float32).reshape(7).copy()
            )
        if arm_pd_kd_per_joint is None:
            self.arm_pd_kd_per_joint = ARM_PD_KD_DEFAULT_PER_JOINT.copy()
        else:
            self.arm_pd_kd_per_joint = (
                np.asarray(arm_pd_kd_per_joint, dtype=np.float32).reshape(7).copy()
            )
        self.arm_tau_clip_nm = (
            float(arm_tau_clip_nm) if arm_tau_clip_nm is not None else float(ARM_TAU_CLIP_NM)
        )
        self._block_density_override = (
            float(block_density) if block_density is not None else None
        )
        self._block_friction_override = (
            tuple(float(x) for x in block_friction)
            if block_friction is not None
            else None
        )
        self._print_arm_tau = bool(print_arm_tau)
        self._arm_tau_peaks_post = np.zeros(max(0, self.n_arm), dtype=np.float64)
        self._arm_tau_clipped_window = 0
        self._arm_tau_total_window = 0
        self._grip_key_step = (
            float(grip_key_step) if grip_key_step is not None else float(GRIP_KEY_STEP_DEFAULT)
        )
        if not (0.0 < self._grip_key_step <= 1.0):
            raise ValueError("grip_key_step must be in (0, 1]")
        self._print_grip_force = bool(print_grip_force)
        self._block_geom_id = -1
        for _geom_name in ("sd_target_block_geom", "target_block_geom"):
            _gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, _geom_name)
            if _gid >= 0:
                self._block_geom_id = int(_gid)
                break
        self._grip_fn_peak_l = 0.0
        self._grip_fn_peak_r = 0.0
        self._grip_fn_peak_misc = 0.0
        self._grip_fn_peak_tot = 0.0
        self._grip_n_con_peak = 0
        self._patch_target_block_physics()
        self.action = np.zeros(self.num_act, dtype=np.float32)
        self.target_dof_pos = self.config["default_angles"].copy()
        # Seed qpos[legs+waist] with the YAML standing pose BEFORE any obs/render.
        # MJCF init leaves these at 0 (locked-straight legs, feet ~14 cm above
        # floor). The Gear WBC leg PD eventually drives them toward
        # ``default_angles``, but the *first* policy observation captures them
        # at zero, which is OOD for every GR00T checkpoint trained on RoboCasa
        # (whose env reset writes ``default_angles`` into qpos before the first
        # ``policy.get_action``). Cf. Bug #8 (May 2026 PnP A/B forensics).
        _da = np.asarray(self.config["default_angles"], dtype=np.float64).reshape(-1)
        if _da.size != self.num_act:
            raise ValueError(
                f"YAML ``default_angles`` length {_da.size} != num_actions ({self.num_act})"
            )
        # Address the first ``num_act`` policy joints by name (legs + waist).
        # ``POLICY_JOINT_NAMES`` lists 29 joints total (legs + waist + arms);
        # ``default_angles`` only covers the first 15. Slicing the resolved
        # addresses keeps this name-based + robust to qpos layout changes.
        self.data.qpos[self._policy_qpos_adr[: self.num_act]] = _da
        if self.has_hands:
            init_gripper_control_dict(self.control_dict, grip_key_step=self._grip_key_step)
        single_obs = compute_single_obs(
            self.data,
            self.config,
            self.action,
            self.control_dict,
            self._policy_qpos_adr,
            self._policy_dof_adr,
            self.torso_index,
        )
        self.single_obs_dim = int(single_obs.shape[0])
        if self.obs_hist_len * self.single_obs_dim != self.num_obs:
            raise ValueError(
                f"obs mismatch: history_len={self.obs_hist_len} * dim={self.single_obs_dim} "
                f"!= num_obs={self.num_obs}"
            )
        self.obs_history: collections.deque[np.ndarray] = collections.deque(
            [np.zeros(self.single_obs_dim, dtype=np.float32) for _ in range(self.obs_hist_len)],
            maxlen=self.obs_hist_len,
        )
        self.stacked_obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0
        self.cmd_lock = threading.Lock()
        self._keyboard_grasp_primitives = bool(keyboard_grasp_primitives)
        self._lock_ee_orient = bool(lock_ee_orient)
        self._freeze_arms_when_walk = bool(freeze_arms_when_walk)

        self.left_ik: Any = None
        self.right_ik: Any = None
        self.arm_qpos_ids: np.ndarray | None = None
        self.arm_ik_rot_weight = float(_ARM_IK_ROT_WEIGHT)
        self.arm_ik_posture_weight = float(_ARM_IK_POSTURE_WEIGHT)
        self.arm_ik_temporal_weight = float(_ARM_IK_TEMPORAL_WEIGHT)
        self.arm_ik_frame_dq_max = float(_ARM_IK_FRAME_DQ_MAX)
        self.arm_ik_frame_dq_l2_cap = float(_ARM_IK_FRAME_DQ_L2_CAP)
        self.arm_ik_step_blend = float(_ARM_IK_STEP_BLEND)
        self.arm_ik_posture_dev_scale = float(_ARM_IK_POSTURE_DEV_SCALE)
        self.arm_ik_posture_dev_boost = float(_ARM_IK_POSTURE_DEV_BOOST)
        self.arm_ik_posture_boost_cap = float(_ARM_IK_POSTURE_BOOST_CAP)
        self._ik_q_prev: np.ndarray | None = None
        self._ik_runaway_count: int = 0

        mujoco.mj_forward(self.model, self.data)
        self._init_arm_state(
            init_ee_for_teleop=bool(
                (teleop or openvr_teleop or self.udp_teleop or self.vr_ik_mode != "off")
                and self.n_arm > 0
            )
        )
        if self._trace is not None:
            lp = Path(log_path).resolve()
            mode = "append" if log_append else "truncate (fresh each run)"
            print(f"Trace log: {lp} ({mode}, hz={self._trace_hz})")

    def close_trace(self) -> None:
        if self._trace is None:
            return
        try:
            self._trace.close()
        finally:
            self._trace = None

    def _ee_decode_quat(self, xmat: np.ndarray, xquat: np.ndarray) -> np.ndarray:
        """Quaternion for ``ee_pose_world_from_ref`` / ``ee_world_targets_for_ik`` decode."""
        if self.vr_teleop and self.vr_ik_mode != "off":
            return torso_xquat_with_receiver_yaw(
                np.asarray(xmat, dtype=np.float64).reshape(3, 3),
                np.asarray(xquat, dtype=np.float64).reshape(4),
                self.vr_receiver_yaw_deg,
            )
        return np.asarray(xquat, dtype=np.float64).reshape(4).copy()

    def _ee_ref_rotmat(self, xmat: np.ndarray, xquat: np.ndarray) -> np.ndarray:
        """Rotation matrix for ref-frame position in ``ee_*`` encode/decode (must match ``_ee_decode_quat``).

        Non-VR: derive **R** from the same quaternion as orientation so ``R`` and ``qr`` are one SO(3).
        VR+IK: keep ``torso_xmat_with_yaw_offset`` paired with ``torso_xquat_with_receiver_yaw``.
        """
        xm = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
        xq = np.asarray(xquat, dtype=np.float64).reshape(4)
        if self.vr_teleop and self.vr_ik_mode != "off":
            return torso_xmat_with_yaw_offset(xm, self.vr_receiver_yaw_deg)
        return quat_wxyz_to_rotmat(normalize_quat(self._ee_decode_quat(xm, xq)))

    def _init_arm_state(self, *, init_ee_for_teleop: bool) -> None:
        if self.n_arm <= 0:
            return
        if "arm_target_q" not in self.control_dict:
            self.control_dict["arm_target_q"] = self.data.qpos[
                7 + self.num_act : 7 + self.n_joints
            ].copy()
        else:
            atq = np.asarray(self.control_dict["arm_target_q"], dtype=np.float32).reshape(-1)
            if atq.size != self.n_arm:
                raise ValueError(
                    f"arm_target_q length {atq.size} != n_joints - num_actions ({self.n_arm})"
                )
            self.control_dict["arm_target_q"] = atq
        if "_arm_target_q_home" not in self.control_dict:
            self.control_dict["_arm_target_q_home"] = np.asarray(
                self.control_dict["arm_target_q"], dtype=np.float32
            ).copy()

        if init_ee_for_teleop:
            self.left_ik, self.right_ik = build_g1_arm_ik_specs(self.model)
            self.arm_qpos_ids = np.concatenate(
                [self.left_ik.qpos_ids, self.right_ik.qpos_ids]
            ).astype(np.int32)
            init_ee_control_dict(
                self.model,
                self.data,
                self.left_ik,
                self.right_ik,
                self.control_dict,
                self.torso_index,
            )

    def _clear_ik_temporal_state(self) -> None:
        """Call on keyboard ``z`` reset: IK temporal smoothing memory (see ``arm_ik_v2``)."""
        self._ik_q_prev = None
        self._ik_runaway_count = 0

    def _refresh_ee_world_anchors_from_home_pose(self) -> None:
        """After ``z``, ``ee_*_quat`` equals torso-frame home; refresh world anchors for lock mode."""
        if not self._lock_ee_orient:
            return
        cd = self.control_dict
        if "_ee_left_quat_world_anchor" not in cd or "_ee_left_quat_home" not in cd:
            return
        ti = int(self.torso_index)
        xm = np.asarray(self.data.xmat[ti], dtype=np.float64).reshape(3, 3)
        xq = np.asarray(self.data.xquat[ti], dtype=np.float64).reshape(4)
        qr = normalize_quat(self._ee_decode_quat(xm, xq))
        ql_h = normalize_quat(np.asarray(cd["_ee_left_quat_home"], dtype=np.float64).reshape(4))
        qr_h = normalize_quat(np.asarray(cd["_ee_right_quat_home"], dtype=np.float64).reshape(4))
        cd["_ee_left_quat_world_anchor"][:] = quat_mul(qr, ql_h)
        cd["_ee_right_quat_world_anchor"][:] = quat_mul(qr, qr_h)

    def _on_teleop_reset(self) -> None:
        self._clear_ik_temporal_state()
        if self._lock_ee_orient:
            with self.cmd_lock:
                self._refresh_ee_world_anchors_from_home_pose()

    def _patch_target_block_physics(self) -> None:
        """Apply CLI/runtime overrides for ``target_block`` mass + friction (no XML edit).

        MuJoCo's Python ``MjModel`` does not expose ``geom_density`` (compile-time hint
        already converted into ``body_mass`` + ``body_inertia`` during XML parse), so we
        patch ``body_mass[bid]`` / ``body_inertia[bid]`` directly from geom volume × override
        density, and ``geom_friction[gid, :]`` for the contact tangential / slip / spin terms.

        Targets ``sd_target_block_geom`` (diner) and ``target_block_geom`` (future scenes).
        """
        if self._block_density_override is None and self._block_friction_override is None:
            return
        gid = -1
        for name in ("sd_target_block_geom", "target_block_geom"):
            cand = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
            if cand >= 0:
                gid = cand
                break
        if gid < 0:
            warnings.warn(
                "block_density / block_friction passed but no target_block_geom found in scene; "
                "ignoring override.",
                UserWarning,
                stacklevel=2,
            )
            return
        if self._block_density_override is not None:
            bid = int(self.model.geom_bodyid[gid])
            size = np.asarray(self.model.geom_size[gid], dtype=np.float64)
            gtype = int(self.model.geom_type[gid])
            if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                volume = 8.0 * float(size[0] * size[1] * size[2])
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                volume = (4.0 / 3.0) * np.pi * float(size[0] ** 3)
            elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                volume = np.pi * float(size[0] ** 2) * 2.0 * float(size[1])
            else:
                volume = float("nan")
            if np.isfinite(volume):
                mass = float(self._block_density_override) * volume
                self.model.body_mass[bid] = mass
                if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                    sx, sy, sz = (2.0 * size[0], 2.0 * size[1], 2.0 * size[2])
                    self.model.body_inertia[bid] = (mass / 12.0) * np.array(
                        [sy * sy + sz * sz, sx * sx + sz * sz, sx * sx + sy * sy],
                        dtype=np.float64,
                    )
                elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    r2 = float(size[0] ** 2)
                    self.model.body_inertia[bid] = (2.0 / 5.0) * mass * r2
                print(
                    f"target_block: density={self._block_density_override:.2f} "
                    f"-> mass={mass:.4f} kg (volume={volume:.6f} m^3)"
                )
            else:
                warnings.warn(
                    f"target_block: density override skipped — unhandled geom type {gtype}",
                    UserWarning,
                    stacklevel=2,
                )
        if self._block_friction_override is not None:
            self.model.geom_friction[gid, :3] = np.asarray(
                self._block_friction_override, dtype=np.float64
            )
            print(
                "target_block: friction (tan, slip, spin) = "
                f"{self._block_friction_override}"
            )

    def _arm_tau_status_snapshot(self) -> str | None:
        """Return a one-line tau-peak summary and reset the peak-hold window. ``None`` if disabled / no data."""
        if not self._print_arm_tau:
            return None
        peaks = self._arm_tau_peaks_post
        if peaks is None or peaks.size == 0:
            return None
        total = int(self._arm_tau_total_window)
        if total == 0:
            return None
        clipped = int(self._arm_tau_clipped_window)
        idx_lwp, idx_rwp = left_right_wrist_pitch_indices(has_hands=self.has_hands)
        peak_lwp = float(peaks[idx_lwp]) if idx_lwp < peaks.size else float("nan")
        peak_rwp = float(peaks[idx_rwp]) if idx_rwp < peaks.size else float("nan")
        peak_overall = float(np.max(peaks)) if peaks.size > 0 else float("nan")
        clip_frac = clipped / max(1, total)
        # Reset window after summarizing so each print covers fresh samples.
        self._arm_tau_peaks_post = np.zeros_like(peaks)
        self._arm_tau_clipped_window = 0
        self._arm_tau_total_window = 0
        return (
            f"arm_tau peaks (post-clip): wristP L={peak_lwp:+.1f} Nm  R={peak_rwp:+.1f} Nm  "
            f"max|τ|={peak_overall:.1f} Nm | clip={self.arm_tau_clip_nm:.0f} Nm "
            f"clip_frac={clip_frac:.2f} ({clipped}/{total} samples)"
        )

    def _geom_side_for_grip_readout(self, geom_id: int) -> str | None:
        """Return ``'L'`` / ``'R'`` if ``geom_id`` belongs to left / right hand chain, else ``None``."""
        bid = int(self.model.geom_bodyid[int(geom_id)])
        for _ in range(48):
            if bid < 0:
                break
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if bname is None:
                bid = int(self.model.body_parentid[bid])
                continue
            if bname.startswith("right_wrist_yaw_link") or "right_hand" in bname:
                return "R"
            if bname.startswith("left_wrist_yaw_link") or "left_hand" in bname:
                return "L"
            bid = int(self.model.body_parentid[bid])
        return None

    def _accumulate_grip_contact_peaks(self) -> None:
        """After ``mj_step``: peak-hold |Fn| sums on ``target_block`` vs hands (for status print)."""
        if not self._print_grip_force or self._block_geom_id < 0:
            return
        bgid = int(self._block_geom_id)
        fn_l = fn_r = fn_misc = 0.0
        n_block = 0
        forces = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 != bgid and g2 != bgid:
                continue
            other = g2 if g1 == bgid else g1
            mujoco.mj_contactForce(self.model, self.data, i, forces)
            fn = abs(float(forces[0]))
            n_block += 1
            side = self._geom_side_for_grip_readout(other)
            if side == "L":
                fn_l += fn
            elif side == "R":
                fn_r += fn
            else:
                fn_misc += fn
        fn_tot = fn_l + fn_r + fn_misc
        self._grip_fn_peak_l = max(self._grip_fn_peak_l, fn_l)
        self._grip_fn_peak_r = max(self._grip_fn_peak_r, fn_r)
        self._grip_fn_peak_misc = max(self._grip_fn_peak_misc, fn_misc)
        self._grip_fn_peak_tot = max(self._grip_fn_peak_tot, fn_tot)
        self._grip_n_con_peak = max(self._grip_n_con_peak, n_block)

    def _grip_force_status_snapshot(self) -> str | None:
        """Peak |normal force| on block since last print; resets peak-hold buffers."""
        if not self._print_grip_force:
            return None
        if self._block_geom_id < 0:
            return "grip_force: no target_block geom in model (Fn N/A)"
        pl = float(self._grip_fn_peak_l)
        pr = float(self._grip_fn_peak_r)
        pm = float(self._grip_fn_peak_misc)
        pt = float(self._grip_fn_peak_tot)
        nmax = int(self._grip_n_con_peak)
        self._grip_fn_peak_l = 0.0
        self._grip_fn_peak_r = 0.0
        self._grip_fn_peak_misc = 0.0
        self._grip_fn_peak_tot = 0.0
        self._grip_n_con_peak = 0
        return (
            f"grip_force (|Fn| on block, peak since last status): L={pl:.2f} N  R={pr:.2f} N  "
            f"other={pm:.2f} N  total={pt:.2f} N | max n_contacts/step={nmax}"
        )

    def start_teleop_listener(self) -> None:
        if self.teleop:
            start_gear_wbc_teleop_listener(
                self.control_dict,
                self.config,
                self.cmd_lock,
                keyboard_grasp_primitives=self._keyboard_grasp_primitives,
                on_teleop_reset=self._on_teleop_reset,
                arm_tau_provider=self._arm_tau_status_snapshot,
                grip_force_provider=(
                    self._grip_force_status_snapshot if self._print_grip_force else None
                ),
            )

    def start_openvr_poller(self) -> None:
        """Optional OpenVR → ``control_dict``; no-op if ``openvr_teleop`` is false."""
        if not self.openvr_teleop:
            return
        try:
            import openvr_teleop as ovt
        except ImportError:
            warnings.warn(
                "openvr_teleop=True but the `openvr_teleop` module is not available "
                "(add `g1_minimal_sim/openvr_teleop.py` or `pip install` deps). "
                "VR arm control disabled.",
                UserWarning,
                stacklevel=2,
            )
            return
        if not hasattr(ovt, "start_poller_for_runtime"):
            warnings.warn(
                "openvr_teleop module has no start_poller_for_runtime(); VR disabled.",
                UserWarning,
                stacklevel=2,
            )
            return
        self._openvr_poller_handle = ovt.start_poller_for_runtime(self)

    def stop_openvr_poller(self) -> None:
        h = self._openvr_poller_handle
        if h is None:
            return
        try:
            import openvr_teleop as ovt

            if hasattr(ovt, "stop_poller"):
                ovt.stop_poller(h)
        finally:
            self._openvr_poller_handle = None

    def stop_udp_teleop(self) -> None:
        if self._udp_receiver is not None:
            self._udp_receiver.stop()
            self._udp_receiver = None

    def stop_vr_teleop(self) -> None:
        if self._vr_teleop_receiver is not None:
            self._vr_teleop_receiver.stop()
            self._vr_teleop_receiver = None

    def reset(self) -> None:
        """Reset MuJoCo state and command buffers (Gym ``reset``)."""
        mujoco.mj_resetData(self.model, self.data)
        # Re-seed qpos[legs+waist] with the YAML standing pose AFTER
        # ``mj_resetData`` zeros qpos back to MJCF defaults. Mirrors the
        # ``__init__`` write so the FIRST policy observation captured post-
        # reset still matches the standing crouch every GR00T checkpoint was
        # trained on. Without this, ``G1GearWBCEnv.reset()`` (which is the
        # entry point the GR00T inference bridge actually uses) silently
        # reverts the runtime to straight legs and re-introduces Bug #8.
        # Cf. ``gr00t_compatible_data_collection.md`` §6.6.
        _da = np.asarray(self.config["default_angles"], dtype=np.float64).reshape(-1)
        self.data.qpos[self._policy_qpos_adr[: self.num_act]] = _da
        mujoco.mj_forward(self.model, self.data)
        self.control_dict["loco_cmd"][:] = self.config["cmd_init"]
        self.control_dict["height_cmd"] = float(self.config["height_cmd"])
        self.control_dict["rpy_cmd"][:] = self.config["rpy_cmd"]
        self.control_dict["freq_cmd"] = float(self.config.get("freq_cmd", 0.75))
        self.action[:] = 0.0
        self.target_dof_pos = self.config["default_angles"].copy()
        self.counter = 0
        for h in self.obs_history:
            h.fill(0.0)
        self.stacked_obs.fill(0.0)
        if self.n_arm > 0:
            self.control_dict["arm_target_q"][:] = self.data.qpos[
                7 + self.num_act : 7 + self.n_joints
            ]
            self.control_dict["_arm_target_q_home"][:] = self.control_dict["arm_target_q"]
        if self.has_hands:
            reset_gripper_control_dict(self.control_dict)
        if (
            (self.teleop or self.openvr_teleop or self.udp_teleop or self.vr_ik_mode != "off")
            and self.n_arm > 0
            and self.left_ik is not None
        ):
            assert self.right_ik is not None and self.arm_qpos_ids is not None
            init_ee_control_dict(
                self.model,
                self.data,
                self.left_ik,
                self.right_ik,
                self.control_dict,
                self.torso_index,
            )
        if self._vr_stream_calib is not None:
            self._vr_stream_calib.reset()
        self._vr_ik_mount_mat4 = None
        self._ik_q_prev = None
        self._ik_runaway_count = 0
        self._forensic_ring.clear()
        self._forensic_active = False
        self._forensic_post_left = 0
        self._forensic_cooldown_left = 0
        self._forensic_row_index = 0

    def step_physics(self) -> None:
        vr_ik_trace: dict[str, Any] | None = None
        q_sol_trace: np.ndarray | None = None
        self._trace_arm_tau_pre_clip = None
        self._trace_arm_tau_post_clip = None

        if self._udp_receiver is not None:
            pkt = self._udp_receiver.consume_latest()
            if pkt is not None:
                with self.cmd_lock:
                    apply_udp_packet_to_control_dict(
                        pkt,
                        data=self.data,
                        control_dict=self.control_dict,
                        torso_index=self.torso_index,
                        left_ik=self.left_ik,
                        right_ik=self.right_ik,
                        has_hands=self.has_hands,
                    )

        vr_snap, vr_t_rx = snapshot_vr_receiver(self)
        apply_vr_stream_locomotion(self, vr_snap, vr_t_rx)
        sync_keyboard_loco_hold(self, vr_snap, vr_t_rx)
        vr_ik_trace = apply_vr_stream_ik_targets(self, vr_snap, vr_t_rx)

        use_ik_sources = (
            self.teleop
            or self.openvr_teleop
            or self.udp_teleop
            or (self.vr_ik_mode != "off" and self.vr_teleop)
        )
        if (
            self._trace is not None
            and self._vr_teleop_receiver is None
            and use_ik_sources
            and self.n_arm > 0
            and self.left_ik is not None
            and vr_ik_trace is None
        ):
            vr_ik_trace = {}
        n_arm = self.n_arm
        with self.cmd_lock:
            _loco_mag = float(
                np.linalg.norm(np.asarray(self.control_dict["loco_cmd"], dtype=np.float32))
            )
        walking = _loco_mag > 0.05
        freeze_walk_arms = (
            self._freeze_arms_when_walk
            and walking
            and use_ik_sources
            and n_arm > 0
            and self.left_ik is not None
            and self.arm_qpos_ids is not None
        )
        # Position-only EE quats: applied inside apply_arm_ik_step under the same cmd_lock as the
        # IK snapshot (avoids pynput racing in between reset and copy).
        if (
            use_ik_sources
            and n_arm > 0
            and self.left_ik is not None
            and self.arm_qpos_ids is not None
        ):
            if freeze_walk_arms:
                with self.cmd_lock:
                    self.control_dict["arm_target_q"][:] = self.control_dict["_arm_target_q_home"]
                self._ik_q_prev = None
                q_sol_trace = None
            else:
                q_sol_trace = apply_arm_ik_step(self, vr_ik_trace)

        leg_tau = pd_control(
            self.target_dof_pos,
            self.data.qpos[7 : 7 + self.num_act],
            self.config["kps"],
            np.zeros_like(self.config["kps"]),
            self.data.qvel[6 : 6 + self.num_act],
            self.config["kds"],
        )
        self.data.ctrl[: self.num_act] = leg_tau

        if self.n_joints > self.num_act:
            if use_ik_sources:
                with self.cmd_lock:
                    arm_tgt = np.asarray(
                        self.control_dict["arm_target_q"], dtype=np.float32
                    ).copy()
            else:
                arm_tgt = np.asarray(self.control_dict["arm_target_q"], dtype=np.float32)
                if self.has_hands:
                    if bool(self.control_dict.get(GR00T_POLICY_HAND_PD_SOURCE_KEY, False)):
                        # GR00T / oracle replay: finger PD targets are the 7-DoF policy vectors
                        # already written into ``arm_target_q`` (not gripper-open/close presets).
                        arm_tgt = arm_tgt.copy()
                    else:
                        hl = hand_target_q(
                            self.model, "left", float(self.control_dict["gripper_left"])
                        )
                        hr = hand_target_q(
                            self.model, "right", float(self.control_dict["gripper_right"])
                        )
                        arm_tgt = arm_tgt.copy()
                        arm_tgt[SL_LEFT_HAND] = hl
                        arm_tgt[SL_RIGHT_HAND] = hr
            n_u = self.n_joints - self.num_act
            s_idx = self._arm_ctrl_to_qslice
            loco_mag = _loco_mag
            walking = loco_mag > 0.05
            kp_by_slice, kd_by_slice = build_arm_pd_per_joint(
                n_u,
                has_hands=self.has_hands,
                kp_per_arm_joint=self.arm_pd_kp_per_joint,
                kd_per_arm_joint=self.arm_pd_kd_per_joint,
                kd_walk_scale=float(_ARM_KD_WALK_SCALE),
                walking=walking,
            )
            q_seg = self.data.qpos[7 + self.num_act : 7 + self.n_joints]
            dq_seg = self.data.qvel[6 + self.num_act : 6 + self.n_joints]
            arm_tau = pd_control(
                arm_tgt[s_idx],
                q_seg[s_idx],
                kp_by_slice[s_idx],
                np.zeros(n_u, dtype=np.float32),
                dq_seg[s_idx],
                kd_by_slice[s_idx],
            )
            arm_tau_pre = np.asarray(arm_tau, dtype=np.float64).copy()
            if self._trace is not None:
                self._trace_arm_tau_pre_clip = arm_tau_pre
            clip_nm = float(self.arm_tau_clip_nm)
            arm_tau = np.clip(arm_tau, -clip_nm, clip_nm)
            arm_tau_post = np.asarray(arm_tau, dtype=np.float64).copy()
            if self._trace is not None:
                self._trace_arm_tau_post_clip = arm_tau_post
            if self._print_arm_tau and self._arm_tau_peaks_post is not None:
                # Track |tau| peak per arm-slice slot (already in qpos-segment order via s_idx
                # gather then ctrl-write; we want peaks indexed by the ``arm_target_q`` layout
                # so wristP at index 5 / 19 lines up. Scatter actuator-ordered tau back through
                # ``s_idx`` to that layout.
                slot_abs = np.zeros(n_u, dtype=np.float64)
                # ``arm_tau`` is in actuator (ctrl) order; ``s_idx[ctrl]`` maps that to the qpos
                # slice index. So slice[s_idx[i]] = |tau[i]|.
                np.maximum.at(slot_abs, np.asarray(s_idx, dtype=np.int64), np.abs(arm_tau_post))
                self._arm_tau_peaks_post = np.maximum(
                    self._arm_tau_peaks_post, slot_abs
                )
                # Count any sample where the pre-clip torque exceeded the clip on any actuator
                # this step as one "clipped" sample (binary per step), and total = +1 step.
                self._arm_tau_total_window += 1
                if np.any(np.abs(arm_tau_pre) > clip_nm + 1e-9):
                    self._arm_tau_clipped_window += 1
            self.data.ctrl[self.num_act :] = arm_tau

        mujoco.mj_step(self.model, self.data)
        self._accumulate_grip_contact_peaks()
        self.counter += 1

        if self._trace is not None and self._vr_teleop_receiver is not None:
            emit_vr_compare_row(
                self,
                vr_snap=vr_snap,
                vr_t_rx=float(vr_t_rx),
                ik_details=vr_ik_trace,
                q_sol=q_sol_trace,
            )
        elif self._trace is not None and self._vr_teleop_receiver is None:
            if (
                use_ik_sources
                and self.n_arm > 0
                and self.left_ik is not None
                and self.arm_qpos_ids is not None
            ):
                emit_teleop_ik_row(self, vr_ik_trace, q_sol_trace)

        if self.counter % self.decim == 0:
            with self.cmd_lock:
                single = compute_single_obs(
                    self.data,
                    self.config,
                    self.action,
                    self.control_dict,
                    self._policy_qpos_adr,
                    self._policy_dof_adr,
                    self.torso_index,
                )
                self.obs_history.append(single)
                for i, hist in enumerate(self.obs_history):
                    self.stacked_obs[
                        i * self.single_obs_dim : (i + 1) * self.single_obs_dim
                    ] = hist
                inp = self.stacked_obs.reshape(1, -1).copy()
                loco = np.asarray(self.control_dict["loco_cmd"], dtype=np.float32).copy()
            if np.linalg.norm(loco) <= 0.05:
                self.action = self.stand_net(inp)
            else:
                self.action = self.walk_net(inp)
            with self.cmd_lock:
                self.target_dof_pos = (
                    self.action * self.config["action_scale"] + self.config["default_angles"]
                )


def run_gear_wbc(
    resources_dir: Path,
    *,
    headless: bool,
    max_steps: int | None,
    control_dict: dict[str, Any] | None = None,
    teleop: bool = False,
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
    config_yaml: str = "g1_gear_wbc.yaml",
    debug_ee_targets: bool = False,
    viewer_camera: str = "default",
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
) -> None:
    """Load MJCF + ONNX and run simulation with PD + decimated policy updates.

    ``control_dict`` defaults to zero locomotion command → **stand** policy branch
    (``np.linalg.norm(loco_cmd) <= 0.05``).

    With ``teleop=True`` (viewer only), a background thread updates ``control_dict`` from
    the keyboard. Legs/waist: same bindings as ``run_mujoco_gear_wbc.py``. Arms: **IK** —
    teleop edits **torso_link**-frame ``ee_{left,right}_{pos,quat}`` (wxyz); each step converts
    to world and fills ``arm_target_q`` via stacked damped least-squares (see ``arm_ik_v2.py``). ``z`` resets locomotion,
    arm joint home, and EE poses.

    **Non-teleop arms:** PD holds ``arm_target_q`` initialized from the loaded pose (no IK).
    """
    if teleop and headless:
        raise ValueError("teleop requires a viewer; do not use --headless with teleop")
    if openvr_teleop and headless:
        raise ValueError("openvr_teleop requires a viewer; do not use --headless with it")
    rt = GearWBCRuntime(
        resources_dir,
        teleop=teleop,
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
        control_dict=control_dict,
        config_yaml=config_yaml,
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
    try:
        if teleop:
            rt.start_teleop_listener()
        if openvr_teleop:
            rt.start_openvr_poller()

        if headless:
            n = max_steps if max_steps is not None else int(
                rt.config.get("simulation_duration", 60.0) / rt.model.opt.timestep
            )
            for _ in range(n):
                rt.step_physics()
            print(
                f"Gear WBC headless done ({n} steps). "
                f"pelvis z={rt.data.qpos[2]:.3f} m"
            )
            return

        from mujoco import viewer as mj_viewer

        if vr_debug_stream:
            from vr_teleop.vr_stream_target_viz import fill_vr_stream_arrow_geoms

        if udp_teleop_bind:
            print(
                f"UDP teleop listening on {udp_teleop_bind!r} "
                f"(run python -m legacy_vr_code.openvr_teleop_client on the VR PC). "
                "Combine with --teleop for keyboard + z reset."
            )
        if vr_teleop_bind:
            print(
                f"VR teleop (openvr_stream) on {vr_teleop_bind!r} — "
                "sticks → loco_cmd; run: python -m vr_teleop.openvr_udp_sender --target <sim-ip>:PORT"
            )

        if teleop:
            if rt.n_arm > 0:
                gh = ""
                if rt.has_hands:
                    gh = (
                        "Hands: 9 open / 0 close left; [=] open / [-] close right; "
                        "h hard-close both; y open both.\n"
                    )
                print(
                    "Teleop: **hold** w/s forward-back, a/d strafe, q/e yaw (release loco keys to stand); "
                    "z full reset. 1/2 height; 3–8 rpy; m/n freq.\n"
                    "Arms (torso-frame IK targets → world each step): left pos i/k j/l u/p; right r/t f/g v/b.\n"
                    "Both-hand orient (wxyz, torso axes): , . = torso X; ; ' = torso Y; [ ] = torso Z.\n"
                    f"{gh}"
                    "Focus terminal for keys. Close viewer to quit."
                )
            else:
                print(
                    "Teleop: **hold** w/s forward-back, a/d strafe, q/e yaw; z reset; "
                    "1/2 height; 3–8 rpy; m/n freq. Focus terminal for keys. Close viewer to quit."
                )
            if keyboard_grasp_primitives:
                print(
                    "Grasp primitives: **x** widen / **c** narrow palms (torso ±Y, ~8 cm span/step); "
                    "**Home** / **End** forward/back both (torso X); "
                    "**Page Up** / **Page Down** raise/lower both (torso Z). Quats & grippers unchanged."
                )
        else:
            print("Gear WBC standing (zero loco_cmd → stand ONNX). Close viewer to quit.")
        if debug_ee_targets:
            print(
                "Debug: green / orange spheres + large arrows = world IK palm targets + target orientation "
                "(torso-frame ee_*)."
            )
        if vr_debug_stream:
            print(
                "Debug VR stream: HMD/controllers as arrows (torso compose). "
                f"Z mode={vr_debug_torso_z_mode} (fixed_world uses Z={vr_debug_z_offset_m:.2f} m); "
                f"receiver yaw={vr_receiver_yaw_deg:.1f} deg; calib orient={vr_stream_calib_orient}; "
                f"head orient={rt.vr_head_orient_mode}."
            )

        def _draw_debug_overlays(viewer: Any) -> None:
            if not debug_ee_targets and not vr_debug_stream:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                return
            with viewer.lock():
                geoms = viewer.user_scn.geoms
                i = 0
                if debug_ee_targets:
                    from teleop_target_viz import fill_ee_target_geoms

                    ti_dbg = int(rt.torso_index)
                    xm_dbg = np.asarray(rt.data.xmat[ti_dbg], dtype=np.float64).reshape(
                        3, 3
                    )
                    xq_dbg = np.asarray(rt.data.xquat[ti_dbg], dtype=np.float64).reshape(
                        4
                    )
                    i = fill_ee_target_geoms(
                        geoms,
                        i,
                        rt.model,
                        rt.data,
                        rt.control_dict,
                        rt.torso_index,
                        enabled=True,
                        n_arm=rt.n_arm,
                        ref_quat_wxyz=rt._ee_decode_quat(xm_dbg, xq_dbg),
                        ref_rotmat=rt._ee_ref_rotmat(xm_dbg, xq_dbg),
                    )
                if vr_debug_stream:
                    recv = rt._vr_teleop_receiver
                    st_vr, _t = recv.snapshot() if recv is not None else (None, 0.0)
                    ti = rt.torso_index
                    d = rt.data
                    pi = int(rt.pelvis_index)
                    if pi >= 0:
                        ppos = np.asarray(d.xpos[pi], dtype=np.float64).reshape(3)
                        pxm = np.asarray(d.xmat[pi], dtype=np.float64).reshape(3, 3)
                    else:
                        ppos, pxm = None, None
                    i = fill_vr_stream_arrow_geoms(
                        geoms,
                        i,
                        st_vr,
                        enabled=True,
                        z_offset_m=vr_debug_z_offset_m,
                        torso_xpos=d.xpos[ti],
                        torso_xmat=d.xmat[ti].reshape(3, 3),
                        calib=rt._vr_stream_calib,
                        torso_z_mode=vr_debug_torso_z_mode,
                        torso_yaw_offset_deg=rt.vr_receiver_yaw_deg,
                        calib_orient_mode=vr_stream_calib_orient,
                        vr_ik_anchor_mode=str(getattr(rt, "vr_ik_anchor_mode", "legacy")),
                        pelvis_xpos=ppos,
                        pelvis_xmat=pxm,
                        mount_T=getattr(rt, "_vr_ik_mount_mat4", None),
                        head_orient_mode=str(getattr(rt, "vr_head_orient_mode", "yaw_only")),
                    )
                viewer.user_scn.ngeom = i

        if max_steps is None:
            with mj_viewer.launch_passive(rt.model, rt.data) as viewer:
                if viewer_camera == "head_pov":
                    from viewer_camera import viewer_use_fixed_camera

                    viewer_use_fixed_camera(rt.model, viewer.cam, "head_pov")
                while viewer.is_running():
                    rt.step_physics()
                    _draw_debug_overlays(viewer)
                    viewer.sync()
        else:
            with mj_viewer.launch_passive(rt.model, rt.data) as viewer:
                if viewer_camera == "head_pov":
                    from viewer_camera import viewer_use_fixed_camera

                    viewer_use_fixed_camera(rt.model, viewer.cam, "head_pov")
                for _ in range(max_steps):
                    if not viewer.is_running():
                        break
                    rt.step_physics()
                    _draw_debug_overlays(viewer)
                    viewer.sync()
    finally:
        rt.stop_openvr_poller()
        rt.stop_udp_teleop()
        rt.stop_vr_teleop()
        rt.close_trace()
