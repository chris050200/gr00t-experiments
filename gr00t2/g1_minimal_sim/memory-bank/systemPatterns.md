# System patterns — g1_minimal_sim

## Two runtime modes (today)

1. **Passive** (`spawn_g1_floor.py` without `--stand`)  
   - Loads `g1_gear_wbc.xml`, steps with **no torques** from our code (robot collapses).  
   - Use: sanity check MJCF / MuJoCo / display.

2. **Gear WBC stand** (`--stand`)  
   - Loads the same MJCF + `g1_gear_wbc.yaml`.  
   - **PD** on controlled joints every step; every `control_decimation` steps runs **ONNX** (stand if `||loco_cmd|| ≤ 0.05`, else walk).  
   - Implementation: **`GearWBCRuntime`** in `gear_wbc_stand.py` (facade) plus `gear_wbc_*` modules (`gear_wbc_config`, `gear_wbc_obs`, `gear_wbc_onnx`, `gear_wbc_pd`, `gear_wbc_teleop`, `gear_wbc_vr_stream`, `gear_wbc_vr_trace`, `gear_wbc_arm_ik`) mirrors `run_mujoco_gear_wbc.py` observation layout and gains; **CPU** ONNX.  
   - **`--stand --teleop`:** optional `pynput` daemon listener + `threading.Lock` on `control_dict`; keymap in `apply_mujoco_gear_wbc_key` (aligned with `run_mujoco_gear_wbc.py`). Headless runs never import teleop. **`--openvr-teleop`:** optional background poller via import **`openvr_teleop`** (`start_poller_for_runtime` / `stop_poller`); **`step_physics`** runs arm IK when **`teleop` or `openvr_teleop`** so EE targets can come from keyboard or VR. **`--udp-teleop`:** **`legacy_vr_code.udp_teleop_receiver`** thread + `apply_udp_packet_to_control_dict` (VR PC: **`python -m legacy_vr_code.openvr_teleop_client`**, JSON **`v: 1`**); IK runs when **`udp_teleop`** too. JSON **`palm_frame`**: **`hmd_relative`** vs **`world`**. Optional **`--debug-ee-targets`** draws world IK targets (`teleop_target_viz.py`). Short ref: **`vr_legacy_history.md`**.  
  - **`--vr-teleop`:** **`vr_teleop.openvr_stream.OpenvrStreamUdpReceiver`** listens for **`vr_teleop.openvr_udp_sender`** packets (JSON **`v:1`**, **`inputs`** + poses). **Phase A:** each **`step_physics`** applies **`sticks_to_loco_cmd`** → **`control_dict["loco_cmd"]`**; if no packet for **`~0.35s`**, **`loco_cmd`** resets to **`cmd_init`**. Mutually exclusive with **`--udp-teleop`** (different protocol). **`play_g1_gear_wbc --debug-vr-stream`:** passive-viewer arrows use torso-mounted compose by default via **`vr_stream_torso_compose`** with receiver-side controls: **`--vr-receiver-yaw-deg`** (default `-90`), **`--debug-vr-stream-torso-z`** (default `full`), **`--debug-vr-stream-calib-orient`** (default `yaw_only`). **Phase B (landed):** `--vr-ik {off,on}` maps stream poses to torso-frame `ee_*` targets using the same compose/calibration path as overlay; **`arm_ik_v2.solve_dual_arm_ik_v2`** fills `arm_target_q`. With `--hands`, stream triggers drive grippers. **Current-HMD orient mode (landed):** `--vr-head-orient {full,yaw_only}` (default `yaw_only`) controls non-legacy (`torso`/`pelvis`) head-relative compose; `yaw_only` keeps world Z shared by dropping HMD pitch/roll inside `inv(T_hmd)`.
  - **VR pose debug (legacy + new):** legacy path is **`legacy_vr_code.vr_teleop_test_streaming`** (UDP JSON **`v: 2`**, port **5006**). New bring-up path is **`vr_teleop/openvr_udp_sender.py`** (VR PC) + **`vr_teleop/vr_lab.py`** (sim PC) using shared mapping in **`vr_teleop/mapping.py`**. Current sender is pose/input streaming only; lab applies virtual-base SE(2) for frame-coherence testing. See **`vr_legacy_history.md`**.  
  - **Integration rule for walking teleop:** treat EE commands as **base/torso-relative** state and compose to world every step. Do not keep world-fixed hand targets as robot base moves.
  - **Validation sequence:** Phase A/B completed in `vr_lab` for XY+yaw coherence (manual). Next step is runtime teleop integration; defer roll/pitch simulation.
   - **Gymnasium (Phase 2a):** **`GearWBCRuntime`** in `gear_wbc_stand.py` (`reset`, `step_physics`); **`G1GearWBC-v0`** / **`G1GearWBC-TablePnP-v0`** in `g1_gear_wbc_env.py`; play CLI `play_g1_gear_wbc.py`. Scene = which yaml/MJCF (`resolve_gear_wbc_config`, `scene` / `config_yaml` kwargs). Teleop still mutates shared **`control_dict`**; `env.step` advances one sim step (`manipulation_and_gr00t_plan.md` §G).
   - **Scenes (`resolve_gear_wbc_config`):** **`stylish_diner`** — default for `play_g1_gear_wbc.py`; merged MJCF in `g1_minimal_sim/scenes/stylish_diner_1/` (world geoms + `target_block` free body). `use_hands` selects between **`g1_gear_wbc_stylish_diner.xml`** (welded fingers) and **`g1_gear_wbc_stylish_diner_hands.xml`** (May 2026 Phase 2 — articulated 14-DoF). **`floor`** / **`table_pnp`** — vendored `g1/` bundle paths; **`table_pnp`** uses `G1GearWBC-TablePnP-v0`. Keyboard hold-loco + optional **`keyboard_grasp_primitives`** + friction-grasp tuning: **`memory-bank/stylish_diner_keyboard_teleop.md`**.
   - **MJCF with freejoint props:** mechanism joint count for PD/arm slices is **`model.nu`**, not `nq−7`, so apple/plate free bodies do not corrupt `arm_target_q` indexing.
   - **Arms:** PD toward `control_dict["arm_target_q"]`. **Without hands:** 14 DoFs (both arms). **With `--hands`:** 28 DoFs in **kinematic `qpos` order**: left arm (7) → left hand (7) → right arm (7) → right hand (7). **`hand_gripper.SL_*`** slices document indices. **Non-teleop:** targets init from `qpos`; `z` restores `_arm_target_q_home` and grippers open. **`--stand --teleop`:** IK writes **only** the two arm blocks (`SL_LEFT_ARM`, `SL_RIGHT_ARM`); each step fills hand blocks from `gripper_left` / `gripper_right` via `hand_target_q()`. **`arm_qpos_ids`** = 14 arm joint addresses only (concat of `ArmSideIK.qpos_ids`), never `np.arange` over the full 28. **Arm PD torques:** `ctrl[num_act:]` is in **actuator XML order**; **`_arm_ctrl_to_qslice`** maps each ctrl slot → `arm_target_q` index so PD uses the correct (target, q, qdot) triple (required because **right-hand motors are thumb→index→middle** while **qpos** is thumb→middle→index). Full detail: **`memory-bank/hands_gripper_teleop.md`**.
   - **Policy ONNX input:** Always **29** joints (`POLICY_JOINT_NAMES` gather), even when the MJCF has 43 mechanism joints; hand `qpos` is not fed to the network.
   - **Leg+waist init contract (Phase 4a + 4a-bis, May 2026).** `GearWBCRuntime` writes `qpos[_policy_qpos_adr[:num_act]] = config["default_angles"]` (legs + waist; 15 values) in **both** `__init__` (before the initial `mj_forward`) and `reset()` (between `mj_resetData` and `mj_forward`). This makes the first observation captured by any consumer (teleop, inference bridge, `G1GearWBC-v0`) match the standing crouch that RoboCasa env reset produces, which every GR00T checkpoint was trained on. The reset mirror is mandatory because `G1GearWBCEnv.reset()` — the entry point the GR00T inference bridge actually uses — calls `mj_resetData` which wipes the `__init__` seed (caught by the first post-Phase-4a numerical A/B dump). Pre-fix the runtime left those slots at zero (straight legs) and relied on the leg PD to drive them toward `default_angles` over hundreds of steps — fine for teleop, but the first `policy.get_action` of a GR00T inference run then saw OOD `state.{left,right}_leg` and returned pathological arm chunks ("hands lift toward face" persisting after Bugs #1–#7). Pinned by `tests/test_runtime_leg_default_pose.py` (5 cases incl. end-to-end pin through `Gr00tObservationBuilder.build()`). Contract + diagnosis: **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md) §8** (Bug #8 archaeology) + **§9** (numerical A/B dump protocol used to find it).

## GR00T VLA vs “not falling over”

- **GR00T (VLA)** outputs high-level **unitree_g1** actions: arm/waist/hand joints + `navigate_command` + `base_height_command` (see `gr00t/configs/data/embodiment_configs.py` in Isaac-GR00T). It does **not** replace the leg balance loop by itself in sim.
- **Gear WBC ONNX** (`Balance`/`ft92`, `Walk`/`ft109`) is the **legged controller** used in the official G1 loco-manip stack; `--stand` uses that layer only.

Future work: **compose** VLA outputs into the same command path the eval stack uses: **`WholeBodyControlWrapper`** + **`concat_action`** (`n1_utils.py`). Official client eval: **`gr00t/eval/rollout_policy.py`** (`get_groot_locomanip_env_fn`, **`MultiStepWrapper`**, `policy.get_action`). See `memory-bank/manipulation_and_gr00t_plan.md` for how that differs from raw `gear_wbc_stand` (dict obs, cameras, `unitree_g1` modalities in `embodiment_configs.py`).

## ONNX filename resolution

`g1_gear_wbc.yaml` names `policy/ft92.onnx` and `policy/ft109.onnx`. Some LFS checkouts provide:

- `GR00T-WholeBodyControl-Balance.onnx` → stand  
- `GR00T-WholeBodyControl-Walk.onnx` → walk  

`load_gear_wbc_config()` resolves either naming scheme automatically.

## Python scoping note

`run_gear_wbc()` must not use `import mujoco.viewer` inside the function before other `mujoco.*` calls (that made `mujoco` a local and caused `UnboundLocalError`). Use `from mujoco import viewer as mj_viewer` in the viewer branch only.

## VR compose contract (do-not-break)

- Keep exactly one rigid compose chain for stream overlays and future IK:
  \[
  ^{W}\!T_{draw} = ^{W}\!T_{torso}^{vr} \left(^{S}\!T_{hmd0}\right)^{-1} \, ^{S}\!T_{dev}
  \]
  where \(^{W}\!T_{torso}^{vr}\) is torso with optional receiver yaw basis offset.
- Apply heading correction as a **single rigid transform** (receiver-side), not by ad-hoc per-device/per-axis hacks.
- Do not rotate torso world position as part of yaw correction (orientation basis only), or walking away from origin can look like drift/warp.
- Prefer `calib-orient=yaw_only` for robust operator startup; `full` remains for debugging.
- For current-HMD head-relative anchors (`torso` / `pelvis`), prefer **`vr-head-orient=yaw_only`** so controller translation does not depend on HMD pitch/roll. This avoids the operator having to tilt the headset to recover intuitive body-forward mapping.
- Current stabilization gap: IK may choose awkward/twisted arm configurations while still tracking targets (damping-only nullspace today).
- Next pattern (optional): merge **`arm_ik_posture.augment_task_with_posture`** into the stacked IK normal equations for a low-weight joint-space bias toward a neutral/home posture — keep implementation in **`arm_ik_posture.py`**, not scattered in the runtime hot path.
- Scope guard: avoid broad new guardrail systems (full collision constraints, major solver refactors, additional frame-path complexity) unless needed after posture re-wiring.

## Reference code (upstream)

- `external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py`  
- `gr00t_wbc/control/utils/n1_utils.py` (`WholeBodyControlWrapper`, `concat_action`)  
- `gr00t_wbc/control/policy/g1_gear_wbc_policy.py` (keyboard cmd semantics for future teleop)

