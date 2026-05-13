# ACTIVE BUG — GR00T inference loop wiring

> **TEMPORARY FILE.** Delete (or graduate; see §12) when the exit criterion in §1 is met. While this file exists it is the single canonical reference for the inference-loop wiring investigation — read it before touching `scripts/run_gr00t_inference.py`, `gear_wbc_stand.GearWBCRuntime`, `gr00t_observation_builder.py`, `_apply_action_step`, or anything in `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/`.

Last touched: May 2026 (Phase 4b lighting + `hybrid_policy_get_action` probe on ego; step-0 metrics improved; qualitative §1 still open).

---

## 1. TL;DR + exit criterion

We are building **teleop → collect → process → train → infer → control**. Right now there is a suspected break in the `infer → control` link. The diagnostic is qualitative: with a **known-good policy** (CloudWalk apple-to-plate checkpoint, `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` or equivalent), the upstream `rollout_policy.py` produces a robot that reaches for the apple, attempts to grasp, walks toward the plate, and opens its hand over the plate (usually misses the apple — that's fine). Our `scripts/run_gr00t_inference.py --scene table_pnp_apple` with the same server + checkpoint historically produced a robot that stood still and brought its hands toward the face; **after Phase 4b MJCF lighting** (May 2026) operator reports **slightly better reach for the first few seconds**, then regression toward face / failure to complete the task. The behavioural divergence is still structural vs Tier-A; not a single-step precision miss only.

**Eight wiring/perceptual/state-init bugs (#1–#8) are already closed** (§8). Step-0 `state.*` and `annotation.*` are now byte-equal between oracle and subject. But the robot still doesn't qualitatively reach. So either:

- (a) one of the wiring contracts we *think* we have right is actually wrong, but the byte-equal A/B harness can't see it at step 0 (e.g. it diverges over time, or it's in a key we're not comparing), or
- (b) one of the wiring rows in §6 / §7 we haven't audited line-by-line yet is the bug.

**Exit criterion (qualitative, video-eval only):** with the CloudWalk apple-to-plate checkpoint and `run_gr00t_inference.py --scene table_pnp_apple`, **one** recorded run must show:

- [ ] Arm reaches toward apple region (not face).
- [ ] Hand closes (attempts grasp) when arm is near apple.
- [ ] Base navigates toward the plate region (not stationary).
- [ ] Hand opens over the plate region.

All four observed in one run. May miss the apple. May fumble. **Delete this file when met.** Lifecycle in §12.

**What is NOT an exit criterion:**

- Byte-equal `state.*` between oracle and subject (already done; was a means, not the end).
- Byte-equal `action.*` between oracle and subject (different RoboCasa scene; mild divergence expected).
- Byte-equal `video.ego_view` (not required; Phase 4b **shrunk** mean RGB gap vs Tier-A — see **§9.5.b**).
- Numerical tolerances on the A/B harness `--state-atol`, `--action-atol` past current values.

**What the byte-equal A/B harness IS useful for:** localizing where in §6 / §7 the bug is hiding. It already paid for itself by finding Bug #8.

---

## 2. Pipeline context (why this bug matters)

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌─────────┐
│ teleop  │ → │ collect │ → │ process │ → │  train  │ → │  INFER   │ → │ CONTROL │ → robot
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └──────────┘   └─────────┘
                                                              │             │
                                                              └─────────────┘
                                                                  THIS FILE
```

`g1_minimal_sim/scripts/run_gr00t_inference.py` is the seam between **INFER** (the policy server returning an action chunk) and **CONTROL** (the action chunk getting translated into MuJoCo PD targets / control_dict writes). The CloudWalk checkpoint is a **known-good policy probe** — if its actions can't drive the robot to vaguely demo the task, the wiring is suspect, not the model. Once the wiring is verified, this same loop runs our own finetuned checkpoints (currently `~/models/GR00T-N1.6-G1-LiftRuns-001-015`, blue-cube lift task) where the eval criterion also stays qualitative ("does it pick up the cube?").

**Upstream eval reference (Tier A):**

```
Isaac-GR00T/gr00t/eval/rollout_policy.py
  └─ env: gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc
     └─ MultiStepWrapper → VideoRecordingWrapper → WholeBodyControlWrapper → SyncEnv
```

**Subject (Tier B, ours):**

```
g1_minimal_sim/scripts/run_gr00t_inference.py
  └─ env: G1GearWBC-v0 with scene=table_pnp_apple
     └─ Gr00tObservationBuilder + _apply_action_step + GearWBCRuntime.step_physics
```

Same checkpoint, same server, different env stacks. Tier A works qualitatively; Tier B doesn't. Goal: get Tier B to qualitatively pass without changing the model.

---

## 3. Symptom + reproduction (qualitative, video-led)

### 3.1 Reference (Tier A) — what "vaguely working" looks like

Saved videos:

- `g1_minimal_sim/data/ab_pnp_20260512/side_a_rollout/sim_eval_videos_gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc_ac30_*/`. Both runs visible; pick either.

Frame-by-frame: stands, pitches torso forward, reaches right arm toward apple, closes fingers around apple, lifts apple slightly, walks left, opens fingers over plate, drops apple (may miss plate — fine). This is **the** behaviour we're trying to mimic.

Reproduce (from `Isaac-GR00T/` with policy server already running on `127.0.0.1:2000`):

```bash
gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  gr00t/eval/rollout_policy.py \
  --policy_client_host 127.0.0.1 --policy_client_port 2000 \
  --model_path "" \
  --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
  --n_episodes 1 --max_episode_steps 5000 --n_action_steps 30 --n_envs 1
```

### 3.2 Observed (Tier B) — what's currently broken

Saved videos:

- `g1_minimal_sim/data/ab_pnp_20260512/side_b_table_pnp.mp4`        (pre-fixes #5–#8)
- `g1_minimal_sim/data/ab_pnp_20260512/side_b_table_pnp_v2.mp4`     (post-fixes #5–#7, pre-fix #8)
- `g1_minimal_sim/data/ab_pnp_20260512/side_b_table_pnp_v3.mp4`     (post-fix #8 / Phase 4a-bis)

Frame-by-frame in the latest (`v3`): stands. Pitches torso slightly. Both arms slowly lift up toward chest/face level. Fingers don't close. Doesn't walk toward plate. Never reaches apple. End of run.

**Post Phase 4b (May 2026):** early seconds can look more apple-directed; failure mode often **reverts later** in the episode. Treat as **time-varying** closed-loop drift, not only step-0 obs. Operator checklist for a **single obs pass**: **`inference_obs_one_pass.md`**.

Reproduce (from `g1_minimal_sim/` with same policy server running):

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  scripts/run_gr00t_inference.py \
  --scene table_pnp_apple \
  --policy-host 127.0.0.1 --policy-port 2000 \
  --max-steps 1500 \
  --save-video data/ab_pnp_20260512/side_b_repro.mp4
```

### 3.3 What "fixed" looks like

Doesn't have to be pretty. Bot moves arm in apple's direction, closes fingers, walks toward plate, opens fingers. Even one out of three tries is enough. Then delete this file.

---

## 4. Reference loop — upstream (oracle) call trace

This section establishes **how the working version routes actions to joints and state to obs**. §5 mirrors it for our subject loop, §6 + §7 diff per-key. Line refs are absolute paths from repo root.

### 4.1 One plan step — call stack

```
rollout_policy.run_rollout_gymnasium_policy()                rollout_policy.py:248
  │
  ├─ actions, _ = policy.get_action(observations)              line 309
  │     # Returns chunk of n_action_steps=30 (per --n_action_steps),
  │     # each step is a dict of 7 keys (left_arm/right_arm/left_hand/
  │     # right_hand/waist/navigate_command/base_height_command).
  │     # Server (Gr00tPolicy._get_action) has ALREADY run
  │     # StateActionProcessor.unapply_action which absolutizes
  │     # RELATIVE-rep keys via _convert_to_absolute_action(action, state[-1])
  │     # → JointActionChunk.to_absolute_chunking. So arm keys arrive
  │     # as ABSOLUTE joint targets, not deltas. (Bugs #1, #4)
  │
  └─ next_obs, _, _, _, _ = env.step(actions)                  line 328
        │
        └─ MultiStepWrapper.step(actions)
              │ Splits the (n_action_steps, D) chunk into N single steps;
              │ for each step calls inner env.step(action_i). Returns stacked obs.
              │
              └─ VideoRecordingWrapper.step(action_i)
                    │ Records a frame, passes through.
                    │
                    └─ WholeBodyControlWrapper.step(action_i)
                          │   n1_utils.py:35
                          │
                          ├─ action_dict = concat_action(robot_model, action_i)
                          │     #   n1_utils.py:127
                          │     # Renames keys (strip "action." prefix), builds:
                          │     #   navigate_cmd               (3,) from action.navigate_command
                          │     #   base_height_command        (1,) from action.base_height_command
                          │     #   target_upper_body_pose     (D_upper,) — joint indices
                          │     #     assembled via robot_model.get_joint_group_indices(group)
                          │     #     for left_arm, right_arm, left_hand, right_hand, waist.
                          │
                          ├─ wbc_policy.set_goal({navigate_cmd, base_height_command,
                          │                       target_upper_body_pose})       n1_utils.py:43
                          │
                          ├─ wbc_action = wbc_policy.get_action()                n1_utils.py:44
                          │     │   G1DecoupledWholeBodyPolicy.get_action()
                          │     │   g1_decoupled_whole_body_policy.py:91
                          │     │
                          │     ├─ upper_body_action = upper_body_policy.get_action()
                          │     │     # IdentityPolicy.get_action just returns self.goal
                          │     │     # verbatim (identity_policy.py:13). NO interpolation.
                          │     │
                          │     ├─ q[upper_body_indices] = upper_body_action["target_upper_body_pose"]
                          │     │     # g1_decoupled_whole_body_policy.py:119
                          │     │     # Scatters arm + hand + waist targets into the full DoF q.
                          │     │
                          │     ├─ robot_model.cache_forward_kinematics(q, auto_clip=False)
                          │     │     # g1_decoupled_whole_body_policy.py:125
                          │     │     # Pinocchio FK on the full q (legs are zero — only
                          │     │     # waist+arms+hands matter for torso/pelvis frame).
                          │     │
                          │     ├─ torso_orientation = frame_placement("torso_link").rotation
                          │     ├─ waist_orientation = frame_placement("pelvis").rotation
                          │     ├─ waist_yaw = atan2(R_pelvis[1,0], R_pelvis[0,0])
                          │     ├─ R_z_pelvis = rpyToMatrix(0, 0, waist_yaw)
                          │     ├─ yaw_only_waist_from_torso = R_z_pelvis.T @ torso_orientation
                          │     ├─ torso_orientation_rpy = matrixToRpy(yaw_only_waist_from_torso)
                          │     │     # g1_decoupled_whole_body_policy.py:126-134
                          │     │     # This is the FK we mirror in _waist_action_to_rpy_cmd
                          │     │     # (Bug #5 fix). YAW-ONLY pelvis frame; not just identity.
                          │     │
                          │     ├─ lower_body_action = lower_body_policy.get_action(
                          │     │       time, q_arms, base_height_command,
                          │     │       torso_orientation_rpy, interpolated_navigate_cmd)
                          │     │     # g1_decoupled_whole_body_policy.py:136
                          │     │     # G1GearWbcPolicy — runs leg ONNX (stand or walk)
                          │     │     # with these as part of the policy obs vector.
                          │     │
                          │     ├─ q[lower_body_indices] = lower_body_action["body_action"][0][:len(lower_body_indices)]
                          │     │     # g1_decoupled_whole_body_policy.py:141
                          │     │     # Legs + waist from the ONNX; if waist appears in both
                          │     │     # upper and lower body groups, LOWER takes preference.
                          │     │
                          │     └─ return {"q": q}     # full (num_dofs,) joint target vector
                          │
                          ├─ result = super().step(wbc_action)               n1_utils.py:46
                          │     │
                          │     └─ SyncEnv.step({"q": full_q})               sync_env.py:201
                          │           │
                          │           └─ queue_action(action)                sync_env.py:244
                          │                 │
                          │                 ├─ action_q = convert_q_to_actuated_joint_order(action["q"])
                          │                 │     # Convert joint-order q → actuator-order q
                          │                 │     # (right hand actuator order ≠ qpos order).
                          │                 │
                          │                 └─ self.env.step({"q": action_q, "tau": tau_q})
                          │                       # → Gr00tLocomanipRoboCasaEnv → RoboCasa env;
                          │                       #   RoboCasa's composite controller PDs the
                          │                       #   q targets onto MuJoCo joints over
                          │                       #   one control tick (control_freq=50, 20 ms).
                          │
                          └─ wbc_policy.set_observation(result[0])         n1_utils.py:47
                                # Feed obs back so the next plan can use it. Note: the
                                # IdentityPolicy upper body ignores observation entirely
                                # (g1_decoupled_whole_body_policy.py:31-32).
```

### 4.2 Observation builder (upstream)

`SyncEnv.observe()` (sync_env.py:144) constructs the per-step observation dict:

| Key | Source | Notes |
|---|---|---|
| `q` | `robot_model.get_configuration_from_actuated_joints(body_q, left_hand_q, right_hand_q)` from RoboCasa `raw_obs` | Actuator → joint order; positions only |
| `dq`, `ddq`, `tau_est` | Same conversion on RoboCasa's `body_dq`, etc. | Debug taps; not policy inputs |
| `floating_base_pose / vel / acc` | From `raw_obs` | Debug taps |
| `wrist_pose` | FK on `q` via `get_eef_obs` (`sync_env.py:428`) | Debug tap |
| `state.*` (7 keys) | `prepare_observation_for_eval(robot_model, obs)` slices `q` by joint group | **THE policy state inputs.** Positions only — no velocities. |
| `annotation.human.task_description` | `raw_obs["language.language_instruction"]` | RoboCasa's per-env instruction (e.g. `LMPnPAppleToPlate._get_instruction`) |
| `video.<name>` | `raw_obs[<name>_image]` for each camera | uint8 RGB; one per camera in `camera_names` |
| `torso_quat`, `torso_ang_vel` | G1-specific (`G1SyncEnv.observe`, sync_env.py:510) | Not policy inputs in current modality config |

The policy's modality config (`MODALITY_CONFIGS["unitree_g1"]` in `embodiment_configs.py`) tells the client which keys it actually consumes. Confirmed inputs: `video.ego_view`, `state.{7 groups}`, `annotation.human.task_description`.

### 4.3 Cadence

- One `policy.get_action` call → chunk of `n_action_steps = 30` plan steps.
- One plan step → one `MultiStepWrapper` iteration → one `WholeBodyControlWrapper.step` → one `wbc_policy.get_action` → one `SyncEnv.step` → one underlying `env.step({"q": action_q, "tau": tau_q})`.
- RoboCasa's `control_freq = 50` (sync_env.py:50) means one underlying `env.step` advances physics by 20 ms, internally as N mj_steps.
- Net: 1 policy call → 30 plan steps × 20 ms = 600 ms of simulated time before the next replan.

---

## 5. Subject loop — ours (`run_gr00t_inference.py`) call trace

Same shape as §4, aligned for diffing.

### 5.1 One plan step — call stack

```
run_gr00t_inference.py    (== run_gr00t_stylish_diner_inference.py)
  │
  ├─ main()  → _run_policy_loop(args, env, rt, obs_builder, n_substeps, ...)
  │            scripts/run_gr00t_stylish_diner_inference.py:1148
  │
  └─ for step_i in range(args.max_steps):                       line 1210
       │
       ├─ obs_flat = obs_builder.build(rt.model, rt.data,
       │                               task_description=args.task_description,
       │                               include_video=True)        line 1211
       │     │   gr00t_observation_builder.py:190
       │     │
       │     ├─ q = mujoco_q_to_robot_model_q(model, data, robot_model)
       │     │     # gr00t_observation_builder.py:54
       │     │     # Reads each named joint's qpos from MuJoCo (by joint name lookup)
       │     │     # into the RobotModel's DoF index order (Pinocchio joint order).
       │     │
       │     ├─ prepare_observation_for_eval(robot_model, obs)
       │     │     # SAME function as upstream uses (n1_utils.py:152).
       │     │     # Splits q into state.{7 groups}; positions only.
       │     │
       │     ├─ obs["annotation.human.task_description"] = task_description
       │     │
       │     └─ if include_video:
       │           rgb = render_ego_rgb_uint8(model, data)         # MuJoCo Renderer
       │           obs["ego_view_image"] = rgb
       │           obs["video.ego_view"] = rgb
       │
       ├─ # Append to history buffers
       ├─ for k in state_keys: state_hist[k].append(obs_flat["state.{k}"])
       ├─ for k in video_keys: video_hist[k].append(obs_flat[video_src])
       │
       ├─ # Replan if needed (chunk exhausted or first step)
       ├─ if (not action_plan) or (action_plan_idx >= args.n_action_steps):
       │     policy_obs = build_history_stacks(state_hist, video_hist,   line 1246
       │                                       language_keys, task_description)
       │     action_raw, _info = client.get_action(policy_obs)            line 1257
       │     # client = PolicyClient(host=..., port=...). Same server,
       │     # same StateActionProcessor.unapply_action absolutization as oracle.
       │     # action_raw is a dict of 7 (1, T, D) tensors.
       │     action_plan = {k: arr[0] for k, arr in action_raw.items()}    line 1259
       │     action_plan_idx = 0
       │     n_replans += 1
       │
       ├─ # Slice out one plan step from the chunk
       ├─ action_step = _select_plan_step(action_plan, action_plan_idx)    line 1294
       │
       ├─ state_now = {f"state.{k}": state_hist[k][-1] for k in state_keys}  line 1296
       │     # Latest state — only used by _apply_action_step in `relative`
       │     # arm-action-mode (NPZ replay branch). On the live branch with
       │     # --arm-action-mode auto resolved to absolute, state_now is unused.
       │
       ├─ applied = _apply_action_step(rt, action_step, state_now,         line 1297
       │                arm_action_mode=arm_mode_resolved,        # = "absolute"
       │                clip_arm_targets_to_limits=...,
       │                apply_waist_fn=apply_waist_fn,            # Bug #5 fix
       │                hand_permutation=hand_permutation)        # Bug #6 fix
       │     │   run_gr00t_stylish_diner_inference.py:573
       │     │
       │     ├─ # action.navigate_command (3,) → loco_cmd[:3]            line 608
       │     ├─ # action.base_height_command (1,) → height_cmd           line 612
       │     ├─ # action.waist (3,) → Pinocchio FK → rpy_cmd[:3]         line 616
       │     │       via _waist_action_to_rpy_cmd(robot_model, w)
       │     │       (mirrors upstream G1DecoupledWholeBodyPolicy FK byte-for-byte)
       │     ├─ # action.{left,right}_arm (7,) → arm_target_q[SL_{SIDE}_ARM]
       │     │       direct write (absolute live; +state_now on replay)  line 694
       │     ├─ # action.{left,right}_hand (7,) → arm_target_q[SL_{SIDE}_HAND]
       │     │       permuted policy→mj order via hand_permutation       line 700
       │     └─ # rt.control_dict mutated in-place under rt.cmd_lock
       │
       ├─ _advance_physics_one_plan_step(env, n_substeps)                  line 1315
       │     │   run_gr00t_stylish_diner_inference.py:1126
       │     │   Calls env.step(np.zeros(1, dtype=np.float32)) n_substeps times
       │     │   (= 4 with diner YAML's sim_dt=0.005s and plan_hz=50.0).
       │     │
       │     └─ for _ in range(n_substeps):  env.step(...)
       │           │   g1_gear_wbc_env.G1GearWBCEnv.step  →  rt.step_physics()
       │           │       gear_wbc_stand.py:742
       │           │
       │           ├─ # Read VR/UDP teleop sources (irrelevant; no teleop active)
       │           ├─ # Apply IK from teleop sources (irrelevant)
       │           │
       │           ├─ # Leg PD                                       line 812
       │           │   leg_tau = pd_control(
       │           │       target_dof_pos = self.action * action_scale + default_angles,
       │           │       q   = data.qpos[7 : 7+num_act],
       │           │       kp  = config["kps"],
       │           │       qd_des = 0,
       │           │       dq  = data.qvel[6 : 6+num_act],
       │           │       kd  = config["kds"])
       │           │   data.ctrl[:num_act] = leg_tau
       │           │
       │           ├─ # Arm + hand PD                                 line 822
       │           │   arm_tgt = control_dict["arm_target_q"]
       │           │   arm_tau = pd_control(arm_tgt[s_idx], q_seg[s_idx],
       │           │                        kp_by_slice, ..., dq_seg[s_idx], kd_by_slice)
       │           │   data.ctrl[num_act:] = clip(arm_tau, -tau_clip, +tau_clip)
       │           │
       │           ├─ mujoco.mj_step(model, data)                     line 885
       │           │
       │           └─ # Every `decim` mj_steps, run leg ONNX          line 906
       │               if counter % decim == 0:
       │                   single = compute_single_obs(data, config, action,
       │                                               control_dict, ...)
       │                       # gear_wbc_obs.py:35 — builds 86-D obs vector:
       │                       #   command[:3] = loco_cmd[:3] * cmd_scale
       │                       #   command[3]  = height_cmd
       │                       #   command[4:7] = rpy_cmd[:3]    (from action.waist via FK)
       │                       #   + omega, gravity, q_scaled, dq_scaled, prev_action
       │                   self.action = self.stand_net(input) if loco_norm <= 0.05
       │                                 else self.walk_net(input)
       │                   target_dof_pos = self.action * action_scale + default_angles
       │
       └─ # Append one video frame per plan step (not per mj_step)        line 1317
          if writer is not None:
              writer.append_data(obs_flat["video.ego_view"])
```

### 5.2 Cadence

- One `client.get_action` → chunk of `args.n_action_steps = 30` plan steps.
- One plan step → one outer loop iteration → one `_apply_action_step` + one `_advance_physics_one_plan_step`.
- `_advance_physics_one_plan_step` calls `env.step` `n_substeps = round(1/(plan_hz*sim_dt)) = 4` times.
- Each `env.step` does one `mj_step` (5 ms with diner YAML).
- Net: 1 plan step = 4 mj_steps = 20 ms simulated time. Matches upstream's 20 ms per plan step. (Bug #3 fix.)

### 5.3 Where the leg ONNX gets its inputs

Inside `gear_wbc_stand.step_physics` (line 906), the leg ONNX runs every `decim` mj_steps using `compute_single_obs` (`gear_wbc_obs.py:35`). Its inputs:

- `command[0:3]` = `loco_cmd[:3] * cmd_scale`  — **driven by `action.navigate_command` via `_apply_action_step` line 608.**
- `command[3]`   = `height_cmd`                — **driven by `action.base_height_command` via line 612.**
- `command[4:7]` = `rpy_cmd[:3]`               — **driven by `action.waist` via FK in `_waist_action_to_rpy_cmd`.**
- `omega`, `gravity`, `qj_scaled`, `dqj_scaled`, `prev_action` — read from MuJoCo state.

This is the analog of upstream's `lower_body_policy.get_action(time, q_arms, base_height_command, torso_orientation_rpy, interpolated_navigate_cmd)`. **Same leg ONNX file, same input layout.** (The Gear WBC `ft92.onnx` / `ft109.onnx` are upstream's exact same files; both stacks consume them with the same `compute_observation` layout.)

---

## 6. Per-action-key wiring (the diff that matters)

For each of the 7 `action.*` keys: where does the value go upstream, where does it go in our subject, and do we have evidence they're equivalent? "??" rows are the most likely sites of the residual qualitative failure.

| `action.*` key | Upstream sink (file:line) | Our sink (file:line) | Equiv? | Evidence / notes |
|---|---|---|---|---|
| `action.left_arm` (7,) | `concat_action` → `target_upper_body_pose[left_arm_idx]` (`n1_utils.py:142-148`) → `IdentityPolicy.get_action` returns verbatim → `q[upper_body_indices]` (`g1_decoupled.py:119`) → `SyncEnv.step({"q": ...})` → RoboCasa composite controller PD | `arm_target_q[SL_LEFT_ARM]` direct write (`run_gr00t_*.py:696`) → `gear_wbc_stand.step_physics` arm PD (`gear_wbc_stand.py:850-883`) → `data.ctrl[num_act:]` → `mj_step` | **YES** | Bug #1 + #4 closed: absolute live, relative replay, resolved per branch. Order matches (both use `robot_model.get_joint_group_indices("left_arm")` and MuJoCo qpos kinematic-tree order; `assert_arm_joint_order_matches` startup guard pins it. Pinned by `test_run_gr00t_inference_cadence.py::test_apply_action_step_*`. |
| `action.right_arm` (7,) | Symmetric to left | Symmetric to left | **YES** | Same as above. |
| `action.left_hand` (7,) | `concat_action` → `target_upper_body_pose[left_hand_idx]` in `robot_model.get_joint_group_indices("left_hand")` order (sorted DoF idx = `[index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2]`) | `arm_target_q[SL_LEFT_HAND]` after permutation `perm = [4,5,6,2,3,0,1]` (`run_gr00t_*.py:702`) gathered via `hand_perm_policy_to_mj("left", policy_names)` | **YES** | Bug #6 closed. Upstream's `target_upper_body_pose` is in sorted-DoF order; our `SL_LEFT_HAND` is MuJoCo qpos order (`hand_gripper.LEFT_HAND_JOINT_NAMES` = thumb→middle→index). Permutation reorders to match. Pinned by `test_apply_action_step_applies_hand_permutation_*` + `test_real_hand_perm_policy_to_mj_matches_canonical`. |
| `action.right_hand` (7,) | Symmetric to left | Symmetric to left | **YES** | Same as above. |
| `action.waist` (3,) | Not written to joints directly. `concat_action` puts it into `target_upper_body_pose[waist_idx]` → `IdentityPolicy` → `q[waist_idx]` → Pinocchio FK in `G1DecoupledWholeBodyPolicy.get_action` (line 125-134) → `torso_orientation_rpy` (yaw-only pelvis frame) → fed to `lower_body_policy.get_action` → leg ONNX obs. | `_waist_action_to_rpy_cmd(robot_model, waist_action)` (`run_gr00t_*.py:292`) does the **same FK byte-for-byte** → `rt.control_dict["rpy_cmd"]` → `compute_single_obs` `command[4:7]` → leg ONNX obs. | **YES** | Bug #5 closed. We do NOT write `action.waist` into qpos directly — neither does upstream. Both routes feed it to the leg ONNX via the same FK. Pinned by `test_waist_action_to_rpy_cmd_axis_aligned` (Pinocchio FK round-trip on yaw / roll / pitch). |
| `action.navigate_command` (3,) | `concat_action` renames → `navigate_cmd` (`n1_utils.py:137`) → `wbc_policy.set_goal({navigate_cmd})` → stashed in `upper_body_goal` → passed through `IdentityPolicy.get_action` → `interpolated_navigate_cmd` (note name suggests interpolation but `IdentityPolicy` doesn't interpolate; `g1_decoupled.py:122`) → `lower_body_policy.get_action(..., interpolated_navigate_cmd)` → leg ONNX obs. | `rt.control_dict["loco_cmd"][:3] = action_step["action.navigate_command"][:3]` direct write (`run_gr00t_*.py:610`). Then `compute_single_obs` `command[:3] = loco_cmd[:3] * cmd_scale` (`gear_wbc_obs.py:51`). | **?? UNVERIFIED** | Same leg ONNX, same `cmd_scale` (YAML), same downstream consumer. But **upstream passes navigate_cmd through G1GearWbcPolicy first** before it lands in the leg ONNX obs; we go directly. If `G1GearWbcPolicy.get_action` applies any rescaling / sign flip / frame transform on navigate_cmd before constructing its own obs vector, we'd silently mismatch. **§10B audits this.** |
| `action.base_height_command` (1,) | `concat_action` → `base_height_command` (`n1_utils.py:138`) → `wbc_policy.set_goal({base_height_command})` → `upper_body_goal["base_height_command"]` → `lower_body_policy.get_action(..., base_height_command, ...)` → leg ONNX. | `rt.control_dict["height_cmd"] = float(action.base_height_command)` (`run_gr00t_*.py:614`). Then `compute_single_obs` `command[3] = height_cmd` (`gear_wbc_obs.py:52`). | **?? UNVERIFIED** | Same shape as `navigate_command`. Same audit applies (§10C). |

### 6.1 What "YES" actually means

The "YES" rows assert that **at step 0**, the value written by both stacks ends up at the same downstream consumer with the same numerical interpretation. They do NOT assert:

- That the value is correct over time. (Drift from policy-step compounding is possible.)
- That the dtype / fp rounding is identical. (Differences in fp32 vs fp64 broadcasting across the dozens of intermediate ops could accumulate.)
- That the FP scheduling within the 4-substep loop is identical. (Upstream's RoboCasa controller runs internal control ticks differently from our `step_physics` decimation.)

The byte-equal `state.*` A/B harness confirms the round trip `MuJoCo qpos → state.* obs → policy → action.*` is byte-equal at step 0 for state. It does NOT confirm `action.* → joint target → mj_step → new qpos → state.*` produces identical evolution across multiple plan steps. **§10A is designed to test exactly that.**

### 6.2 What "?? UNVERIFIED" actually means

`navigate_command` and `base_height_command` are the two action keys where upstream's path adds a non-trivial layer (`G1GearWbcPolicy.get_action`) between the wrapper write and the leg-ONNX obs construction. We go straight to the leg-ONNX obs via `rt.control_dict["loco_cmd"]` / `["height_cmd"]`. If that policy layer does any of:

- Scale `navigate_cmd` by something other than `cmd_scale` (e.g. clamping, smoothing across plan steps, frame rotation through current torso yaw)
- Modify `base_height_command` relative to current pelvis height
- Cache or filter command values across substeps

…then we'd silently see a different control regime in the legs. The leg ONNX is what makes the robot walk and stabilize — if it gets wrong navigate inputs, the robot can sit motionless or shuffle in place while the upper body tries to reach. **This is the highest-probability unaudited row.**

To verify: read `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/policy/g1_gear_wbc_policy.py` end-to-end. We have NOT done this yet for the inference investigation; we leaned on the same-ONNX-file argument. §10B is the audit.

---

## 7. Per-state-key wiring (the diff that matters)

Same shape as §6 for the seven `state.*` keys that the modality config feeds the policy.

| `state.*` key | Upstream source (file:line) | Our source (file:line) | Equiv? | Evidence / notes |
|---|---|---|---|---|
| `state.left_arm` (7,) | `SyncEnv.observe` → `q = get_configuration_from_actuated_joints(...)` (`sync_env.py:152`) → `prepare_observation_for_eval` slices `q[left_arm_idx]` (`n1_utils.py:159, 167`) | `Gr00tObservationBuilder.build` → `mujoco_q_to_robot_model_q(model, data, robot_model)` (`gr00t_observation_builder.py:54, 202`) → `prepare_observation_for_eval` slices `q[left_arm_idx]` | **YES** | Same slicing function. Joint order matches (`robot_model.joint_names`). Positions only. Byte-equal at step 0 in A/B harness (post Bug #8). |
| `state.right_arm` (7,) | Symmetric | Symmetric | **YES** | Same. |
| `state.left_hand` (7,) | `prepare_observation_for_eval` slices `q[left_hand_idx]` (sorted DoF idx; index→middle→thumb) | Same | **YES** | Same. (Note: this is in policy DoF order, NOT MuJoCo qpos order. The hand permutation only matters on the action side; state side is consistent.) Byte-equal at step 0. |
| `state.right_hand` (7,) | Symmetric | Symmetric | **YES** | Same. |
| `state.left_leg` (6,) | `prepare_observation_for_eval` slices `q[left_leg_idx]` | Same. Initial value is YAML `default_angles[0:6]` post-`reset()` (Phase 4a-bis) | **YES** | Bug #8 closed. Both `__init__` and `reset()` seed `default_angles`. Pinned by `test_runtime_leg_default_pose.py`. Byte-equal at step 0 in A/B harness. |
| `state.right_leg` (6,) | Symmetric | Symmetric (`default_angles[6:12]`) | **YES** | Same. |
| `state.waist` (3,) | `prepare_observation_for_eval` slices `q[waist_idx]` | Same (`default_angles[12:15]` post-reset, all zeros) | **YES** | Bug #8 closed. Byte-equal at step 0. |

### 7.1 Velocity question (closed)

Earlier hypothesis: maybe `state.*` carries positions+velocities and we're sending zeros. **Resolution:** `prepare_observation_for_eval` reads only `obs["q"]` (positions). The function never touches `dq`/`ddq`. Both upstream and ours consume positions only. Not a bug candidate.

### 7.2 What "YES" doesn't cover

At step 0 we're byte-equal. But each subsequent plan step, the policy sees `state.*` that depends on what the physics did in response to the previous action. If our `_apply_action_step → step_physics → mj_step` produces different qpos evolution than upstream's `SyncEnv.step → RoboCasa controller → mj_step`, then by step 1 the policy sees different state, returns different actions, and the trajectories fork — **possibly to "stand and lift hand to face" instead of "reach and grab"**. §10A is the test that probes this.

---

## 8. Closed bugs (#1–#8) — chronological summary

This table is the canonical archaeology. Detailed per-bug writeups (root cause, diagnostic, fix, pinning test) used to live in `activeContext.md` "Finetuned checkpoint bring-up" and `gr00t_compatible_data_collection.md` §6.1–§6.7. Both of those were moved into this file when it was created.

| # | Name | Symptom | Root cause | Fix location | Pinned by |
|---|---|---|---|---|---|
| 1 | Action-rep mismatch (replay path) | Replay-arms idle | `--arm-action-mode absolute` was the default; on-disk NPZ stores `target − state` deltas (logger default `relative_arm_actions=True`); writing delta directly into `arm_target_q` froze arms near zero | `resolve_arm_action_mode(mode, branch)` — `auto → relative` on replay branch | `test_run_gr00t_inference_cadence.py::test_resolve_arm_action_mode_*` |
| 2 | Dtype-cast write loss | All PD writes silently dropped | `_apply_action_step` did `atq = np.asarray(arm_target_q, dtype=np.float32)`; runtime `arm_target_q` is fp64 from `data.qpos.copy()`; the dtype request triggered a fresh fp32 buffer, throw-away | `_apply_action_step` reads raw `rt.control_dict["arm_target_q"]` buffer (no dtype cast) — `run_gr00t_*.py:636` | Diagnostic confirmation via `step=… atq.la=… qpos.la=…` debug line; no dedicated test (rolled into Bug #1 e2e test) |
| 3 | Plan-cadence mismatch | Live arms exploded ~11 rad in 50 plan steps | Inference bridge consumed one plan step per `mj_step` (5 ms); training data sampled at 50 Hz (20 ms / plan). Running 4× too fast compounded relative-action state error | `_advance_physics_one_plan_step(env, n_substeps=4)`; `--plan-hz 50.0` default; pinned by `resolve_plan_substeps` | `test_run_gr00t_inference_cadence.py` (20 cases, incl. `DEFAULT_PLAN_HZ == Gr00tTeleopEpisodeLogger.sample_hz`) |
| 4 | Live-branch double-add (the structural pretzel) | Live arms pretzeled within ~50 plan steps despite #1 fixed | Server's `StateActionProcessor.unapply_action` already absolutizes RELATIVE keys via `_convert_to_absolute_action(action, state[-1])` → `to_absolute_chunking`. After Bug #1 fix, script defaulted to `relative` on BOTH branches → live branch added `state_now` on top → `state_server_ref + delta_predicted + state_now` | Split default by branch via `resolve_arm_action_mode(mode, branch)`; `auto → absolute` on live | Same test as Bug #1 (resolver + e2e-apply-step cases) |
| 5 | `action.waist` silently dropped (hands lift toward face — CloudWalk PnP A/B) | Persistent symptom: arms climb to chest/face, fingers never react. `pelvis_z` pinned at 0.74 m | Our `_apply_action_step` had NO `action.waist` branch. Upstream `G1DecoupledWholeBodyPolicy.get_action` runs Pinocchio FK on `q[upper_body_idx]` to derive `torso_orientation_rpy` (yaw-only pelvis frame) which feeds the leg ONNX obs (`command[4:7]`). Without this, `rpy_cmd` stayed at YAML default `[0,0,0]`, legs kept torso upright, GR00T's arm policy compensated by swinging arms up | `_waist_action_to_rpy_cmd(robot_model, w)` mirrors upstream FK byte-for-byte; threaded into `_apply_action_step(apply_waist_fn=...)`; `--no-apply-waist` reproduces broken path | `test_waist_action_to_rpy_cmd_axis_aligned` (Pinocchio FK round-trip on yaw / roll / pitch) + plumbing cases |
| 6 | Hand-joint permutation (index commands actuate thumbs) | Right hand chunk `[+0.032 +0.053 +0.044 +0.026 −0.013 −0.019 −0.014]` — index/middle positive then thumb negative — routed onto thumb-first qpos | `robot_model.get_joint_group_indices("{side}_hand")` returns sorted DoF idx: `[index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2]`. Our `SL_*_HAND` indexes MuJoCo qpos tree order: `[thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]`. Direct write swapped signs onto wrong fingers | `hand_perm_policy_to_mj(side, policy_names)` derives `[4,5,6,2,3,0,1]` from joint names at startup; `_apply_action_step(hand_permutation=...)` gathers `action.{side}_hand[perm]` before write; `--no-permute-hands` reproduces broken path; `assert_arm_joint_order_matches` startup guard | `test_apply_action_step_applies_hand_permutation_left_and_right` + `test_real_hand_perm_policy_to_mj_matches_canonical` |
| 7 | Visual / language alignment | "Hands lift toward face" persisted on Tier-B PnP after #1–#6 even with bridge correct | Three concurrent distribution shifts on `scenes/table_pnp_apple/g1_gear_wbc{,_hands}_table_pnp_apple.xml`: (a) only `head_pov` camera defined (~18° pitch-down, fovy=110°) vs upstream `oak_egoview` (~45° pitch-down, fovy=79.5°); (b) hand-mesh `rgba="0.7 0.7 0.7 1"` (light grey) vs upstream `rgba="0.1 0.1 0.1 1"` (near-black); (c) task description was a truncated paraphrase missing "walk left and" directive | (a) Added `<camera name="oak_egoview"...>` with exact upstream pos/quat/fovy to both XMLs; (b) recoloured 32 hand-mesh geoms to near-black in hands XML; (c) `DEFAULT_TASK_BY_SCENE` carries upstream string verbatim | `test_table_pnp_apple_mjcf_has_oak_egoview_camera`, `test_hands_mjcf_recolours_hand_meshes_to_near_black`, `test_cli_default_camera_name_is_oak_egoview`, `test_resolve_task_description_scene_defaults` |
| 8 | Leg-pose init mismatch (Phase 4a + 4a-bis) | "Hands lift toward face" still persisted after #1–#7 (`side_b_table_pnp_v2.mp4`) | RoboCasa env reset writes YAML `default_angles` into qpos BEFORE first `policy.get_action`. Our `GearWBCRuntime.__init__` left `qpos[7 : 7+15]` at zero (straight legs). GR00T checkpoint had never seen straight-leg pose in training → policy returned pathological arm chunks; `~14 cm` pelvis-height mismatch also brightened ego frame +10 RGB mean. Phase 4a fixed `__init__` but A/B dump showed legs still zero — root cause was `G1GearWBCEnv.reset()` → `mj_resetData` wiping the `__init__` seed | Phase 4a: write `qpos[_policy_qpos_adr[:num_act]] = config["default_angles"]` in `__init__`. Phase 4a-bis: mirror same write into `reset()` between `mj_resetData` and `mj_forward` | `test_runtime_leg_default_pose.py` (5 cases: post-init, default values, pelvis-z, post-reset, end-to-end through `Gr00tObservationBuilder`) |

### 8.1 Phase 1+2 forensic verdict (ruled out, NOT a bug)

After Bugs #1–#7 the operator asked whether the relative-vs-absolute mode or CW-vs-CCW joint conventions were actually correct. A read-only re-evaluation conclusively ruled both out:

- **Phase 1 (action contract):** `rollout_policy.py` consumes `action.left_arm` / `action.right_arm` as absolute joint targets because `StateActionProcessor.unapply_action` (server-side) absolutizes RELATIVE keys via `_convert_to_absolute_action` → `to_absolute_chunking`. `WholeBodyControlWrapper.step` + `concat_action` write them directly into `target_upper_body_pose`, and `IdentityPolicy` passes them through to the upper-body branch. Our `_apply_action_step` absolute write on the live branch (post Bug #4 fix) is correct.
- **Phase 2 (joint kinematics):** All 14 arm joints + 3 waist joints are byte-identical between Tier-A `g1_29dof_rev_1_0.xml` and Tier-B `g1_gear_wbc_hands_table_pnp_apple.xml` (axes, ranges, parent bodies, qpos addressing). URDF the `RobotModel` loads matches both MJCFs. No sign/range/parent-frame discrepancy.

This is why §10 next-experiments focus on observation evolution over time and on the two unverified `action.*` rows (§6 ?? entries), not on rerunning the action-mode debate.

---

## 9. Diagnostic tools available (NOT exit criteria)

### 9.1 `policy_ab_dump.py` — module

`g1_minimal_sim/policy_ab_dump.py` exposes `dump_policy_roundtrip_pack(out_dir, *, side, policy_obs, actions, extra)`. Serializes the policy input dict + raw returned action chunks to disk at the **first** `policy.get_action` of a run:

```
out_dir/
├── manifest.json                       # side, timestamp, shape table, extra
├── obs/
│   ├── state_left_arm.npy              # one per state.* key (batch dim stripped)
│   ├── state_left_leg.npy
│   ├── …
│   ├── annotation_human_task_description.npy
│   ├── video_ego_view.npy              # full t0 frame stack (uint8)
│   └── video_ego_view_t0.png           # rendered preview
└── action/
    ├── action_left_arm.npy             # one per action.* key (batch dim stripped)
    ├── action_navigate_command.npy
    └── …
```

Hooked into both clients via `--policy-ab-dump DIR`. Pinned by `tests/test_policy_ab_dump.py`.

### 9.2 `scripts/run_policy_ab_dump.py` — one-command harness

Bundles "probe server → run oracle with `--policy-ab-dump <run-id>/oracle` → run subject with `--policy-ab-dump <run-id>/subject` → load both → print colourised per-key diff table → exit non-zero on regression" into one command. From `g1_minimal_sim/`:

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  scripts/run_policy_ab_dump.py --run-id <name>
```

Flags worth knowing:

- `--diff-only` — re-print diff for existing run-id (fast iteration on tolerances).
- `--skip-oracle` / `--skip-subject` — re-run one side against existing other-side dump.
- `--save-video PATH` — also record subject MP4 for side-by-side viewing.
- `--state-atol 1e-4` (default) — hard Bug-#8 regression gate.
- `--action-atol 5e-2` (default) — soft gate; exceeded → exit non-zero.
- `--oracle-arg` / `--subject-arg` — pass-through extra argv (repeatable).

Exit codes: `0` OK / `1` REGRESSION / `2` Isaac-GR00T root missing / `3` server probe failed / `4` dump dirs missing post-run.

### 9.3 Interpretation matrix (what differs → where to look)

| Differs | Likely root cause | Where in this doc |
|---|---|---|
| `state.left_leg` / `state.right_leg` | Robot-state init drift | §8 Bug #8; `GearWBCRuntime.__init__` AND `.reset()` |
| `state.waist` | Waist init drift | §8 Bug #8; `_arm_target_q_home` not mutated by prior run |
| `state.{left,right}_arm` / `_hand` | Arm/hand init drift | `_init_arm_state`; hand permutation Bug #6 |
| `video.ego_view` mean / brightness | Camera pose / lighting / torso height | §8 Bug #7 camera contract; if torso height wrong recheck Bug #8 |
| `annotation.human.task_description` | Prompt mismatch | §8 Bug #7 task description row; `DEFAULT_TASK_BY_SCENE` |
| Action chunk only (obs all match) | Server-side decoder drift, or checkpoint variant | Compare `processor_config.json`; check both clients hit same port |

### 9.4 Other ablation toggles on `run_gr00t_inference.py`

- `--debug-policy-prints` — modality config dump; first-N replan action stats including hand chunk summary; periodic `rpy=(r,p,y)` + `|atq.lh|/|qpos.lh|`; one-time warnings if `action.waist` or `action.{side}_hand` missing on hands MJCF.
- `--replay-npz PATH` — bypass policy server, feed `action.*` from recorded NPZ through the same `_apply_action_step`. Use to sanity-check the action-application path without the policy.
- `--no-apply-waist` — reproduce Bug #5 pre-fix path for A/B.
- `--no-permute-hands` — reproduce Bug #6 pre-fix path for A/B.
- `--camera-name head_pov` — reproduce Bug #7 camera-fallback path for A/B.
- `--arm-action-mode {auto,relative,absolute}` — explicit override of branch resolver.

### 9.5 Current verdict

#### 9.5.a Historical snapshot — `post4a_bis` (May 12 2026, pre–Phase 4b lighting pass)

`scripts/run_policy_ab_dump.py --run-id post4a_bis` post-Phase-4a-bis output:

- `state.*` (all 7 keys): **OK** (byte-equal).
- `annotation.human.task_description`: **OK**.
- `video.ego_view`: **WARN** — per-channel mean RGB delta `[+10.5, +10.3, +10.1]` (minimal brighter than Tier-A).
- `action.left_arm` / `action.right_arm` / `action.navigate_command`: **WARN** vs Tier-A recorded first chunk.

#### 9.5.b Post Phase 4b MJCF + hybrid probe (May 2026, paired fresh dumps)

**MJCF (Tier-B only):** darker `<visual><headlight>` + attenuated worldbody directional `<light>` in:

- `scenes/table_pnp_apple/g1_gear_wbc_table_pnp_apple.xml`
- `scenes/table_pnp_apple/g1_gear_wbc_hands_table_pnp_apple.xml`
- `scenes/lab_dc_layout/lab_dc_world.xml` + **`gen_lab_dc_world_xml.py`** (regen parity)

Upstream vendored `arenas/gear_lab/gear_lab.xml` is **unchanged** — parity tuning lives in **minimal merged scenes** only.

**Representative paired dumps** (`compare_policy_ab_dumps.py --exit-on never`, rollout vs minimal after 4b):

- `video.ego_view` mean RGB delta vs pre-4b **shrunk** from ~**+11 / +10 / +10** per channel to ~**+4.8 / +4.7 / +4.7** (minimal still slightly brighter).
- First-chunk **`action.navigate_command`** diff between rollout-recorded and minimal-recorded actions moved into **OK** at default action atol (**~0.043** `max|diff|`). Arm rows still **WARN** but improved vs §9.5.a.

**`scripts/hybrid_policy_get_action.py`** (same two dump roots, live `get_action`):

- **`minimal`** vs Tier-A ref: `navigate_command` still **WARN**; arms **WARN**.
- **`hybrid_ego`** (Tier-B `state.*` + language + **Tier-A `video.ego_view`**) vs Tier-A ref: **`navigate_command` OK**, **`left_arm` OK**, **`right_arm` WARN** but much smaller than pure minimal — **ego pixels remain strongly causal at step 0**.

**Server repeatability noise floor:** `rollout` mode vs rollout’s own dumped actions can still show small **WARN** on `navigate_command` (~0.067) — treat as nondeterminism / float noise when reading OK/WARN bands.

**Qualitative:** §1 exit **still unchecked**; early-episode behavior **somewhat improved** per operator.

**Conclusion:** Observation / lighting was a **real partial lever**; remaining gap is **not** guaranteed to be “one more MJCF knob” — if hybrid is already tight vs Tier-A ref at step 0 but live fails later, shift effort to **§10** time-resolved tools (trace, oracle chunk replay, multi-step dumps).

---

## 10. NEXT EXPERIMENTS to localize the qualitative failure

Ranked by leverage. The byte-equal A/B has done all it can at step 0; from here the question is **how do the trajectories evolve differently after step 0**.

**Operator shortcut for obs-only iteration:** **`inference_obs_one_pass.md`** (dump → compare → hybrid → MJCF scope → re-measure → qualitative §1, with a **stop rule** when hybrid is already tight but live still fails over time).

### 10A. Action-trajectory replay (highest leverage; ~1 hour, clean answer either way)

**Hypothesis:** if our action-consumption is right, then feeding upstream's exact action chunks into our env should reproduce reach-and-grab behaviour.

**Protocol:**

0. **Already implemented (Variant A):** record Tier-A `replan_*.npz` via `rollout_policy.py --record-oracle-action-chunks DIR` (`oracle_action_chunk_io.py`); replay on Tier-B with **`run_gr00t_inference.py --replay-oracle-action-chunks DIR`** (no `PolicyClient`). Same `_apply_action_step` + physics cadence as live. Use this before building a new `chunk_*.npz` format.
1. **Capture phase (optional extension).** Add a hook to `Isaac-GR00T/gr00t/eval/rollout_policy.py` (or a fork) that records every `actions, _ = policy.get_action(observations)` chunk for the first N plan steps (say N=200, ~4 s simulated time) of a successful Tier-A run. Save to disk as a sequence of 7-key dicts.
2. **Replay phase.** Add `--replay-action-chunks PATH` flag to `scripts/run_gr00t_inference.py` that loads the captured sequence and consumes the recorded action chunks **instead of calling the policy**. Everything else runs normally: same `Gr00tObservationBuilder`, same `_apply_action_step` (including waist FK + hand permutation), same `_advance_physics_one_plan_step`, same MJCF, same camera.
3. **Record video.** `--save-video data/replay_oracle_actions.mp4`.
4. **Compare to Tier-A reference video qualitatively.**

**Outcomes:**

- **Subject reaches / grabs / walks:** our action consumption is correct; the residual bug is in the *observation* we build (camera render path, state slicing, language prompt — most likely camera/lighting since `state.*` is byte-equal at step 0). **Phase 4b** (§9.5.b) already improved ego / step-0 actions; continue **`inference_obs_one_pass.md`** or shift to **time-resolved** tools (trace, multi-step dumps) if replay is good but live still drifts.
- **Subject still stands and lifts to face:** our action consumption itself has a wiring bug not caught at step 0. Most likely candidates: `navigate_command` / `base_height_command` routing (§10B/C), or upstream's `WholeBodyControlWrapper.step` does something between `concat_action` and `wbc_policy.get_action` that we don't mirror (interpolation across plan steps via `target_time` / `interpolation_garbage_collection_time` — see `n1_utils.py:36-43`).

**Why this is the right first test:** it cleaves the hypothesis space in half with a single recording. Either action consumption is right or wrong; we currently don't know which.

**Implementation skeleton (~1.5 hours):**

```python
# In rollout_policy.run_rollout_gymnasium_policy, after `actions, _ = policy.get_action(...)`:
if record_action_chunks_dir:
    np.savez(
        Path(record_action_chunks_dir) / f"chunk_{i:05d}.npz",
        **{k: np.asarray(v) for k, v in actions.items()},
    )

# In run_gr00t_inference._run_replay_action_chunks_loop:
chunks = sorted(Path(args.replay_action_chunks).glob("chunk_*.npz"))
action_plan: dict[str, np.ndarray] = {}
action_plan_idx = 0
for step_i in range(args.max_steps):
    obs_flat = obs_builder.build(rt.model, rt.data, task_description=..., include_video=True)
    if (not action_plan) or (action_plan_idx >= args.n_action_steps):
        chunk_path = chunks.pop(0)
        with np.load(chunk_path) as z:
            action_plan = {k: z[k][0] for k in z.files}  # strip batch dim, (T,D)
        action_plan_idx = 0
    action_step = _select_plan_step(action_plan, action_plan_idx)
    action_plan_idx += 1
    state_now = {f"state.{k}": ... for k in state_keys}
    _apply_action_step(rt, action_step, state_now, arm_action_mode="absolute",
                       apply_waist_fn=..., hand_permutation=...)
    _advance_physics_one_plan_step(env, n_substeps)
    if writer: writer.append_data(obs_flat["video.ego_view"])
```

### 10B. `action.navigate_command` routing audit

**Hypothesis:** upstream's path through `G1GearWbcPolicy.get_action` does something to `navigate_cmd` between `WholeBodyControlWrapper.set_goal` and the leg-ONNX obs that we skip by going `loco_cmd → compute_single_obs.command[:3]` directly.

**Protocol (read-only, no code changes):**

1. Read `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/policy/g1_gear_wbc_policy.py` end-to-end. Trace how `navigate_cmd` (and `base_height_command`, `torso_orientation_rpy`) flow from `lower_body_policy.get_action(time, q_arms, base_height_command, torso_orientation_rpy, interpolated_navigate_cmd)` (`g1_decoupled_whole_body_policy.py:136`) into the leg ONNX obs vector.
2. Diff against `gear_wbc_obs.compute_single_obs` (`gear_wbc_obs.py:35`). Check:
   - Is `cmd_scale` applied identically? (YAML config — should be identical because both stacks load the same `g1_gear_wbc.yaml`.)
   - Is `navigate_cmd` filtered / smoothed / clamped across plan steps upstream? (Look for `prev_cmd`, `cmd_history`, `low_pass`, etc.)
   - Is `navigate_cmd` rotated through current torso yaw? (Body-frame vs world-frame conventions.)
   - Are sign conventions identical for `[vx, vy, vyaw]`? (Should be — same ONNX file — but worth pinning.)
3. Run a controlled experiment: with the subject's leg ONNX inputs frozen to a known walking-forward profile (set `loco_cmd = [0.3, 0, 0]` manually for 200 steps), verify the robot walks forward. If it does, navigate routing is wired correctly for the standard case.

**If a divergence is found in step 2:** that's Bug #9. Fix in `_apply_action_step` or in `gear_wbc_stand.step_physics` to mirror upstream's transform.

### 10C. `action.base_height_command` routing audit

Same shape as §10B. Likely a smaller leverage win than §10B because `base_height_command` is a scalar with less room for frame / scaling confusion, but worth ruling out.

**Protocol:** Same as §10B but focused on the `base_height_command` row in `g1_gear_wbc_policy.py`. Check if upstream maintains a target-vs-current pelvis-height error, applies any smoothing, or applies it as a delta vs current rather than absolute target height.

### 10D. Camera / lighting harmonization (Phase 4b)

**Hypothesis (only chase if §10A points back here):** the +10 RGB mean shift on `video.ego_view` is large enough to push the vision-language head out of distribution, even though the geometric camera pose matches upstream exactly post Bug #7.

**Protocol:**

1. Read upstream RoboCasa scene's lighting block (`scenes/table_pnp_apple/`-equivalent in `gr00trobocasa/`) and mirror into our `scenes/table_pnp_apple/g1_gear_wbc_hands_table_pnp_apple.xml`. Likely candidates: `<light>` configs, ambient / diffuse / specular intensities, background colour.
2. Re-render and compare `video_ego_view_t0.png` previews.
3. Re-run A/B harness; expect `video.ego_view` WARN to shrink. Re-run qualitative test.

**Note:** this is intentionally lower-priority than §10A/B because the user has explicitly de-prioritized "make the egoview byte-equal" — what matters is whether the policy's *behavioural* output changes when lighting is fixed. §10A indirectly tests this: if the policy commands the right actions under replay, lighting probably isn't blocking us.

### 10E. Upper-body target interpolation across substeps (lowest priority)

**Hypothesis (low probability):** upstream's `G1DecoupledWholeBodyPolicy.get_action` is supposed to interpolate upper-body targets across the 4 substeps inside one plan step, but `IdentityPolicy.get_action` (`identity_policy.py:13`) just returns `self.goal` verbatim — no interpolation. So both stacks apply the target directly. Manifestation would be jerkier reaches in our version, NOT "doesn't reach at all".

**Status:** **ruled out as the qualitative-failure cause** based on `identity_policy.py:13`. Kept in this list for completeness; do not investigate further unless §10A/B fail to localize the bug.

### 10F. Time / cadence drift across many plan steps

**Hypothesis:** even with `n_substeps=4` correct on a single plan step, accumulated wall-clock or simulated-time drift over 200+ plan steps could push the trajectory to a fundamentally different basin.

**Protocol:** instrument both clients with `time.monotonic()` measurements around `policy.get_action` and `env.step`. Compare cumulative simulated time at step 200. Bug if they differ by more than ~1 plan period (20 ms).

**Priority:** low. Bug #3 fix pins `n_substeps` per plan step; cadence is right unless upstream's RoboCasa env secretly adds substeps we don't know about.

---

## 11. Hypotheses ruled out (don't re-investigate)

The following have been audited line-by-line and confirmed not to be the residual root cause. **Do not re-run these investigations** unless §10 fails and you've exhausted all of §10A–F.

- **Action-rep mismatch on the live branch.** (Bug #1 + #4 closed; server absolutizes RELATIVE keys; script does NOT add `state_now` on live.)
- **Plan-cadence mismatch.** (Bug #3 closed; `n_substeps = 4`.)
- **dtype-cast write loss.** (Bug #2 closed; raw buffer read.)
- **Hand permutation.** (Bug #6 closed; `[4,5,6,2,3,0,1]`.)
- **Waist FK drop.** (Bug #5 closed; `_waist_action_to_rpy_cmd` mirrors upstream byte-for-byte.)
- **Camera / hand colour / task prompt.** (Bug #7 closed; pin tests pass.)
- **Leg-init mismatch.** (Bug #8 closed; both `__init__` and `reset()` seed `default_angles`.)
- **Joint axes / ranges / parent frames.** (Phase 2 forensic audit — byte-identical between MJCFs.)
- **`state.*` velocities.** (Audit closed in §7.1 — `prepare_observation_for_eval` reads positions only; both stacks.)
- **`IdentityPolicy` interpolation.** (Closed in §10E — `IdentityPolicy.get_action` is a literal pass-through.)
- **`navigate_cmd` shape and key naming.** (Audit closed in §6 — both stacks consume `[vx, vy, vyaw]`; key is `navigate_cmd` upstream, `loco_cmd` in our control_dict; semantics confirmed identical in YAML.)

If a future investigation rediscovers any of these, **add a one-line entry here** explaining why the previous closure was wrong, rather than silently re-debugging.

---

## 12. Lifecycle on close

When the §1 exit criterion is met (four checkboxes ticked in one Tier-B run), do **one** of:

### Option A — Delete outright

If every contract has graduated into tests (`tests/test_*`) + brief invariant notes in `systemPatterns.md` / `techContext.md` / `gr00t-g1-wholebody.md`, just `git rm ACTIVE_gr00t_inference_wiring.md`. Update the banner at the top of `activeContext.md` and the `INDEX.md` "Currently active" line in the same commit. This is the preferred outcome — the doc was always meant to be temporary.

### Option B — Graduate to permanent reference

If you decide the per-key wiring tables (§6 / §7) and line-numbered loop walk-throughs (§4 / §5) are worth keeping as a long-term reference for future GR00T-related debugging:

1. Rename `ACTIVE_gr00t_inference_wiring.md` → `gr00t_inference_wiring.md` (drop the `ACTIVE_` prefix).
2. Trim aggressively:
   - **Keep:** §4, §5, §6, §7, §9 (just the tools, not the current verdict), §11 (durable ruled-out hypotheses).
   - **Discard:** §1 (exit criterion no longer applies), §2 (pipeline context belongs in `projectbrief.md`), §3 (symptom is historical), §8 (bug archaeology moves to `progress.md` appendix), §10 (next-experiments are stale), §12 (this section).
   - Target: ~300 lines vs current ~800.
3. Update `INDEX.md` to list it under "Topic docs (read on demand)" instead of "Currently active".
4. Update the banner in `activeContext.md` to remove the ACTIVE bug pointer.

### Either way: update cross-refs

After deletion or rename, grep the memory bank for `ACTIVE_gr00t_inference_wiring.md` and update any stale pointers:

```bash
rg "ACTIVE_gr00t_inference_wiring" g1_minimal_sim/memory-bank/
```

Expected callers: `activeContext.md` banner, `INDEX.md` currently-active section, `progress.md` open-items list, `systemPatterns.md` leg-init contract note, `gr00t-g1-wholebody.md` robot-state init contract note, `techContext.md` Phase-3 forensics paragraph, `gr00t_compatible_data_collection.md` §6 pointer.

---

## 13. Tests pinning closed contracts

The following tests are the regression gates for each closed bug. If any of these starts failing in CI, the corresponding contract has regressed and we'd reintroduce the matching symptom:

| Test file | Cases | Pins | Bug(s) |
|---|---|---|---|
| `tests/test_run_gr00t_inference_cadence.py` | 20+ | `resolve_plan_substeps`, `DEFAULT_PLAN_HZ == Gr00tTeleopEpisodeLogger.sample_hz`, `resolve_arm_action_mode` per branch, `_apply_action_step` e2e (incl. hand permutation, waist apply, navigate write, height write), `assert_arm_joint_order_matches`, `_clip_qpos_segment_to_joint_limits`, MJCF oak_egoview camera presence, hand mesh recolour, CLI default camera, scene task description | #1, #3, #4, #5, #6, #7 |
| `tests/test_waist_action_to_rpy_cmd.py` (or merged into above) | 3 (yaw, roll, pitch) | `_waist_action_to_rpy_cmd` Pinocchio FK round-trip | #5 |
| `tests/test_runtime_leg_default_pose.py` | 5 | `qpos[legs+waist]` matches `default_angles` post-`__init__`; YAML defaults match upstream standing crouch; pelvis-z in `[0.74, 0.83]` m; post-`reset()` qpos matches; end-to-end through `Gr00tObservationBuilder.build()` | #8 |
| `tests/test_policy_ab_dump.py` | 1 | `dump_policy_roundtrip_pack` writes expected manifest + .npy + .png layout | — (tool contract) |
| `tests/test_run_gr00t_inference_cadence.py::test_real_hand_perm_policy_to_mj_matches_canonical` | 1 | `[4,5,6,2,3,0,1]` derivation resilient to upstream joint-name reorder | #6 |

Run from `g1_minimal_sim/`:

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  -m pytest tests/test_run_gr00t_inference_cadence.py \
            tests/test_runtime_leg_default_pose.py \
            tests/test_policy_ab_dump.py -q
```

Expected: all green. If any fails after a change to the inference path, you've regressed one of the closed contracts — fix that before continuing investigation.

---

## File pointers (canonical paths)

| Layer | Path |
|---|---|
| Subject loop | `g1_minimal_sim/scripts/run_gr00t_inference.py` (alias for `run_gr00t_stylish_diner_inference.py`) |
| Subject runtime | `g1_minimal_sim/gear_wbc_stand.py` (`GearWBCRuntime.{__init__, reset, step_physics}`) |
| Subject obs | `g1_minimal_sim/gr00t_observation_builder.py` |
| Subject leg ONNX obs | `g1_minimal_sim/gear_wbc_obs.py` (`compute_single_obs`) |
| Subject Gym wrapper | `g1_minimal_sim/g1_gear_wbc_env.py` |
| Upstream loop | `Isaac-GR00T/gr00t/eval/rollout_policy.py` |
| Upstream WBC wrapper | `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/utils/n1_utils.py` |
| Upstream policy combiner | `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/policy/g1_decoupled_whole_body_policy.py` |
| Upstream upper-body policy | `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/policy/identity_policy.py` |
| Upstream lower-body policy | `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/policy/g1_gear_wbc_policy.py` |
| Upstream Gym env | `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/envs/robocasa/sync_env.py` |
| Server-side action decode | `Isaac-GR00T/gr00t/data/state_action/state_action_processor.py` (`unapply_action`) |
| Modality config | `Isaac-GR00T/gr00t/configs/data/embodiment_configs.py` (`MODALITY_CONFIGS["unitree_g1"]`) |
| A/B dump tool | `g1_minimal_sim/policy_ab_dump.py` |
| A/B harness | `g1_minimal_sim/scripts/run_policy_ab_dump.py` |
| RoboCasa task class | `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobocasa/robocasa/environments/locomanipulation/locomanip_dc.py` (`LMPnPAppleToPlateDC`) |
| Tier-B scenes | `g1_minimal_sim/scenes/table_pnp_apple/g1_gear_wbc{,_hands}_table_pnp_apple.xml` |
| Sample saved videos | `g1_minimal_sim/data/ab_pnp_20260512/` (`side_a_rollout/...` = Tier-A; `side_b_*.mp4` = Tier-B) |
