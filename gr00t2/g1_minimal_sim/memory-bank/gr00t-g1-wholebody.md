# GR00T × Unitree G1 whole-body control (reference)

Persistent notes from architecture discussion (Isaac-GR00T + GR00T-WholeBodyControl). **Read this when working on G1 loco-manip, teleop, or new sim tasks.**

## Stack

- **GR00T-WholeBodyControl demo** is **MuJoCo** (RoboCasa / Robosuite), not Isaac Lab. Eval: `rollout_policy.py` ↔ GR00T server ↔ Gym env `gr00tlocomanip_g1_sim/...` (`SyncEnv`, etc.).

## How walking works (not one monolithic policy)

- **GR00T VLA** (`EmbodimentTag.UNITREE_G1` / modality `unitree_g1` in `gr00t/configs/data/embodiment_configs.py`) predicts **actions**:
  - **Upper body**: `left_arm`, `right_arm`, `left_hand`, `right_hand`, `waist` — **joint space** (`NON_EEF`), not Cartesian EE targets by default.
  - **Base**: `navigate_command` (**3** floats) and `base_height_command` (scalar).
- **Legged locomotion** is a **separate module**: **`G1GearWbcPolicy`** — two **ONNX** policies (stand vs walk), driven by `cmd` / height / torso orientation and full proprioception; outputs **lower-body** joint targets.
- **`G1DecoupledWholeBodyPolicy`** combines **upper-body** (interpolation / joint targets) with **lower-body** (Gear WBC). The lower body receives arm pose, `base_height_command`, torso RPY, and interpolated `navigate_cmd` so legs co-react with the upper body.

## Action → WBC wiring

- **`WholeBodyControlWrapper`** + **`concat_action`** (`gr00t_wbc/control/utils/n1_utils.py`): maps flat keys like `action.navigate_command` → `navigate_cmd`, builds `target_upper_body_pose` from arm/waist/hand groups, then `wbc_policy.set_goal` / `get_action` → full `q` for the sim step.

## Semantics

- **`navigate_command`**: 3D high-level command for Gear WBC (not "full world [vx, vy, vz]" as a separate API); **vertical** posture is mostly **`base_height_command`**. Exact frame/convention matches Gear WBC / training (see `g1_gear_wbc.yaml`, `cmd_scale`).

## Teleop / "base + end effectors"

- **Keyboard locomotion** already exists on **`G1GearWbcPolicy.handle_keyboard_button`** (e.g. WASD-style `cmd` updates).
- For **Cartesian EE control**, add **IK** (or use RoboCasa teleop / wrist-pose paths) to produce **arm joints** fed as `target_upper_body_pose`; GR00T's shipped G1 action space is **joints**, not EE poses.

## New tasks / data

- New RoboCasa envs + prompts: register env IDs, match **modality config** and dataset layout; **MuJoCo path is lowest friction** vs porting to Isaac Lab.

## Key paths (Isaac-GR00T checkout)

- Modality: `gr00t/configs/data/embodiment_configs.py` (`unitree_g1`).
- WBC wrapper: `external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/utils/n1_utils.py`.
- Decoupled policy: `.../g1_decoupled_whole_body_policy.py`.
- Gear WBC: `.../g1_gear_wbc_policy.py`.
- Sync Gym env: `.../control/envs/robocasa/sync_env.py`.

## Local sandbox (`g1_minimal_sim`)

Minimal MuJoCo spawn + optional Gear WBC `--stand`. **Swappable scenes:** default **`stylish_diner`** (merged diner MJCF in-repo under `g1_minimal_sim/scenes/stylish_diner_1/`), **`floor`**, and **table PnP-style** (`--scene table_pnp`, `G1GearWBC-TablePnP-v0`) using the same Gear WBC bundle under `sim2mujoco/resources/robots/g1/` for floor/table — lighter than RoboCasa `LMPnPAppleToPlateDC` but same leg policy stack. **Keyboard teleop** (hold W/S A/D Q/E), optional **grasp primitives**, and **planned push-box smoke test:** `stylish_diner_keyboard_teleop.md`. **Roadmap, commands, tests:** `INDEX.md`.

## Robot-state init contract for GR00T eval (Phase 4a, May 2026)

When running any GR00T checkpoint against an in-repo MuJoCo scene the **first observation** the policy consumes MUST come from the same standing posture the policy was trained on. RoboCasa's env reset writes the YAML's `default_angles` into qpos before the first `policy.get_action`; **`gear_wbc_stand.GearWBCRuntime.__init__`** now mirrors that by writing `self.data.qpos[self._policy_qpos_adr[:num_act]] = config["default_angles"]` (legs + waist) immediately before the initial `mj_forward`. Pre-fix the runtime left those slots at zero (straight legs), and `state.{left,right}_leg` showed up OOD in the very first policy input → pathological arm chunks (the "hands lift toward face" symptom that persisted after fixing all wiring + perceptual mismatches). Contract + numerical A/B diagnosis: **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md) §8** (Bug #8 archaeology, leg-pose contract) + **§9** (`policy_ab_dump.py` + `scripts/run_policy_ab_dump.py` harness + `--policy-ab-dump` flag on both `rollout_policy.py` and `run_gr00t_inference.py`). Tests: `tests/test_runtime_leg_default_pose.py` + `tests/test_policy_ab_dump.py`.
