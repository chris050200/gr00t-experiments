# Plan: Gymnasium spine, manipulation scene, GR00T inference

This doc complements `progress.md`. **Execution order (locked in):** build the **Gymnasium spine + teleop/play driver first**; then manipulation scenes; then GR00T / eval parity.

---

## 0. Execution order (do this sequence)

1. **§G — Gymnasium spine** — Register a `gymnasium.Env` around the existing MJCo + Gear WBC stack; **teleop, visualization, and future policies** all go through this env (or thin wrappers), not only `spawn_g1_floor.py`’s bespoke loop.
2. **§A — Manipulation environment** — New MJCF or loaded XML variant; **swapped in via the same Env** (`env_cfg`, `xml_path`, or `gym.make` id).
3. **Gripper / hand** — Extend action/`control_dict` inside the same actuation path the env uses. **Done in minimal sim (Apr 2026):** teleop grippers + `g1_gear_wbc_hands.xml` + `hand_gripper.py` / `GearWBCRuntime` (`--hands`, `use_hands`); see **`g1_minimal_sim/memory-bank/hands_gripper_teleop.md`**. Remaining: policy / `unitree_g1` action wiring (**D2**).
4. **§B–E — GR00T** — `MultiStepWrapper` + `rollout_policy.py` or local policy against the **same** env class.

---

## G. Gymnasium spine (teleop + play + eval prep) — **first**

**Goal:** One **spine** for everything interactive: physics + WBC live behind a **`gymnasium.Env`** so you can swap **environment** (XML / scene id), **command source** (keyboard teleop, GR00T policy, scripted zeros), and **visualization** (`render`, MuJoCo viewer) without forking the control loop.

**Design (three layers — all behind Gym):**

1. **Same env, two (or more) loops**  
   - **Play mode:** a small driver (e.g. `play.py` or CLI flag) steps the env; **teleop** writes actions or internal command buffer → `step`.  
   - **Eval mode:** `rollout_policy.py` (or local script) steps the same env; **policy** writes actions → `step`.  
   Same physics contract; only the **command source** changes.

2. **`gymnasium.Env` as the public API**  
   - `reset()` / `step(action)` / `render()` (and optional `metadata` for render mode).  
   - Refactor **`gear_wbc_stand`** so the heavy logic is **called from Env methods**, not the only entrypoint. `spawn_g1_floor.py` can remain as a thin CLI: `gym.make(...)` + run the appropriate driver.

3. **Injectable pieces (keep modular)**  
   - **Environment** — which MJCF / scene (future: table PnP XML vs floor-only).  
   - **Command source** — teleop keyboard (pynput), later VR/webcam, or policy.  
   - **Actuation** — map command → torques (existing `control_dict` + Gear WBC + arm IK path).  
   - **Optional `ObservationBuilder`** — when GR00T is wired: build dict obs matching `unitree_g1` inside env or a wrapper.

**Teleop specifically:** Implement as a **driver** that uses the Env: read keys → produce `action` (or write into env-internal `control_dict` before `step`, if action space is structured that way). Avoid leaving teleop **only** inside a non-Gym codepath long term; **Gym-first for teleop** is the target.

**Deliverables (checklist)**

- [x] `gymnasium` env **`G1GearWBCEnv`** + register **`G1GearWBC-v0`** (`g1_gear_wbc_env.py`).
- [x] Shared sim **`GearWBCRuntime`** (`gear_wbc_stand.py`); `run_gear_wbc` uses it.
- [x] **Play / teleop CLI** `play_g1_gear_wbc.py`.
- [x] Headless pytest + pelvis z band (`tests/test_g1_gear_wbc_env_headless.py`).
- [x] `techContext.md` + `spawn_g1_floor.py` doc mention.

**Tests**

- Headless: `N` steps after `reset` without error; pelvis z in same band as `spawn_g1_floor.py --stand --headless`.
- Manual: teleop through new play entrypoint matches current behavior (loco + torso-frame EE IK).

**Migration note:** It is OK to keep `run_gear_wbc()` as an internal helper the Env calls; the **user-facing** story becomes “install env, run play or rollout,” not only “run spawn script.”

---

## A. Manipulation environment (PnP / apple–plate style) — **after §G**

**Goal:** Same robot + Gear WBC stack, richer scene: floor + table + manipuland + goal (e.g. plate).

**Current minimal sim MJCF:** `g1_gear_wbc_table.xml` / `g1_gear_wbc_hands_table.xml` — static table + **freejoint** apple (sphere) and plate (cylinder); tabletop ~**0.95 m**; place manipuland centers **inside the table top’s x/y extent** or they fall at spawn.

**Options (unchanged; now both load through the Gym env):**

1. **Composite MJCF** — Include robot from `g1_gear_wbc.xml` bundle; add `g1_table_task.xml` (or similar) with table / object / receptacle. Env takes `xml_path` or a second registered id.
2. **Official task env** — `gym.make("gr00tlocomanip_g1_sim/...")` for benchmark parity (heavier deps). Your **play driver** can target this id too once the spine pattern exists.

**Suggested order:** Implement **(1)** in `g1_minimal_sim` first (fast iteration); use **(2)** for checkpoint benchmarking.

**Deliverables**

- [x] Env parameters / second env id for table PnP MJCF (`G1GearWBC-TablePnP-v0`, `scene`, `config_yaml`).
- [x] Headless smoke on new scene (`test_g1_gear_wbc_env_headless.py` table case).
- Optional geometric success checks later.

**Tests**

- Stand regression on new MJCF (pelvis z band).

---

## B. GR00T inference: how `rollout_policy.py` fits

Official client eval (see `Isaac-GR00T/examples/GR00T-WholeBodyControl/README.md`):

- **Server:** `gr00t/eval/run_gr00t_server.py` with `--embodiment-tag UNITREE_G1` and checkpoint.
- **Client:** `gr00t/eval/rollout_policy.py` with `--env_name ...`, etc.

Pipeline inside `rollout_policy.py` (simplified):

1. `get_gym_env` → for `UNITREE_G1`, `get_groot_locomanip_env_fn`: `gym.make(..., camera_names=[...])` then **`WholeBodyControlWrapper`** (Gear WBC in the loop).
2. Optional `VideoRecordingWrapper`.
3. **`MultiStepWrapper`**: stacks video/state, chunks **`n_action_steps`** for the policy.
4. Rollout loop: **`policy.get_action(observations)`** → `env.step(actions)`.

Once **§G** is done, your **minimal env** is already Gym-shaped; adding **`MultiStepWrapper`** and policy-ready **obs dict** becomes an incremental wrapper on the same class (or a registered variant), instead of a one-off bridge from scratch.

**`g1_minimal_sim` pre-§G:** raw MuJoCo + `gear_wbc_stand.py` + `control_dict` — no policy dict obs yet. **Post-§G:** Env exposes `step`/`reset`; obs/action spaces can be extended toward `unitree_g1` without a second parallel loop.

---

## C. Observation / action contract (`unitree_g1`)

Source of truth: `Isaac-GR00T/gr00t/configs/data/embodiment_configs.py` → **`MODALITY_CONFIGS["unitree_g1"]`**.

- **Video:** `ego_view` (delta indices as in config).
- **State:** `left_leg`, `right_leg`, `waist`, `left_arm`, `right_arm`, `left_hand`, `right_hand`.
- **Action:** `left_arm`, `right_arm`, `left_hand`, `right_hand`, `waist`, `base_height_command`, `navigate_command` (relative/absolute per `ActionConfig` entries).
- **Language:** `annotation.human.task_description`.

Implement **inside or beside the Env** (e.g. `_get_obs()` building the dict) so play and eval share one implementation.

**Teleop → finetune pipeline (detailed):** how to build **GR00T-compatible** dict obs, log **concat_action**-shaped actions, pick **ego** resolution, and export **LeRobot** — see **`gr00t_compatible_data_collection.md`** (phases P0–P4, MJCF/camera caveats). **Code:** **`gr00t_observation_builder.py`** (P1 state + optional **`head_pov`** video keys).

---

## D. Implementation strategies (updated)

| Strategy | What you build | Good when |
|----------|----------------|-----------|
| **D1 — Official gym env only** | `rollout_policy.py` + `LMPnP...` | Benchmark parity, no custom scene. |
| **D2 — Gym env over minimal sim** (**primary for this repo**) | Registered Env (§G) + optional `MultiStepWrapper` | Custom scene, teleop, eval on **one** spine. |
| **D3 — In-process bridge only** | Policy called inside non-Gym loop | Short hacks only; not the long-term direction once §G exists. |

**Recommendation:** **D2** is the default path for `g1_minimal_sim`. Use **D1** to validate checkpoint install. Avoid relying on **D3** as the main architecture after §G lands.

**Camera note:** Trace **`observation_schema`** / `SyncEnv` for `ego_view` mapping, sizes, and normalization when implementing obs in the Env.

---

## E. Suggested sequencing (full stack)

1. **§G** — Gymnasium env + play/teleop driver + tests (**before** new scenes).
2. **§A** — Manipulation MJCF / env variant; teleop in scene. **Follow-up:** §H **diner box push** smoke (keyboard-only) now that the arm-walk / live-VR wrist instability was resolved by the `arm_ik_v2` `mj_jac` point-frame fix (`stylish_diner_keyboard_teleop.md`).
3. **Gripper / hand** in env action / `control_dict` — **teleop path done** in `g1_minimal_sim`; GR00T hand modalities next.
4. **GR00T:** **D1** smoke; then extend **your Env** with obs dict + `MultiStepWrapper` / `rollout_policy` compatibility (**D2**).
5. **Tests:** pytest for env step; golden obs compare vs official env optional.

---

## F. File pointers

- Minimal table scene: `Isaac-GR00T/.../sim2mujoco/resources/robots/g1/g1_gear_wbc_table.xml`, `g1_gear_wbc_hands_table.xml`, `g1_gear_wbc_table.yaml`, `g1_gear_wbc_hands_table.yaml`; `g1_minimal_sim/gear_wbc_stand.py` (`resolve_gear_wbc_config`).
- Stylish diner: `g1_minimal_sim/scenes/stylish_diner_1/stylish_diner.xml` (layout), `g1_gear_wbc_stylish_diner.xml` / `.yaml` (robot + scene). Teleop / bug / box plan: **`stylish_diner_keyboard_teleop.md`**.
- Rollout: `Isaac-GR00T/gr00t/eval/rollout_policy.py` (`create_eval_env`, `get_groot_locomanip_env_fn`, `MultiStepWrapper`).
- Embodiment modalities: `Isaac-GR00T/gr00t/configs/data/embodiment_configs.py` (`unitree_g1`).
- Example eval commands: `Isaac-GR00T/examples/GR00T-WholeBodyControl/README.md`.
- WBC action glue (upstream): `GR00T-WholeBodyControl/.../n1_utils.py` (`concat_action`, `WholeBodyControlWrapper`).

---

## H. Stylish diner — keyboard-first manipulation smoke (planned)

**Goal:** Before VR or GR00T, validate **locomotion + arm IK + clutter** using only **`play_g1_gear_wbc.py --teleop`** (default **`--scene stylish_diner`**).

**Task:** Start a few feet from the central table, approach the free `target_block` placed on that table, and use the **rigid palms** (no articulated fingers in merged diner MJCF) plus optional `--keyboard-grasp-primitives` to **push/steer** it toward the kitchen / blue prep table — **very easy** success = visible translation under contact.

**Headless tests (not a substitute):** `tests/test_stylish_diner_hands_mjcf_integration.py` and `tests/test_g1_gear_wbc_env_headless.py` guard MJCF merge, config resolution, and stand-in-scene; they do **not** validate manipulation feel, slip, or carry. Keep this §H as the **operator / teleop** checklist; see `stylish_diner_keyboard_teleop.md` for the same distinction.

**Dependencies / risks:** The previous arm-walk / live-VR wrist instability is resolved by the `arm_ik_v2` `mj_jac` point-frame fix and live headset re-validation. If the smoke still exposes residual instability, narrow first to **`--scene floor`** + duplicated free box before debugging full diner clutter. Optional **`--keyboard-grasp-primitives`** for coarse palm spacing only.
