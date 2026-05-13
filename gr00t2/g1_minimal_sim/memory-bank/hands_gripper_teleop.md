# Hands & gripper teleop (G1 minimal sim)

**Status:** Working (Apr 2026), user-verified. Read this file after a context reset before touching `gear_wbc_stand.py`, `hand_gripper.py`, or MJCF hands.

---

## 1. What was added (one paragraph)

The stock **`g1_gear_wbc.xml`** has **welded fingers** (hand joints commented out). We added **`g1_gear_wbc_hands.xml`** + **`g1_gear_wbc_hands.yaml`** in the same vendored `.../robots/g1/` folder: articulated **14 hand DoFs**, **incremental gripper teleop**, and PD on fingers **without breaking** the pretrained **Gear WBC ONNX** (still 29 leg/waist/arm joints in the policy observation).

Enable with **`--hands`** on `spawn_g1_floor.py --stand` or **`play_g1_gear_wbc.py`** / Gym **`use_hands=True`**.

---

## 2. Files (canonical)

| Path | Role |
|------|------|
| `Isaac-GR00T/.../g1/g1_gear_wbc_hands.xml` | Copy of gear MJCF with hand `<joint>` + `<motor>` uncommented |
| `Isaac-GR00T/.../g1/g1_gear_wbc_hands.yaml` | Same as `g1_gear_wbc.yaml` but `xml_path: g1_gear_wbc_hands.xml` |
| `Isaac-GR00T/.../g1/g1_gear_wbc_hands_table.xml` / `.yaml` | Hands + table / apple / plate; yaml `xml_path: g1_gear_wbc_hands_table.xml`; use with `--scene table_pnp` or `G1GearWBC-TablePnP-v0` + `use_hands=True` |
| `g1_minimal_sim/scenes/stylish_diner_1/g1_gear_wbc_stylish_diner_hands.xml` / `.yaml` | **Diner + articulated hands** (May 2026 Phase 2). Built by appending the diner worldbody/lights suffix to `g1_gear_wbc_hands.xml`. `target_block` matches the welded-finger merged variant (half-size 0.14, density 5, friction `"3.0 0.1 0.005"`). Selected by `resolve_gear_wbc_config(scene="stylish_diner", use_hands=True)`. |
| `g1_minimal_sim/hand_gripper.py` | `gripper_left` / `gripper_right` in \([0,1]\), `hand_target_q()`, teleop keys, **`SL_*` slices** into `arm_target_q` |
| `g1_minimal_sim/gear_wbc_stand.py` | `GearWBCRuntime(..., config_yaml=...)`, policy obs gather, IK + hand fill + **actuator-ordered PD** |

Default sim still uses **`g1_gear_wbc.yaml`** (no articulated hands).

---

## 3. `control_dict` (hands mode)

| Key | Meaning |
|-----|---------|
| `gripper_left`, `gripper_right` | Scalar **1 = more open**, **0 = more closed**; clamped \([0,1]\); **incremental** key steps (`hand_gripper._GRIP_STEP`) plus hard open/close presets |
| `arm_target_q` | Length **28** when `has_hands`: see §4 |
| EE / IK keys | Unchanged: `ee_*` in `torso_link`, `z` resets loco + arms + EE home + **grippers → open** |

---

## 4. `arm_target_q` layout (critical)

Segment **`qpos[7 + num_act : 7 + n_joints]`** (and matching **`arm_target_q`**) follows **MuJoCo kinematic tree order**, not “both arms then both hands”:

| Slice (index in 28-vector) | Joints |
|----------------------------|--------|
| `[0:7]` `SL_LEFT_ARM` | Left arm (shoulder → wrist) |
| `[7:14]` `SL_LEFT_HAND` | Left hand (tree order: thumb → middle → index) |
| `[14:21]` `SL_RIGHT_ARM` | Right arm |
| `[21:28]` `SL_RIGHT_HAND` | Right hand (tree order: **thumb → middle → index**) |

**IK** (`arm_ik_v2.solve_dual_arm_ik_v2`) returns **14** values: left arm, then right arm. With hands:

- Write **`arm_target_q[SL_LEFT_ARM] = q_sol[:7]`**
- Write **`arm_target_q[SL_RIGHT_ARM] = q_sol[7:14]`**
- Never assign **`q_sol` to `arm_target_q[:14]` contiguously** — that would put right-arm targets into **left-hand** qpos slots (violent motion).

**Hands:** `hand_target_q(model, "left"|"right", g)` returns **7** angles in **the same order as qpos** for that side. **`RIGHT_HAND_JOINT_NAMES`** must be **thumb, middle, index** (matches tree). Do **not** follow the `<actuator>` block order for the right hand (see §6).

---

## 5. ONNX observation (29 joints only)

With **43** mechanism joints, the policy still expects a **fixed 86-D** single step (× history = 516). **`POLICY_JOINT_NAMES`** lists the **29** leg + waist + arm joints **by name**; `_compute_single_obs` **gathers** `qpos`/`qvel` at those addresses. Hand joints are **excluded** from the policy input.

---

## 6. Actuator order ≠ qpos order (right hand PD)

**Problem:** `data.ctrl[i]` is indexed by **`<actuator>` declaration order**. For the **right hand only**, MJCF lists motors **thumb → index → middle**, while **`qpos`** is **thumb → middle → index**.

**Fix:** `_arm_actuator_qpos_slice_indices()` builds **`_arm_ctrl_to_qslice`**: for each upper-body ctrl slot `i`, the index `s` into the contiguous **`arm_target_q` / qpos segment** for the joint that actuator actually drives (`actuator_trnid` → `jnt_qposadr`). Arm PD uses **gathered** arrays:

`arm_tgt[s_idx]`, `qpos[s_idx]`, `qvel[s_idx]`, `kp[s_idx]`, `kd[s_idx]` → `arm_tau` written to `ctrl[num_act:]` in **actuator order**.

Slots **0–23** are identity; **24–27** permute (index/middle swap vs naive index).

Without this, **only some fingers respond** and others look “frozen” or fight the wrong torque.

---

## 7. IK and `arm_qpos_ids`

**`arm_qpos_ids`** must list **only the 14 arm joint qpos addresses** (`left_ik.qpos_ids` ∪ `right_ik.qpos_ids`). Do **not** use `np.arange(7+num_act, 7+n_joints)` when hands exist — that includes **finger** qpos and breaks **`solve_dual_arm_ik_v2`** (expects 14).

---

## 8. Teleop keys (gripper)

| Key | Effect |
|-----|--------|
| **`9`** | Left gripper **more open** (+ step) |
| **`0`** | Left gripper **more closed** (− step) |
| **`=`** | Right gripper **more open** |
| **`-`** | Right gripper **more closed** |
| **`h`** | **Hard-close both** grippers (`gripper_left = gripper_right = 0`) |
| **`y`** | **Open both** grippers (`gripper_left = gripper_right = 1`) |

Focus **terminal** (pynput). **`z`** full reset includes grippers → fully open. These keys only affect hands-enabled scenes (`--hands` / `use_hands=True`). On no-hand scenes (welded MJCF), `h` and `y` print a one-time `[hand_gripper] h/y ignored: scene has no articulated grippers` warning and no-op — there is **no kinematic grasp-assist fallback**, by design (physics-only contract; see `activeContext.md`). Articulated diner hands are now available via `--scene stylish_diner --hands`.

---

## 9. Tunables (`hand_gripper.py`)

- **`_GRIP_STEP`**: incremental sensitivity (~0.07).
- **`_open_closed_for_joint`**: blends “closed” slightly inside joint limits (**0.82** scale) to limit PD overshoot.

Finger PD gains in `gear_wbc_stand.py` are softer than arms on hand slices (`kp` ~45, `kd` ~0.35 on hand `qpos` indices via gathered gains).

## 13. Per-arm-joint PD profile (Phase 1 friction-grasp tuning, May 2026)

`gear_wbc_pd.build_arm_pd_per_joint` builds the `kp_by_slice` / `kd_by_slice` arrays the runtime PD consumes. Layout in `arm_target_q` order:

- No hands (`n_u = 14`): `[LEFT_ARM(7), RIGHT_ARM(7)]` — both arms get the same per-arm-joint vector.
- Hands (`n_u = 28`): `[LEFT_ARM(7), LEFT_HAND(7), RIGHT_ARM(7), RIGHT_HAND(7)]` — hand slots use `HAND_PD_KP_DEFAULT = 45` / `HAND_PD_KD_DEFAULT = 0.35`.

Per-arm-joint order is the IK qpos order (matches `arm_ik.LEFT_ARM_JOINTS`):

| index | joint           | default kp | default kd | grasp kp | grasp kd |
|-------|-----------------|------------|------------|----------|----------|
| 0     | shoulder_pitch  | 45         | 1.2        | 45       | 1.2      |
| 1     | shoulder_roll   | 45         | 1.2        | 45       | 1.2      |
| 2     | shoulder_yaw    | 45         | 1.2        | 45       | 1.2      |
| 3     | elbow           | 45         | 1.2        | 60       | 1.5      |
| 4     | wrist_roll      | 45         | 1.2        | 90       | 2.0      |
| 5     | wrist_pitch     | 45         | 1.2        | 120      | 2.5      |
| 6     | wrist_yaw       | 45         | 1.2        | 60       | 1.6      |

Constants: `ARM_PD_KP_DEFAULT_PER_JOINT`, `ARM_PD_KP_GRASP_PER_JOINT`, `ARM_PD_KD_DEFAULT_PER_JOINT`, `ARM_PD_KD_GRASP_PER_JOINT`, `ARM_TAU_CLIP_NM = 30` (default), `ARM_TAU_CLIP_GRASP_NM = 40` (grasp). Pinned by `tests/test_arm_pd_per_joint_gains.py`.

The `_ARM_KD_WALK_SCALE = 1.45` multiplier applies **only to the arm slice while walking** (`||loco_cmd|| > 0.05`); hand kd is unaffected.

### CLI flags (`play_g1_gear_wbc.py`, mirrored on `spawn_g1_floor.py --stand`)

- `--arm-gain-profile {default, grasp}` — opt in to the table above wholesale. Default mode is byte-for-byte identical to pre-Phase-1 behaviour, so existing pelvis-band / walk regressions don't shift.
- Per-axis kp/kd: `--arm-kp-shoulder`, `--arm-kp-elbow`, `--arm-kp-wrist-roll`, `--arm-kp-wrist-pitch`, `--arm-kp-wrist-yaw`, plus matching `--arm-kd-*`. Each takes a single float and overrides the profile-resolved value for that joint. `--arm-kp-shoulder` writes pitch/roll/yaw together.
- `--arm-tau-clip-nm` — per-actuator torque clip; default 30, grasp 40.
- `--diner-block-density KG_PER_M3` and `--diner-block-friction "TAN SLIP SPIN"` — patch `target_block` mass/inertia (recomputed from geom volume × density) and `geom_friction` at load time. No XML edit. Diner default: `density=15` (=> 0.21 kg cube), friction `"3.0 0.1 0.005"`.
- `--print-arm-tau` — with `--teleop`: append post-clip wrist_pitch L/R peak + `clip_frac` to the periodic teleop status print so you can tell when the wrist is saturating mid-grasp.

Implementation: `play_g1_gear_wbc._resolve_arm_pd_overrides` (CLI → `(kp_per_joint, kd_per_joint, tau_clip_nm)` triple, all `None` when nothing is overridden); flags forwarded into `G1GearWBCEnv` → `GearWBCRuntime`. Status printer takes an optional `arm_tau_provider: Callable[[], str | None]` closure (`_arm_tau_status_snapshot`) so the listener thread can append a one-line summary without touching the runtime.

### Tuning protocol (current operator workflow)

1. Baseline: `play_g1_gear_wbc.py --teleop --keyboard-grasp-primitives` (no profile flag). Approach the diner table, palms close, `c` to pinch. Watch wrists back-drive — confirms baseline kp=45 / clip=30 cannot sustain contact.
2. Step up: same with `--arm-gain-profile grasp --print-arm-tau`. Pinch and lift. If `clip_frac < 0.2` and wrist pitch L/R peaks are well under 40 Nm, you're solidly in PD-not-saturated regime.
3. If still slipping or `clip_frac > 0.3`: `--arm-kp-wrist-pitch 160 --arm-tau-clip-nm 60`, then if needed `--arm-kp-wrist-pitch 200`.
4. If wrists hold but block still escapes: drop `--diner-block-density 8` (mass 0.11 kg). If that lifts, the PD profile is right and the 0.21 kg default is just heavy for these gains; trade up density gradually.
5. If contact looks oscillatory: raise `--arm-kd-wrist-pitch 3.5` (or 5.0). Don't chase oscillation by raising kp without kd — same trick as a normal PD tuning loop.

---

## 10. Gym / CLI

```bash
python spawn_g1_floor.py --stand --teleop --hands
python play_g1_gear_wbc.py --teleop --hands
```

```python
gym.make("G1GearWBC-v0", use_hands=True, enable_teleop=True)
# After register_g1_gear_wbc_env()
```

---

## 11. Upstream note

**`run_mujoco_gear_wbc.py`** (GR00T-WholeBodyControl) still assumes the **no-hand** MJCF for a contiguous `n_joints` observation. The **hands variant** is intended for **`g1_minimal_sim`** paths that implement §5–§6.

---

## 12. Future / GR00T

- Wire **`left_hand` / `right_hand`** from `unitree_g1` into the same `control_dict` / `arm_target_q` hand slices (or a small adapter).
- Optional pytest: hand joint motion vs `gripper_*` in headless with hands MJCF.
- **Slip-margin telemetry (next telemetry priority).** Extend `_accumulate_grip_contact_peaks` with tangential force from `mj_contactForce` so status emits `|Ft| / (μ|Fn|)` per side — the actual slip indicator. See **`INDEX.md`** (MuJoCo grasp / contact) + `activeContext.md` "Future cleanups" #1.
- **Closed-loop grip-force control (downstream).** Today the controller squeezes by stiffening `wrist_pitch` PD until contact holds; future iteration is a normal-force-setpoint controller that uses the telemetry above as feedback.

---

## 14. Diner + articulated hands (Phase 2, May 2026)

**What changed.** The diner scene used to be **welded fingers only** and `h`/`y` triggered a kinematic grasp-assist (a temporary hack that pinned `target_block` to the palm midpoint via `data.qpos`). The user rejected that as cheating because it doesn't generalize to real tools / bussing. Phase 2:

1. Built **`scenes/stylish_diner_1/g1_gear_wbc_stylish_diner_hands.{xml,yaml}`** — same merge pattern as the welded-finger merged diner XML, but with the articulated `g1_gear_wbc_hands.xml` body (compiler `meshdir` rewritten for the new file's location). `target_block`: half-size `0.14`, density `5`, friction `"3.0 0.1 0.005"` — matches the welded-finger merged variant (one source of truth for sizing/mass/friction across `stylish_diner.xml`, `g1_gear_wbc_stylish_diner.xml`, `g1_gear_wbc_stylish_diner_hands.xml`).
2. Lifted the gate in **`gear_wbc_config.resolve_gear_wbc_config`**: `scene="stylish_diner"` + `use_hands=True` now returns the new pair instead of raising. The `play_g1_gear_wbc.py` `--hands ignored on stylish_diner` warning + downgrade is gone.
3. **Deleted** the kinematic grasp-assist:
    - `GearWBCRuntime.__init__`: removed `_grasp_assist_active` / `_grasp_assist_offset_world` / `_grasp_assist_quat_wxyz` / `_grasp_assist_body_id` / `_grasp_assist_qpos_adr` / `_grasp_assist_qvel_adr` initialization.
    - `GearWBCRuntime`: removed `_grasp_assist_available`, `_grasp_assist_palm_midpoint`, `_apply_grasp_assist`, the `_on_teleop_reset` reset of `_grasp_assist_active`, and both `_apply_grasp_assist` calls in `step_physics`.
    - `gear_wbc_stand.run_gear_wbc` + `play_g1_gear_wbc`: dropped the "Grasp assist: h attach target_block to palm midpoint; y release." banner branch.
    - `hand_gripper.apply_gripper_teleop_key`: dropped the `_grasp_assist_request` writes; `h`/`y` now no-op + warn once on no-hand scenes (`_HY_NO_HANDS_WARNED` guard).
    - Tests: deleted `tests/test_diner_grasp_assist.py`; flipped `tests/test_hand_gripper_keys.py::test_hy_noop_without_articulated_grippers` to assert **no** cheat key lands in `control_dict`. `tests/test_stylish_diner_hands_mjcf_integration.py` (formerly `test_diner_hands_smoke.py`) covers config resolution, finger-joint presence, target_block structural invariants, no `_grasp_assist_*` regression, and 20 physics steps — MJCF/integration scope only.

**Operator command (Phase 1 PD + Phase 2 fingers + Phase 3 solver options + grip-force telemetry, real physics):**

```bash
play_g1_gear_wbc.py --teleop --keyboard-grasp-primitives --hands \
  --arm-gain-profile grasp --print-arm-tau --print-grip-force
```

`h` is now a real fist close (uniform `g=0` across both hands); use it to curl thumbs out of the way before pinching with `c`. If a per-finger override turns out to be needed (e.g. thumb still wrong-poses), add a `_FIST_OVERRIDES` map in `hand_gripper.py` keyed off `LEFT_/RIGHT_HAND_JOINT_NAMES`. As of Phase 2 landing, the uniform close was sufficient to start tuning friction / lift behavior. Per-finger curl primitives are listed as a near-term cleanup in `activeContext.md`.

## 15. Contact / solver options for grasping (Phase 3, May 2026)

Carry-while-walking smoke kept failing with "10-15 N grip but block slips after 1-2 steps" through every PD / friction / mass tuning combination. Root cause was **MuJoCo contact solver defaults**, not anything in this file or §13: empty `<option>` block in the diner MJCFs meant `impratio=1` + pyramidal friction cone, which under-conditions tangential force inside cone-bound contacts during gait shocks. Fix landed in:

- `scenes/stylish_diner_1/g1_gear_wbc_stylish_diner_hands.xml` (articulated)
- `scenes/stylish_diner_1/g1_gear_wbc_stylish_diner.xml` (welded)

```xml
<option impratio="10" cone="elliptic" noslip_iterations="3"/>
```

Plus per-`target_block` (only): `solref="0.005 1"` `solimp="0.95 0.99 0.001"`. Leg-floor contacts left at MuJoCo defaults so the Gear WBC ONNX leg policy stays in distribution.

**Operator implication.** When PD + friction tuning isn't holding a grasp under perturbation in MuJoCo, **check solver options first** before sweeping more gains. Five-line checklist: **`INDEX.md`** (section **MuJoCo grasp / contact**); telemetry gaps and debounce notes: **`activeContext.md`**.

**Telemetry today:** `--print-grip-force` exposes peak `|Fn|` per side; `|Ft| / (μ|Fn|)` slip-margin is the next telemetry priority and is the highest-leverage manipulation gap (see `activeContext.md` "Future cleanups" #1).
