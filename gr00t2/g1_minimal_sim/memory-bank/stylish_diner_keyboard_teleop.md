# Stylish diner scene + keyboard teleop

Central reference for the **default play scene** (`stylish_diner`), **keyboard locomotion / arm IK**, optional **keyboard grasp primitives**, and **known arm instability** when walking in clutter. Also tracks a **manual operator checklist** (“push a box around the diner”) — not a substitute for automated tests, which only cover MJCF merge / config wiring (see `tests/test_stylish_diner_hands_mjcf_integration.py`).

---

## Files (what lives where)

| Path | Role |
|------|------|
| `g1_minimal_sim/scenes/stylish_diner_1/stylish_diner.xml` | **Static world only**: floor, walls, bar, tables, booths, kitchen shell, **lights**. No robot. Compiler `angle="degree"`, `timestep="0.002"` in this file; merged robot scene uses the Gear WBC yaml `simulation_dt` at runtime. |
| `g1_minimal_sim/scenes/stylish_diner_1/g1_gear_wbc_stylish_diner.xml` | **Merged** G1 + diner (welded fingers): robot + actuators first, then a **second** `worldbody` block with diner geoms (names prefixed `sd_*`), lights, and **`body name="target_block"`** (`freejoint` + `sd_target_block_geom`, half-size **0.14**, density 5, friction `"3.0 0.1 0.005"`, plus `solref="0.005 1"` / `solimp="0.95 0.99 0.001"`) on the central table. `<option impratio="10" cone="elliptic" noslip_iterations="3"/>` for grasp-friendly contact (May 2026 — see **[INDEX.md](INDEX.md)** § MuJoCo grasp / contact). G1 spawns at `pelvis pos="-2.8 0.4 0.793"`. Source layout matches `stylish_diner.xml` but is **inlined**, not `<include>`-d. Finger joints/motors are commented out; `GearWBCRuntime.has_hands` is **false**. |
| `g1_minimal_sim/scenes/stylish_diner_1/g1_gear_wbc_stylish_diner_hands.xml` | **Merged** G1 + diner with **articulated 14-DoF hands** (May 2026 Phase 2). Built from `g1_gear_wbc_hands.xml` + the same diner suffix (`sd_*` geoms + `target_block`). `target_block`: half-size **0.11** (intentionally smaller than the welded variant, ~53 g cube), density 5, friction `"3.0 0.1 0.005"`, plus `solref="0.005 1"` / `solimp="0.95 0.99 0.001"`. Same `<option impratio="10" cone="elliptic" noslip_iterations="3"/>` as the welded twin. Selected by `--scene stylish_diner --hands` / `use_hands=True`. |
| `g1_minimal_sim/scenes/stylish_diner_1/g1_gear_wbc_stylish_diner.yaml` | Scene-local yaml for the welded-finger variant (paths relative to `Isaac-GR00T/.../robots/g1/policy/` etc.). Selected by `resolve_gear_wbc_config(scene="stylish_diner", use_hands=False)`. |
| `g1_minimal_sim/scenes/stylish_diner_1/g1_gear_wbc_stylish_diner_hands.yaml` | Hands-enabled twin of the above; `xml_path: g1_gear_wbc_stylish_diner_hands.xml`. Selected by `resolve_gear_wbc_config(scene="stylish_diner", use_hands=True)`. |

**Config gate (May 2026 Phase 2 — lifted):** `gear_wbc_config.resolve_gear_wbc_config(scene="stylish_diner", use_hands=True)` now returns the new merged hands+diner pair instead of raising. The previous `play_g1_gear_wbc.py` `--hands ignored on stylish_diner` downgrade warning is gone.

**`target_block` sizing — drift-tolerant (May 2026):** half-size has been iterated between 0.14 and 0.11 during friction-grasp work. Current state: **welded variant 0.14** (~110 g), **hands variant 0.11** (~53 g). Density 5 / friction `"3.0 0.1 0.005"` is consistent across both — and is **explicitly a starting point** for prototyping the grasp pipeline, not realistic manipulation parameters. `tests/test_stylish_diner_hands_mjcf_integration.py` reads sizing from the loaded model and asserts structural invariants (cube, density 1–2000 kg/m³) only — it does **not** prove grasp quality. Realism phase-up plan: **[INDEX.md](INDEX.md)** + **`activeContext.md`**.

**Solver options for grasping (May 2026):** Both diner MJCFs declare `<option impratio="10" cone="elliptic" noslip_iterations="3"/>`; `target_block` (only) tightens `solref="0.005 1"` / `solimp="0.95 0.99 0.001"`. This was the actual fix for "10-15 N grip but slips after 1-2 steps" — leg-floor contacts intact for ONNX leg policy. Short checklist: **[INDEX.md](INDEX.md)** (same §); rationale for these three knobs is **Solver options for grasping** above.

---

## CLI defaults (`play_g1_gear_wbc.py`)

- **`--scene`** default is **`stylish_diner`** (not empty floor). Startup prints `scene` / `env_id` for sanity.
- **`--teleop`**: viewer + terminal focus; **not** combinable with `--headless`.
- **`--keyboard-grasp-primitives`**: requires **`--teleop`**; default **off**. See §Grasp primitives below.

---

## Keyboard teleop (legs + arms)

**Locomotion (hold-to-move, Apr 2026):** keys are **held**, not tap-to-nudge. **`sync_keyboard_loco_hold`** runs after VR stick mapping each step so keyboard and VR do not fight.

- **W / S** — forward / back along commanded heading.
- **A / D** — strafe left / right.
- **Q / E** — yaw left / right.
- Release **all** of the above → `loco_cmd` returns to stand init (`cmd_init`).

**Other teleop:** height / rpy / freq per existing keymap; **`z`** — full reset (loco + arm + EE homes). **Arm IK** (torso-frame `ee_*`): left `i/k` `j/l` `u/p`; right `r/t` `f/g` `v/b`; shared orient `,` `.` `;` `'` `[` `]`. Implementation: `gear_wbc_teleop.py`, `gear_wbc_stand.step_physics`, `gear_wbc_arm_ik.py`, `arm_ik.py` / `arm_ik_v2.py`, `ee_frame.py`.

**Debounce / OS-keyrepeat semantics (deliberate, May 2026):** three teleop key families intentionally behave differently — document this when onboarding a new operator.

| Family | Keys | Semantics | OS-keyrepeat behaviour |
|---|---|---|---|
| Loco | `w/s a/d q/e` | **Hold-to-move**; release → stand | Pressed-keys set drives `loco_cmd` each step |
| Grasp primitives | `x/c Home/End PgUp/PgDn` | **Tap = one nudge**; 150 ms debounce per token | Mash → ~6.6 nudges/s (fine; intentional throttle) |
| IK arm + posture | `i/k j/l u/p`, `r/t f/g v/b`, `,./;'[]`, `1-8 m n` | **Continuous-while-held** (no debounce); fine motion via OS-keyrepeat | ~30 nudges/s × 12 mm ≈ 0.36 m/s drift while held |
| Gripper | `9/0 =/-`, `h/y` | `9/0=/-` per-tap; `h/y` are absolute set | `9/0=/-` step 0.018 → ~1.85 s full traversal at 30 Hz; `h/y` idempotent |

If this surprises you, it is by design (held = continuous; tap = quantized) but **not** auto-discoverable from the help text. Debounce / telemetry gaps: **`activeContext.md`** (Future cleanups).

**Diner `h`/`y` (May 2026 Phase 2 — articulated):** with `--hands` on the diner, `h` hard-closes both grippers into a fist and `y` opens them — physics-only, no kinematic glue. The previous welded-finger kinematic grasp-assist (where `h` pinned `target_block` to the palm midpoint via `data.qpos`) was **deleted**: `_grasp_assist_*` is gone from `gear_wbc_stand`, `hand_gripper`, `play_g1_gear_wbc`, and `tests/test_diner_grasp_assist.py`. On no-hand scenes (welded MJCF, `--hands` not set), `h`/`y` print `[hand_gripper] h/y ignored: scene has no articulated grippers` once and no-op (no cheat fallback by design).

---

## Optional keyboard grasp primitives (motion primitive)

**Module:** `g1_minimal_sim/keyboard_grasp_primitives.py`  
**Flag:** `--keyboard-grasp-primitives` (with `--teleop` only).

Symmetric **torso-frame** palm **position** nudges only (does **not** change `ee_*_quat` or grippers):

| Key | Effect |
|-----|--------|
| **x** | Widen: palms move apart along torso **+Y** (~0.04 m per side per press). |
| **c** | Narrow: opposite of **x**. |
| **Home** / **End** | Both palms **+X** / **−X** in torso frame (~0.04 m per press) for forward/back reaching. |
| **Page Up** / **Page Down** | Both palms **+Z** / **−Z** in torso frame (~0.025 m per press). |

Clamped to **±0.35 m** from `_ee_*_pos_home` per implementation. Intended for coarse “box grasp spacing” without dirtying core IK math.

---

## Friction-grasp PD tuning (Phase 1) + articulated fingers on diner (Phase 2, May 2026)

Real physics, no kinematic glue. `--arm-gain-profile grasp` opts into the stiffer per-arm-joint PD; `--hands` opts into articulated 14-DoF fingers on the diner (Phase 2 landed):

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  play_g1_gear_wbc.py --teleop --keyboard-grasp-primitives --hands \
  --arm-gain-profile grasp --print-arm-tau
```

The grasp profile stiffens elbow + wrist trio so wrists can sustain palm-pinch normal force without back-driving (table + flag list: `hands_gripper_teleop.md` §13). The most contact-loaded DoF is **`wrist_pitch`** (kp default 45 → grasp 120, clip 30 → 40 Nm). `--print-arm-tau` adds a post-clip wrist_pitch L/R peak + `clip_frac` line to the periodic teleop status print so you can tell when you're saturating.

`target_block` defaults (current): hands variant **half-size 0.11** (~53 g), welded variant 0.14 (~110 g); both density 5, friction `"3.0 0.1 0.005"`. Block tuning at load time, no XML edit: `--diner-block-density KG_PER_M3`, `--diner-block-friction "TAN SLIP SPIN"`. Use these to falsify "PD is too soft" vs "block is too heavy / slick" vs "solver is mis-conditioning friction" before chasing more gains. **The current defaults are starting points, not realistic** — see **[INDEX.md](INDEX.md)** + **`activeContext.md`**.

Phase 2 specifics (`hands_gripper_teleop.md` §14 has the full deletion + merge writeup):

- New merged MJCF/yaml `g1_gear_wbc_stylish_diner_hands.{xml,yaml}` selected via `--hands` on diner. The kinematic grasp-assist path was **deleted** end-to-end (`_grasp_assist_*` removed from `gear_wbc_stand`, `hand_gripper`, `play_g1_gear_wbc`; `tests/test_diner_grasp_assist.py` removed; `tests/test_hand_gripper_keys.py` flipped to assert **no cheat key lands in control_dict**).
- `h` is a real fist close (uniform `g=0` across all 14 finger DoFs) so the thumb stops obstructing the front face of the cube; combine with `c` palm pinch for the actual grasp.
- Successor (next iteration): **grip-force detection** so the controller squeezes with the right normal force instead of relying on operator-tuned wrist PD.

## Long-walk arm instability (RESOLVED May 2026)

The "arms go unstable after walking a few feet" symptom (in diner *and* on bare floor) was diagnosed as a wrong-Jacobian bug in `arm_ik_v2.solve_dual_arm_ik_v2` — `mujoco.mj_jac` was being called with body-frame palm offset instead of world-frame palm position, so descent direction error scaled with `||x_body||`. Fix landed (one line per side; see `progress.md` history appendix and `ik_lock_transform_forensics.md` for the chain map and seam invariants). Pinned by `tests/test_arm_ik_v2_jacobian.py` + extended `tests/test_arm_ik_v2_smoke.py`. Live VR re-validation confirmed the previous wrist / ~180° spin issue is resolved too.

## Carry-while-walking slip (RESOLVED May 2026)

The "10-15 N grip force seems right but block slips out after 1-2 walking steps" symptom was diagnosed as **MuJoCo solver mis-conditioning** of friction during gait shocks: empty `<option>` block in the diner MJCFs meant `impratio=1` + pyramidal cone, which under-delivers tangential force inside cone-bound contacts under perturbation. Fix: per-MJCF `<option impratio="10" cone="elliptic" noslip_iterations="3"/>` plus per-block `solref="0.005 1"` / `solimp="0.95 0.99 0.001"` (leg-floor contacts intact). Validated headless 4000-step stand and live operator carry. **The fix was solver options, not PD or friction or mass** — those layers were already correct. Canonical summary: **[INDEX.md](INDEX.md)** + **`activeContext.md`**.

---

## Manual checklist: “move the box around the diner”

**Intent:** Operator validates **keyboard teleop** (walk + arm IK + optional grasp primitives) in **clutter** without VR or policy. Robot starts in front of the central table; operator approaches the free **`target_block`** on that table (see `stylish_diner.xml` — `body name="target_block"` with `freejoint` and `target_block_geom`), then uses **rigid palms**, optional **`--keyboard-grasp-primitives`**, and on **`--hands`** scenes physics-only **`h`** (fist close) / **`y`** (open) — **not** the old kinematic grasp-assist (removed) — to **push, carry, or steer** the block toward the blue prep table / kitchen area.

**Automated tests (scope):** `tests/test_stylish_diner_hands_mjcf_integration.py` only checks merged MJCF + `GearWBCRuntime` load, finger joint names, no `_grasp_assist_*` regression, and coarse `target_block` structure after a few steps. **`tests/test_g1_gear_wbc_env_headless.py`** exercises stand-in-diner via the Gymnasium env. Neither replaces this teleop checklist for manipulation feel or slip behavior.

**Success criteria (informal):**

- Robot remains standing; no runaway IK requiring `z` reset every few seconds.
- Block moves measurably under **contact + pushing** (stable grasp optional depending on scene and fingers).

**Implementation notes (when picked up):**

- No new deps; may add **`--max-viewer-steps`** or a tiny scripted “episode end” later.
- If arm bug (above) triggers during this task, **fix or mitigate instability first** or narrow test to **floor scene** with a duplicated free box.

---

## Quick commands

```bash
cd ~/Documents/gr00t_experiments/gr00t2/g1_minimal_sim

# Default diner + keyboard teleop (hold W/S A/D Q/E for loco)
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --teleop

# Same + optional symmetric palm spacing / height nudges
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --teleop --keyboard-grasp-primitives

# Empty floor (isolates scene contacts vs locomotion)
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --teleop --scene floor
```

---

## Cross-links

- Roadmap / phases: `progress.md`  
- Current priorities: `activeContext.md`  
- All flags and venv: `techContext.md`  
- Hands (incl. diner+hands Phase 2): `hands_gripper_teleop.md` (§14)
