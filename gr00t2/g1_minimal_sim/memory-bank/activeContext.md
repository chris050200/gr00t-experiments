# Active context — g1_minimal_sim

> **🔥 ACTIVE BUG (May 2026) — GR00T inference loop wiring.** Tier-A apple PnP works; Tier-B **`--scene table_pnp_apple`** still misses qualitative §1, but **Phase 4b** MJCF lighting **shrunk** ego mean-RGB gap vs Tier-A (order **~+11 → ~+5** per channel on representative paired dumps); first-chunk **`action.navigate_command`** dump diff vs Tier-A is now **OK** (~0.043); **`scripts/hybrid_policy_get_action.py`** shows **ego pixels** still strongly causal at step 0. Operator: **slightly better first seconds**, then **regression** (time-varying). Full audit + next experiments: **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md)** §9.5.b / §10. **One-pass obs checklist:** **[`inference_obs_one_pass.md`](inference_obs_one_pass.md)**. Read ACTIVE before touching `scripts/run_gr00t_inference.py`, `GearWBCRuntime`, `_apply_action_step`, `Gr00tObservationBuilder`, or `gr00t_wbc/control/`. **Delete ACTIVE** when §1 exit is met.

---

**Current top priority:** **GR00T inference parity** — lock the full chain **teleop → dataset → training → inference** so observations and decoded actions match checkpoint semantics. **Active investigation:** **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md)** (banner above). **Canonical pipeline map:** **[`gr00t_compatible_data_collection.md`](gr00t_compatible_data_collection.md) §6.4** (mermaid + tables). **Minimal-sim inference:** `scripts/run_gr00t_inference.py` (`--scene` / `--prompt`; `run_gr00t_stylish_diner_inference.py` is the same module); **server:** `Isaac-GR00T/gr00t/eval/run_gr00t_server.py` with **`--use-sim-policy-wrapper`** and **`embodiment_tag`** matching the checkpoint (e.g. **`UNITREE_G1`**). **PnP wiring check:** compare **same PnP checkpoint**, **A** = Isaac-GR00T **`rollout_policy.py`** + `LMPnPAppleToPlateDC_G1_gear_wbc`, **B** = **`run_gr00t_inference.py --scene table_pnp_apple`**. Interpretation table: **`techContext.md`** § “Canonical PnP A/B”. **Diff two dumps:** `scripts/compare_policy_ab_dumps.py`; **vision causality:** `scripts/hybrid_policy_get_action.py` (see **`inference_obs_one_pass.md`**). **OG-only eval** doc: § “OG apple-to-plate reference eval (Tier A)”. **Still active alongside:** carry-while-walking / contact stack; **GR00T data wiring** P0–P2 as in **[`gr00t_compatible_data_collection.md`](gr00t_compatible_data_collection.md)** §7; P3 LeRobot export / P4 eval wrapper. The "10-15 N grip but slips after 1-2 steps" failure mode was *not* PD or friction — it was the empty `<option>` block in the diner MJCFs (default `impratio=1`, pyramidal cone) under-conditioning friction during walk shocks. Quick checklist: **[`INDEX.md`](INDEX.md)** (section **MuJoCo grasp / contact**). Operator command:

```
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  play_g1_gear_wbc.py --teleop --keyboard-grasp-primitives --hands \
  --arm-gain-profile grasp --print-arm-tau --print-grip-force
```

`--print-grip-force` adds peak `|Fn|` on `target_block` (L / R / other / total + `max n_contacts/step`) to the periodic status print so squeeze-vs-slip can be correlated with `wrist_pitch` torque. Approach with held W; reach palms with `i/k j/l u/p` / `r/t f/g v/b`; `c` to pinch palms; `h` to fist-close both. Block sizing is currently **half-size 0.11 m, density 5, friction 3.0** in the hands MJCF (~53 g cube) — explicitly a **starting point** for prototyping the pipeline, not realistic manipulation. Phase up with `--diner-block-density 50/100/200/500` to stress-test once telemetry + control are mature.

The earlier kinematic grasp-assist path is **deleted** (`_grasp_assist_*` removed from `gear_wbc_stand`, `hand_gripper`, `play_g1_gear_wbc`, plus `tests/test_diner_grasp_assist.py`); on no-hand scenes `h`/`y` print a one-time warning and no-op (physics-only contract).

---

## Finetuned checkpoint bring-up — see ACTIVE bug file

The end-to-end bring-up story for `~/models/GR00T-N1.6-G1-LiftRuns-001-015` (blue-cube lift, local) and the CloudWalk PnP A/B (apple/plate, public checkpoint) — including all eight closed wiring/perceptual/state-init bugs, the Phase 1+2 forensic verdict ruling out action-mode and joint-kinematics regressions, the Phase 3 numerical A/B dump tooling, and the Phase 4a/4a-bis leg-init reset-path mirror — lives in **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md) §8 + §10**. The qualitative apple-to-plate test still fails despite `state.*` byte-equal at step 0; ranked next experiments are in §10 of that file (top of list: **action-trajectory replay**, ~1 hr, cleaves the remaining hypothesis space). Delete that file when its §1 exit criterion is met.

---

## GR00T eval strategy (project policy)

Established May 2026 so resets do not default back to “Tier-B must match Tier-A behavior” as the main success metric.

- **Cross-sim PnP parity (minimal MuJoCo vs Isaac / RoboCasa) is intentionally deprioritized** as an end goal. Different sims and rollouts make full behavioral match a weak, expensive signal compared to targeted diagnostics.
- **Tier-A vs Tier-B stays useful as forensics:** policy A/B dumps, hybrid step-0 probes, cadence and decode checks cleave **inference loop + obs + wiring** from **policy and training distribution**. Treat qualitative apple-to-plate on Tier-B as one diagnostic thread, not proof the whole stack is wrong if it diverges from Tier-A.
- **The parity-style experiments worth doing** freeze one side of the boundary: **replay oracle trajectories or action chunks**, or pipe oracle observations through the subject client, so only one of {sim dynamics, logged distribution} changes at a time. Ranked experiments: **`ACTIVE_gr00t_inference_wiring.md`** §10.
- **Working hypothesis:** persistent live drift after matched step-0 `state.*` is largely **dataset / domain** (vision, temporal context, dynamics), not leftover joint-index bugs—use replay cleaves plus a **locally owned** finetune smoke test (scripted headless episodes, e.g. `scripts/collect_dummy_gr00t_batch.py`, then export/train per **`gr00t_compatible_data_collection.md`** §6.4) before scaling hundreds of human teleop trajectories.

---

## Pre-feature cleanup posture (current)

The team agreed on a Phase 0 hygiene pass *before* adding new features. Already landed: empty `g1_diner_lab/` / `legacy/` / `ik/` directories deleted; `MUJOCO_LOG.TXT` removed; memory bank consolidated under `g1_minimal_sim/memory-bank/`; `progress.md` and this file compressed; legacy VR docs merged into `vr_legacy_history.md`. Larger refactors (entry-point dedup, config dataclasses, `CommandSource` abstraction, IK package reshape) are explicitly *deferred* to dedicated follow-up phases — see "Future cleanups" below.

## User-validated baseline commands

Keep these working through any refactor — they are the operator-validated paths.

- **Best current teleop:** `play_g1_gear_wbc.py --vr-teleop --vr-ik on --vr-ik-anchor torso --vr-head-orient yaw_only --debug-vr-stream --hands`
- **Quick VR debug overlay:** `play_g1_gear_wbc.py --vr-teleop --debug-vr-stream`
- **Keyboard teleop (hold W/S A/D Q/E):** `play_g1_gear_wbc.py --teleop`
- **Diner friction-grasp smoke (current, articulated fingers):** `play_g1_gear_wbc.py --teleop --keyboard-grasp-primitives --hands --arm-gain-profile grasp --print-arm-tau` (see `stylish_diner_keyboard_teleop.md`). Diner box-push without articulated fingers (welded MJCF) is the same minus `--hands` and `--arm-gain-profile`.

Defaults already tuned for the user setup: `--vr-receiver-yaw-deg=-90`, `--debug-vr-stream-torso-z=full`, `--debug-vr-stream-calib-orient=yaw_only`, `--vr-head-orient=yaw_only`.

## Active design decisions

- **Lock orientation contract.** Keep one rigid compose chain shared by debug overlays and IK (`vr_stream_torso_compose`); apply heading correction as a single rigid transform. No partial per-device or per-axis hacks. See `systemPatterns.md` "VR compose contract" and `ik_lock_transform_forensics.md`.
- **Keyboard EE keys live in `arm_ik.apply_ik_arm_teleop_key`.** Treat any swap-in teleop source (VR / motion primitive / policy) as a peer that mutates the same `control_dict` keys; do not fork a parallel control path.
- **`--vr-ik {off,on}`** is canonical; legacy `pos` / `pose` map to `on` in `vr_stream_ik_mode.normalize_vr_ik_mode`. CLI help no longer advertises the legacy aliases.
- **Stylish diner now supports `--hands`** (Phase 2, May 2026). `resolve_gear_wbc_config(scene="stylish_diner", use_hands=True)` returns `g1_gear_wbc_stylish_diner_hands.{xml,yaml}` (merged: articulated 14-DoF hands + diner worldbody + `target_block`). The kinematic grasp-assist path was **deleted from `gear_wbc_stand`, `hand_gripper`, `play_g1_gear_wbc`** and from tests; `h` / `y` are physics-only fist-close / open on hands-enabled scenes and a one-time warning + no-op without articulated grippers. **Physics-only contract.** `--scene table_pnp --hands` is still available for the apple-on-plate environment.
- **Phase 1 friction-grasp PD knobs (May 2026, opt-in).** `--arm-gain-profile {default, grasp}` selects per-arm-joint kp/kd; default preserves historical uniform `kp=45, kd=1.2, clip=30 Nm`. Grasp profile stiffens elbow + wrists (`kp=[45,45,45,60,90,120,60], kd=[1.2,1.2,1.2,1.5,2.0,2.5,1.6], clip=40 Nm`). Per-joint overrides: `--arm-kp-wrist-pitch`, `--arm-kp-wrist-roll`, `--arm-kp-wrist-yaw`, `--arm-kp-elbow`, `--arm-kp-shoulder` (and matching `--arm-kd-*`), plus `--arm-tau-clip-nm`. Block tuning (no XML edit): `--diner-block-density KG_PER_M3` and `--diner-block-friction "TAN SLIP SPIN"`. `--print-arm-tau` appends post-clip wrist_pitch L/R peaks + clip fraction to the periodic teleop status print. Builder is `gear_wbc_pd.build_arm_pd_per_joint`; runtime attrs `rt.arm_pd_kp_per_joint`, `rt.arm_pd_kd_per_joint`, `rt.arm_tau_clip_nm`. See `hands_gripper_teleop.md` §13 and `stylish_diner_keyboard_teleop.md` for the tuning protocol.
- **Solver options for grasping (May 2026).** Both diner MJCFs (`g1_gear_wbc_stylish_diner_hands.xml`, `g1_gear_wbc_stylish_diner.xml`) declare `<option impratio="10" cone="elliptic" noslip_iterations="3"/>`; `target_block` only tightens `solref="0.005 1"` / `solimp="0.95 0.99 0.001"` (leg-floor contacts intact for ONNX leg policy). This — not PD or friction — was the root cause of "carry slips after 1-2 steps". `target_block` sizing is **drift-tolerant** in tests: `tests/test_stylish_diner_hands_mjcf_integration.py` reads from the model and asserts structural invariants (cube, plausible density), not hard-coded numbers — integration only, not manipulation proof. Same story as **INDEX § MuJoCo grasp / contact**; telemetry / debounce gaps stay in this file.
- **Grip-force telemetry (May 2026, opt-in).** `--print-grip-force` (with `--teleop --hands`) appends peak `|Fn|` on `target_block` per side to the periodic status print. Implementation: `gear_wbc_stand._accumulate_grip_contact_peaks` after each `mj_step` walks `data.contact[*]` filtered to the block geom, calls `mj_contactForce`, classifies by side via `_geom_side_for_grip_readout` (parent-body name walk: any ancestor matching `left_wrist_yaw_link` / `*left_hand*` → L; right symmetric). Status snapshot resets peak-hold buffers each print. **Known gap:** only normal force is exposed; tangential / slip-margin (`|Ft| / (μ|Fn|)`) is the actual slip indicator and is the next telemetry priority — see **Future cleanups** #1 below.

## Future cleanups (deferred, in priority order)

These came out of the May 2026 critique pass. Listed here so they don't get rediscovered each reset.

1. **Manipulation telemetry (slip margin + per-finger Fn).** Highest-leverage manipulation gap. `_accumulate_grip_contact_peaks` exposes `|Fn|` only; slip-margin `|Ft| / (μ|Fn|)` is the direct slip indicator and is not surfaced. Per-finger lump (thumb / index / middle / palm) instead of L / R also unblocks geometry-dependent grasp diagnostics. Land this before tuning realistic mass / friction.
2. **Manipulation control structure (per-finger curls + grasp macro).** Today `h` is a uniform fist close (g=0 across all 14 finger DoFs); thumb obstructs cube faces, operator works around with `c` palm-narrow first. Add `_FIST_OVERRIDES` in `hand_gripper.py` so default thumb pose opposes the palm; consider a `g`-key "close around bounding box" macro.
3. **Teleop debounce inconsistency (document or unify).** Grasp primitives (`x/c Home/End PgUp/PgDn`) debounce at 150 ms; IK arm keys (`i/k j/l u/p`, `r/t f/g v/b`, `,./;'[]`) and posture keys (`1-8 m n`) do **not**. Continuous-while-held drift on IK keys is intentional but undocumented. Recommendation: keep behaviour, document it explicitly in `stylish_diner_keyboard_teleop.md` and `hands_gripper_teleop.md` so operators are not surprised.
4. **Carry-while-walking regression test.** Currently no test asserts `target_block.xpos[2]` stays above table top after N walking steps; "slips after N steps" can re-regress silently. Add `tests/test_diner_carry_walk_smoke.py` once telemetry (slip margin) is wired so the test can also assert `max |Ft| / (μ|Fn|) < 0.95`.
5. **`CommandSource` abstraction.** Single protocol that keyboard, VR, scripted motion primitives, and (later) GR00T policy all implement; ordered list inside `GearWBCRuntime.step_physics`. Unblocks "swap teleop with one flag" and "motion primitives compose with teleop". Single highest-leverage refactor for upcoming work.
6. **Config dataclasses.** Group the 30+ kwargs on `GearWBCRuntime.__init__` / `run_gear_wbc` / `G1GearWBCEnv.__init__` into `VRConfig` / `IKConfig` / `TeleopConfig` / `TraceConfig`. Argparse builds them. Required first if `CommandSource` is going to land cleanly.
7. **Entry-point consolidation.** `play_g1_gear_wbc.py` and `spawn_g1_floor.py` duplicate ~70% of CLI flags and validation. Either delete the `--stand` path on `spawn_g1_floor.py` (keep it as the passive-physics demo) or share `_argparse_common.py`.
8. **IK package reshape.** Pull keyboard EE keymap out of `arm_ik.py` into a `teleop/keyboard.py`; group solver / specs / FK / posture / step into an `arm/` package. Strict invariant going forward: IK code never imports keyboard or VR code.
9. **`legacy_vr_code/` rename.** It is mislabeled — `--udp-teleop` v1 still imports from it at runtime. Either move `udp_teleop_receiver.py` into `vr_teleop/` (alongside `openvr_stream.py`) or rename `legacy_vr_code/` → `vr_teleop/legacy_v1/`. Decide together with whether `--udp-teleop` is still wanted at all.
10. **Mirror solver fix to vendored manipulation MJCFs.** `g1_gear_wbc_table.xml` / `g1_gear_wbc_hands_table.xml` (apple/plate scene) still uses MuJoCo defaults. Mirror the option block + per-object `solref`/`solimp` once table_pnp manipulation becomes active. Vendored-asset write requires care — document in `techContext.md` as a re-apply step if upstream is refreshed.

## Stable baselines / contracts

- **Joint counts:** `n_joints = model.nu` for arm slices (do not use `nq − 7`).
- **Arm `qpos` indexing:** `arm_qpos_ids = concat(left_ik.qpos_ids, right_ik.qpos_ids)`. Never `np.arange` over the full 28 with hands.
- **Arm PD via `_arm_ctrl_to_qslice`** because right-hand actuator order ≠ qpos order.
- **Policy ONNX input is always 29 joints** (`POLICY_JOINT_NAMES` gather), even with hands MJCF (43 joints).
- **Optional ONNX import** is lazy in `make_onnx_runner`; lightweight imports (e.g. `default_g1_resources_dir`) work without `onnxruntime`.
- **Trace contract:** `play_g1_gear_wbc.py --log PATH` → `vr_compare` rows when VR receiver active, otherwise `teleop_ik` rows when an IK source is active. `--forensic-burst` writes Tier-B windows in the same JSONL.
