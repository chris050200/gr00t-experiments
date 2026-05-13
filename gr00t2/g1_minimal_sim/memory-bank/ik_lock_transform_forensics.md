# IK Lock Transform Forensics (keyboard teleop)

Purpose: document the exact orientation/transform chain for `--teleop --lock-ee-orient`, define seam invariants, and standardize forensic logging.

---

## Why this exists

The floor+diner instability can look like "targets are wrong" while the root may be "tracking diverged after targets were formed." This file defines a precise chain so we can localize failures without guessing.

---

## Orientation chain (lock mode)

Symbols:

- `q_r`: reference-body orientation quaternion used for decode (`rt._ee_decode_quat(...)`)
- `Q_lw`, `Q_rw`: stored world anchor quaternions (`_ee_*_quat_world_anchor`)
- `q_le`, `q_re`: torso-frame commanded EE quaternions (`control_dict["ee_*_quat"]`)
- `q_lw_tgt`, `q_rw_tgt`: world target quaternions sent to IK
- `q_lw_meas`, `q_rw_meas`: measured world palm quaternions from FK (`body_palm_pose`)

Lock step (`gear_wbc_arm_ik._lock_ee_quats_to_home_under_teleop`):

- `q_le = conj(q_r) * Q_lw`
- `q_re = conj(q_r) * Q_rw`

Decode to world targets (`arm_ik.ee_world_targets_for_ik` + `ee_frame.ee_pose_world_from_ref`):

- `q_lw_tgt = q_r * q_le`
- `q_rw_tgt = q_r * q_re`

In ideal lock mode:

- `q_lw_tgt == Q_lw`
- `q_rw_tgt == Q_rw`

---

## End-to-end IK chain map (keyboard -> joint targets)

This is the authoritative flow that must be preserved and logged.

1. Keyboard teleop updates torso-frame commands in `control_dict`
   - `ee_left_pos`, `ee_left_quat`, `ee_right_pos`, `ee_right_quat`
   - Source: `gear_wbc_teleop.apply_mujoco_gear_wbc_key(...)`

2. Optional lock rewrite (keyboard lock mode only)
   - `gear_wbc_arm_ik._lock_ee_quats_to_home_under_teleop(...)`
   - Uses current torso decode quaternion `q_r` and world anchors `Q_lw`, `Q_rw`
   - Writes torso-frame `ee_*_quat` so world target orientation remains fixed

3. Torso-frame -> world target compose for IK
   - `arm_ik.ee_world_targets_for_ik(...)`
   - Internally same frame contract as `ee_frame.ee_pose_world_from_ref(...)`
   - Produces world-space targets:
     - `wp_l`, `wq_l`
     - `wp_r`, `wq_r`

4. Pre-step measured world palm state (FK)
   - `arm_ik.body_palm_pose(...)`
   - Produces:
     - `pml`, `qml`
     - `pmr`, `qmr`
   - Used for measured-vs-target error and seam checks

5. Dual-arm IK solve
   - `arm_ik_v2.solve_dual_arm_ik_v2(...)`
   - Inputs:
     - world targets (`wp_*`, `wq_*`)
     - arm joint slice (`arm_qpos_ids`)
     - posture/temporal weighting knobs
   - Output:
     - `q_sol` (14D arm solution: left 7, right 7)
   - **Internal contract (May 2026 fix, pinned by `tests/test_arm_ik_v2_jacobian.py`):** the inner
     `mujoco.mj_jac(model, data, jacp, jacr, point, body_id)` calls MUST pass `point` as the **palm
     WORLD position** (`p_l`/`p_r` returned by `body_palm_pose`), not the body-local `palm_local`
     offset. MuJoCo's API specifies `point` is global; passing body-local makes the position
     Jacobian's lever arm grow with `||x_body||` (i.e. with how far the robot has walked in world)
     and was the root cause of the May 2026 long-walk arm spasms / VR ~180° spin instability.
     Regression test `test_arm_ik_v2_passes_world_palm_point_to_mj_jac` monkeypatches `mj_jac` and
     asserts the actual `point` argument equals `palm_world` (within 5 mm) and is far from
     `palm_local`; if a future edit reverts the bug it fails immediately with a self-explanatory
     diff message.

6. Per-step output limiting (anti-spike)
   - `gear_wbc_arm_ik._limit_arm_ik_step(...)`
   - Uses current arm qpos `q_cur` and `q_sol`
   - Output:
     - `q_out` (final commanded arm joint targets for this step)
     - diagnostics: `ik_arm_delta_raw_l2`, `ik_arm_delta_out_l2`

7. Write commanded targets
   - `control_dict["arm_target_q"]`
   - With hands:
     - `SL_LEFT_ARM <- q_out[:7]`
     - `SL_RIGHT_ARM <- q_out[7:14]`
     - hand slices from `hand_gripper.hand_target_q(...)`
   - Without hands:
     - `arm_target_q[:14] <- q_out`

8. PD and torque application
   - Runtime PD path (`gear_wbc_pd` / `gear_wbc_stand`) reads `arm_target_q`
   - Writes torques to `data.ctrl[...]`
   - `mj_step` advances plant to next measured state

---

## Transform chain map (local/global seams)

For each side (left/right), required frame path is:

- `ee_*` in torso/reference frame
- decode torso basis (`xmat`, `xquat`) through `_ee_decode_quat` / `_ee_ref_rotmat`
- compose world target pose (`ee_pose_world_from_ref` math)
- solve IK in world target space
- compare against world FK measured palm pose

Seams that must remain explicit in logs:

1. Command seam: `control_dict.ee_*` (torso frame)
2. Decode seam: torso pose + decoded ref basis at solve time
3. Compose seam: world target pose from torso-frame command
4. IK seam: raw solver output vs step-limited output
5. Tracking seam: measured FK world palm vs world target

---

## Imperative forensic logger contract (next improvements)

This is not optional for future forensic mode work.

1. Burst logger must capture the whole chain, not only error summaries.
2. Every forensic episode must include enough fields to replay seam-by-seam diagnosis.
3. Coverage must include keyboard command -> transform compose -> IK solve -> joint command -> measured tracking.

Minimum required field groups per forensic row:

- Command inputs: `ee_*`, `loco_cmd`, lock mode flags/anchors
- Torso solve-time basis: `torso_xpos`, `torso_xmat`, `torso_xquat`, decoded ref basis
- World targets: `ik_tgt_world_*` (pos + quat)
- IK solver outputs: `ik_q_sol`, `ik_arm_delta_raw_l2`, `ik_arm_delta_out_l2`, `ik_step_limit_scale`
- Joint command outputs: arm slices written to `arm_target_q`
- Measured outputs: `meas_palm_world_*` (pos + quat)
- Error invariants: seam metrics (`ik_dbg_*`, measured-vs-target position/quat errors)
- Actuation stress: clip stats (`arm_tau_*`)

Definition of done for forensic mode:

- A single triggered episode JSONL window is sufficient to identify the first failing seam without rerunning with new instrumentation.

---

## Seam invariants (what should stay near zero)

For each side (left/right):

1. Lock-anchor consistency
   - `geodesic_deg(Q_w_anchor, q_w_tgt)`
2. Reconstruction consistency
   - `geodesic_deg(q_w_tgt, q_w_recon_from_ee)`
3. Pre-step tracking gap
   - `geodesic_deg(q_w_meas_pre, q_w_tgt)`

Current runtime traces expose these via `ik_dbg_*` fields in `ik` payloads.

Interpretation:

- (1) high -> lock compose bug.
- (1) low, (2) high -> decode/reconstruct mismatch bug.
- (1)(2) low, (3) high -> post-target tracking/feasibility issue.

---

## IK solver demand/saturation signals

Use together with seam invariants:

- `ik_arm_delta_raw_l2` (solver wants this much change)
- `ik_arm_delta_out_l2` (after per-step caps)
- `ik_step_limit_scale = out/raw`

Pattern for "bad attractor":

- `raw` high, `scale` low, and measured-vs-target orientation large for long windows.

---

## Logging policy (tiered)

Do not log all raw values all the time.

Tier A (always-on, cheap):

- seam invariants (`ik_dbg_*`)
- demand/saturation (`ik_arm_delta_*`, `ik_step_limit_scale`)
- high-level errors (`err_meas_minus_ik_tgt_world_*`, quaternion error deg)

Tier B (forensic burst, event-window only):

- raw pose quaternions at each seam
- torso decode basis (`xmat/xquat` at solve time)
- optional IK internals around trigger windows

Reason: Tier A catches onset with low overhead; Tier B gives depth only when needed.

---

## Recommended forensic workflow

1. Reproduce with normal teleop command and `--log`.
2. Run `tests/teleop_regression/extract_ik_episodes.py`.
3. Identify first bad episode and first invariant that departs nominal.
4. Patch one seam only.
5. Re-run same scenario; compare episode count/duration and onset time.

