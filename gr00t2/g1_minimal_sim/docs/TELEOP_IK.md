# Teleop, targets, and IK (plain English)

## What you are building

- **Legs / balance:** Gear WBC — two ONNX policies (stand vs walk) + PD, same idea as the GR00T G1 sim.
- **Where the hands should be:** “End effector” (`ee_*`) **targets in the moving torso frame** — the same frame as the chest. When the robot walks, the **torso moves**, so the **same numbers** mean **different positions in the world**. Your **debug spheres** show those **world** targets.
- **Getting the arms there:** **IK** turns world palm targets into **joint angles** (`arm_target_q`). **Motors** (PD) try to hold those angles.

So there are **three** layers: **targets → IK → PD**. If spheres look right but arms wig out, either IK is producing bad angles or **PD / physics** cannot track them (often worse while walking).

## Modular teleop (swap inputs)

All backends edit the **same** `control_dict`:

| Input | How it hooks in |
|-------|-------------------|
| Keyboard | `gear_wbc_teleop` → `loco_cmd`, `ee_*`, height/rpy, **`z` reset** |
| VR stream | `gear_wbc_vr_stream` → sticks → `loco_cmd`; optional VR IK → `ee_*` |
| Legacy UDP | `legacy_vr_code/udp_teleop_receiver` → `ee_*` / loco from packets |

You generally use **one** locomotion path at a time (e.g. don’t mix VR stream with legacy UDP for the same thing). Keyboard + **optional** VR is supported (see `play_g1_gear_wbc.py` help).

## One mental image

```
teleop (keyboard / VR / UDP)
        ↓
control_dict["ee_*"]  (torso frame)  +  loco_cmd, etc.
        ↓
ee_world_targets_for_ik  +  torso pose  →  world palm targets
        ↓
solve_dual_arm_ik_v2  →  arm_target_q
        ↓
PD on arm joints  →  torques  →  physics
```

Frame consistency lives in **`ee_frame.py`** + **`arm_ik.py`** (`ee_world_targets_for_ik`). Do not mix **torso rotation matrix** and **torso quaternion** from different sources for position vs orientation — use **one** SO(3) (see code: `quat_wxyz_to_rotmat`).

## Keyboard (boxes / manipulation)

With **`--teleop`**, keys adjust **`ee_*`** in torso frame (see `arm_ik.apply_ik_arm_teleop_key`). **`z`** resets locomotion, arm joints home, and **end-effector targets** back to spawn defaults.

Optional **`--keyboard-grasp-primitives`**: widen/narrow, forward/back, and raise/lower palms (`keyboard_grasp_primitives.py`) without changing full IK wiring.

Optional **`--lock-ee-orient`** (with **`--teleop`**): keeps **world** palm orientation fixed (see `_ee_*_quat_world_anchor` in `init_ee_control_dict`). Each step the torso-frame `ee_*_quat` is updated so `R_world_palm` matches that anchor as the chest moves while walking. Only **position** keys change targets in a lasting way; orientation keys are overridden. Skipped when **`--vr-ik`** is on. **`z`** refresh re-anchors from the home torso-frame pose and current torso.

## Reset behavior

- **`z`:** Resets commands **and** clears **IK temporal smoothing** (`_ik_q_prev`) so the solver does not “remember” a bad previous pose after you recover.

## Tuning notes

- While **walking** (`‖loco_cmd‖ > 0.05`), arm joint **damping** is scaled up slightly so arms shake less when the base is moving (`gear_wbc_stand`: `_ARM_KD_WALK_SCALE`).
- If arms still diverge with correct-looking spheres, use **`--log`** (overwrites each run unless `--log-append`) and compare measured palm vs target fields in JSONL.

## Where to read code

| Piece | Module |
|-------|--------|
| Torso ↔ world EE math | `ee_frame.py`, `arm_ik.py` |
| IK solve | `arm_ik_v2.py`, `gear_wbc_arm_ik.py` |
| Runtime loop | `gear_wbc_stand.py` |
| Keyboard | `gear_wbc_teleop.py` |
| Debug spheres | `teleop_target_viz.py`, `--debug-ee-targets` |
