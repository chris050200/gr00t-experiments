# `table_box` task scaffold

Purpose: first repeatable manipulation benchmark for teleop and later GR00T eval.

Target flow:

1. Approach table.
2. Align end-effector over box.
3. Grasp and lift above threshold.
4. (Optional) place at goal marker.

Initial scope:

- Keep using existing Gear WBC scene plumbing (`scene` / `config_yaml`) while this folder stores task-specific metadata and notes.
- Add MJCF/YAML assets here only when the task diverges enough from `table_pnp` to justify dedicated files.

Success metrics (draft):

- Binary success: box lifted above `min_lift_height_m` for `hold_steps`.
- Time-to-first-grasp.
- Failure category: miss, slip, instability, reset required.
