# Project brief — g1_minimal_sim

## Purpose

A **small, local sandbox** next to `Isaac-GR00T` for experimenting with **Unitree G1** in **MuJoCo**: spawn the same MJCF bundle used by GR00T-WholeBodyControl / Gear WBC, then grow toward **locomotion**, **arm control**, **teleop**, **manipulation scenes**, and **PolicyClient inference** against a checkpoint served from Isaac-GR00T—each step **gated by tests** (see `progress.md`). **Teleop → dataset → training → inference** semantics for `unitree_g1`: **`memory-bank/gr00t_compatible_data_collection.md` §6.4**; broader GR00T plan: **`memory-bank/manipulation_and_gr00t_plan.md`**.

## In scope

- Minimal entrypoint: `spawn_g1_floor.py` (passive physics or `--stand` via Gear WBC ONNX); **`--scene table_pnp`** for table + manipuland MJCF (see `techContext.md`).
- **`play_g1_gear_wbc.py`** default scene is **`stylish_diner`** (merged G1 + diner MJCF under `scenes/stylish_diner_1/`). Diner layout source: `stylish_diner.xml`; keyboard teleop, optional grasp primitives, known arm-walk bug, and planned “push box in diner” test: **`memory-bank/stylish_diner_keyboard_teleop.md`**.
- Optional **`--hands`**: loads articulated-finger MJCF — `g1_gear_wbc_hands.xml` (floor), `g1_gear_wbc_hands_table.xml` (table_pnp), or **`g1_gear_wbc_stylish_diner_hands.xml`** (May 2026 Phase 2 — diner+articulated). 14-DoF gripper teleop on all three. Detail: `memory-bank/hands_gripper_teleop.md`.
- Optional **split-machine VR over UDP** — **`legacy_vr_code/`** (v1: **`--udp-teleop`**); **`vr_teleop/`** (**`--vr-teleop`**, **`openvr_udp_sender`**, **`openvr_stream`**, play **`--debug-vr-stream`**) for the rebuilt stream; see **`vr_legacy_history.md`** and **`vr_stream_debug_overlays.md`**.
- Implementation detail: `gear_wbc_stand.py` (facade: **PD** + decimated **ONNX**, CPU `onnxruntime`) plus split `gear_wbc_*.py` modules (config, obs, teleop, VR stream/trace, arm IK).
- Dependency on vendored assets under `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/` (MJCF, yaml, meshes, policy ONNX). **Table scene** adds `g1_gear_wbc_table.xml`, `g1_gear_wbc_hands_table.xml`, and matching **`g1_gear_wbc_table.yaml`**, **`g1_gear_wbc_hands_table.yaml`** (local additions — preserve if refreshing dependencies).

## Out of scope (for this folder, unless roadmap says otherwise)

- **Training** the GR00T VLA inside this folder (finetune lives in Isaac-GR00T). **Running** a checkpoint via **`gr00t/eval/run_gr00t_server.py`** + **`scripts/run_gr00t_stylish_diner_inference.py`** is in scope; hosting policy servers in production is not a goal of this sandbox.
- Isaac Lab ports.
- Replacing upstream MJCF without documenting the change in `techContext.md`.

## Layout assumption

`g1_minimal_sim/` and `Isaac-GR00T/` are **siblings** under `gr00t2/` so `default_g1_xml_path()` resolves correctly.

## Success (for “phase 0”)

- One command runs **passive** sim (headless or viewer when `DISPLAY` exists).
- One command runs **`--stand`** with ONNX present; headless run reports stable pelvis height after sufficient steps.
