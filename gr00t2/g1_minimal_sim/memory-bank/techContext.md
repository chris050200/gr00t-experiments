# Tech context — g1_minimal_sim

## Repository paths (canonical)

- This sandbox: `gr00t2/g1_minimal_sim/`
- Sibling: `gr00t2/Isaac-GR00T/`
- G1 MJCF + yaml + meshes + `policy/*.onnx`:

  `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/`

## Recommended Python environment

**Preferred:** uv venv created by Isaac-GR00T setup:

```text
Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python
```

Create/update it (from **Isaac-GR00T** repo root):

```bash
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

**Minimal pip set** (any venv): `mujoco`, `onnxruntime`, `pyyaml`, **`gymnasium`** (Phase 2a env). Optional: `pillow` for `--save-frame`; **`pynput`** for `--stand --teleop` / `play_g1_gear_wbc.py --teleop` (on Linux, pynput needs **DISPLAY** / X11 for its keyboard backend). Optional: **`--openvr-teleop`** — requires **`openvr`** (pip) and SteamVR; optional Python module **`openvr_teleop`** in `g1_minimal_sim` implementing **`start_poller_for_runtime`** / **`stop_poller`**; if missing, a **warning** is printed and sim still runs (keyboard-only).

**Do not** rely on `conda install mujoco` from defaults (package often missing).

### Pytest (Gear WBC / ONNX)

Use the **same interpreter** as `play_g1_gear_wbc.py` / `spawn_g1_floor.py` — the Isaac-GR00T **GR00T-WholeBodyControl** venv — so `onnxruntime`, `mujoco`, and `gymnasium` match. Running `pytest` from **conda `(base)`** or another env without those packages will fail tests that construct **`GearWBCRuntime`** (e.g. `tests/test_g1_gear_wbc_env_headless.py`), even though `import default_g1_resources_dir` can work after lazy ONNX import.

From **`g1_minimal_sim/`** (adjust the leading `..` if your checkout path differs):

```bash
cd ~/Documents/gr00t_experiments/gr00t2/g1_minimal_sim

../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python -m pytest tests/ -q

# Headless env smoke only:
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python -m pytest tests/test_g1_gear_wbc_env_headless.py -q
```

Arm-only tests (`tests/test_arm_ik_*.py`) may pass with fewer deps, but **full** `g1_minimal_sim` CI-style runs should use the venv above.

### GR00T `unitree_g1` state shapes (P0)

Requires **`gr00t_wbc`** (WholeBodyControl install). Prints `RobotModel` + `prepare_observation_for_eval` slice sizes (`state.left_leg`, …, `state.right_hand`) and optional MuJoCo `nq`/`nu` for stylish_diner MJCF:

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  scripts/validate_gr00t_unitree_g1_shapes.py --waist-ik --mujoco-nq --hands
```

Pytest: `tests/test_validate_gr00t_unitree_g1_shapes.py`. Detail: **`gr00t_compatible_data_collection.md`** §7.

### GR00T observation builder (P1)

Module **`gr00t_observation_builder.py`**: **`Gr00tObservationBuilder.for_locomanip_default()`** → **`build(model, data, task_description=..., include_video=True/False)`** → dict with **`state.*`**, **`annotation.human.task_description`**, and optionally **`ego_view_image`** / **`video.ego_view`** ( **`head_pov`**, default **640×480**). Requires **`gr00t_wbc`** (same venv as above). Headless **`pytest`** uses **`include_video=False`**; ego rendering needs **GL** (interactive machine or **`MUJOCO_GL`**).

**P2 recording:** **`gr00t_teleop_logger.py`** (`Gr00tTeleopEpisodeLogger`, **`save_npz`**). **`play_g1_gear_wbc.py`** flags **`--gr00t-record-dir DIR`**, **`--gr00t-task "..."`**, optional **`--gr00t-record-video`**, optional **`--headless --headless-scripted-demo wiggle`** (synthetic motion for P2/export smoke, not real demos). Writes **`episode_000.npz`** + **`episode_000_metadata.json`** when the viewer closes or headless run finishes. Quick inspect: **`scripts/preview_gr00t_npz.py`** (optional **`--plot-dir`** for matplotlib PNGs).

## Commands (from `g1_minimal_sim/`)

Always use a **real** path to `python`—never type literal `...` in bash.

```bash
cd ~/Documents/gr00t_experiments/gr00t2/g1_minimal_sim

# Passive
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --headless

# Stand (Gear WBC)
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --stand --headless --max-steps 20000
```

Viewer (needs `DISPLAY` or X11 forward):

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --stand
```

Keyboard teleop (viewer + **terminal focus** for keys; `pip install pynput` if needed):

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --stand --teleop
```

**OpenVR teleop** (SteamVR + Quest Link / PC VR; `pip install openvr`; viewer only — no `--headless`):

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --stand --openvr-teleop
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --stand --teleop --openvr-teleop --hands
```

Combine **`--teleop`** with **`--openvr-teleop`** for keyboard locomotion / **`z`** reset while controllers drive arms (and triggers drive grippers when **`--hands`**).

**Split-machine VR over UDP v1** (OpenVR on **VR PC**, MuJoCo on **sim PC**):

- Sim: `spawn_g1_floor.py --stand --udp-teleop 0.0.0.0:5005` (optional `--teleop`, `--hands`; `--headless` OK).
- VR PC (from `g1_minimal_sim`): `pip install openvr numpy` then  
  `python -m legacy_vr_code.openvr_teleop_client --host <sim-ip> --port 5005`  
  (deploy the **`legacy_vr_code/`** package or full checkout).
- **`--debug-ee-targets`:** world IK palm spheres (`teleop_target_viz.py`). Notes: **`vr_legacy_history.md`**.

**Split-machine VR (`vr_teleop` stream, locomotion Phase A):**

- Sim: `spawn_g1_floor.py --stand --vr-teleop` (default bind `0.0.0.0:5006`) or `play_g1_gear_wbc.py --vr-teleop` / `--headless` OK. **Not** combinable with `--udp-teleop`.
- VR PC: `python -m vr_teleop.openvr_udp_sender --target <sim-ip>:5006` (`pip install openvr`). Sticks → **`loco_cmd`**; combine `--teleop` on sim for keyboard height / rpy / **`z`** if needed.
- IK from stream: **`--vr-ik on`** (legacy **`pos`/`pose`** accepted as **`on`**). Reuses torso compose/calibration from debug overlay path; palm orientation from stream when IK is on. **Controller→palm** orientation uses fixed intrinsic offsets (Z→Y′→X″ deg, post-multiply on composed world quat) in **`gear_wbc_vr_stream.py`**: `VR_IK_HAND_INTRINSIC_{YAW,PITCH,ROLL}_DEG` / `VR_IK_HAND_OFFSET_QUAT_WXYZ` (edit there if a different headset grip feels twisted). JSONL trace echoes the same `vr_ik_hand_*` fields for reproducibility.
- Current-HMD orient mode for non-legacy anchors: **`--vr-head-orient {full,yaw_only}`** (default **`yaw_only`**). `yaw_only` keeps world Z shared and removes HMD pitch/roll coupling from head-relative `inv(T_hmd)` translation.
- Optional later: joint-space posture via **`arm_ik_posture.py`** merged into IK (not wired today; avoids extra CLI until needed).

**VR pose stream v2 (legacy debug viewer**, UDP **5006**, not `control_dict`):

- Sim: `python -m legacy_vr_code.vr_teleop_test_streaming receive --bind 0.0.0.0:5006` (MuJoCo + **DISPLAY**; optional **`--pov-preview`** + OpenCV).
- VR PC: `python -m legacy_vr_code.vr_teleop_test_streaming send --host <sim-ip> --port 5006`.
- New scaffold: **`vr_teleop/`**. Short ref: **`vr_legacy_history.md`**.
- Tests: `pytest tests/test_vr_steamvr_mujoco_mapping.py tests/test_vr_teleop_root_motion.py tests/test_vr_pov_rotate.py` (numpy-only); `tests/test_vr_pov_pose.py` needs MuJoCo. Some sandboxes: `OMP_NUM_THREADS=1`.

**New split-machine VR lab (`vr_teleop/`)**:

- Sim PC (receiver/viewer):
  `python -m vr_teleop.vr_lab --bind 0.0.0.0:5006`
- VR PC (Quest2 + SteamVR/OpenVR sender):
  `python -m vr_teleop.openvr_udp_sender --target <sim-ip>:5006 --tracking-space seated --hz 90`
- Optional sender tuning:
  - `--offset-mj X Y Z` (global MuJoCo-frame position offset),
  - `--controller-yaw-offset-deg D` (local wrist neutral tweak).
- Notes:
  - `vr_lab` is intentionally simple (floor + origin XYZ marker + HMD/left/right arrows) for mapping verification.
  - In some environments/tests use `MKL_THREADING_LAYER=SEQUENTIAL OMP_NUM_THREADS=1` to avoid MKL/OpenMP shared-memory aborts.

Planned test mode (next): translation-only mobile-frame validation in `vr_lab` (virtual-base XY drive, yaw disabled) to verify base-relative pose invariance before adding yaw and before wiring to humanoid walking teleop.

**Articulated hands + gripper** (same venv; uses `g1_gear_wbc_hands.yaml` + `g1_gear_wbc_hands.xml` in the vendored `g1/` folder):

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --stand --teleop --hands
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --teleop --hands
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --teleop --hands --scene table_pnp
```

**Gymnasium:** after `register_g1_gear_wbc_env()`: `gym.make("G1GearWBC-v0", ...)` or `gym.make("G1GearWBC-TablePnP-v0", use_hands=True, enable_teleop=True)` (or `G1GearWBC-v0`, `scene="table_pnp"`).

**Gymnasium spine (Phase 2a)** — same physics, `gymnasium` API + optional teleop:

**`play_g1_gear_wbc.py` scene default:** `--scene stylish_diner` (merged diner). **`--hands`** with default scene resolves to the merged **`g1_gear_wbc_stylish_diner_hands.{xml,yaml}`** (May 2026 Phase 2 — articulated 14-DoF fingers on diner). **`--keyboard-grasp-primitives`** requires **`--teleop`** (symmetric EE position nudges: **x/c**, **Home/End**, **Page Up/Down**). Diner world file reference + current friction-grasp + Phase 2 hands writeup: **`memory-bank/stylish_diner_keyboard_teleop.md`** and **`memory-bank/hands_gripper_teleop.md`** §14.

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --headless --max-steps 2000
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --teleop
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --teleop --scene floor
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --teleop --keyboard-grasp-primitives
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --vr-teleop --debug-vr-stream
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --vr-teleop --vr-ik on --debug-vr-stream
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --vr-teleop --vr-ik on --debug-vr-stream --hands
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --vr-teleop --vr-ik on --vr-ik-anchor torso --vr-head-orient yaw_only --debug-vr-stream --hands
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --vr-teleop --debug-vr-stream --vr-ik pose --hands
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --task table_box --vr-teleop --debug-vr-stream --vr-ik pose --hands
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --vr-teleop --debug-vr-stream --debug-vr-stream-torso-z frozen_calib
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --vr-teleop --debug-vr-stream --vr-ik on --hands --log /tmp/g1_vr_trace.jsonl --log-hz 60
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --scene table_pnp --headless --max-steps 2000
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python spawn_g1_floor.py --stand --scene table_pnp --headless --max-steps 5000
```

**VR/IK JSONL trace (`play_g1_gear_wbc.py`):**

- Flags: `--log PATH` (append JSONL), `--log-hz HZ` (default **30** in argparse; operator often uses **60**).
- Emitted from `gear_wbc_stand.GearWBCRuntime` as **`{"kind":"vr_compare",...}`** rows (plus `{"kind":"meta",...}` open/close).
- Key fields (names only):
  - **Stream:** `vr_stream.*`
  - **Overlay (debug arrows path):** `overlay.*`
  - **IK target pipeline:** `ik.*` (includes `ik_pw_*` composed world targets, `ee_*`, `ik_pr_*` torso-frame intermediates, `ik_world_*_from_ee` reconstruction)
  - **Cross-checks:** `dbg_pw_minus_overlay_*`, `dbg_overlay_minus_ik_world_*`, `dbg_pw_minus_ik_world_*`
  - **Measured vs commanded (post-step FK):** `meas_palm_world_*`, `ik_tgt_world_*_from_ee`, `err_meas_minus_ik_tgt_world_*`, `err_meas_minus_ik_pw_world_*`
  - **Spin / tracking diagnostics:** `torso_omega_world`, `torso_omega_norm`; `err_quat_meas_minus_ik_tgt_world_{left,right}_deg`; `dbg_ik_tgt_world_pos_post_minus_ik_solve_{left,right}_l2` (same `ee_*` with post-step vs IK-time torso); `arm_tau_pre_clip_l2`, `arm_tau_post_clip_l2`, `arm_tau_n_clip`, `arm_tau_max_abs_clip_delta`; in `ik.*`: `ik_solve_torso_xpos`, `ik_solve_torso_xmat`, `ik_solve_torso_xquat`
  - **Deltas:** `delta_*` (between logged samples; not the same as sim dt if `--log-hz` < stepping rate)

**Keyboard teleop JSONL (`kind: teleop_ik`) plotting (May 2026):**

- Script: `tests/teleop_regression/plot_teleop_jsonl.py`
- Input: any `play_g1_gear_wbc.py --log /tmp/foo.jsonl` run
- Output: PNG with position/quaternion tracking errors, arm clip counts, and pelvis XY path
- Current break references in plotting/regression:
  - position break: **0.15 m**
  - position warn: **0.10 m**
  - quaternion break: **45 deg**

```bash
# Plot a manual teleop log
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
tests/teleop_regression/plot_teleop_jsonl.py \
  --log /tmp/lockeeorientdebugeetargets.jsonl \
  --out /tmp/lockeeorientdebugeetargets_plot.png
```

**Headless teleop regression harness (May 2026):**

- Script: `tests/teleop_regression/table_box_walk_regression.py`
- Supports scenes (`floor`, `table_pnp`, `stylish_diner`), profiles (`baseline`, `aggressive_explode`, `forward_floor`, `manual_like_long_walk`), and lock modes (`off`, `full`, `yaw_only`).
- Emits JSONL + summary JSON + PNG with pelvis XY path.

```bash
# Manual-like floor sequence: forward/back/turn/forward
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
tests/teleop_regression/table_box_walk_regression.py \
  --out-dir /tmp/floor_manual_like_ab \
  --scene floor \
  --profile manual_like_long_walk \
  --steps 8600
```

**Episode extractor (forensic triage):**

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
tests/teleop_regression/extract_ik_episodes.py \
  --log /tmp/lockeeorientdebugeetargets.jsonl \
  --out /tmp/lockee_episodes_summary.json \
  --dump-first-episode-rows /tmp/lockee_first_episode_rows.jsonl
```

**Forensic burst logging (Tier B windows):**

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
play_g1_gear_wbc.py \
  --scene floor --teleop --lock-ee-orient --debug-ee-targets \
  --log /tmp/lockeeorientdebugeetargets.jsonl \
  --forensic-burst \
  --forensic-pre-steps 180 \
  --forensic-post-steps 240
```

Burst data is written into the same JSONL with:
- `kind: teleop_forensic_event` (`trigger_start` / `trigger_end`)
- `kind: teleop_forensic_row` (`phase: pre|post`, `episode_id`)

For complete forensic-chain requirements and seam map, see:
- `memory-bank/ik_lock_transform_forensics.md` (end-to-end keyboard->IK->joint-target map + required burst field groups).

**Quick analysis recipes:**

```bash
# max tracking error (needs jq)
jq -s 'map(select(.kind=="vr_compare") | .err_meas_minus_ik_tgt_world_left_l2) | max' /tmp/g1_vr_trace.jsonl

# show worst-left-error row (jq)
jq -s 'map(select(.kind=="vr_compare")) | max_by(.err_meas_minus_ik_tgt_world_left_l2)' /tmp/g1_vr_trace.jsonl
```

```bash
# crude grep without jq: largest left tracking errors (sorted)
grep '"kind":"vr_compare"' /tmp/g1_vr_trace.jsonl | grep 'err_meas_minus_ik_tgt_world_left_l2' | sed 's/.*"err_meas_minus_ik_tgt_world_left_l2":\([0-9.eE+-]*\).*/\1 &/' | sort -nr | head
```

Registered ids: **`G1GearWBC-v0`** (floor), **`G1GearWBC-TablePnP-v0`** (table + free apple/plate) — `g1_gear_wbc_env.register_g1_gear_wbc_env()` then `gymnasium.make`. Module: `g1_gear_wbc_env.py`; shared sim: `gear_wbc_stand.GearWBCRuntime`; scene resolution: **`gear_wbc_stand.resolve_gear_wbc_config()`**.

**VR debug overlay defaults (Apr 2026):**

- `--vr-receiver-yaw-deg`: **`-90`** (receiver-side yaw basis correction)
- `--debug-vr-stream-torso-z`: **`full`**
- `--debug-vr-stream-calib-orient`: **`yaw_only`**
- `--vr-head-orient`: **`yaw_only`** (for non-legacy `--vr-ik-anchor`)

These defaults were user-validated for stable co-motion and no drift in walking/turning on the current setup.

**Arms + IK (``--stand --teleop``):** Keys adjust **`torso_link`**-frame palm targets; each step `ee_frame` → world, then IK → `arm_target_q`. Left pos: `i/k` `j/l` `u/p`; right: `r/t` `f/g` `v/b`; both-hand orient (wxyz, torso-fixed axes): `,` `.` ; `;` `'` ; `[` `]`. `z` resets locomotion + arm home + EE home. Status line prints `ee_*(torso)` (and `grip L/R` when `--hands`). Modules: `arm_ik.py`, `ee_frame.py`. Without `--teleop`, arms stay at loaded-pose `arm_target_q` (no IK).

**Gripper (only with `--hands` / `use_hands`):** scalars in `control_dict` — **`9`** / **`0`** open / close **left**; **`=`** / **`-`** open / close **right** (small step per keypress; clamped 0–1); **`h`** hard-closes both; **`y`** opens both. Design notes and pitfalls: **`memory-bank/hands_gripper_teleop.md`**.

## Flags (`spawn_g1_floor.py`)

| Flag | Role |
|------|------|
| `--stand` | Gear WBC loop (`gear_wbc_stand.run_gear_wbc`) |
| `--headless` | No GLFW |
| `--max-steps N` | Step count when headless (passive or stand) |
| `--xml PATH` | Override MJCF (parent dir used as resources for `--stand`) |
| `--save-frame` | Passive headless only; needs `MUJOCO_GL=egl` (or osmesa) often |
| `--teleop` | With `--stand` only, not with `--headless`; legs → `loco_cmd` / height / rpy / freq; arms → **IK** EE targets + **`z`** reset (needs `pynput`, DISPLAY on Linux) |
| `--openvr-teleop` | With `--stand`, viewer only: optional VR → `control_dict` (see `gear_wbc_stand.start_openvr_poller`) |
| `--udp-teleop HOST:PORT` | With `--stand`: listen for JSON from **`legacy_vr_code.openvr_teleop_client`** (e.g. `0.0.0.0:5005`); allowed with `--headless`; not with **`--vr-teleop`** |
| `--vr-teleop [HOST:PORT]` | With `--stand`: listen for **`vr_teleop.openvr_udp_sender`** (default `0.0.0.0:5006`); sticks → **`loco_cmd`**; allowed with `--headless` |
| `--vr-ik {off,pos,pose}` | With `--stand --vr-teleop`: stream controllers update torso-frame `ee_*` targets (`pos`=position only, `pose`=position+orientation) |
| `--vr-head-orient {full,yaw_only}` | With `--stand --vr-teleop` and non-legacy `--vr-ik-anchor`: orientation used for current-HMD `inv(T_hmd)` in head-relative compose |
| `--vr-receiver-yaw-deg DEG` | With `--stand`: receiver-side yaw basis offset for stream compose (default `-90`) |
| `--debug-ee-targets` | With `--stand`, viewer only: spheres = world IK palm targets (`teleop_target_viz.py`) |
| `--hands` | With `--stand`: load **`g1_gear_wbc_hands.yaml`** (articulated fingers + gripper keys **9/0** left, **=/-** right); requires assets next to default `g1_gear_wbc.yaml` |
| `--scene` | With `--stand` on **`spawn_g1_floor.py`** or **`play_g1_gear_wbc.py`**: default **`stylish_diner`** (merged `scenes/stylish_diner_1/`); **`floor`** (empty checker); **`table_pnp`** (table + apple + plate, `G1GearWBC-TablePnP-v0` when used from play). |
| `--task` | With `--stand`: task preset key from `scenes/<task>/task_spec.yaml` (currently `table_box`); overrides `--scene` and `--config-yaml` |
| `--config-yaml` | With `--stand`: yaml basename in the MJCF directory, or **absolute** path (resources dir = parent of that file); overrides `--scene` |

## ONNX / LFS

If policies missing, from **GR00T-WholeBodyControl** repo root:

```bash
git lfs pull
```

Expect under `.../g1/policy/`: `ft92.onnx` + `ft109.onnx` **or** `GR00T-WholeBodyControl-Balance.onnx` + `GR00T-WholeBodyControl-Walk.onnx`.

## Common failures

| Symptom | Cause | Mitigation |
|---------|--------|------------|
| `No such file .../.venv/bin/python` | Wrong path or setup not run | Run `setup_GR00T_WholeBodyControl.sh` or use full path |
| `DISPLAY` / GLFW | Headless or SSH without X11 | `--headless` or `ssh -X`, or viewer on desktop |
| `ONNX policy not found` | LFS not pulled | `git lfs pull` in dependency repo |
| `Missing .../g1_gear_wbc_*_table.yaml` | Table yamls not in vendored `g1/` (upstream reset) | Re-add `g1_gear_wbc_table.yaml` + `g1_gear_wbc_hands_table.yaml` next to `*_table.xml` (same keys as floor yamls, different `xml_path`) |
| onnxruntime DRM warning | Headless GPU discovery | Harmless on CPU EP |

## Sim timestep

`g1_gear_wbc.yaml` sets `simulation_dt: 0.005` (applied in `run_gear_wbc`).

## GR00T eval (official stack, not `g1_minimal_sim` entrypoint)

Full VLA + loco-manip benchmark uses **server + client** and a **Gymnasium** env with **`WholeBodyControlWrapper`**. Example client command (from `Isaac-GR00T/examples/GR00T-WholeBodyControl/README.md`; adjust ports to match server):

```bash
# From Isaac-GR00T repo root; server must be running first.
gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
  --policy_client_host 127.0.0.1 \
  --policy_client_port 2000 \
  --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
  --n_episodes 5 \
  --max_episode_steps 5000 \
  --n_action_steps 30 \
  --n_envs 1
```

### OG apple-to-plate reference eval (Tier A — fair vs shipped scene)

Use this stack to validate **original** Digital Cousin apple-to-plate **weights + RoboCasa scene** (not `g1_minimal_sim` MJCF). **No need to copy checkpoints into `g1_minimal_sim`** — point `--model-path` at Hugging Face ids or a local cache directory.

| What | Where under `gr00t2/Isaac-GR00T` (or HF) |
|------|-------------------------------------------|
| Operator doc + exact commands | `examples/GR00T-WholeBodyControl/README.md` |
| Shipped eval checkpoint | HF [`nvidia/GR00T-N1.6-G1-PnPAppleToPlate`](https://huggingface.co/nvidia/GR00T-N1.6-G1-PnPAppleToPlate) or README **Option 1** path `cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate` |
| Finetune / dataset layout | `examples/GR00T-WholeBodyControl/finetune_g1.sh` → dataset dir `examples/GR00T-WholeBodyControl/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/unitree_g1.LMPnPAppleToPlateDC` (clone via README sparse-checkout of `unitree_g1.LMPnPAppleToPlateDC` from `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`) |
| RoboCasa task class | `external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobocasa/robocasa/environments/locomanipulation/locomanip_dc.py` — `LMPnPAppleToPlateDC` |
| Gym env id | `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc` |
| Policy server | `gr00t/eval/run_gr00t_server.py` — **`--embodiment-tag UNITREE_G1`**, **`--use-sim-policy-wrapper`**, **`--model-path`** as above; **`--port`** must match **`rollout_policy.py` `--policy_client_port`** (README examples use `2000` for client; default server port in code is `5555` if unset — align explicitly). |
| Client | `gr00t/eval/rollout_policy.py` using **WholeBodyControl** venv Python (same as block above). |
| Sim setup (once) | `bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh` (see README). |

**Tier B (minimal sim proxy):** `g1_minimal_sim/scripts/run_gr00t_inference.py --scene table_pnp` with the **same** server + PnP checkpoint tests **client + obs builder + cadence** in-repo; it is **not** pixel-identical to `LMPnPAppleToPlateDC` (see **`gr00t_compatible_data_collection.md`** §3–4). Use Tier A pass / Tier B fail to blame domain mismatch; Tier A fail to blame install, ports, or checkpoint path.

### Canonical PnP A/B (same model, apple/plate task — **not** diner vs OG)

Use this when debugging **minimal-sim inference wiring** against a **known-good** Isaac-GR00T eval. **Do not** compare `rollout_policy` on RoboCasa to `run_gr00t_inference` on **`stylish_diner`** if the goal is “PnP checkpoint parity”: diner is a **different** task and usually a **different** finetune.

| Side | Client | Scene / env | Checkpoint |
|------|--------|-------------|------------|
| **A (reference)** | `Isaac-GR00T/gr00t/eval/rollout_policy.py` | `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc` (RoboCasa Digital Cousin apple → plate) | `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` (or README paths) |
| **B (ours)** | `g1_minimal_sim/scripts/run_gr00t_inference.py` | `--scene table_pnp` (minimal-sim MJCF apple/plate; **not** RoboCasa) | **Same** `--model-path` / HF id as A |

**Same server process** for both runs (or restart between runs with identical flags): `run_gr00t_server.py` with **`--embodiment-tag UNITREE_G1`**, **`--use-sim-policy-wrapper`**, ports aligned with client.

**How to read results**

- **A works, B bad:** chase **observation alignment** in minimal sim (ego camera mount / FOV / resolution, `Gr00tObservationBuilder`, `--plan-hz` / substeps, prompts). Iterating camera and builder until B improves is the intended loop.
- **A bad:** fix **Isaac-GR00T setup**, checkpoint download, or ports before tuning minimal sim.
- **“Same scene”** here means **same task intent** (pick apple, place on plate), **not** identical meshes or pixels between RoboCasa and `table_pnp`.

Local minimal-sim command shape (second terminal after server):

```bash
cd g1_minimal_sim
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  scripts/run_gr00t_inference.py \
  --policy-host 127.0.0.1 --policy-port 2000 \
  --scene table_pnp \
  --prompt "pick up the apple and place it on the plate" \
  --max-steps 5000 --debug-policy-prints
```

Observation/action layout for **`unitree_g1`:** `gr00t/configs/data/embodiment_configs.py`. **Teleop → dataset → training → inference** checklist (wrapper, `decode_action`, state alignment): **`g1_minimal_sim/memory-bank/gr00t_compatible_data_collection.md` §6.4**. Broader plan: `manipulation_and_gr00t_plan.md`.

## Local finetuned checkpoint bring-up (stylish diner)

Checkpoint used for local bring-up:

```bash
~/models/GR00T-N1.6-G1-LiftRuns-001-015
```

Prompt strings used from local lift datasets:

- `pick up the blue block`
- `pick up the block and lift it above the table`

Official policy server (Isaac-GR00T):

```bash
cd ~/Documents/gr00t_experiments/gr00t2/Isaac-GR00T

gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  gr00t/eval/run_gr00t_server.py \
  --model-path ~/models/GR00T-N1.6-G1-LiftRuns-001-015 \
  --embodiment-tag UNITREE_G1 \
  --use-sim-policy-wrapper \
  --device cuda \
  --port 2000
```

Local stylish-diner inference bridge:

```bash
cd ~/Documents/gr00t_experiments/gr00t2/g1_minimal_sim

../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  scripts/run_gr00t_inference.py \
  --policy-host 127.0.0.1 \
  --policy-port 2000 \
  --scene stylish_diner \
  --prompt "pick up the blue block" \
  --max-steps 1200 \
  --debug-policy-prints \
  --save-video ~/Documents/gr00t_experiments/gr00t2/g1_minimal_sim/data/g1_diner_infer.mp4
```

(`scripts/run_gr00t_stylish_diner_inference.py` is the same implementation; `run_gr00t_inference.py` is the preferred name. Both prepend `g1_minimal_sim` to `sys.path` when run as files.)

Defaults (May 2026+): **`--arm-action-mode auto`** (live → absolute after server decode); **`--plan-hz 50`** with **`n_substeps`** pinned to logger; **`--n-action-steps 30`**; **`--video-fps`** unset → matches **`plan_hz`** for real-time playback; optional **`--replay-npz`** to validate bridge without the model. Wiring contract + diagnostics live in **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md)** §6 / §9 while the parity bug is open; the durable pipeline map stays in **`gr00t_compatible_data_collection.md` §6.4**.

Current status:

- Bridge bugs (dtype write, cadence, live double-add) are **fixed** and pinned by **`tests/test_run_gr00t_inference_cadence.py`**.
- Remaining gap is mostly **policy / dataset / domain** (hands, grasp quality, scene pixels) — verify server returns hand keys via **`--debug-policy-prints`**.

## GR00T policy I/O numerical A/B dump

> Full protocol, interpretation matrix, exit-code semantics, and current verdict live in **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md) §9** while the qualitative inference-wiring bug is still active. This section just keeps the command pair so an operator can re-run the harness without grepping for it.

One-command harness (from `g1_minimal_sim/`):

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  scripts/run_policy_ab_dump.py --run-id post4a_bis
```

Raw CLI form (both clients accept `--policy-ab-dump DIR`):

```bash
# Oracle (run from Isaac-GR00T repo root):
gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  gr00t/eval/rollout_policy.py \
  --policy_client_host 127.0.0.1 --policy_client_port 2000 \
  --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
  --n_episodes 1 --max_episode_steps 200 --n_action_steps 30 --n_envs 1 \
  --policy-ab-dump /tmp/gr00t_ab_rollout_step0

# Subject (run from g1_minimal_sim/):
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  scripts/run_gr00t_inference.py --scene table_pnp_apple \
  --policy-host 127.0.0.1 --policy-port 2000 --max-steps 200 --n-action-steps 30 \
  --policy-ab-dump /tmp/gr00t_ab_minimal_step0
```

Module pin: `g1_minimal_sim/policy_ab_dump.py` (`dump_policy_roundtrip_pack`), `g1_minimal_sim/scripts/run_policy_ab_dump.py`, tests `tests/test_policy_ab_dump.py`. Caveat: a trailing space after a `\` continuation breaks argparse with `unrecognized arguments: --policy-ab-dump` — put the command on one line or use bare continuations.

**Diff two existing dump roots** (no server; exploratory vs strict):

```bash
cd g1_minimal_sim
python scripts/compare_policy_ab_dumps.py \
  -a /tmp/gr00t_ab_rollout_after_phase4b \
  -b /tmp/gr00t_ab_minimal_after_phase4b \
  --exit-on never
```

**Hybrid `get_action` at step 0** (needs running policy server; vision causality probe):

```bash
cd g1_minimal_sim
python scripts/hybrid_policy_get_action.py \
  --rollout /tmp/gr00t_ab_rollout_after_phase4b \
  --minimal /tmp/gr00t_ab_minimal_after_phase4b \
  --policy-host 127.0.0.1 --policy-port 2000 \
  --modes minimal hybrid_ego rollout
```

**Phase 4b MJCF scope (May 2026):** ego lighting parity edits live only under **`g1_minimal_sim/scenes/table_pnp_apple/g1_gear_wbc_{table,hands}_table_pnp_apple.xml`** and **`g1_minimal_sim/scenes/lab_dc_layout/{lab_dc_world.xml, gen_lab_dc_world_xml.py}`** — not the upstream Isaac `arenas/gear_lab/gear_lab.xml`. Operator checklist: **`memory-bank/inference_obs_one_pass.md`**.
