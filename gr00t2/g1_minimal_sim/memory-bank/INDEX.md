# g1_minimal_sim memory bank

This is the canonical project memory bank. Read these on entry to any task in this workspace (especially after a context reset).

## Currently active

> **🔥 [ACTIVE_gr00t_inference_wiring.md](ACTIVE_gr00t_inference_wiring.md)** — GR00T inference-loop audit (Tier-A apple PnP vs `run_gr00t_inference.py --scene table_pnp_apple`). **May 2026:** eight wiring bugs closed + **Phase 4b** lighting on Tier-B MJCF **shrunk** ego mean-RGB gap vs Tier-A; **`scripts/hybrid_policy_get_action.py`** proves **ego** still causal at step 0; qualitative §1 **still open** (early seconds improved, then regression). **One-pass obs checklist:** [inference_obs_one_pass.md](inference_obs_one_pass.md). Read ACTIVE before touching `scripts/run_gr00t_inference.py`, `gear_wbc_stand.py`, `gr00t_observation_builder.py`, or policy-server wiring. Delete ACTIVE when §1 exit is met; graduate durable notes into `systemPatterns.md` / `tests/`.

## Core (read first)

| File | Purpose |
|------|---------|
| [projectbrief.md](projectbrief.md) | Scope and success criteria for this sandbox |
| [systemPatterns.md](systemPatterns.md) | Passive sim vs Gear WBC stand; GR00T VLA vs leg policy; VR compose contract |
| [techContext.md](techContext.md) | Paths, commands, dependencies, common failures |
| [activeContext.md](activeContext.md) | Current focus + future cleanups (deferred refactors) |
| [progress.md](progress.md) | Current state + roadmap; resolved bugs in compact appendix |

## MuJoCo grasp / contact (read before maxing friction or PD)

- Try `<option impratio="10" cone="elliptic" noslip_iterations="3"/>` first — default `impratio=1` under-delivers tangential vs normal contact force in the cone.
- Tighten `solref` / `solimp` on the **manipuland** only; keep leg–floor defaults until you re-validate the Gear WBC ONNX stand/walk.
- Diner `target_block` density ~5 and friction ~3.0 are **prototype starting points**; harder tasks ramp mass and μ deliberately.
- `--print-grip-force` peaks `|Fn|`; slip margin needs `|Ft|` vs `μ|Fn|` — gaps and roadmap: `activeContext.md`.
- Mesh palm collisions → noisy contacts; if still slipping after the solver fix, try primitive palm pads or heavier objects before absurd friction.

## GR00T PnP parity experiment (same checkpoint, two clients)

To validate **`run_gr00t_inference.py`** against **known-good** Isaac-GR00T eval: same **`nvidia/GR00T-N1.6-G1-PnPAppleToPlate`** (or equivalent), **Side A** = `rollout_policy.py` + `LMPnPAppleToPlateDC_G1_gear_wbc`, **Side B** = `run_gr00t_inference.py --scene table_pnp` (**not** `stylish_diner`). Full table + interpretation: **[`techContext.md`](techContext.md)** § "Canonical PnP A/B".

**This parity test is still failing qualitatively** (oracle ≈ apple-to-plate; subject ≈ reach-then-regress / face-level failure), but **step-0 metrics improved** after Phase 4b (see ACTIVE §9.5.b). Live audit + next experiments: **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md)**. **Obs one-pass:** **[`inference_obs_one_pass.md`](inference_obs_one_pass.md)**. Forensic dump tooling (`policy_ab_dump.py`, `scripts/run_policy_ab_dump.py --run-id <name>`, `--policy-ab-dump DIR` on both clients) is catalogued there in §9; **`scripts/hybrid_policy_get_action.py`** probes vision causality at step 0; the command pair lives in **[`techContext.md`](techContext.md)** § "GR00T policy I/O numerical A/B dump" for muscle-memory access.

## Related repo (GR00T2 root)

| Location | Purpose |
|----------|---------|
| [`Isaac-GR00T`](../../Isaac-GR00T/) (sibling under `gr00t2/`) | Official **`rollout_policy.py`**, **`run_gr00t_server.py`**, WholeBodyControl README, PnP checkpoint / RoboCasa env. |
| [`scene_builder`](../scenes/scene_builder/README.md) (under `g1_minimal_sim/scenes/`) | **Scene builder**: GLB/USD preview, MJCF export, assets installed under `g1_minimal_sim/scenes/`. Memory bank: [`scene_builder/memory-bank/INDEX.md`](../scenes/scene_builder/memory-bank/INDEX.md). |

## Topic docs (read on demand)

| File | When to read |
|------|--------------|
| [gr00t-g1-wholebody.md](gr00t-g1-wholebody.md) | GR00T × G1 whole-body reference (VLA actions vs Gear WBC; `navigate_command` / `base_height_command`; eval path) |
| [gr00t_compatible_data_collection.md](gr00t_compatible_data_collection.md) | **LeRobot / `unitree_g1` obs + action logging** — P0 shape script, **`gr00t_observation_builder.py`** (P1), **`gr00t_teleop_logger.py`** + **`play_g1_gear_wbc.py --gr00t-record-dir`** (P2), **§6.4 teleop→dataset→train→inference map**, server wrapper / `decode_action` checklist, P3–P4 export/eval |
| [manipulation_and_gr00t_plan.md](manipulation_and_gr00t_plan.md) | Gymnasium spine + manipulation + GR00T inference plan (§G then §A then §B–E) |
| [stylish_diner_keyboard_teleop.md](stylish_diner_keyboard_teleop.md) | Default diner scene, hold-to-move keyboard loco, optional grasp primitives, friction-grasp PD + Phase 2 articulated-fingers-on-diner; distinguishes teleop checklists vs headless MJCF/integration tests |
| [hands_gripper_teleop.md](hands_gripper_teleop.md) | Articulated hands MJCF (incl. **diner+hands**, May 2026), `arm_target_q` layout, ONNX 29-joint gather, actuator ↔ qpos PD mapping, **§13 per-arm-joint PD profile + flags**, **§15 solver / grasp notes** (see also **INDEX § MuJoCo grasp / contact** above) |
| [ik_lock_transform_forensics.md](ik_lock_transform_forensics.md) | Lock-orient transform / IK chain map, seam invariants, tiered forensic logging policy. Canonical writeup of the May 2026 `mj_jac` point-frame fix |
| [vr_stream_debug_overlays.md](vr_stream_debug_overlays.md) | `--debug-vr-stream` arrows on `play_g1_gear_wbc.py`; torso-follow compose + IK handoff notes |
| [vr_legacy_history.md](vr_legacy_history.md) | UDP `--udp-teleop` v1 (still wired) + `vr_teleop_test_streaming` v2 (debug viewer) reference; lives under `legacy_vr_code/` |
| [inference_obs_one_pass.md](inference_obs_one_pass.md) | **Tier-B PnP obs iteration:** paired dumps → compare → hybrid `get_action` → Phase 4b MJCF scope → re-measure; **stop rule** when hybrid tight but live still drifts |

## Code packages worth knowing

- **`vr_teleop/`** — current VR path: `mapping`, `openvr_udp_sender`, `openvr_stream`, `vr_lab`, `viewer_arrows`, `vr_stream_target_viz`, `vr_stream_torso_compose`.
- **`legacy_vr_code/`** — frozen v1 + v2 reference; `udp_teleop_receiver.py` is still imported by `gear_wbc_stand` for `--udp-teleop`. See `vr_legacy_history.md`.
- **`tests/teleop_regression/`** — headless ONNX-walk regression (`table_box_walk_regression.py`), JSONL plotter (`plot_teleop_jsonl.py`), episode extractor (`extract_ik_episodes.py`).
- **`tests/test_stylish_diner_hands_mjcf_integration.py`** — stylish_diner + `--hands` YAML/XML resolve, runtime load, no kinematic grasp-assist regression, coarse `target_block` invariants; **not** a grasp-quality test (use teleop/VR for that). See `stylish_diner_keyboard_teleop.md`.
- **`policy_ab_dump.py` + `tests/test_policy_ab_dump.py`** — GR00T policy I/O dump tooling (`dump_policy_roundtrip_pack`); hooked into `--policy-ab-dump DIR` on both `Isaac-GR00T/gr00t/eval/rollout_policy.py` (oracle) and `scripts/run_gr00t_stylish_diner_inference.py` (subject). Protocol / interpretation matrix: **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md) §9**.
- **`scripts/run_policy_ab_dump.py`** — one-command A/B harness. Probes server, runs both clients with `--policy-ab-dump`, loads dumps, prints colourised per-key max-|diff| table, exits non-zero on `state.*` / `action.*` regression. `--diff-only` re-prints diff for an existing run-id. Protocol: **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md) §9**.
- **`scripts/hybrid_policy_get_action.py`** — loads two `policy_ab_dump` roots, builds `minimal` / `hybrid_ego` / `rollout` obs for `PolicyClient.get_action`, prints action diff vs dumped refs (vision causality at step 0). See **`inference_obs_one_pass.md`**.
- **`tests/test_runtime_leg_default_pose.py`** — pins the Phase 4a + 4a-bis leg-init contract: `GearWBCRuntime.__init__` AND `GearWBCRuntime.reset()` both write YAML `default_angles` into `qpos[legs+waist]` before the relevant `mj_forward`. Includes end-to-end pin through `Gr00tObservationBuilder.build()` (`state.{left_leg,right_leg,waist}` must equal standing crouch post-`env.reset()`). Contract / Bug #8 archaeology: **[`ACTIVE_gr00t_inference_wiring.md`](ACTIVE_gr00t_inference_wiring.md) §8**.

## Operator quick-reference (current best teleop path)

```
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
  play_g1_gear_wbc.py --vr-teleop --vr-ik on --vr-ik-anchor torso \
  --vr-head-orient yaw_only --debug-vr-stream --hands
```

Defaults already tuned: `--vr-receiver-yaw-deg=-90`, `--debug-vr-stream-torso-z=full`, `--debug-vr-stream-calib-orient=yaw_only`; live VR re-validation passed in May 2026. See `activeContext.md` for the current task and other validated commands.
