# VR stream debug overlays (`play_g1_gear_wbc.py`)

## Current behavior (streamlined)

- **Flags:** `--vr-teleop --debug-vr-stream` (viewer only).
- **Default frame:** torso-mounted compose is now the default overlay path.
- **Code path:** `vr_teleop/vr_stream_target_viz.py` + `vr_teleop/vr_stream_torso_compose.py`.
- **Calibration:** first valid streamed HMD after `env.reset()` defines \(^{S}\!T_{hmd,0}\) in `VrTorsoVizCalib`.
- **Shared runtime calibration:** `GearWBCRuntime` now owns one `VrTorsoVizCalib` reused by both debug arrows and `--vr-ik`, and reset on env/runtime reset.
- **Defaults (user-validated):**
  - `--vr-receiver-yaw-deg=-90`
  - `--debug-vr-stream-torso-z=full`
  - `--debug-vr-stream-calib-orient=yaw_only`
  - `--vr-head-orient=yaw_only` (for non-legacy `--vr-ik-anchor`)
- **Z policy:** `--debug-vr-stream-torso-z` = `full` (default) / `fixed_world` / `frozen_calib`; `--debug-vr-stream-z-offset` is used for world Z in `fixed_world`.

## Frame math (authoritative)

Let \(^{S}\!T_{dev}\) be stream pose for HMD/controller and \(^{W}\!T_{torso}(t)\) be `torso_link` world pose from MuJoCo.

\[
^{W}\!T_{dev}^{draw}(t) \;=\; ^{W}\!T_{torso}(t)\; \big(^{S}\!T_{hmd,0}\big)^{-1}\; ^{S}\!T_{dev}(t)
\]

This guarantees arrows translate and rotate with the robot's moving local torso frame.

Receiver-side yaw correction is applied by building a VR torso orientation basis
\(^{W}\!R_{torso}^{vr} = R_z(\theta)\,^{W}\!R_{torso}\) (position unchanged), then composing as above.

Calibration orientation mode:

- `full`: store full HMD quaternion in \(^{S}\!T_{hmd,0}\).
- `yaw_only` (default): store yaw-only quaternion (drops startup pitch/roll coupling).

Current-HMD orientation mode for non-legacy head-relative anchors (`torso` / `pelvis`):

- `full`: use full current HMD quaternion in `inv(T_hmd)`.
- `yaw_only` (recommended): use yaw-only current HMD quaternion in `inv(T_hmd)` so world Z stays shared and hand translation does not depend on headset pitch/roll.

## Validation

- Unit tests: `tests/test_vr_stream_torso_compose.py` (numpy-only).
- Manual check: walk + pivot with `--debug-vr-stream`; markers should stay co-moving with the torso.

## Troubleshooting quick map

- **Constant heading offset (~90°):** tune `--vr-receiver-yaw-deg`.
- **No visible vertical movement:** use `--debug-vr-stream-torso-z full` (avoid `fixed_world` for pose debugging).
- **Vertical movement appears as arc (up/down coupled with fwd/back):** use `--debug-vr-stream-calib-orient yaw_only`.
- **Hand motion feels correct only when physically tilting HMD:** use `--vr-head-orient yaw_only` (with `--vr-ik-anchor torso` or `pelvis`).
- **Drift/warp while robot walks/turns:** verify only one rigid correction path is active; avoid ad-hoc per-device/per-axis offsets.

## Known-good command (current setup)

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --vr-teleop --debug-vr-stream
```

## Current best teleop command (user-validated)

```bash
../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python play_g1_gear_wbc.py --vr-teleop --vr-ik on --vr-ik-anchor torso --vr-head-orient yaw_only --debug-vr-stream --hands
```

May 2026 live headset re-validation confirmed this path is fully functional after the `arm_ik_v2` `mj_jac` point-frame fix: the previous wrist / ~180° spin issue appears fixed and posing is very stable. Treat future VR work as ergonomics / ease-of-control tuning unless new stability symptoms appear.

## Phase B status: VR poses → IK targets

1. **Landed:** stream source per arm (`left`, `right`) composes to world with the same torso chain used by overlay.
2. **Landed:** composed world wrist pose converts to torso-frame target via `ee_pose_ref_from_world(...)`.
3. **Landed:** torso-frame targets write to `control_dict["ee_*"]` under **`--vr-ik on`** (position + palm quaternion from stream).
4. **Landed:** stale stream policy resets `loco_cmd`; IK stream path currently resets EE homes when stale and keyboard teleop is not active.
5. **Next hardening focus:** optional ergonomics / ease-of-control tuning and optional **`arm_ik_posture`** merge for twistiness; keep this compose contract unchanged.

The debug overlay and IK path should share the same calibration and compose helper to avoid frame drift.

## Consolidation checklist (next)

1. Keep one compose/calibration helper for both overlay and IK (already landed; avoid drift/duplication).
2. Add per-hand distance clamp around torso-frame anchor before IK solve.
3. Add trigger-gated tracking enable/disable per hand.
4. Tune IK/PD aggressiveness for unreachable targets.
5. Remove duplicate/legacy frame knobs once behavior is stable.
