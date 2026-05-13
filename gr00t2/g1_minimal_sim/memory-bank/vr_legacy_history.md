# VR teleop legacy history (pointer)

Both flavours of legacy VR plumbing are kept under `legacy_vr_code/` for reference. The active VR path lives in **`vr_teleop/`** (`mapping`, `openvr_udp_sender`, `openvr_stream`, `vr_lab`, `viewer_arrows`, `vr_stream_target_viz`, `vr_stream_torso_compose`). For the live VR debug overlay + IK compose, see **`vr_stream_debug_overlays.md`**.

## v1 — UDP `--udp-teleop` (still wired at runtime)

Despite the "legacy" label, this path is still imported by `gear_wbc_stand` (`legacy_vr_code.udp_teleop_receiver.start_udp_receiver_for_runtime` + `apply_udp_packet_to_control_dict`). Decide explicitly whether to retire or rename in a future cleanup phase (see `activeContext.md` "Future cleanups").

- **Sim:** `play_g1_gear_wbc.py --udp-teleop 0.0.0.0:5005` (or `spawn_g1_floor.py --stand --udp-teleop ...`). Allowed with `--headless`. **Not** combinable with `--vr-teleop` (different protocol).
- **VR PC (from `g1_minimal_sim`):** `python -m legacy_vr_code.openvr_teleop_client --host <sim-ip> --port 5005`.
- **JSON v1.** Optional `palm_frame: hmd_relative` vs `world` per packet; `--debug-ee-targets` + `teleop_target_viz.py` stay in the main tree.
- **Tests still pinning this path:** `tests/test_udp_teleop_bind.py`, `tests/test_hmd_relative_compose.py`.

## v2 — `legacy_vr_code.vr_teleop_test_streaming` (debug viewer only)

Reference UDP JSON v2 send/receive with a MuJoCo viewer. **Not** wired into `control_dict`; superseded by `vr_teleop/vr_lab.py` for new bring-up.

- **Sim:** `python -m legacy_vr_code.vr_teleop_test_streaming receive --bind 0.0.0.0:5006`
- **VR PC:** `python -m legacy_vr_code.vr_teleop_test_streaming send --host <sim-ip> --port 5006`
- **Tests still pinning this path:** `tests/test_vr_steamvr_mujoco_mapping.py`, `tests/test_vr_teleop_root_motion.py`, `tests/test_vr_pov_rotate.py`, `tests/test_vr_pov_pose.py` (last one needs MuJoCo).

## Replacement contract for v1 UDP teleop

When v1 is finally retired, the new sim-side receiver should:

1. Compose EE world targets from torso pose each step (no world-fixed hand targets while the base moves).
2. Use the same `vr_teleop/mapping.py` axis convention as `openvr_udp_sender` so both protocols share one OpenVR → MuJoCo path.
3. Plug into a future `CommandSource` abstraction so VR / keyboard / motion-primitive / policy sources compose by ordered priority instead of mutually-exclusive flags.
