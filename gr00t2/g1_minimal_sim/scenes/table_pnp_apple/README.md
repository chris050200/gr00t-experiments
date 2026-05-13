# `table_pnp_apple` (B1 — RoboCasa-style lab layout + G1)

MJCF merges the **G1 29-DoF** model with the **RoboCasa-derived** scene from `../lab_dc_layout/` (same assets as `LabArena` + `LMPnPAppleToPlate` intent):

- **Arena:** `gear_lab` floor + back wall + fixed cameras (`frontview`, `birdview`, `agentview`, `sideview`).
- **Two `lab_table` instances:** `dc_table_pick` @ `(0.5, 0, 0)` and `dc_table_target` @ `(0.5, 1.2, 0)` (yaw π/2).
- **`manip_apple`:** `apple_0` meshes + collision + `freejoint` (`manip_apple_free`).
- **`manip_plate`:** `plate_1` meshes + collision, static on the target table.

**Path rules (important):** `compiler meshdir` points at vendored G1 STLs. **Robocasa `<mesh file="...">` paths** are therefore **relative to `meshdir`** (`../../../../../dexmg/gr00trobocasa/robocasa/models/assets/...`). **`<texture file="...">`** paths are **relative to this MJCF directory** (`../../../Isaac-GR00T/.../robocasa/models/assets/...`). The checkout must keep **`gr00t2/g1_minimal_sim`** and **`gr00t2/Isaac-GR00T`** as siblings.

ONNX policies still resolve from the default G1 bundle via `policy_bundle_dir: "__DEFAULT_G1_RESOURCES__"` in the yaml.

**Scene id:** `scene="table_pnp_apple"` for `G1GearWBCEnv` / `run_gr00t_inference.py --scene table_pnp_apple`.

**Hands:** `g1_gear_wbc_hands_table_pnp_apple.xml` uses the same world layout block as the no-hands MJCF.

## View (no policy / WBC)

From this directory, either:

```bash
python -m mujoco.viewer g1_gear_wbc_table_pnp_apple.xml
```

or the small helper (steps physics so the apple can move):

```bash
python3 view_table_pnp_apple.py
python3 view_table_pnp_apple.py --hands
```
