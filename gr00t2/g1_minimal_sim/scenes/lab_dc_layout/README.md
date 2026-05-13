# Lab DC world (standalone MJCF)

**`lab_dc_world.xml`** is a **no-robot** scene that stitches together the same **on-disk assets** RoboCasa uses for **`LabArena` + `LMPnPAppleToPlate`**:

- **Arena:** `arenas/gear_lab/gear_lab.xml` (floor + wall + light + cameras + visual block)
- **Two lab tables:** `objects/omniverse/locomanip/lab_table/model.xml` (meshes + colliders ×2, different `pos`)
- **Apple:** `objects/omniverse/locomanip/apple_0/` (visual + collision meshes, **freejoint**)
- **Plate:** `objects/omniverse/locomanip/plate_1/` (static body on the target table)

All `file="..."` paths are **relative to this directory** and point into the sibling **`Isaac-GR00T/.../robocasa/models/assets/`** tree. The **G1 merged** scene under `../table_pnp_apple/` uses different path bases for **meshes** (relative to G1 `meshdir`) vs **textures** (relative to the MJCF file); see that folder’s `README.md`.

## Regenerate

```bash
python3 g1_minimal_sim/scenes/lab_dc_layout/gen_lab_dc_world_xml.py
```

## View

```bash
cd g1_minimal_sim/scenes/lab_dc_layout
python -m mujoco.viewer lab_dc_world.xml
```

## Merged G1 scene

The same layout (with `manip_apple` / `manip_plate` naming) is merged into **`../table_pnp_apple/g1_gear_wbc_table_pnp_apple.xml`** and **`g1_gear_wbc_hands_table_pnp_apple.xml`** for `run_gr00t_inference --scene table_pnp_apple`.

## Placement note

Table positions **`(0.5, 0, 0)`** and **`(0.5, 1.2, 0)`** match the **nominal** `reference_pos` from `LMPnPBottle._get_objects` / `table_target`. Apple / plate **xyz** are **mid-sampler approximations**; tune after visual compare with Side A video or by logging one RoboCasa `reset()` pose.
