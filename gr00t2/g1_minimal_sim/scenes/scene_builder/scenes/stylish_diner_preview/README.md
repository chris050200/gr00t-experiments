# Stylish diner preview (scene_builder)

MuJoCo clone of the **static** `g1_minimal_sim/scenes/stylish_diner_1/stylish_diner.xml` shell.

The **big central table** is removed; **three GLB booths** sit in a row at the former table spot
(**Y ≈ 0.4**), spaced along **X** (`−2.90`, `0`, `2.90` m — ~12 in tighter than an earlier `±3.2` draft), each **`euler="0 0 90"`** so they read as a
diner bench row (tweak euler/pos in XML if the mesh faces the wrong way). **`target_block`** is
placed on the **middle** booth (~`z = 1.12`).

## Shared assets (efficient)

- **`stylish_diner_preview.xml`** lists each booth mesh/texture/material **once** in `<asset>`.
- **`booth_inst_0` … `_2`** are three `<body>` nodes whose geoms reference the **same**
  `mesh="sb_booth_mesh00"` … ids. MuJoCo keeps **one copy** of each mesh/texture in memory (similar
  idea to instancing).

For **three.js / web**: load **one** glTF (or one set of buffers), then use **`InstancedMesh`** (or
thin wrapper) for the three transforms — do **not** duplicate the whole GLB URL three times.

## Floor texture

The floor uses a **tileable wood-style albedo** PNG, `assets/booth/textures/stylish_diner_floor_planks.png`
(same file is copied under `g1_minimal_sim/scenes/stylish_diner_1/assets/floor/` for the G1 MJCFs).
Regenerate both from `g1_minimal_sim/scenes/scene_builder/tools/gen_stylish_diner_floor_texture.py` (default **2048×2048**, tileable noise + wavy grain; `--seed` to vary).

## Booth files on disk

`assets/booth/` is produced by `export_glb_mjcf.py` (or auto-run from the preview script if missing).

**Scale:** The GLB often imports undersized (unitless / inches confusion). Defaults use
**`--booth-scale 18`** (~3× the older default of 6). Tune with `--booth-scale N`.

If vertex coordinates are **feet** instead of meters, run:

```bash
python preview_stylish_diner_scene.py --refresh-assets --vertex-units feet --booth-scale 18
```

That applies **×0.3048** (ft→m) on top of `--booth-scale`.

Refresh after changing the GLB:

```bash
cd ../..   # gr00t2/g1_minimal_sim/scenes/scene_builder/
python preview_stylish_diner_scene.py --refresh-assets
```

## Preview

```bash
python preview_stylish_diner_scene.py
python preview_stylish_diner_scene.py --headless
```
