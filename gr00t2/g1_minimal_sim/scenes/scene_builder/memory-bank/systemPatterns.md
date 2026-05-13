# System patterns — scene_builder

## Package vs experiments

| `scenebuilder/` | Importable modules |
|-----------------|-------------------|
| `glb_pipeline.py` | `prepare_parts_from_glb`, `export_part_bundle`, `build_preview_spec`, … |
| `mjcf_library.py` | `export_glb_asset_library` — writes disk + `{prefix}_standalone.xml` |
| `gl_guard.py` | `install()` — MuJoCo GL import guard |

| `experiments/` | Runnable CLIs + `imports/` drop folder |

## MJCF merge contract

Exports emit **`meshdir="meshes"`** and **`texturedir="textures"`**. Mesh and texture `file=`
attributes are **filenames only** (paths relative to those dirs).

Standalone XML compiles alone for validation. Merging into G1 requires **globally unique**
`prefix` per instance (e.g. `sd_boothglb_0`, `sd_boothglb_1`).

Geoms default to **visual-only** (`contype=0`, `conaffinity=0`). Add collision proxies in G1 if needed.
