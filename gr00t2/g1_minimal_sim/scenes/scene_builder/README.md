# Scene builder (GR00T2)

Tools to **preview** and **export** 3D assets (GLB, USDZ) for MuJoCo, then merge snippets into
`g1_minimal_sim/scenes/` (see `memory-bank/`).

## Layout

| Path | Role |
|------|------|
| `scenebuilder/` | Python package: `glb_pipeline`, `mjcf_library`, `gl_guard` |
| `experiments/` | CLI scripts: `visualize_glb_mujoco.py`, `visualize_usdz_mujoco.py`, `imports/` |
| `export_glb_mjcf.py` | Export GLB → folder + `{prefix}_standalone.xml` + `README_MERGE.txt` |
| `tests/` | Pytest (requires `trimesh`, `pillow`, `mujoco`) |

## Commands

From this directory (`gr00t2/g1_minimal_sim/scenes/scene_builder/`):

```bash
# Preview booth (paths relative to scene_builder/)
python experiments/visualize_glb_mujoco.py experiments/imports/2k_textured_diner_booth_w_table.glb --flip-x-180

# Stylish diner shell + 3× shared booth instances (see scenes/stylish_diner_preview/README.md)
python preview_stylish_diner_scene.py
python preview_stylish_diner_scene.py --headless

# Export library + MJCF for merging into stylish_diner
python export_glb_mjcf.py experiments/imports/2k_textured_diner_booth_w_table.glb \
  --out-dir ./out/booth_2k --prefix sd_boothglb_0 --flip-x-180 \
  --install-to-g1-scenes ../
```

## Tests

```bash
python -m pytest tests/ -v
```

PYTHONPATH is set via `pyproject.toml` for pytest.
