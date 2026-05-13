# Progress — scene_builder

## Working

- **`experiments/visualize_glb_mujoco.py`** — preview (imports `scenebuilder.glb_pipeline`).
- **`experiments/visualize_usdz_mujoco.py`** — USD mesh preview (shared `gl_guard`).
- **`export_glb_mjcf.py`** — export library folder + standalone MJCF + optional install into G1 `scenes/<out_dir_name>/`.
- **`tests/`** — GLB pipeline smoke + standalone MJCF compile test.

## Known limits

- Diffuse texture only (no normal maps).
- Exported geoms are visual-only unless edited in G1.
