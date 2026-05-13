# Scene builder — memory bank

Canonical docs for **`gr00t2/g1_minimal_sim/scenes/scene_builder/`** (GLB/USD → MuJoCo assets, MJCF export for G1 scenes).

## Core

| File | Purpose |
|------|---------|
| [projectbrief.md](projectbrief.md) | Scope: preview tools, library export, hand-off to `g1_minimal_sim/scenes` |
| [techContext.md](techContext.md) | Paths, commands, pytest, dependencies |
| [systemPatterns.md](systemPatterns.md) | Package layout, MJCF merge pattern |
| [activeContext.md](activeContext.md) | Current focus |
| [progress.md](progress.md) | Status |

## Operator commands

Preview GLB (from `scene_builder/`):

```bash
python experiments/visualize_glb_mujoco.py experiments/imports/2k_textured_diner_booth_w_table.glb --flip-x-180
```

Stylish diner + **3× instanced** booths (shared `sb_booth_*` assets):

```bash
python preview_stylish_diner_scene.py
```
See `scenes/stylish_diner_preview/README.md`.

Export to library + install copy under G1 scenes:

```bash
python export_glb_mjcf.py experiments/imports/2k_textured_diner_booth_w_table.glb \
  --out-dir ./out/booth_2k --prefix sd_boothglb_0 --flip-x-180 \
  --install-to-g1-scenes ../g1_minimal_sim/scenes
```

Tests:

```bash
python -m pytest tests/ -v
```
