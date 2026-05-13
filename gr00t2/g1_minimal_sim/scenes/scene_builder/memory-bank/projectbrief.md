# Project brief — scene_builder

## Goal

Provide a **dedicated workspace** under GR00T2 for turning arbitrary **glTF / USD** props into
**MuJoCo-ready meshes + textures + MJCF snippets**, then dropping outputs into
**`g1_minimal_sim/scenes/`** for merged Gear WBC scenes (e.g. stylish diner).

## Success criteria

- **Preview** GLB/USD in MuJoCo without touching `play_g1_gear_wbc` plumbing.
- **Export** a repeatable folder layout: `meshes/`, `textures/`, `{prefix}_standalone.xml`, merge README.
- **Document** how to paste or include assets into `g1_gear_wbc_stylish_diner*.xml`.
- **Regression tests** that run headlessly (no viewer).

## Non-goals

- Replacing the full diner MJCF generator or Gear WBC runtime (those stay in `g1_minimal_sim`).
- Full PBR (normal/metal/rough) in MuJoCo — diffuse-only unless extended later.
