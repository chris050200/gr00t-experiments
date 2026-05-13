# Merge `mcd_rest` into a G1 stylish_diner MJCF

Exported from scene_builder (see repo `scene_builder/memory-bank/`).

1. Copy this entire folder under `g1_minimal_sim/scenes/` (or symlink), e.g.
   `g1_minimal_sim/scenes/mcd_rest_export/`.
2. In `g1_gear_wbc_stylish_diner.xml` (or hands variant), inside the top-level `<asset>` block,
   paste the `<mesh>`, `<texture>`, and `<material>` lines from `mcd_rest_standalone.xml`
   (only the contents between `<asset>` and `</asset>`), **or** use `<include file="..."/>` if your
   MuJoCo version resolves paths relative to the including file (test compile).
3. Inside `<worldbody>`, paste the `<body name="mcd_rest_root" ...>...</body>` subtree (adjust `pos`
   for multiple instances; **rename** `mcd_rest` for copy 2 and 3 so asset names stay unique).
4. Collision: these geoms are visual-only. Add box/cylinder collision geoms if the robot should
   physically interact.

Compile check::

   mujoco.MjModel.from_xml_path("mcd_rest_standalone.xml")  # run from this directory
