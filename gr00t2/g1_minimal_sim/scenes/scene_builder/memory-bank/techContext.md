# Tech context — scene_builder

## Location

Repository root: **`gr00t2/g1_minimal_sim/scenes/scene_builder/`** (sibling to **`gr00t2/g1_minimal_sim/`**).

## Dependencies

- **MuJoCo** Python bindings (`import mujoco`)
- **trimesh**, **pillow** for GLB
- **pxr** (OpenUSD) optional — USD preview only

## Python path

Scripts assume cwd **`scene_builder/`** or insert **`scene_builder/`** on `sys.path` so
`import scenebuilder` resolves.

## GL backend

`scenebuilder.gl_guard.install()` runs before `import mujoco` in CLIs. Avoid
`MUJOCO_GL=osmesa` in broken conda envs; prefer **`unset MUJOCO_GL`** or **`MUJOCO_GL=egl`**
headless.

## Tests

```bash
cd gr00t2/g1_minimal_sim/scenes/scene_builder && python -m pytest tests/ -v
```

`pyproject.toml` sets `pythonpath = ["."]` for pytest.

## Export install target

`export_glb_mjcf.py --install-to-g1-scenes` copies the output folder to:

`g1_minimal_sim/scenes/<out_dir_name>/`
