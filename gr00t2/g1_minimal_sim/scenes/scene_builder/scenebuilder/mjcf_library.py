"""Copy exported GLB parts to a stable folder and emit MJCF snippets for merging into G1 scenes."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from scenebuilder.glb_pipeline import (
    ExportBundle,
    center_scale_meshes,
    export_part_bundle,
    prepare_parts_from_glb,
)


def sanitize_prefix(prefix: str) -> str:
    p = re.sub(r"[^0-9a-zA-Z_]", "_", prefix.strip())
    return p or "asset"


def export_glb_asset_library(
    glb_path: Path,
    out_dir: Path,
    prefix: str,
    *,
    up_axis: str = "y",
    flip_x_180: bool = False,
    scale: float = 1.0,
    thicken_eps: float = 0.0,
    body_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    verbose: bool = False,
) -> Path:
    """Export GLB parts under ``out_dir`` and write MJCF merge snippets.

    Layout::

        out_dir/
          meshes/   PREFIX_m00.obj | .stl
          textures/ PREFIX_t00.png (when textured)
          PREFIX_standalone.xml   (minimal mujoco — compile check)
          README_MERGE.txt        (how to paste into stylish_diner)

    Geoms are **visual-only** (``contype=0`` ``conaffinity=0``). Add collision proxies separately.
    """
    px = sanitize_prefix(prefix)
    glb_path = glb_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "meshes"
    tex_dir = out_dir / "textures"
    mesh_dir.mkdir(exist_ok=True)
    tex_dir.mkdir(exist_ok=True)

    parts = prepare_parts_from_glb(glb_path, up_axis=up_axis, flip_x_180=flip_x_180)
    center_scale_meshes(parts, scale, thicken_eps)

    bundles: list[ExportBundle] = []
    tex_rel_by_part: list[str | None] = []
    with tempfile.TemporaryDirectory(prefix="glb_export_") as tmp:
        root = Path(tmp)
        for i, p in enumerate(parts):
            b = export_part_bundle(i, p.mesh, root, use_texture=True, verbose=verbose)
            bundles.append(b)

        for i, b in enumerate(bundles):
            suf = b.mesh_path.suffix.lower()
            dst_mesh = mesh_dir / f"{px}_m{i:02d}{suf}"
            shutil.copy2(b.mesh_path, dst_mesh)
            rel_tex: str | None = None
            if b.mode == "obj" and b.texture_path is not None:
                ext = b.texture_path.suffix.lower() or ".png"
                dst_tex = tex_dir / f"{px}_t{i:02d}{ext}"
                shutil.copy2(b.texture_path, dst_tex)
                rel_tex = f"textures/{px}_t{i:02d}{ext}"
            tex_rel_by_part.append(rel_tex)

    # --- MJCF generation (paths relative to this XML file; meshdir/texturedir subdirs)
    asset_lines: list[str] = []
    body_geoms: list[str] = []
    px_esc = escape(px)

    for i, b in enumerate(bundles):
        mname = f"{px}_mesh{i:02d}"
        # Paths relative to meshdir= / texturedir= (not including those folder names).
        mesh_file = f"{px}_m{i:02d}{b.mesh_path.suffix.lower()}"
        asset_lines.append(f'    <mesh name="{mname}" file="{escape(mesh_file)}"/>')

        if b.mode == "obj" and tex_rel_by_part[i] is not None:
            tex_file = Path(tex_rel_by_part[i]).name
            tname = f"{px}_tex{i:02d}"
            matname = f"{px}_mat{i:02d}"
            asset_lines.append(
                f'    <texture name="{tname}" type="2d" file="{escape(tex_file)}"/>'
            )
            asset_lines.append(
                f'    <material name="{matname}" texture="{tname}" '
                f'reflectance="0.5" specular="0.5" shininess="0.5"/>'
            )
            body_geoms.append(
                f'      <geom name="{px_esc}_g{i:02d}" type="mesh" mesh="{mname}" '
                f'material="{matname}" contype="0" conaffinity="0"/>'
            )
        else:
            r, g, b_, a = b.rgba
            body_geoms.append(
                f'      <geom name="{px_esc}_g{i:02d}" type="mesh" mesh="{mname}" '
                f'rgba="{r:.6g} {g:.6g} {b_:.6g} {a:.6g}" contype="0" conaffinity="0"/>'
            )

    bx, by, bz = body_pos
    standalone = f"""<mujoco model="{px_esc}_library">
  <compiler angle="radian" meshdir="meshes" texturedir="textures"/>
  <asset>
{chr(10).join(asset_lines)}
  </asset>
  <worldbody>
    <body name="{px_esc}_root" pos="{bx} {by} {bz}">
{chr(10).join(body_geoms)}
    </body>
  </worldbody>
</mujoco>
"""
    (out_dir / f"{px}_standalone.xml").write_text(standalone, encoding="utf-8")

    merge_readme = f"""# Merge `{px}` into a G1 stylish_diner MJCF

Exported from scene_builder (see repo `scene_builder/memory-bank/`).

1. Copy this entire folder under `g1_minimal_sim/scenes/` (or symlink), e.g.
   `g1_minimal_sim/scenes/{px}_export/`.
2. In `g1_gear_wbc_stylish_diner.xml` (or hands variant), inside the top-level `<asset>` block,
   paste the `<mesh>`, `<texture>`, and `<material>` lines from `{px}_standalone.xml`
   (only the contents between `<asset>` and `</asset>`), **or** use `<include file="..."/>` if your
   MuJoCo version resolves paths relative to the including file (test compile).
3. Inside `<worldbody>`, paste the `<body name="{px}_root" ...>...</body>` subtree (adjust `pos`
   for multiple instances; **rename** `{px}` for copy 2 and 3 so asset names stay unique).
4. Collision: these geoms are visual-only. Add box/cylinder collision geoms if the robot should
   physically interact.

Compile check::

   mujoco.MjModel.from_xml_path("{px}_standalone.xml")  # run from this directory
"""
    (out_dir / "README_MERGE.txt").write_text(merge_readme, encoding="utf-8")

    return out_dir
