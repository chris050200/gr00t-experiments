#!/usr/bin/env python3
"""Load ``scenes/mcdonalds_restaurant_preview/mcdonalds_restaurant_preview.xml`` in MuJoCo.

With ``--refresh-assets``, re-exports ``experiments/imports/mcdonald_restaurant.glb`` into
``assets/mcdonalds/``. Export uses scale ``0.01`` (GLB units treated as centimeters → meters) and
``--thicken-eps 1e-5`` so very thin / degenerate mesh parts compile (MuJoCo qhull / volume checks).
Exports apply ``--flip-x-180`` by default (180° about X: Y/Z negated — building flipped “upside down”).
Use ``--no-flip-x-180`` to disable.
The preview XML is regenerated from ``mcd_rest_standalone.xml`` (paths + floor + lights).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scenebuilder.gl_guard import install as _install_gl_guard

_install_gl_guard()

import mujoco


def _default_xml() -> Path:
    return _ROOT / "scenes" / "mcdonalds_restaurant_preview" / "mcdonalds_restaurant_preview.xml"


def _default_glb() -> Path:
    return _ROOT / "experiments" / "imports" / "mcdonald_restaurant.glb"


def _asset_out() -> Path:
    return _ROOT / "scenes" / "mcdonalds_restaurant_preview" / "assets" / "mcdonalds"


def _regenerate_preview_xml_from_standalone() -> None:
    standalone = _asset_out() / "mcd_rest_standalone.xml"
    out = _ROOT / "scenes" / "mcdonalds_restaurant_preview" / "mcdonalds_restaurant_preview.xml"
    if not standalone.is_file():
        raise SystemExit(f"Missing export: {standalone} (run export or --refresh-assets)")
    text = standalone.read_text(encoding="utf-8")
    text = text.replace('model="mcd_rest_library"', 'model="mcdonalds_restaurant_preview"')
    text = text.replace('meshdir="meshes"', 'meshdir="assets/mcdonalds/meshes"')
    text = text.replace('texturedir="textures"', 'texturedir="assets/mcdonalds/textures"')
    inject_head = """  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.75 0.75 0.72" ambient="0.38 0.36 0.34"/>
    <rgba haze="0.05 0.05 0.05 1"/>
  </visual>
"""
    text = text.replace(
        '<mujoco model="mcdonalds_restaurant_preview">\n  <compiler angle="radian" meshdir="assets/mcdonalds/meshes" texturedir="assets/mcdonalds/textures"/>',
        '<mujoco model="mcdonalds_restaurant_preview">\n  <compiler angle="radian" meshdir="assets/mcdonalds/meshes" texturedir="assets/mcdonalds/textures"/>'
        + "\n"
        + inject_head.rstrip(),
    )
    floor = """    <geom name="mcd_ground" type="plane" pos="0 0 0" size="60 60 0.01" rgba="0.18 0.19 0.22 1" contype="0" conaffinity="0"/>
    <light name="mcd_sun" pos="8 8 12" dir="-0.3 -0.3 -1" directional="true" diffuse="0.92 0.9 0.85" specular="0.35 0.35 0.32"/>
"""
    text = text.replace("  <worldbody>\n", "  <worldbody>\n" + floor)
    out.write_text(text, encoding="utf-8")


def _refresh_assets(
    glb: Path,
    *,
    scale: float,
    thicken_eps: float,
    flip_x_180: bool,
) -> None:
    if not glb.is_file():
        raise SystemExit(f"GLB not found: {glb}")
    out = _asset_out()
    cmd = [
        sys.executable,
        str(_ROOT / "export_glb_mjcf.py"),
        str(glb),
        "--out-dir",
        str(out),
        "--prefix",
        "mcd_rest",
        "--scale",
        str(scale),
        "--thicken-eps",
        str(thicken_eps),
        "--verbose",
    ]
    if flip_x_180:
        cmd.append("--flip-x-180")
    print("Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=str(_ROOT), check=True)
    _regenerate_preview_xml_from_standalone()


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview McDonald's restaurant GLB scene in MuJoCo.")
    ap.add_argument("--xml", type=Path, default=None, help="MJCF path (default: mcdonalds_restaurant_preview.xml)")
    ap.add_argument("--headless", action="store_true", help="Compile + mj_forward only.")
    ap.add_argument(
        "--refresh-assets",
        action="store_true",
        help="Re-export GLB into assets/mcdonalds/ and regenerate preview XML.",
    )
    ap.add_argument("--glb", type=Path, default=None, help="Source GLB when refreshing.")
    ap.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Uniform scale for export (default 0.01: cm → m).",
    )
    ap.add_argument(
        "--thicken-eps",
        type=float,
        default=1e-5,
        help="Vertex noise vs bbox diagonal; avoids degenerate mesh compile failures.",
    )
    ap.add_argument(
        "--no-flip-x-180",
        action="store_true",
        help="Disable 180° X-axis vertex flip (default: flip enabled).",
    )
    args = ap.parse_args()
    flip_x_180 = not args.no_flip_x_180

    xml_path = (args.xml or _default_xml()).expanduser().resolve()
    glb_path = (args.glb or _default_glb()).expanduser().resolve()

    if args.refresh_assets:
        _refresh_assets(
            glb_path,
            scale=args.scale,
            thicken_eps=args.thicken_eps,
            flip_x_180=flip_x_180,
        )
    elif not (_asset_out() / "mcd_rest_standalone.xml").is_file():
        print("Missing assets; exporting GLB once...", file=sys.stderr)
        _refresh_assets(
            glb_path,
            scale=args.scale,
            thicken_eps=args.thicken_eps,
            flip_x_180=flip_x_180,
        )
    elif not xml_path.is_file():
        _regenerate_preview_xml_from_standalone()

    if not xml_path.is_file():
        raise SystemExit(f"MJCF not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.headless:
        print(f"Headless OK: {xml_path} (nq={model.nq}, ngeom={model.ngeom})")
        return

    from mujoco import viewer as mj_viewer

    print(f"Opening viewer for {xml_path} (close window to exit).")
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
