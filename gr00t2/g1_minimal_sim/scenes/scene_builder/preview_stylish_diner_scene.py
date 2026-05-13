#!/usr/bin/env python3
"""Load ``scenes/stylish_diner_preview/stylish_diner_preview.xml`` and open the MuJoCo viewer.

On first use (or with ``--refresh-assets``), re-exports the booth GLB from ``experiments/imports/``
so ``assets/booth/`` exists. The MJCF references each mesh/texture **once**; three booth bodies
share those asset ids (efficient in MuJoCo; for three.js use one glTF + instancing — see scene README).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scenebuilder.gl_guard import install as _install_gl_guard

_install_gl_guard()

import mujoco


def _default_xml() -> Path:
    return _ROOT / "scenes" / "stylish_diner_preview" / "stylish_diner_preview.xml"


def _default_glb() -> Path:
    return _ROOT / "experiments" / "imports" / "2k_textured_diner_booth_w_table.glb"


def _default_booth_assets() -> Path:
    return _ROOT / "scenes" / "stylish_diner_preview" / "assets" / "booth"


# Default export scale: triple prior default (6 → 18). GLB often arrives undersized vs diner meters.
DEFAULT_BOOTH_SCALE = 18.0
# If vertex coordinates are in **feet**, multiply by this to get meters (1 ft = 0.3048 m).
FEET_TO_METERS = 0.3048


def _refresh_booth_assets(
    glb: Path,
    out: Path,
    scale: float,
    *,
    vertex_units: str,
) -> None:
    if not glb.is_file():
        raise SystemExit(f"Booth GLB not found: {glb}\nCopy a .glb into experiments/imports/ or pass --glb.")
    from scenebuilder.mjcf_library import export_glb_asset_library

    eff = float(scale)
    if vertex_units == "feet":
        eff *= FEET_TO_METERS
        print(f"vertex-units=feet: effective scale = {scale} × {FEET_TO_METERS} = {eff}", file=sys.stderr)

    export_glb_asset_library(
        glb,
        out,
        "sb_booth",
        flip_x_180=True,
        scale=eff,
        body_pos=(0.0, 0.0, 0.0),
        verbose=True,
    )
    print(f"Refreshed booth assets under {out.resolve()} (scale={eff})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview stylish_diner_preview MuJoCo scene.")
    ap.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Path to MJCF (default: scenes/stylish_diner_preview/stylish_diner_preview.xml)",
    )
    ap.add_argument("--headless", action="store_true", help="Compile + mj_forward only.")
    ap.add_argument(
        "--refresh-assets",
        action="store_true",
        help="Re-export booth GLB into assets/booth/ before loading.",
    )
    ap.add_argument(
        "--glb",
        type=Path,
        default=None,
        help="Booth GLB used when --refresh-assets (default: experiments/imports/...booth...glb)",
    )
    ap.add_argument(
        "--booth-scale",
        type=float,
        default=DEFAULT_BOOTH_SCALE,
        help=f"Uniform scale for booth export when refreshing (default: {DEFAULT_BOOTH_SCALE}, ~3× prior 6).",
    )
    ap.add_argument(
        "--vertex-units",
        choices=("meters", "feet"),
        default="meters",
        help="If 'feet', multiply booth export by 0.3048 (feet→meters) on top of --booth-scale.",
    )
    args = ap.parse_args()

    xml_path = (args.xml or _default_xml()).expanduser().resolve()
    booth_out = _default_booth_assets()
    glb_path = (args.glb or _default_glb()).expanduser().resolve()

    vu = args.vertex_units
    if args.refresh_assets:
        _refresh_booth_assets(glb_path, booth_out, args.booth_scale, vertex_units=vu)
    elif not (booth_out / "meshes").is_dir():
        print("Missing assets/booth/meshes; exporting booth GLB once...", file=sys.stderr)
        _refresh_booth_assets(glb_path, booth_out, args.booth_scale, vertex_units=vu)

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
