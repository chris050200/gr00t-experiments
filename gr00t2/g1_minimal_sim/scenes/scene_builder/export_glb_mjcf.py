#!/usr/bin/env python3
"""Export a GLB into a persistent folder + MJCF standalone snippet for merging into G1 scenes."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scenebuilder.gl_guard import install as _install_gl_guard

_install_gl_guard()

from scenebuilder.mjcf_library import export_glb_asset_library


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export GLB → meshes/, textures/, PREFIX_standalone.xml under --out-dir."
    )
    p.add_argument("glb_path", type=Path, help="Input .glb / .gltf")
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (created). Typically under gr00t2/g1_minimal_sim/scenes/scene_builder/out or g1_minimal_sim/scenes.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Unique MJCF name prefix (e.g. sd_boothglb_a). Sanitized to letters/digits/underscore.",
    )
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--up-axis", choices=("y", "z"), default="y")
    p.add_argument("--flip-x-180", action="store_true")
    p.add_argument("--thicken-eps", type=float, default=0.0)
    p.add_argument("--body-pos", type=str, default="0 0 0", help='Three floats, e.g. "1.2 3 0"')
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--install-to-g1-scenes",
        type=Path,
        default=None,
        help="If set, copy the finished folder into this directory (e.g. path to g1_minimal_sim/scenes).",
    )
    args = p.parse_args()

    glb = args.glb_path.expanduser().resolve()
    if not glb.is_file():
        raise SystemExit(f"Not found: {glb}")

    parts = [float(x) for x in args.body_pos.split()]
    if len(parts) != 3:
        raise SystemExit("--body-pos must be three floats")
    pos = (parts[0], parts[1], parts[2])

    out = export_glb_asset_library(
        glb,
        args.out_dir.expanduser().resolve(),
        args.prefix,
        up_axis=args.up_axis,
        flip_x_180=args.flip_x_180,
        scale=args.scale,
        thicken_eps=args.thicken_eps,
        body_pos=pos,
        verbose=args.verbose,
    )
    print(f"Wrote asset library: {out}")

    if args.install_to_g1_scenes is not None:
        dest_root = args.install_to_g1_scenes.expanduser().resolve()
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / out.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        print(f"Installed copy to: {dest}")


if __name__ == "__main__":
    main()
