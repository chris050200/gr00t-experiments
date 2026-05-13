#!/usr/bin/env python3
"""CLI: preview a GLB in MuJoCo. Implementation lives in ``scenebuilder.glb_pipeline``."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scenebuilder.gl_guard import install as _install_gl_guard

_install_gl_guard()

import mujoco
import numpy as np

from scenebuilder.glb_pipeline import (
    build_preview_spec,
    center_scale_meshes,
    export_part_bundle,
    prepare_parts_from_glb,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a GLB/GLTF asset in MuJoCo.")
    parser.add_argument("glb_path", type=Path, help="Path to .glb or .gltf")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale after centering (default: 1)")
    parser.add_argument(
        "--up-axis",
        choices=("y", "z"),
        default="y",
        help="Source asset up-axis (default: y for glTF).",
    )
    parser.add_argument(
        "--flip-x-180",
        action="store_true",
        help="Rotate imported mesh by 180 deg around +X (fix upside-down cases).",
    )
    parser.add_argument(
        "--thicken-eps",
        type=float,
        default=0.0,
        help="Relative jitter per part to avoid qhull issues on thin/open meshes (default: 0).",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Disable textured OBJ export; always use STL + flat RGBA.",
    )
    parser.add_argument("--headless", action="store_true", help="Compile + mj_forward only (no viewer).")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print why any part fell back to STL (texture path debugging).",
    )
    parser.add_argument(
        "--keep-mesh-dir",
        type=Path,
        default=None,
        help="If set, copy the entire export directory tree here for inspection.",
    )
    args = parser.parse_args()

    glb_path = args.glb_path.expanduser().resolve()
    if not glb_path.is_file():
        raise SystemExit(f"File not found: {glb_path}")

    parts = prepare_parts_from_glb(
        glb_path,
        up_axis=args.up_axis,
        flip_x_180=args.flip_x_180,
    )
    center_scale_meshes(parts, args.scale, args.thicken_eps)

    tri_count = int(sum(len(np.asarray(p.mesh.faces)) for p in parts))
    vert_count = int(sum(len(np.asarray(p.mesh.vertices)) for p in parts))
    print(f"Expanded GLB: {len(parts)} parts, {vert_count} vertices, {tri_count} triangles")

    with tempfile.TemporaryDirectory(prefix="glb_mujoco_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        bundles = []
        tex_parts = 0
        for i, p in enumerate(parts):
            b = export_part_bundle(
                i,
                p.mesh,
                tmpdir_path,
                use_texture=not args.no_texture,
                verbose=args.verbose,
            )
            bundles.append(b)
            if b.mode == "obj":
                tex_parts += 1
        print(f"Export: {tex_parts} textured OBJ part(s), {len(bundles) - tex_parts} STL fallback part(s)")

        if args.keep_mesh_dir is not None:
            args.keep_mesh_dir.mkdir(parents=True, exist_ok=True)
            dest = args.keep_mesh_dir.resolve()
            shutil.copytree(tmpdir_path, dest, dirs_exist_ok=True)
            print(f"Wrote export tree to: {dest}")

        spec = build_preview_spec(bundles)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        if args.headless:
            print("Headless compile OK (mj_forward succeeded).")
            return

        from mujoco import viewer as mj_viewer

        print("Opening MuJoCo viewer (close window to exit).")
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
