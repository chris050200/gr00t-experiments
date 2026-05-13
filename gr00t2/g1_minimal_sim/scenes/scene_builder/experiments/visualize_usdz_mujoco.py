#!/usr/bin/env python3
"""Load a ``.usdz`` (or ``.usd`` / ``.usda``) asset with OpenUSD and preview it in MuJoCo.

MuJoCo does not read USD directly. This tool walks ``UsdGeom.Mesh`` prims, merges all
triangles into a temporary binary STL, then builds a minimal ``MjSpec`` scene (floor +
mesh geom, visual-only contact) and opens the interactive viewer.

**Dependency:** Python bindings for OpenUSD (import name ``pxr``). Typical installs::

    conda install -c conda-forge openusd

If ``pxr`` is missing, the script exits with a short install hint.

**Coordinate frames:** USD stages are usually **Y-up**. MuJoCo worlds are **Z-up**. By default
this script applies a fixed basis change (same idea as swapping Y/Z with a sign flip so +Z is up).

**Flat / sheet geometry:** MuJoCo compiles meshes with qhull; perfectly flat vertex sets can
fail with ``qhull error``. Use ``--thicken-eps`` (default small jitter) to break degeneracy.

Example (from ``gr00t2/g1_minimal_sim/scenes/scene_builder/``)::

    python experiments/visualize_usdz_mujoco.py /path/to/model.usdz

Headless smoke (compiles model, no window; needs ``MUJOCO_GL`` set appropriately on servers)::

    MUJOCO_GL=egl python experiments/visualize_usdz_mujoco.py model.usdz --headless

**Import-time GL:** ``MUJOCO_GL=osmesa`` often breaks ``import mujoco`` in conda. Use
``scenebuilder.gl_guard`` (via shared package) to fall back to **glfw** unless
``MUJOCO_USE_OSMESA=1``. Prefer **egl** for headless GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scenebuilder.gl_guard import install as _install_gl_guard

_install_gl_guard()

import argparse
import os
import struct
import tempfile

import mujoco
import numpy as np


def _mj_mesh_inertia_exact() -> mujoco.mjtMeshInertia:
    """``mjtMeshInertia`` renamed ``mjINERTIA_*`` → ``mjMESH_INERTIA_*`` in MuJoCo 3.3+."""
    mi = mujoco.mjtMeshInertia
    for name in ("mjMESH_INERTIA_EXACT", "mjINERTIA_EXACT"):
        v = getattr(mi, name, None)
        if v is not None:
            return v
    for name in ("mjMESH_INERTIA_LEGACY", "mjINERTIA_LEGACY"):
        v = getattr(mi, name, None)
        if v is not None:
            return v
    raise RuntimeError("mujoco.mjtMeshInertia: no compatible mesh inertia enum found")


def _require_usd():
    try:
        from pxr import Gf, Usd, UsdGeom  # type: ignore[import-not-found]

        return Gf, Usd, UsdGeom
    except ImportError:
        print(
            "Missing OpenUSD Python module ``pxr``.\n"
            "Install OpenUSD for Python (then re-run), for example:\n"
            "  conda install -c conda-forge openusd\n"
            "Some studios ship ``pxr`` with their USD toolchain; ensure your PYTHONPATH "
            "includes the USD ``lib/python`` directory.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def _write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write binary STL (little-endian). vertices: (N,3) float32; faces: (T,3) uint32."""
    nv = np.asarray(vertices, dtype=np.float32)
    nf = np.asarray(faces, dtype=np.uint32)
    with path.open("wb") as f:
        f.write(b" " * 80)
        f.write(struct.pack("<I", len(nf)))
        for tri in nf:
            v0, v1, v2 = nv[int(tri[0])], nv[int(tri[1])], nv[int(tri[2])]
            e1, e2 = v1 - v0, v2 - v0
            n = np.cross(e1, e2)
            ln = float(np.linalg.norm(n))
            if ln > 1e-20:
                n = (n / ln).astype(np.float32)
            else:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            f.write(n.tobytes())
            f.write(v0.tobytes())
            f.write(v1.tobytes())
            f.write(v2.tobytes())
            f.write(struct.pack("<H", 0))


def _usd_y_up_to_mujoco_z_up(points: np.ndarray) -> np.ndarray:
    """Map common Y-up USD coordinates into MuJoCo Z-up (right-handed)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    # x' = x, y' = z, z' = -y  (rotate -90 deg about +X)
    out = np.empty_like(p)
    out[:, 0] = p[:, 0]
    out[:, 1] = p[:, 2]
    out[:, 2] = -p[:, 1]
    return out


def _transform_points_local_to_world(
    Gf, Usd, UsdGeom, prim, points_local: np.ndarray
) -> np.ndarray:
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    out = np.empty_like(points_local, dtype=np.float64)
    for i in range(len(points_local)):
        pl = points_local[i]
        v = Gf.Vec3d(float(pl[0]), float(pl[1]), float(pl[2]))
        tw = m.Transform(v)
        out[i] = (tw[0], tw[1], tw[2])
    return out


def _triangulate_usd_mesh_indices(counts: list[int], indices: list[int]) -> list[tuple[int, int, int]]:
    tris: list[tuple[int, int, int]] = []
    offset = 0
    for c in counts:
        if c < 3:
            offset += c
            continue
        verts = indices[offset : offset + c]
        offset += c
        if c == 3:
            tris.append((verts[0], verts[1], verts[2]))
            continue
        # Fan triangulation for n-gons (preview-quality).
        for i in range(1, c - 1):
            tris.append((verts[0], verts[i], verts[i + 1]))
    return tris


def _collect_mesh_triangles(Gf, Usd, UsdGeom, stage_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stage = Usd.Stage.Open(os.fspath(stage_path))
    if not stage:
        raise RuntimeError(f"Usd.Stage.Open failed for {stage_path}")

    all_tris: list[tuple[int, int, int]] = []
    verts_out: list[np.ndarray] = []
    vert_base = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get()
        counts_attr = mesh.GetFaceVertexCountsAttr().Get()
        idx_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if pts_attr is None or counts_attr is None or idx_attr is None:
            continue

        pts_local = np.array([(float(p[0]), float(p[1]), float(p[2])) for p in pts_attr], dtype=np.float64)
        if pts_local.size == 0:
            continue

        pts_world = _transform_points_local_to_world(Gf, Usd, UsdGeom, prim, pts_local)
        pts_mj = _usd_y_up_to_mujoco_z_up(pts_world)

        tris = _triangulate_usd_mesh_indices(list(counts_attr), list(idx_attr))
        if not tris:
            continue

        verts_out.append(pts_mj.astype(np.float64))
        for a, b, c in tris:
            all_tris.append((vert_base + a, vert_base + b, vert_base + c))
        vert_base += pts_mj.shape[0]

    if not verts_out:
        raise RuntimeError(
            "No UsdGeom.Mesh geometry found (or meshes had no faces). "
            "Instancers / BasisCurves / implicit surfaces are not handled yet."
        )

    vertices = np.vstack(verts_out).astype(np.float64)
    faces = np.array(all_tris, dtype=np.int64)
    if faces.size == 0:
        raise RuntimeError("Meshes contained points but no triangles after triangulation.")

    vmax = int(vertices.shape[0])
    if (faces < 0).any() or (faces >= vmax).any():
        raise RuntimeError("Face indices out of range after merge — unexpected USD topology.")

    return vertices, faces.astype(np.uint32)


def _bbox_diag(vertices: np.ndarray) -> float:
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def _maybe_thicken_vertices(vertices: np.ndarray, eps_scale: float) -> np.ndarray:
    """Break qhull degeneracy for nearly-flat meshes by tiny deterministic jitter."""
    v = np.asarray(vertices, dtype=np.float64).copy()
    diag = _bbox_diag(v)
    span = v.max(0) - v.min(0)
    min_span = float(np.min(span))
    max_span = float(np.max(span))
    thin = max_span > 0 and (min_span / max_span) < 1e-5
    if eps_scale <= 0 and not thin:
        return v

    eps = max(1e-9, eps_scale * diag) if eps_scale > 0 else max(1e-9, 1e-6 * diag)
    rng = np.random.default_rng(0)
    jitter = rng.normal(scale=eps, size=v.shape)
    return v + jitter


def _center_and_scale(vertices: np.ndarray, scale: float) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64).copy()
    center = 0.5 * (v.min(axis=0) + v.max(axis=0))
    v -= center
    v *= float(scale)
    return v


def _flip_x_180(vertices: np.ndarray) -> np.ndarray:
    """Rotate points by 180 deg about +X (useful when imported asset is upside down)."""
    v = np.asarray(vertices, dtype=np.float64).copy()
    v[:, 1] *= -1.0
    v[:, 2] *= -1.0
    return v


def _build_preview_spec(stl_path: Path, mesh_name: str = "usdz_mesh") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "usdz_preview"
    spec.option.timestep = 0.01
    spec.option.gravity[:] = (0.0, 0.0, -9.81)

    spec.visual.headlight.active = True
    spec.visual.headlight.diffuse[:] = (0.85, 0.85, 0.85)
    spec.visual.headlight.ambient[:] = (0.25, 0.25, 0.25)

    wb = spec.worldbody
    wb.add_light(pos=(0.0, 0.0, 4.0), dir=(0.0, 0.0, -1.0), diffuse=(0.9, 0.9, 0.9))
    wb.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(8.0, 8.0, 0.01),
        rgba=(0.18, 0.20, 0.24, 1.0),
    )

    ms = spec.add_mesh(name=mesh_name, file=os.fspath(stl_path))
    ms.inertia = _mj_mesh_inertia_exact()

    root = wb.add_body(name="asset", pos=(0.0, 0.0, 0.05))
    root.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        contype=0,
        conaffinity=0,
        rgba=(0.72, 0.74, 0.78, 1.0),
        mass=0.0,
    )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a USDZ/USD stage in MuJoCo.")
    parser.add_argument(
        "usdz_path",
        type=Path,
        help="Path to .usdz, .usd, or .usda",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale applied after centering the merged mesh (default: 1).",
    )
    parser.add_argument(
        "--flip-x-180",
        action="store_true",
        help="Rotate imported mesh by 180 deg about +X (helpful if it appears upside down).",
    )
    parser.add_argument(
        "--thicken-eps",
        type=float,
        default=1e-5,
        help="Vertex jitter amplitude relative to bbox diagonal to avoid qhull failures on flat "
        "geometry (default: 1e-5). Use 0 to disable unless auto-thin triggers.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Compile and mj_forward once; do not open the interactive viewer.",
    )
    parser.add_argument(
        "--keep-stl",
        type=Path,
        default=None,
        help="If set, copy the generated STL to this path for debugging.",
    )
    args = parser.parse_args()

    usdz_path = args.usdz_path.expanduser().resolve()
    if not usdz_path.is_file():
        raise SystemExit(f"File not found: {usdz_path}")

    Gf, Usd, UsdGeom = _require_usd()

    vertices, faces = _collect_mesh_triangles(Gf, Usd, UsdGeom, usdz_path)
    print(
        f"Merged USD meshes: {vertices.shape[0]} vertices, {faces.shape[0]} triangles "
        f"(diag ~ {_bbox_diag(vertices):.6g})"
    )

    vertices = _center_and_scale(vertices, args.scale)
    if args.flip_x_180:
        vertices = _flip_x_180(vertices)
    thin = False
    span = vertices.max(0) - vertices.min(0)
    max_span = float(np.max(span))
    min_span = float(np.min(span))
    if max_span > 0 and (min_span / max_span) < 1e-5:
        thin = True
        print("Detected nearly-flat geometry; applying thicken jitter for MuJoCo qhull.", file=sys.stderr)

    if thin or args.thicken_eps > 0:
        vertices = _maybe_thicken_vertices(vertices, args.thicken_eps if args.thicken_eps > 0 else 1e-6)

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".stl", prefix="usdz_mujoco_")
    os.close(tmp_fd)
    stl_path = Path(tmp_name)
    try:
        _write_binary_stl(stl_path, vertices.astype(np.float32), faces)
        if args.keep_stl is not None:
            args.keep_stl.parent.mkdir(parents=True, exist_ok=True)
            args.keep_stl.write_bytes(stl_path.read_bytes())
            print(f"Wrote STL: {args.keep_stl.resolve()}")

        spec = _build_preview_spec(stl_path)
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
    finally:
        try:
            stl_path.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
