"""GLB/GLTF → meshes + :class:`ExportBundle` + MuJoCo :class:`MjSpec` preview builder."""

from __future__ import annotations

import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scenebuilder.gl_guard import install as _install_gl_guard

_install_gl_guard()

import mujoco
import numpy as np

@dataclass
class PreparedPart:
    name: str
    mesh: object  # trimesh.Trimesh (typed loosely to avoid hard import at runtime)


@dataclass
class ExportBundle:
    mode: Literal["obj", "stl"]
    mesh_path: Path
    texture_path: Path | None
    rgba: tuple[float, float, float, float]


def _require_trimesh():
    try:
        import trimesh  # type: ignore[import-not-found]

        return trimesh
    except ImportError:
        print(
            "Missing dependency: trimesh\n"
            "Install in your current env, then rerun:\n"
            "  python -m pip install trimesh pillow",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def _mj_mesh_inertia_exact() -> mujoco.mjtMeshInertia:
    mi = mujoco.mjtMeshInertia
    for name in ("mjMESH_INERTIA_EXACT", "mjINERTIA_EXACT"):
        v = getattr(mi, name, None)
        if v is not None:
            return v
    for name in ("mjMESH_INERTIA_LEGACY", "mjINERTIA_LEGACY"):
        v = getattr(mi, name, None)
        if v is not None:
            return v
    raise RuntimeError("mujoco.mjtMeshInertia: no compatible enum found")


def _write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
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


def _to_rgba_tuple(v: object) -> tuple[float, float, float, float] | None:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    if arr.size == 3:
        arr = np.concatenate([arr, np.array([1.0], dtype=np.float64)])
    arr = arr[:4]
    if np.max(arr) > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def _extract_part_rgba(mesh) -> tuple[float, float, float, float]:
    fallback = (0.72, 0.74, 0.78, 1.0)
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return fallback

    material = getattr(visual, "material", None)
    if material is not None:
        for attr in ("main_color", "baseColorFactor", "diffuse"):
            value = getattr(material, attr, None)
            if value is None:
                continue
            rgba = _to_rgba_tuple(value)
            if rgba is not None:
                return rgba

    face_colors = getattr(visual, "face_colors", None)
    if face_colors is not None:
        fc = np.asarray(face_colors)
        if fc.size >= 3:
            rgba = _to_rgba_tuple(np.median(fc, axis=0))
            if rgba is not None:
                return rgba

    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None:
        vc = np.asarray(vertex_colors)
        if vc.size >= 3:
            rgba = _to_rgba_tuple(np.median(vc, axis=0))
            if rgba is not None:
                return rgba

    return fallback


def _iter_scene_instances(scene) -> list[tuple[str, str, np.ndarray]]:
    out: list[tuple[str, str, np.ndarray]] = []
    nodes = list(getattr(scene.graph, "nodes_geometry", []))
    for node_name in nodes:
        transform = np.eye(4, dtype=np.float64)
        geom_name = None

        try:
            item = scene.graph[node_name]
            if isinstance(item, tuple) and len(item) >= 2:
                transform = np.asarray(item[0], dtype=np.float64).reshape(4, 4)
                geom_name = str(item[1])
        except Exception:
            pass

        if geom_name is None:
            try:
                transform2, geom_name2 = scene.graph.get(node_name)
                transform = np.asarray(transform2, dtype=np.float64).reshape(4, 4)
                geom_name = str(geom_name2)
            except Exception:
                continue

        out.append((str(node_name), geom_name, transform))

    if out:
        return out

    for geom_name in scene.geometry.keys():
        out.append((str(geom_name), str(geom_name), np.eye(4, dtype=np.float64)))
    return out


def _y_up_to_z_up(vertices: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    out = np.empty_like(v)
    out[:, 0] = v[:, 0]
    out[:, 1] = v[:, 2]
    out[:, 2] = -v[:, 1]
    return out


def _flip_x_180(vertices: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64).copy()
    v[:, 1] *= -1.0
    v[:, 2] *= -1.0
    return v


def _bbox_diag(vertices: np.ndarray) -> float:
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def _maybe_thicken(vertices: np.ndarray, eps_scale: float) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64).copy()
    diag = _bbox_diag(v)
    if diag <= 0:
        return v
    eps = max(1e-9, eps_scale * diag)
    rng = np.random.default_rng(0)
    return v + rng.normal(scale=eps, size=v.shape)


def _mesh_has_texture_uv(mesh) -> bool:
    vis = getattr(mesh, "visual", None)
    if vis is None or getattr(vis, "kind", None) != "texture":
        return False
    uv = getattr(vis, "uv", None)
    if uv is None:
        return False
    n_uv = int(len(np.asarray(uv)))
    n_v = int(len(mesh.vertices))
    # Trimesh usually stores one UV per vertex; some assets use per-corner UVs (3 per face).
    if n_uv == n_v:
        return True
    n_f = int(len(mesh.faces))
    if n_f > 0 and n_uv == n_f * 3:
        return True
    return False


def _strip_obj_mtllib_lines(obj_path: Path) -> None:
    """Remove ``mtllib`` / ``usemtl`` directives so MuJoCo uses only the MJCF geom ``material``.

    Trimesh writes an MTL with ``Kd 0.4 0.4 0.4`` (typical glTF default). Some MuJoCo builds
    honor that for shading and the mesh looks flat gray even when a ``map_Kd`` texture is bound
    on the geom.
    """
    try:
        raw = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return
    out: list[str] = []
    for line in raw:
        ls = line.strip().lower()
        if ls.startswith("mtllib") or ls.startswith("usemtl"):
            continue
        out.append(line)
    obj_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")


def _obj_mtllib_name(obj_path: Path) -> str | None:
    try:
        for raw in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if line.lower().startswith("mtllib"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    return parts[1].strip()
    except OSError:
        return None
    return None


def first_map_kd_from_mtl(mtl_path: Path) -> str | None:
    """Return the diffuse map filename from an MTL file, if any.

    Handles ``map_Kd`` / ``map_kd``, optional Blender-style flags (``-s``, ``-o``, …), and
    strips quotes. Avoids ``line[len("map_kd"):]`` on mixed-case lines (length mismatch).
    """
    try:
        text = mtl_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(?i)map_kd\s+(.*)$", line)
        if not m:
            continue
        rest = m.group(1).strip().strip('"').strip("'")
        if not rest:
            continue
        # Blender / some exporters: map_Kd -o u v w file.png  or  map_Kd -s 1 file.png
        tokens = re.split(r"\s+", rest)
        image_suffixes = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
        image_hits: list[str] = []
        for tok in tokens:
            if not tok or tok.startswith("#"):
                continue
            if tok.startswith("-"):
                continue
            low = tok.lower()
            if any(low.endswith(suf) for suf in image_suffixes):
                image_hits.append(tok)
        if image_hits:
            return image_hits[-1]
        for tok in tokens:
            if not tok or tok.startswith("#"):
                continue
            if tok.startswith("-"):
                continue
            return tok
    return None


def _resolve_existing_file_case_insensitive(subdir: Path, name: str) -> Path | None:
    """Linux-friendly lookup when ``mtllib`` name casing does not match the file on disk."""
    direct = (subdir / name).resolve()
    if direct.is_file():
        return direct
    name_lower = name.lower()
    for p in subdir.iterdir():
        if p.is_file() and p.name.lower() == name_lower:
            return p.resolve()
    return None


def _infer_texture_image_in_dir(subdir: Path) -> Path | None:
    """If ``map_Kd`` parsing fails, use the lone image next to the OBJ (trimesh always writes one)."""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
    images: list[Path] = []
    for p in subdir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            images.append(p)
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    images.sort(key=lambda p: p.stat().st_size, reverse=True)
    return images[0]


def prepare_parts_from_glb(
    glb_path: Path,
    *,
    up_axis: str,
    flip_x_180: bool,
) -> list[PreparedPart]:
    trimesh = _require_trimesh()
    loaded = trimesh.load(os.fspath(glb_path), force="scene")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)

    prepared: list[PreparedPart] = []
    instances = _iter_scene_instances(scene)
    for idx, (node_name, geom_name, transform) in enumerate(instances):
        geom = scene.geometry.get(geom_name)
        if geom is None:
            continue

        mesh = geom.copy()
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            continue

        try:
            mesh.apply_transform(transform)
        except Exception:
            continue

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            continue
        if faces.ndim != 2 or faces.shape[1] != 3:
            continue
        if len(vertices) < 4 or len(faces) == 0:
            continue

        if up_axis == "y":
            mesh.vertices = _y_up_to_z_up(mesh.vertices)
        elif up_axis != "z":
            raise ValueError(f"Unsupported up-axis: {up_axis}")

        if flip_x_180:
            mesh.vertices = _flip_x_180(mesh.vertices)

        prepared.append(
            PreparedPart(name=f"part_{idx}_{node_name}".replace(" ", "_"), mesh=mesh)
        )

    if not prepared:
        raise RuntimeError("No usable triangle meshes found in GLB/GLTF scene.")
    return prepared


def center_scale_meshes(meshes: list[PreparedPart], scale: float, thicken_eps: float) -> None:
    all_vertices = np.vstack([np.asarray(p.mesh.vertices, dtype=np.float64) for p in meshes])
    center = 0.5 * (all_vertices.min(axis=0) + all_vertices.max(axis=0))
    for p in meshes:
        v = (np.asarray(p.mesh.vertices, dtype=np.float64) - center) * float(scale)
        if thicken_eps > 0:
            v = _maybe_thicken(v, thicken_eps)
        p.mesh.vertices = v


def export_part_bundle(
    idx: int,
    mesh,
    out_root: Path,
    *,
    use_texture: bool,
    verbose: bool = False,
) -> ExportBundle:
    rgba = _extract_part_rgba(mesh)
    subdir = out_root / f"part_{idx:04d}"
    subdir.mkdir(parents=True, exist_ok=True)

    if use_texture and _mesh_has_texture_uv(mesh):
        obj_path = subdir / "mesh.obj"
        try:
            mesh.export(os.fspath(obj_path))
        except Exception as exc:
            if verbose:
                print(f"part_{idx:04d}: mesh.export(OBJ) failed ({exc!r}); using STL.", file=sys.stderr)
            obj_path = None
        if obj_path is not None and obj_path.is_file():
            mtl_name = _obj_mtllib_name(obj_path)
            tex_rel: str | None = None
            if mtl_name:
                mtl_path = _resolve_existing_file_case_insensitive(subdir, mtl_name)
                if mtl_path is not None:
                    tex_rel = first_map_kd_from_mtl(mtl_path)
            tex_path: Path | None = None
            if tex_rel:
                cand = _resolve_existing_file_case_insensitive(subdir, tex_rel)
                if cand is not None:
                    tex_path = cand
            if tex_path is None:
                tex_path = _infer_texture_image_in_dir(subdir)
            if tex_path is not None:
                _strip_obj_mtllib_lines(obj_path)
                return ExportBundle(mode="obj", mesh_path=obj_path, texture_path=tex_path, rgba=rgba)
            if verbose:
                print(
                    f"part_{idx:04d}: textured export produced OBJ but no resolvable map_Kd image "
                    f"(mtl={mtl_name!r}, map={tex_rel!r}); using STL.",
                    file=sys.stderr,
                )
        elif verbose and use_texture and _mesh_has_texture_uv(mesh):
            print(f"part_{idx:04d}: OBJ export missing after mesh.export; using STL.", file=sys.stderr)
    elif verbose and use_texture:
        print(
            f"part_{idx:04d}: no TextureVisuals UV layout this tool understands "
            f"(verts={len(mesh.vertices)}); using STL.",
            file=sys.stderr,
        )

    stl_path = subdir / "mesh.stl"
    _write_binary_stl(
        stl_path,
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
    )
    return ExportBundle(mode="stl", mesh_path=stl_path, texture_path=None, rgba=rgba)


def _mj_texrole_rgb():
    role = getattr(mujoco, "mjtTextureRole", None)
    if role is None:
        return None
    v = getattr(role, "mjTEXROLE_RGB", None)
    return v


def _bind_rgb_material(spec: mujoco.MjSpec, *, tex_name: str, png_path: Path) -> str:
    tex = spec.add_texture(name=tex_name)
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.file = os.fspath(png_path.resolve())

    mat_name = f"{tex_name}_mat"
    mat = spec.add_material(name=mat_name)
    role = _mj_texrole_rgb()
    if role is None:
        raise RuntimeError("mujoco.mjtTextureRole missing; cannot bind diffuse texture.")
    mat.textures[role] = tex_name
    return mat_name


def build_preview_spec(bundles: list[ExportBundle]) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "glb_preview"
    spec.option.timestep = 0.01
    spec.option.gravity[:] = (0.0, 0.0, -9.81)

    spec.visual.headlight.active = True
    spec.visual.headlight.diffuse[:] = (0.85, 0.85, 0.85)
    spec.visual.headlight.ambient[:] = (0.45, 0.45, 0.45)

    wb = spec.worldbody
    wb.add_light(pos=(0.0, 0.0, 4.0), dir=(0.0, 0.0, -1.0), diffuse=(0.9, 0.9, 0.9))
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(8.0, 8.0, 0.01), rgba=(0.18, 0.20, 0.24, 1.0))
    root = wb.add_body(name="asset", pos=(0.0, 0.0, 0.05))

    for i, bundle in enumerate(bundles):
        mesh_name = f"mesh_{i}"
        ms = spec.add_mesh(name=mesh_name, file=os.fspath(bundle.mesh_path.resolve()))
        ms.inertia = _mj_mesh_inertia_exact()

        geom_kwargs = dict(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            contype=0,
            conaffinity=0,
            mass=0.0,
        )

        if bundle.mode == "obj" and bundle.texture_path is not None:
            mat_name = _bind_rgb_material(spec, tex_name=f"tex_{i}", png_path=bundle.texture_path)
            geom_kwargs["material"] = mat_name
            geom_kwargs["rgba"] = (1.0, 1.0, 1.0, 1.0)
        else:
            geom_kwargs["rgba"] = bundle.rgba

        root.add_geom(**geom_kwargs)

    return spec
