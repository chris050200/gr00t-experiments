"""Headless smoke for GLB → MuJoCo (``scenebuilder.glb_pipeline``)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mujoco
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BOOTH_GLB = _ROOT / "experiments" / "imports" / "2k_textured_diner_booth_w_table.glb"


def test_mtl_map_kd_parses_blender_style_flags() -> None:
    from scenebuilder import glb_pipeline as gp

    import tempfile as tf

    with tf.NamedTemporaryFile(mode="w", suffix=".mtl", delete=False, encoding="utf-8") as f:
        f.write("newmtl m\nmap_Kd -s 1 lambert1.png\n")
        path = Path(f.name)
    try:
        assert gp.first_map_kd_from_mtl(path) == "lambert1.png"
    finally:
        path.unlink(missing_ok=True)


def test_booth_glb_textured_pipeline_headless() -> None:
    pytest.importorskip("trimesh")
    if not _BOOTH_GLB.is_file():
        pytest.skip(f"Booth GLB fixture not in tree: {_BOOTH_GLB}")

    from scenebuilder import glb_pipeline as gp

    parts = gp.prepare_parts_from_glb(_BOOTH_GLB, up_axis="y", flip_x_180=True)
    gp.center_scale_meshes(parts, 1.0, 0.0)

    with tempfile.TemporaryDirectory(prefix="glb_smoke_") as tmp:
        root = Path(tmp)
        bundles = [
            gp.export_part_bundle(i, p.mesh, root, use_texture=True, verbose=False)
            for i, p in enumerate(parts)
        ]
        tex_parts = sum(
            1 for b in bundles if b.mode == "obj" and b.texture_path is not None
        )
        assert tex_parts == len(parts), (
            f"expected every part as textured OBJ ({len(parts)}), got {tex_parts} "
            f"(modes={[b.mode for b in bundles]})"
        )

        spec = gp.build_preview_spec(bundles)
        model = spec.compile()
        assert model.ntex == tex_parts, (model.ntex, tex_parts)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)


def test_export_mjcf_standalone_compiles() -> None:
    pytest.importorskip("trimesh")
    if not _BOOTH_GLB.is_file():
        pytest.skip(f"Booth GLB fixture not in tree: {_BOOTH_GLB}")

    from scenebuilder.mjcf_library import export_glb_asset_library

    with tempfile.TemporaryDirectory(prefix="mjcf_lib_") as tmp:
        out = Path(tmp) / "booth_export"
        export_glb_asset_library(
            _BOOTH_GLB,
            out,
            "sd_boothglb_test",
            flip_x_180=True,
            verbose=False,
        )
        xml = next(out.glob("*_standalone.xml"))
        mujoco.MjModel.from_xml_path(str(xml))
