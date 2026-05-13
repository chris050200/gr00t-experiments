"""Import-time GL backend guard for MuJoCo (optional; use before ``import mujoco``)."""

from __future__ import annotations

import os
import sys


def install() -> None:
    """``MUJOCO_GL=osmesa`` + incomplete PyOpenGL often breaks before user code runs."""
    if os.environ.get("MUJOCO_GL", "").strip().lower() != "osmesa":
        return
    if os.environ.get("MUJOCO_USE_OSMESA") == "1":
        return
    print(
        "scene_builder: MUJOCO_GL=osmesa + PyOpenGL often raises AttributeError on import; "
        "using MUJOCO_GL=glfw. For headless GPU use MUJOCO_GL=egl; to keep osmesa set "
        "MUJOCO_USE_OSMESA=1.",
        file=sys.stderr,
    )
    os.environ["MUJOCO_GL"] = "glfw"
