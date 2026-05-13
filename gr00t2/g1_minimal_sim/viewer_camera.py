"""MuJoCo passive viewer: select built-in MJCF cameras (``mjCAMERA_FIXED``)."""

from __future__ import annotations

import mujoco


def viewer_use_fixed_camera(
    model: mujoco.MjModel, viewer_cam: mujoco.MjvCamera, name: str
) -> None:
    """Point ``viewer_cam`` at model camera ``name`` (must exist in MJCF)."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        raise ValueError(
            f"MJCF camera {name!r} not found (ncam={model.ncam}); "
            "expected egocentric camera on robot."
        )
    viewer_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer_cam.fixedcamid = cid
