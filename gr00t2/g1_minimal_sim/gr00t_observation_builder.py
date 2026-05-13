"""Build GR00T / ``unitree_g1``-shaped observation dicts from MuJoCo + :class:`GearWBCRuntime`.

P1 — state slices via ``prepare_observation_for_eval``, ego RGB from MJCF camera (default ``oak_egoview``,
fallback ``head_pov``).

Requires ``gr00t_wbc`` (Isaac-GR00T WholeBodyControl install). Use the same venv as
``play_g1_gear_wbc.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mujoco
    from gr00t_wbc.control.robot_model.robot_model import RobotModel


def _wholebody_control_root() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Isaac-GR00T"
        / "external_dependencies"
        / "GR00T-WholeBodyControl"
    ).resolve()


def ensure_gr00t_wbc_importable() -> Path:
    """Insert WholeBodyControl repo root onto ``sys.path`` for ``gr00t_wbc``."""
    root = _wholebody_control_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected GR00T-WholeBodyControl at {root} (sibling: gr00t2/Isaac-GR00T/...)."
        )
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    return root


def default_robot_model_g1_locomanip():
    """``RobotModel`` matching ``rollout_policy`` / ``enable_waist=True`` (like gym kwargs)."""
    ensure_gr00t_wbc_importable()
    from gr00t_wbc.control.robot_model.instantiation import get_robot_type_and_model

    _t, model = get_robot_type_and_model("G1", enable_waist_ik=True)
    return model


def mujoco_q_to_robot_model_q(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    robot_model: "RobotModel",
) -> np.ndarray:
    """Fill Pinocchio configuration ``q`` (length ``num_joints``) from MuJoCo joint positions."""
    import mujoco

    q = np.zeros(robot_model.num_joints, dtype=np.float64)
    if robot_model.num_joints != robot_model.num_dofs:
        raise ValueError(
            "Expected num_joints == num_dofs for single-DoF manipulator model; "
            f"got {robot_model.num_joints} vs {robot_model.num_dofs}"
        )
    for name in robot_model.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint {name!r} missing from MuJoCo model")
        qadr = int(model.jnt_qposadr[jid])
        jnt_type = model.jnt_type[jid]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"Joint {name!r} is free; floating-base alignment is not handled here"
            )
        q_next = (
            int(model.jnt_qposadr[jid + 1])
            if jid + 1 < model.njnt
            else int(model.nq)
        )
        nq_j = q_next - qadr
        if nq_j != 1:
            raise ValueError(
                f"Joint {name!r} spans nq={nq_j}; only single-DoF joints supported"
            )
        qi = robot_model.dof_index(name)
        q[qi] = float(data.qpos[qadr])
    return q


class Gr00tObservationBuilder:
    """Produces dict observations compatible with ``prepare_observation_for_eval`` + ego video keys."""

    def __init__(
        self,
        robot_model: "RobotModel",
        *,
        camera_name: str = "oak_egoview",
        image_width: int = 640,
        image_height: int = 480,
    ) -> None:
        self._robot_model = robot_model
        self._camera_name = camera_name
        self._image_width = int(image_width)
        self._image_height = int(image_height)
        self._renderer: Any | None = None
        # MuJoCo 3.x Python ``MjModel`` has no ``.ptr``; use object identity to decide Renderer reuse.
        self._renderer_model_id: int | None = None
        self._resolved_camera_name: str | None = None

    @classmethod
    def for_locomanip_default(cls, **kwargs: Any) -> Gr00tObservationBuilder:
        """Build with :func:`default_robot_model_g1_locomanip` (waist IK on)."""
        return cls(default_robot_model_g1_locomanip(), **kwargs)

    @property
    def robot_model(self) -> "RobotModel":
        """Underlying Pinocchio ``RobotModel`` (joint group indices, FK, etc.).

        Exposed so the inference bridge can mirror upstream's
        ``G1DecoupledWholeBodyPolicy.get_action`` FK to translate
        ``action.waist`` joint targets into the lower-body policy's
        ``rpy_cmd`` torso orientation command (see
        ``scripts/run_gr00t_stylish_diner_inference.py::_waist_action_to_rpy_cmd``).
        """
        return self._robot_model

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_model_id = None
        self._resolved_camera_name = None

    def _resolve_camera_name(self, model: "mujoco.MjModel") -> str:
        import mujoco

        preferred = self._camera_name
        candidates = [preferred]
        if preferred != "head_pov":
            candidates.append("head_pov")
        for name in candidates:
            cid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name))
            if cid >= 0:
                return name
        raise ValueError(
            f"None of camera candidates {candidates!r} exist in MJCF (mjOBJ_CAMERA)."
        )

    def render_ego_rgb_uint8(
        self, model: "mujoco.MjModel", data: "mujoco.MjData"
    ) -> np.ndarray:
        """Render ``camera_name`` (``RGB uint8`` ``H×W×3``)."""
        import mujoco

        mid = id(model)
        if self._renderer is None or self._renderer_model_id != mid:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(
                model,
                height=self._image_height,
                width=self._image_width,
            )
            self._renderer_model_id = mid
            self._resolved_camera_name = self._resolve_camera_name(model)

        assert self._resolved_camera_name is not None
        cid = int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, self._resolved_camera_name
            )
        )
        if cid < 0:
            # Guard against a model hot-swap where camera name changed after renderer setup.
            self._resolved_camera_name = self._resolve_camera_name(model)
            cid = int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_CAMERA, self._resolved_camera_name
                )
            )
        self._renderer.update_scene(data, camera=cid)
        rgb = self._renderer.render()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgb

    def build(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        *,
        task_description: str = "",
        include_video: bool = True,
    ) -> dict[str, Any]:
        """Return observation dict with ``state.*``, ``annotation...``, and optional ego images."""
        ensure_gr00t_wbc_importable()
        from gr00t_wbc.control.utils.n1_utils import prepare_observation_for_eval

        q = mujoco_q_to_robot_model_q(model, data, self._robot_model)
        obs: dict[str, Any] = {"q": q}
        prepare_observation_for_eval(self._robot_model, obs)
        obs["annotation.human.task_description"] = task_description

        if include_video:
            rgb = self.render_ego_rgb_uint8(model, data)
            obs["ego_view_image"] = rgb
            obs["video.ego_view"] = rgb

        return obs
