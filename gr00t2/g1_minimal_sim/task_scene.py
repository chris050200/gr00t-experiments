"""Task-spec to scene/config resolution for CLI entrypoints.

This keeps task selection lightweight: a task currently resolves to either a
named built-in scene preset or a concrete config yaml path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_TASK_SPECS: dict[str, Path] = {
    "table_box": Path(__file__).resolve().parent / "scenes" / "table_box" / "task_spec.yaml",
}


def available_tasks() -> tuple[str, ...]:
    """Return supported task keys for CLI help/validation."""
    return tuple(sorted(_TASK_SPECS.keys()))


def resolve_task_scene(task: str) -> tuple[str, str | None, dict[str, Any]]:
    """Resolve task key to ``(scene, config_yaml, metadata)``.

    Current policy:
    - ``scene_source.mode: existing_table_pnp`` -> ``scene='table_pnp'``
    - ``scene_source.mode: existing_floor`` -> ``scene='floor'``
    - ``scene_source.mode: config_yaml`` -> ``config_yaml=<string path>``
    """
    if task not in _TASK_SPECS:
        known = ", ".join(available_tasks()) or "<none>"
        raise ValueError(f"unknown task {task!r}; expected one of: {known}")

    spec_path = _TASK_SPECS[task]
    if not spec_path.is_file():
        raise FileNotFoundError(f"task spec not found: {spec_path}")

    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    scene_src = spec.get("scene_source", {})
    mode = str(scene_src.get("mode", "existing_floor"))

    if mode == "existing_table_pnp":
        return "table_pnp", None, spec
    if mode == "existing_floor":
        return "floor", None, spec
    if mode == "config_yaml":
        cfg = scene_src.get("config_yaml", None)
        if cfg is None:
            raise ValueError(
                f"task {task!r} scene_source.mode=config_yaml requires scene_source.config_yaml"
            )
        return "floor", str(cfg), spec

    raise ValueError(
        f"unsupported scene_source.mode {mode!r} in task {task!r}; "
        "expected existing_table_pnp, existing_floor, or config_yaml"
    )
