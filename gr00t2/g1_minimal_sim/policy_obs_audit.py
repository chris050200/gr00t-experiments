"""Classify observation keys for rollout vs minimal-sim policy A/B dumps.

**Rollout** (``Isaac-GR00T/gr00t/eval/rollout_policy.py``) dumps the **raw**
``observations`` dict from the vector env immediately before
``policy.get_action(observations)``. That dict includes RoboCasa / wrapper debug
taps (``q``, ``dq``, ``floating_base_*``, …) and parallel image keys
(``ego_view_image``, …).

**Minimal sim** (``scripts/run_gr00t_stylish_diner_inference.py``) dumps
``policy_obs``: only tensors built for ``PolicyClient.get_action(policy_obs)`` —
``video.{k}`` and ``state.{k}`` from ``client.get_modality_config()`` plus
``annotation.*`` language keys. It does **not** re-serialize ``q`` / ``dq`` /
raw ego aliases unless they appear in that batched policy dict.

So keys that are **rollout-only** in ``compare_policy_ab_dumps`` are often
*expected*, not a missing-policy-input bug.
"""

from __future__ import annotations

import json
from pathlib import Path


def minimal_policy_obs_contract_summary() -> str:
    """One-paragraph summary of what minimal live inference sends to ``get_action``."""
    return (
        "Minimal live inference builds ``policy_obs`` inside "
        "``_run_policy_loop`` from ``PolicyClient.get_modality_config()`` only: "
        "each ``video.<modality_key>`` as a (B,T,H,W,C) stack, each ``state.<modality_key>`` "
        "as (B,T,D), and each ``annotation.*`` language key. Raw vecenv keys like "
        "``q`` / ``dq`` / ``ego_view_image`` are not copied into ``policy_obs`` unless "
        "they are part of that modality contract."
    )


def classify_rollout_only_obs_key(key: str) -> tuple[str, str]:
    """Return ``(category, one_line_explanation)`` for a rollout-only observation key.

    Used when a key appears in a rollout ``policy_ab_dump`` manifest but not in
    a minimal-sim dump of ``policy_obs``.
    """
    k = str(key)

    if k == "video.tpp_view":
        return (
            "optional_third_person_video",
            "Rollout vec env exposes ``video.tpp_view``; minimal stack may omit this camera / key.",
        )

    if k.startswith("state.") or k.startswith("video.") or k.startswith("annotation."):
        return (
            "modality_subset",
            "Policy-shaped key absent on minimal side for this run (unexpected if both dumps use the same modality list).",
        )

    if k in _VECENV_LOWLEVEL_KEYS:
        return (
            "vecenv_lowlevel",
            "Raw sim / estimator taps from the vec env; not included in minimal ``policy_obs``.",
        )

    if k in _PARALLEL_IMAGE_KEYS:
        return (
            "parallel_image_alias",
            "Parallel RGB key from vec env; minimal stacks ``video.<name>`` from ``ego_view_image`` / render source instead.",
        )

    if k.endswith("_image") and k != "ego_view_image":
        return (
            "parallel_image_alias",
            "Likely a non-batched camera alias alongside ``video.*`` tensors.",
        )

    if k in ("wrist_pose", "torso_quat", "torso_ang_vel"):
        return (
            "vecenv_pose_debug",
            "Extra pose / IMU style taps from rollout stack; not forwarded in minimal ``policy_obs``.",
        )

    return (
        "rollout_only_unclassified",
        "Not in the built-in hint table; inspect vec env / wrapper that builds ``observations``.",
    )


_VECENV_LOWLEVEL_KEYS = frozenset(
    {
        "q",
        "dq",
        "ddq",
        "tau_est",
        "floating_base_pose",
        "floating_base_vel",
        "floating_base_acc",
    }
)

_PARALLEL_IMAGE_KEYS = frozenset(
    {
        "ego_view_image",
        "tpp_view_image",
    }
)


def load_manifest_obs_key_names(root: Path) -> list[str]:
    """Return sorted observation key names from ``manifest.json``."""
    root = root.expanduser().resolve()
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return sorted(str(e["key"]) for e in m.get("obs_keys", []))


def rollout_only_keys(rollout_root: Path, minimal_root: Path) -> list[str]:
    """Observation keys present in rollout dump but absent in minimal dump."""
    ra = set(load_manifest_obs_key_names(rollout_root))
    mb = set(load_manifest_obs_key_names(minimal_root))
    return sorted(ra - mb)


def format_rollout_only_audit(rollout_root: Path, minimal_root: Path | None) -> str:
    """Human-readable report (no colour codes)."""
    lines: list[str] = []
    lines.append("GR00T policy observation key audit")
    lines.append("")
    lines.append(minimal_policy_obs_contract_summary())
    lines.append("")
    lines.append(f"rollout dump: {rollout_root.resolve()}")
    if minimal_root is not None:
        lines.append(f"minimal dump: {minimal_root.resolve()}")
        only = rollout_only_keys(rollout_root, minimal_root)
        lines.append("")
        lines.append(f"rollout-only obs keys ({len(only)}):")
        for k in only:
            cat, expl = classify_rollout_only_obs_key(k)
            lines.append(f"  - {k}")
            lines.append(f"      category: {cat}")
            lines.append(f"      note: {expl}")
    else:
        lines.append("")
        lines.append("All rollout obs keys (pass minimal_root to diff):")
        for k in load_manifest_obs_key_names(rollout_root):
            cat, expl = classify_rollout_only_obs_key(k)
            lines.append(f"  - {k}  [{cat}]")
    lines.append("")
    return "\n".join(lines) + "\n"


def audit_rollout_vs_minimal_cli(
    rollout_root: Path,
    minimal_root: Path | None,
    *,
    print_summary: bool = True,
) -> int:
    """CLI helper: print audit; return 0."""
    text = format_rollout_only_audit(rollout_root, minimal_root)
    if print_summary:
        print(text, end="")
    return 0
