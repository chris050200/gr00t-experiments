"""Load and diff ``policy_ab_dump`` packs (oracle vs subject, or any two roots).

Used by ``scripts/run_policy_ab_dump.py`` and ``scripts/compare_policy_ab_dumps.py``.
When scenes differ (non-identical XML), use ``exit_on="never"`` to inspect keys and
numeric gaps without a failing exit code; use ``exit_on="strict"`` for the Bug #8
regression gate (same rules as the original harness).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_USE_COLOUR = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    if not _USE_COLOUR:
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def _green(s: str) -> str:
    return _c("32", s)


def _red(s: str) -> str:
    return _c("31", s)


def _yellow(s: str) -> str:
    return _c("33", s)


def _bold(s: str) -> str:
    return _c("1", s)


def load_pack(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Load ``(manifest, obs_dict, action_dict)`` from a dump pack root."""
    root = root.expanduser().resolve()
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    obs: dict[str, Any] = {}
    for e in m["obs_keys"]:
        if "strings" in e:
            obs[e["key"]] = ("strings", tuple(e["strings"]))
        elif "file" in e:
            obs[e["key"]] = ("array", np.load(root / e["file"], allow_pickle=False))
    act: dict[str, np.ndarray] = {}
    for e in m.get("action_keys", []):
        if "file" in e:
            act[e["key"]] = np.load(root / e["file"], allow_pickle=False)
    return m, obs, act


def tagged_obs_to_batched_policy_obs(tagged: dict[str, Any]) -> dict[str, Any]:
    """Turn ``load_pack`` observation entries into a ``PolicyClient.get_action`` dict.

    ``load_pack`` stores each observation as ``("array", ndarray)`` without the
    leading batch dimension, or ``("strings", tuple[str, ...])`` for language keys.
    This prepends ``B=1`` for every array (``state.*`` → ``(1, T, D)``,
    ``video.*`` → ``(1, T, H, W, C)``).
    """
    out: dict[str, Any] = {}
    for k, v in tagged.items():
        if isinstance(v, tuple) and v[0] == "strings":
            out[k] = v[1]
            continue
        if isinstance(v, tuple) and v[0] == "array":
            arr = np.asarray(v[1])
            out[k] = arr[None, ...]
            continue
        raise TypeError(f"unexpected obs entry for {k!r}: {type(v).__name__}")
    return out


def build_policy_obs_from_ab_dumps(
    modality: Any,
    obs_minimal: dict[str, Any],
    obs_rollout: dict[str, Any],
    mode: str,
    *,
    video_swap_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Assemble a tagged ``policy_obs`` (``load_pack`` shape) for hybrid A/B experiments.

    Parameters
    ----------
    modality
        Return value of ``PolicyClient.get_modality_config()`` — mapping with
        ``"video"``, ``"state"``, ``"language"`` entries each having ``modality_keys``.
    obs_minimal, obs_rollout
        Observation dicts from ``load_pack`` (second component).
    mode
        ``"minimal"`` — all tensors from the minimal dump.
        ``"rollout"`` — all tensors from the rollout dump (must contain every key).
        ``"hybrid_ego"`` — ``state.*`` and ``annotation.*`` from minimal; each
        ``video.*`` from minimal unless the key is in ``video_swap_keys`` (default
        ``video.ego_view``), then from rollout. Missing non-swap video keys fall
        back to rollout (so ``video.tpp_view`` can be filled when minimal omits it).
    video_swap_keys
        Full keys such as ``"video.ego_view"`` to take from ``obs_rollout`` when
        ``mode == "hybrid_ego"``. Default: ``frozenset({"video.ego_view"})``.
    """
    if mode not in ("minimal", "rollout", "hybrid_ego"):
        raise ValueError(f"mode must be minimal|rollout|hybrid_ego, got {mode!r}")
    swap = video_swap_keys if video_swap_keys is not None else frozenset({"video.ego_view"})

    video_keys = list(modality["video"].modality_keys)
    state_keys = list(modality["state"].modality_keys)
    language_keys = list(modality["language"].modality_keys)

    tagged: dict[str, Any] = {}

    def _pick_state(full_k: str) -> Any:
        if mode == "rollout":
            return obs_rollout[full_k]
        return obs_minimal[full_k]

    def _pick_video(full_k: str) -> Any:
        if mode == "minimal":
            return obs_minimal[full_k]
        if mode == "rollout":
            return obs_rollout[full_k]
        # hybrid_ego
        if full_k in swap:
            if full_k not in obs_rollout:
                raise KeyError(
                    f"hybrid mode needs {full_k!r} in rollout dump for swap set {swap!r}"
                )
            return obs_rollout[full_k]
        if full_k in obs_minimal:
            return obs_minimal[full_k]
        if full_k in obs_rollout:
            return obs_rollout[full_k]
        raise KeyError(
            f"no {full_k!r} in minimal or rollout dump; server modality still requires it"
        )

    def _pick_lang(full_k: str) -> Any:
        if mode == "rollout":
            return obs_rollout[full_k]
        return obs_minimal[full_k]

    for sk in state_keys:
        tagged[f"state.{sk}"] = _pick_state(f"state.{sk}")
    for vk in video_keys:
        tagged[f"video.{vk}"] = _pick_video(f"video.{vk}")
    for lk in language_keys:
        tagged[lk] = _pick_lang(lk)

    return tagged


def max_abs_action_diff(
    ref: dict[str, np.ndarray], got: dict[str, np.ndarray]
) -> dict[str, float]:
    """Per-key ``max|ref - got|`` for action dicts (missing keys skipped)."""
    out: dict[str, float] = {}
    for k, a in ref.items():
        if k not in got:
            out[k] = float("nan")
            continue
        b = got[k]
        if a.shape != b.shape:
            out[k] = float("nan")
            continue
        out[k] = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    return out


def diff_arrays(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return ``(max|diff|, L2(diff))`` in fp64."""
    d = b.astype(np.float64) - a.astype(np.float64)
    return float(np.abs(d).max()), float(np.linalg.norm(d))


def classify(max_abs: float, atol: float) -> tuple[str, str]:
    """Return ``(label, colour_wrapped_label)`` given ``max|diff|`` and tolerance."""
    if max_abs <= atol:
        return ("OK", _green("OK     "))
    if max_abs <= atol * 100.0:
        return ("WARN", _yellow("WARN   "))
    return ("FAIL", _red("FAIL   "))


def diff_pack_roots(
    root_a: Path,
    root_b: Path,
    *,
    label_a: str = "A",
    label_b: str = "B",
    state_atol: float = 1e-4,
    action_atol: float = 5e-2,
    video_atol: float = 10.0,
    exit_on: str = "strict",
) -> int:
    """Print a per-key diff table; return exit code (0 = OK, 1 = regression).

    Parameters
    ----------
    exit_on
        ``"strict"`` — fail on state/annotation/action tolerances (harness behaviour).
        ``"never"`` — always exit 0 (exploratory / cross-XML obs diagnosis).
    """
    root_a = root_a.expanduser().resolve()
    root_b = root_b.expanduser().resolve()
    strict = exit_on == "strict"

    print()
    print(_bold("=" * 78))
    print(_bold("  GR00T policy I/O dump diff"))
    print(_bold("=" * 78))
    print(f"  {label_a:8s}: {root_a}")
    print(f"  {label_b:8s}: {root_b}")
    print(
        f"  tolerances: state_atol={state_atol:g}  "
        f"action_atol={action_atol:g}  video_atol={video_atol:g}  exit_on={exit_on!r}"
    )
    print()

    _, oa, aa = load_pack(root_a)
    _, ob, ab = load_pack(root_b)

    def obs_keys_by_prefix(obs: dict[str, Any], prefix: str) -> set[str]:
        return {k for k in obs if k.startswith(prefix)}

    state_a = obs_keys_by_prefix(oa, "state.")
    state_b = obs_keys_by_prefix(ob, "state.")
    only_state_a = sorted(state_a - state_b)
    only_state_b = sorted(state_b - state_a)
    state_keys = sorted(state_a | state_b)

    video_a = obs_keys_by_prefix(oa, "video.")
    video_b = obs_keys_by_prefix(ob, "video.")
    only_video_a = sorted(video_a - video_b)
    only_video_b = sorted(video_b - video_a)

    annot_a = obs_keys_by_prefix(oa, "annotation.")
    annot_b = obs_keys_by_prefix(ob, "annotation.")

    other_a = set(oa) - state_a - video_a - annot_a
    other_b = set(ob) - state_b - video_b - annot_b
    other_keys = sorted(other_a | other_b)

    if only_state_a or only_state_b or only_video_a or only_video_b:
        print(_bold("[key set]  (obs present on one side only)"))
        if only_state_a:
            print(f"  state.* only {label_a}: {only_state_a}")
        if only_state_b:
            print(f"  state.* only {label_b}: {only_state_b}")
        if only_video_a:
            print(f"  video.* only {label_a}: {only_video_a}")
        if only_video_b:
            print(f"  video.* only {label_b}: {only_video_b}")
        print()

    fail = False
    print(_bold(f"{'KEY':40s} {'STATUS':8s} {'SHAPE':16s} {'max|diff|':>12s} {'L2':>12s}"))
    print("-" * 90)

    def row(key: str, status_colour: str, shape: str, max_abs: float | None, l2: float | None) -> None:
        max_s = "n/a" if max_abs is None else f"{max_abs:.6g}"
        l2_s = "n/a" if l2 is None else f"{l2:.6g}"
        print(f"{key:40s} {status_colour} {shape:16s} {max_s:>12s} {l2_s:>12s}")

    print(_bold("[state.*]  (Bug #8 invariant when scenes align)"))
    for k in state_keys:
        if k not in oa:
            row(k, _yellow("B_ONLY "), "-", None, None)
            continue
        if k not in ob:
            row(k, _red("MISSING ") if strict else _yellow("B_MISS  "), "-", None, None)
            if strict:
                fail = True
            continue
        ta, va = oa[k]
        tb, vb = ob[k]
        if ta != "array" or tb != "array":
            row(k, _yellow("SKIP   "), "non-array", None, None)
            continue
        if va.shape != vb.shape:
            row(k, _red("SHAPE  "), f"{va.shape}!={vb.shape}", None, None)
            if strict:
                fail = True
            continue
        max_abs, l2 = diff_arrays(va, vb)
        label, coloured = classify(max_abs, state_atol)
        if strict and label == "FAIL":
            fail = True
        row(k, coloured, str(va.shape), max_abs, l2)

    print(_bold("[annotation.*]"))
    annot_iter = sorted(annot_a | annot_b) if not strict else sorted(annot_a)
    for k in annot_iter:
        if k not in oa:
            row(k, _yellow("B_ONLY "), "-", None, None)
            continue
        if k not in ob:
            row(k, _red("MISSING ") if strict else _yellow("B_MISS  "), "-", None, None)
            if strict:
                fail = True
            continue
        va = oa[k]
        vb = ob[k]
        if va == vb:
            row(k, _green("OK     "), "strings", 0.0, 0.0)
        else:
            row(k, _red("FAIL   "), "strings", float("nan"), float("nan"))
            print(f"    {label_a}: {va}")
            print(f"    {label_b}: {vb}")
            if strict:
                fail = True

    print(_bold("[video.*]  (soft tolerance; lighting / scene shift expected across XML)"))
    vid_keys = sorted(video_a | video_b) if not strict else sorted(video_a)
    for k in vid_keys:
        if k not in oa:
            row(k, _yellow("B_ONLY "), "-", None, None)
            continue
        if k not in ob:
            row(k, _yellow("SKIP   "), "-", None, None)
            continue
        ta, va = oa[k]
        tb, vb = ob[k]
        if ta != "array" or tb != "array":
            row(k, _yellow("SKIP   "), "non-array", None, None)
            continue
        if va.shape != vb.shape:
            row(k, _red("SHAPE  "), f"{va.shape}!={vb.shape}", None, None)
            continue
        max_abs, l2 = diff_arrays(va, vb)
        _, coloured = classify(max_abs, video_atol)
        row(k, coloured, str(va.shape), max_abs, l2)
        if k == "video.ego_view":
            ma = va.astype(np.float64).reshape(-1, 3).mean(axis=0)
            mb = vb.astype(np.float64).reshape(-1, 3).mean(axis=0)
            print(
                f"    {label_a}  per-ch mean RGB: [{ma[0]:.1f}, {ma[1]:.1f}, {ma[2]:.1f}]    "
                f"{label_b}: [{mb[0]:.1f}, {mb[1]:.1f}, {mb[2]:.1f}]    "
                f"delta: [{mb[0] - ma[0]:+.1f}, {mb[1] - ma[1]:+.1f}, {mb[2] - ma[2]:+.1f}]"
            )

    skipped_other = sorted(k for k in other_keys if k in oa and k not in ob)
    if skipped_other:
        print(_bold(f"[other obs (only {label_a}; often debug taps)]"))
        for k in skipped_other:
            row(k, _yellow("A_ONLY "), "-", None, None)
    skipped_other_b = sorted(k for k in other_keys if k in ob and k not in oa)
    if skipped_other_b:
        print(_bold(f"[other obs (only {label_b})]"))
        for k in skipped_other_b:
            row(k, _yellow("B_ONLY "), "-", None, None)

    print(_bold("[action.*]  (consequence of obs + same server/checkpoint)"))
    act_keys = sorted(set(aa) | set(ab)) if not strict else sorted(aa.keys())
    for k in act_keys:
        if k not in aa:
            row(k, _yellow("B_ONLY "), "-", None, None)
            continue
        if k not in ab:
            row(k, _red("MISSING ") if strict else _yellow("B_MISS  "), "-", None, None)
            if strict:
                fail = True
            continue
        va = aa[k]
        vb = ab[k]
        if va.shape != vb.shape:
            row(k, _red("SHAPE  "), f"{va.shape}!={vb.shape}", None, None)
            continue
        max_abs, l2 = diff_arrays(va, vb)
        label, coloured = classify(max_abs, action_atol)
        if strict and label == "FAIL":
            fail = True
        row(k, coloured, str(va.shape), max_abs, l2)

    print()
    if strict and fail:
        print(_red(_bold("VERDICT: REGRESSION  (one or more keys exceed tolerance)")))
        return 1
    if strict:
        print(_green(_bold("VERDICT: OK  (all checked keys within tolerance)")))
    else:
        print(_bold("VERDICT: exploratory (--exit-on never); exit code 0"))
    return 0
