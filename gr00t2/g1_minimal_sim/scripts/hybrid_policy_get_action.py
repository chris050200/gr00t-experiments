#!/usr/bin/env python3
"""Hybrid ``PolicyClient.get_action`` probe using two ``policy_ab_dump`` trees.

Loads dumps from **rollout** (Tier-A vecenv) and **minimal** (Tier-B
``policy_obs``), builds an observation dict matching
``get_modality_config()``, optionally swaps ``video.ego_view`` (or other keys)
from rollout into an otherwise-minimal package, calls the live policy server
once per ``--modes`` entry, and prints per-action-key ``max|diff|`` vs the
recorded dump actions.

**Typical use** (policy server already running; from ``g1_minimal_sim/``)::

    python scripts/hybrid_policy_get_action.py \\
      --rollout /tmp/gr00t_ab_rollout_step0 \\
      --minimal /tmp/gr00t_ab_minimal_step0_post4a \\
      --policy-host 127.0.0.1 --policy-port 2000 \\
      --modes minimal hybrid_ego rollout

**Interpretation**

- If ``hybrid_ego`` actions move much closer to the **rollout** recorded actions
  than ``minimal`` does, pixels (ego RGB) are likely driving the policy gap.
- If ``hybrid_ego`` still matches ``minimal`` poorly vs rollout, look for missing
  video keys, dtype/layout issues, or non-vision causes.

**Dry run** (no server; print minimal dump obs shapes)::

    python scripts/hybrid_policy_get_action.py -a /path/to/rollout -b /path/to/minimal --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_MIN_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(_MIN_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIN_SIM_ROOT))

_ISAAC = _MIN_SIM_ROOT.parent / "Isaac-GR00T"
if _ISAAC.is_dir() and str(_ISAAC) not in sys.path:
    sys.path.insert(0, str(_ISAAC))


def _strip_batch0_actions(actions: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in actions.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and int(arr.shape[0]) == 1:
            arr = arr[0]
        out[k] = arr
    return out


def _print_modality(modality: dict[str, Any]) -> None:
    print("Policy server modality_config:")
    for group in ("video", "state", "language"):
        if group not in modality:
            print(f"  [{group}]  (missing)")
            continue
        cfg = modality[group]
        keys = list(getattr(cfg, "modality_keys", []) or [])
        deltas = getattr(cfg, "delta_indices", None)
        print(f"  [{group}]  keys={keys!r}  delta_indices={deltas!r}")


def _print_action_diff_table(
    label: str,
    diffs: dict[str, float],
    *,
    action_atol: float,
) -> None:
    print()
    print(f"--- {label} ---  (max|server_action - ref_dump_action|)  atol={action_atol}")
    for k in sorted(diffs.keys()):
        d = diffs[k]
        if d != d:  # NaN
            status = "nan"
        elif d <= action_atol:
            status = "OK"
        elif d <= action_atol * 100.0:
            status = "WARN"
        else:
            status = "FAIL"
        print(f"  {k:32s}  {status:6s}  max_abs={d:.6g}")


def main(argv: list[str] | None = None) -> int:
    from policy_ab_compare import (
        build_policy_obs_from_ab_dumps,
        load_pack,
        max_abs_action_diff,
        tagged_obs_to_batched_policy_obs,
    )

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rollout", "-a", type=Path, required=True, help="Rollout policy_ab_dump root.")
    p.add_argument("--minimal", "-b", type=Path, required=True, help="Minimal-sim policy_ab_dump root.")
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=2000)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["minimal", "hybrid_ego", "rollout"],
        choices=("minimal", "hybrid_ego", "rollout"),
        help="Which observation packages to try (each triggers one get_action).",
    )
    p.add_argument(
        "--video-swap-keys",
        nargs="*",
        default=["video.ego_view"],
        metavar="KEY",
        help="For hybrid_ego: full obs keys to take from rollout (default: video.ego_view).",
    )
    p.add_argument("--action-atol", type=float, default=5e-2, help="OK/WARN band for action diff print.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the server; only print minimal dump obs array shapes.",
    )
    args = p.parse_args(argv)

    try:
        from gr00t.policy.server_client import PolicyClient
    except ImportError as e:
        print(
            "error: could not import gr00t.policy.server_client "
            f"({e}). Use the Isaac-GR00T venv or set PYTHONPATH to Isaac-GR00T.",
            file=sys.stderr,
        )
        return 2

    _, obs_roll, act_roll = load_pack(args.rollout)
    _, obs_min, act_min = load_pack(args.minimal)

    if args.dry_run:
        man_path = args.minimal / "manifest.json"
        if not man_path.is_file():
            print(f"error: missing {man_path}", file=sys.stderr)
            return 4
        man = json.loads(man_path.read_text(encoding="utf-8"))
        print("[dry-run] skipping server; minimal manifest obs keys + array shapes:")
        for e in man.get("obs_keys", []):
            if "file" in e:
                arr = np.load(args.minimal / e["file"], allow_pickle=False)
                print(f"  {e['key']}: shape={arr.shape} dtype={arr.dtype}")
            else:
                print(f"  {e['key']}: {e}")
        print("[dry-run] omit --dry-run and start the policy server to run hybrid get_action.")
        return 0

    client = PolicyClient(host=args.policy_host, port=args.policy_port, strict=False)
    if not client.ping():
        print(
            f"error: policy server not reachable at {args.policy_host}:{args.policy_port}",
            file=sys.stderr,
        )
        return 3
    modality = client.get_modality_config()
    _print_modality(modality)

    swap_set = frozenset(str(k) for k in args.video_swap_keys)

    for mode in args.modes:
        tagged = build_policy_obs_from_ab_dumps(
            modality,
            obs_min,
            obs_roll,
            mode,
            video_swap_keys=swap_set,
        )
        policy_obs = tagged_obs_to_batched_policy_obs(tagged)
        print()
        print(f"=== mode={mode!r}  batched policy_obs shapes ===")
        for k in sorted(policy_obs.keys()):
            v = policy_obs[k]
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} {v!r}")

        client.reset()
        raw_actions, _info = client.get_action(policy_obs)
        server_actions = _strip_batch0_actions(raw_actions)

        dr = max_abs_action_diff(act_roll, server_actions)
        dm = max_abs_action_diff(act_min, server_actions)
        _print_action_diff_table(
            f"{mode}: vs rollout dump actions",
            dr,
            action_atol=float(args.action_atol),
        )
        _print_action_diff_table(
            f"{mode}: vs minimal dump actions",
            dm,
            action_atol=float(args.action_atol),
        )

    print()
    print("Done. If hybrid_ego tracks rollout much better than minimal does, prioritize")
    print("MJCF / render parity for ego RGB (memory-bank ACTIVE_gr00t_inference_wiring.md §10D).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
