#!/usr/bin/env python3
"""Diff two existing ``policy_ab_dump`` directories (no server, no subprocess).

Use when oracle and subject packs were produced separately, or when XML/scenes
differ and you want **per-key stats + key-set gaps** without treating every
state mismatch as a hard failure.

Examples::

    cd g1_minimal_sim
    python scripts/compare_policy_ab_dumps.py \\
        --a /tmp/gr00t_ab_minimal_step0_post4a --b /tmp/gr00t_ab_rollout_step0 \\
        --label-a minimal --label-b rollout

    # Same tolerances as ``run_policy_ab_dump.py`` but never fail the process:
    python scripts/compare_policy_ab_dumps.py -a DIR1 -b DIR2 --exit-on never

    # Strict regression gate (identical to harness diff):
    python scripts/compare_policy_ab_dumps.py -a DIR1 -b DIR2 --exit-on strict

Rollout-only keys: see ``scripts/audit_policy_obs_keys.py`` (uses
``policy_obs_audit``) for a per-key glossary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from policy_ab_compare import diff_pack_roots  # noqa: E402


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-a",
        "--a",
        type=Path,
        required=True,
        help="First dump root (must contain manifest.json + obs/).",
    )
    p.add_argument(
        "-b",
        "--b",
        type=Path,
        required=True,
        help="Second dump root.",
    )
    p.add_argument("--label-a", default="A", metavar="NAME", help="Column label for --a.")
    p.add_argument("--label-b", default="B", metavar="NAME", help="Column label for --b.")
    p.add_argument(
        "--exit-on",
        choices=("never", "strict"),
        default="never",
        help="'strict' = fail on tolerances (like run_policy_ab_dump). "
        "'never' = always exit 0 for cross-scene inspection.",
    )
    p.add_argument("--state-atol", type=float, default=1e-4)
    p.add_argument("--action-atol", type=float, default=5e-2)
    p.add_argument("--video-atol", type=float, default=10.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    return diff_pack_roots(
        args.a,
        args.b,
        label_a=str(args.label_a),
        label_b=str(args.label_b),
        state_atol=float(args.state_atol),
        action_atol=float(args.action_atol),
        video_atol=float(args.video_atol),
        exit_on=str(args.exit_on),
    )


if __name__ == "__main__":
    raise SystemExit(main())
