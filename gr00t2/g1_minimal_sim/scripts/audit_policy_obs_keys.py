#!/usr/bin/env python3
"""Audit rollout vs minimal ``policy_ab_dump`` observation keys.

Explains why many keys are **rollout-only**: rollout dumps the full vec-env
``observations`` dict; minimal dumps only the ``policy_obs`` tensor package sent
to ``PolicyClient.get_action``.

Run from ``g1_minimal_sim/``::

    python scripts/audit_policy_obs_keys.py \\
        --rollout /tmp/gr00t_ab_rollout_step0 \\
        --minimal /tmp/gr00t_ab_minimal_step0_post4a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from policy_obs_audit import audit_rollout_vs_minimal_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rollout",
        type=Path,
        required=True,
        help="Rollout ``policy_ab_dump`` root (contains manifest.json).",
    )
    p.add_argument(
        "--minimal",
        type=Path,
        default=None,
        help="Minimal-sim dump root. If omitted, only list rollout keys with categories.",
    )
    args = p.parse_args(argv)
    return audit_rollout_vs_minimal_cli(args.rollout, args.minimal)


if __name__ == "__main__":
    raise SystemExit(main())
