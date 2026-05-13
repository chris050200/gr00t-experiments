#!/usr/bin/env python3
"""Modular GR00T policy-client inference for g1_minimal_sim (``--scene``, ``--prompt``).

Implementation lives in :mod:`run_gr00t_stylish_diner_inference`; this file is the
preferred entrypoint name. See that module's docstring for cadence, arm-action
contracts, and examples.

Run from ``g1_minimal_sim/``::

    python scripts/run_gr00t_inference.py --scene table_pnp --prompt \"...\" \\
      --policy-host 127.0.0.1 --policy-port 2000
"""

from __future__ import annotations

from run_gr00t_stylish_diner_inference import build_argparser, main

__all__ = ["build_argparser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
