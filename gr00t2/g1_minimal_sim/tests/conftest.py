"""Pytest defaults for g1_minimal_sim (avoid MKL/OpenMP aborts in some sandboxes)."""

from __future__ import annotations

import os

# See memory-bank/techContext.md — some environments abort without this.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
