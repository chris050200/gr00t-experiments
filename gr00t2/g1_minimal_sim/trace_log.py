"""Append-only JSONL trace logger (generic flight-recorder for sim debugging)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO


class JsonlTrace:
    """Write one JSON object per line.

    Default (**append=False**) opens with truncate — **one process run = one file**, no mixing of sessions.
    Use ``append=True`` only when intentionally continuing an existing log.
    """

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._append = bool(append)
        self._fp: TextIO = self.path.open(mode, encoding="utf-8")
        self._start = time.time()
        self.write(
            {
                "kind": "meta",
                "event": "trace_open",
                "t_wall": time.time(),
                "path": str(self.path.resolve()),
                "write_mode": "append" if append else "truncate",
            }
        )

    def write(self, obj: dict[str, Any]) -> None:
        self._fp.write(json.dumps(obj, separators=(",", ":"), default=_json_default))
        self._fp.write("\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self.write(
                {
                    "kind": "meta",
                    "event": "trace_close",
                    "t_wall": time.time(),
                    "dt_open_s": float(time.time() - self._start),
                }
            )
        finally:
            self._fp.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort flush
        try:
            if hasattr(self, "_fp") and not self._fp.closed:
                self._fp.flush()
        except Exception:
            pass


def _json_default(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    # numpy arrays / scalars
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.astype(float).tolist()
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
    except Exception:
        pass
    return str(x)
