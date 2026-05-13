"""JsonlTrace overwrite vs append."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trace_log import JsonlTrace  # noqa: E402


def test_jsonl_trace_truncates_by_default():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.jsonl"
        j = JsonlTrace(p)
        j.write({"kind": "x", "v": 1})
        j.close()
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 3  # meta open, x, meta close
        assert '"write_mode":"truncate"' in lines[0].replace(" ", "")

        j2 = JsonlTrace(p)
        j2.write({"kind": "y", "v": 2})
        j2.close()
        lines2 = p.read_text(encoding="utf-8").strip().splitlines()
        kinds = [json.loads(L).get("kind") for L in lines2]
        assert "x" not in kinds
        assert "y" in kinds


def test_jsonl_trace_append_keeps_prior_rows():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.jsonl"
        JsonlTrace(p).close()
        n0 = len(p.read_text(encoding="utf-8").strip().splitlines())
        JsonlTrace(p, append=True).close()
        n1 = len(p.read_text(encoding="utf-8").strip().splitlines())
        assert n1 > n0
        assert p.read_text(encoding="utf-8").count('"event":"trace_open"') == 2
