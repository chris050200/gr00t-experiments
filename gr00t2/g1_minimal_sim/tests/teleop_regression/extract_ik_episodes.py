"""Extract trigger/untrigger IK failure episodes from teleop JSONL logs.

Usage:
  python tests/teleop_regression/extract_ik_episodes.py \
    --log /tmp/lockeeorientdebugeetargets.jsonl

Optional:
  --out /tmp/episodes.json
  --dump-first-episode-rows /tmp/episode_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def _scalar_row(r: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = r.get(key)
    return float(v) if isinstance(v, (int, float)) else float(default)


def _scalar_ik(r: dict[str, Any], key: str) -> float:
    d = r.get("ik")
    if isinstance(d, dict) and isinstance(d.get(key), (int, float, bool)):
        return float(d[key])
    return float("nan")


@dataclass
class EpisodeSummary:
    idx_start: int
    idx_end: int
    sim_time_start: float
    sim_time_end: float
    duration_s: float
    n_steps: int
    max_pos_err_m: float
    p95_pos_err_m: float
    max_quat_err_deg: float
    p95_quat_err_deg: float
    p95_ik_raw_step_l2: float
    p95_ik_step_scale: float
    p95_dbg_target_vs_recon_deg: float
    p95_dbg_meas_vs_target_deg: float
    p95_dbg_lock_anchor_vs_target_deg: float


def _episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    out: list[tuple[int, int]] = []
    s = int(idx[0])
    e = int(idx[0])
    for i in idx[1:]:
        ii = int(i)
        if ii == e + 1:
            e = ii
        else:
            out.append((s, e))
            s = e = ii
    out.append((s, e))
    return out


def _p95(xs: np.ndarray) -> float:
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return 0.0
    return float(np.percentile(xs, 95.0))


def analyze_log(
    log_path: Path,
    *,
    kind: str = "teleop_ik",
    pos_break_m: float = 0.15,
    quat_break_deg: float = 45.0,
) -> dict[str, Any]:
    rows = _read_rows(log_path)
    tele = [r for r in rows if r.get("kind") == kind]
    if not tele:
        raise RuntimeError(f"No rows with kind={kind!r} in {log_path}")

    pos_err = np.asarray(
        [
            max(
                _scalar_row(r, "err_meas_minus_ik_tgt_world_left_l2"),
                _scalar_row(r, "err_meas_minus_ik_tgt_world_right_l2"),
            )
            for r in tele
        ],
        dtype=np.float64,
    )
    quat_err = np.asarray(
        [
            max(
                _scalar_row(r, "err_quat_meas_minus_ik_tgt_world_left_deg"),
                _scalar_row(r, "err_quat_meas_minus_ik_tgt_world_right_deg"),
            )
            for r in tele
        ],
        dtype=np.float64,
    )
    bad = (pos_err > float(pos_break_m)) | (quat_err > float(quat_break_deg))
    eps = _episodes(bad)

    # Seam-level diagnostics (prefer max over left/right where relevant).
    ik_raw = np.asarray([_scalar_ik(r, "ik_arm_delta_raw_l2") for r in tele], dtype=np.float64)
    ik_scale = np.asarray([_scalar_ik(r, "ik_step_limit_scale") for r in tele], dtype=np.float64)
    tgt_vs_recon = np.asarray(
        [
            max(
                _scalar_ik(r, "ik_dbg_target_quat_vs_recon_left_deg"),
                _scalar_ik(r, "ik_dbg_target_quat_vs_recon_right_deg"),
            )
            for r in tele
        ],
        dtype=np.float64,
    )
    meas_vs_tgt = np.asarray(
        [
            max(
                _scalar_ik(r, "ik_dbg_meas_pre_vs_target_left_deg"),
                _scalar_ik(r, "ik_dbg_meas_pre_vs_target_right_deg"),
            )
            for r in tele
        ],
        dtype=np.float64,
    )
    anchor_vs_tgt = np.asarray(
        [
            max(
                _scalar_ik(r, "ik_dbg_lock_anchor_vs_target_left_deg"),
                _scalar_ik(r, "ik_dbg_lock_anchor_vs_target_right_deg"),
            )
            for r in tele
        ],
        dtype=np.float64,
    )

    episodes: list[EpisodeSummary] = []
    for s, e in eps:
        sl = slice(s, e + 1)
        t0 = _scalar_row(tele[s], "sim_time")
        t1 = _scalar_row(tele[e], "sim_time")
        episodes.append(
            EpisodeSummary(
                idx_start=s,
                idx_end=e,
                sim_time_start=t0,
                sim_time_end=t1,
                duration_s=max(0.0, t1 - t0),
                n_steps=(e - s + 1),
                max_pos_err_m=float(np.max(pos_err[sl])),
                p95_pos_err_m=_p95(pos_err[sl]),
                max_quat_err_deg=float(np.max(quat_err[sl])),
                p95_quat_err_deg=_p95(quat_err[sl]),
                p95_ik_raw_step_l2=_p95(ik_raw[sl]),
                p95_ik_step_scale=_p95(ik_scale[sl]),
                p95_dbg_target_vs_recon_deg=_p95(tgt_vs_recon[sl]),
                p95_dbg_meas_vs_target_deg=_p95(meas_vs_tgt[sl]),
                p95_dbg_lock_anchor_vs_target_deg=_p95(anchor_vs_tgt[sl]),
            )
        )

    first_bad_idx = int(eps[0][0]) if eps else -1
    first_bad_time = _scalar_row(tele[first_bad_idx], "sim_time") if eps else float("nan")
    summary = {
        "log_path": str(log_path),
        "rows_total": len(rows),
        "rows_kind": len(tele),
        "pos_break_m": float(pos_break_m),
        "quat_break_deg": float(quat_break_deg),
        "bad_steps": int(np.sum(bad)),
        "episodes_n": len(episodes),
        "first_bad_idx": first_bad_idx,
        "first_bad_sim_time": float(first_bad_time),
        "max_pos_err_m": float(np.max(pos_err)),
        "p95_pos_err_m": _p95(pos_err),
        "max_quat_err_deg": float(np.max(quat_err)),
        "p95_quat_err_deg": _p95(quat_err),
        "episodes": [asdict(ep) for ep in episodes],
    }
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--kind", type=str, default="teleop_ik")
    parser.add_argument("--pos-break-m", type=float, default=0.15)
    parser.add_argument("--quat-break-deg", type=float, default=45.0)
    parser.add_argument("--out", type=Path, default=None, help="Optional summary JSON path")
    parser.add_argument(
        "--dump-first-episode-rows",
        type=Path,
        default=None,
        help="Optional JSONL dump of rows in first bad episode (for deep forensic review).",
    )
    args = parser.parse_args()

    summary = analyze_log(
        args.log,
        kind=args.kind,
        pos_break_m=float(args.pos_break_m),
        quat_break_deg=float(args.quat_break_deg),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dump_first_episode_rows is not None:
        rows = _read_rows(args.log)
        tele = [r for r in rows if r.get("kind") == args.kind]
        eps = summary.get("episodes") or []
        if eps:
            s = int(eps[0]["idx_start"])
            e = int(eps[0]["idx_end"])
            args.dump_first_episode_rows.parent.mkdir(parents=True, exist_ok=True)
            args.dump_first_episode_rows.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in tele[s : e + 1]) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    _main()

