"""Plot teleop JSONL logs (from play_g1_gear_wbc --log) into PNG diagnostics.

Usage:
  python tests/teleop_regression/plot_teleop_jsonl.py \
    --log /tmp/my_teleop.jsonl \
    --out /tmp/my_teleop_plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

POS_ERR_BREAK_M = 0.15
POS_ERR_WARN_M = 0.10
QUAT_ERR_BREAK_DEG = 45.0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def _extract_time(rows: list[dict[str, Any]]) -> np.ndarray:
    vals = [float(r["sim_time"]) for r in rows if isinstance(r.get("sim_time"), (int, float))]
    if not vals:
        return np.zeros(0, dtype=np.float64)
    t = np.asarray(vals, dtype=np.float64)
    return t - t[0]


def _series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(vals, dtype=np.float64)


def _pelvis_xy(rows: list[dict[str, Any]]) -> np.ndarray:
    pts: list[list[float]] = []
    for r in rows:
        p = r.get("pelvis_pos")
        if isinstance(p, list) and len(p) >= 2:
            pts.append([float(p[0]), float(p[1])])
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _p95(a: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, 95.0))


def plot_log(log_path: Path, out_png: Path, *, kind: str = "teleop_ik") -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_rows(log_path)
    rows_k = [r for r in rows if r.get("kind") == kind]
    if not rows_k:
        raise RuntimeError(f"No rows with kind={kind!r} found in {log_path}")

    t = _extract_time(rows_k)
    e_l = _series(rows_k, "err_meas_minus_ik_tgt_world_left_l2")
    e_r = _series(rows_k, "err_meas_minus_ik_tgt_world_right_l2")
    q_l = _series(rows_k, "err_quat_meas_minus_ik_tgt_world_left_deg")
    q_r = _series(rows_k, "err_quat_meas_minus_ik_tgt_world_right_deg")
    clip_n = _series(rows_k, "arm_tau_n_clip")
    xy = _pelvis_xy(rows_k)

    n = min(len(t), len(e_l), len(e_r), len(q_l), len(q_r))
    if n == 0:
        raise RuntimeError("Missing required error fields in log rows.")
    t = t[:n]
    e_l = e_l[:n]
    e_r = e_r[:n]
    q_l = q_l[:n]
    q_r = q_r[:n]
    if len(clip_n) < n:
        clip_n = np.pad(clip_n, (0, n - len(clip_n)))
    clip_n = clip_n[:n]

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(t, e_l, color="tab:green", label="left pos err")
    axes[0].plot(t, e_r, color="tab:orange", label="right pos err")
    axes[0].axhline(POS_ERR_BREAK_M, linestyle="--", color="0.45", linewidth=1.0, label=f"{POS_ERR_BREAK_M:.2f}m break")
    axes[0].axhline(POS_ERR_WARN_M, linestyle=":", color="0.60", linewidth=1.0, label=f"{POS_ERR_WARN_M:.2f}m warn")
    axes[0].set_ylabel("Pos err (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, q_l, color="tab:blue", label="left quat err")
    axes[1].plot(t, q_r, color="tab:red", label="right quat err")
    axes[1].axhline(QUAT_ERR_BREAK_DEG, linestyle="--", color="0.45", linewidth=1.0, label=f"{QUAT_ERR_BREAK_DEG:.0f}deg break")
    axes[1].set_ylabel("Quat err (deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, clip_n, color="tab:purple", label="arm_tau_n_clip")
    axes[2].set_ylabel("Clipped joints")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    if xy.shape[0] > 0:
        axes[3].plot(xy[:, 0], xy[:, 1], color="tab:brown", label="pelvis XY path")
        axes[3].scatter([float(xy[0, 0])], [float(xy[0, 1])], s=16, color="tab:green", label="start")
        axes[3].scatter([float(xy[-1, 0])], [float(xy[-1, 1])], s=16, color="tab:red", label="end")
    axes[3].set_ylabel("Pelvis Y (m)")
    axes[3].set_xlabel("Pelvis X (m)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    fig.suptitle(f"{log_path.name} ({kind})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    summary = {
        "rows_total": float(len(rows)),
        "rows_kind": float(len(rows_k)),
        "max_pos_err_left_m": float(np.max(e_l)),
        "max_pos_err_right_m": float(np.max(e_r)),
        "p95_pos_err_left_m": _p95(e_l),
        "p95_pos_err_right_m": _p95(e_r),
        "max_quat_err_left_deg": float(np.max(q_l)),
        "max_quat_err_right_deg": float(np.max(q_r)),
        "p95_quat_err_left_deg": _p95(q_l),
        "p95_quat_err_right_deg": _p95(q_r),
        "pos_err_over_015m_steps": float(np.sum(np.maximum(e_l, e_r) > POS_ERR_BREAK_M)),
        "pos_err_over_010m_steps": float(np.sum(np.maximum(e_l, e_r) > POS_ERR_WARN_M)),
        "quat_err_over_45deg_steps": float(np.sum(np.maximum(q_l, q_r) > QUAT_ERR_BREAK_DEG)),
        "max_arm_tau_n_clip": float(np.max(clip_n)) if clip_n.size else 0.0,
    }
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Path to JSONL log from --log")
    parser.add_argument("--out", type=Path, required=True, help="PNG output path")
    parser.add_argument(
        "--kind",
        type=str,
        default="teleop_ik",
        help="Row kind to plot (default: teleop_ik)",
    )
    args = parser.parse_args()

    summary = plot_log(args.log, args.out, kind=args.kind)
    print(f"plot: {args.out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
