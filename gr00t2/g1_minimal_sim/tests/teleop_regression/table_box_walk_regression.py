"""Headless teleop-regression runner for table_box walk + arm tracking.

This module drives the real Gear WBC runtime (ONNX stand/walk branch) through
the Gym wrapper, logs ``teleop_ik`` JSONL rows, and saves a PNG of key error
signals over time.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ee_frame import normalize_quat, quat_conj, quat_mul
from g1_gear_wbc_env import G1GearWBCEnv
from task_scene import resolve_task_scene

POS_ERR_BREAK_M = 0.15
POS_ERR_WARN_M = 0.10
QUAT_ERR_BREAK_DEG = 45.0


@dataclass(frozen=True)
class WalkRegressionSummary:
    scene: str
    profile: str
    lock_mode: str
    steps_total: int
    walk_steps: int
    walk_fraction: float
    pelvis_xy_displacement_m: float
    max_err_left_m: float
    max_err_right_m: float
    p95_err_left_m: float
    p95_err_right_m: float
    max_err_quat_left_deg: float
    max_err_quat_right_deg: float
    max_arm_tau_n_clip: int
    max_arm_tau_clip_delta_nm: float
    pos_err_over_015m_steps: int
    quat_err_over_45deg_steps: int
    reproduced_bug: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": str(self.scene),
            "profile": str(self.profile),
            "lock_mode": str(self.lock_mode),
            "steps_total": int(self.steps_total),
            "walk_steps": int(self.walk_steps),
            "walk_fraction": float(self.walk_fraction),
            "pelvis_xy_displacement_m": float(self.pelvis_xy_displacement_m),
            "max_err_left_m": float(self.max_err_left_m),
            "max_err_right_m": float(self.max_err_right_m),
            "p95_err_left_m": float(self.p95_err_left_m),
            "p95_err_right_m": float(self.p95_err_right_m),
            "max_err_quat_left_deg": float(self.max_err_quat_left_deg),
            "max_err_quat_right_deg": float(self.max_err_quat_right_deg),
            "max_arm_tau_n_clip": int(self.max_arm_tau_n_clip),
            "max_arm_tau_clip_delta_nm": float(self.max_arm_tau_clip_delta_nm),
            "pos_err_over_015m_steps": int(self.pos_err_over_015m_steps),
            "quat_err_over_45deg_steps": int(self.quat_err_over_45deg_steps),
            "reproduced_bug": bool(self.reproduced_bug),
        }


def _set_arm_targets(rt: Any, i: int, profile: str) -> None:
    """Set non-home EE targets so IK stays active while walking."""
    with rt.cmd_lock:
        if "ee_left_pos" not in rt.control_dict:
            return
        if profile == "aggressive_explode":
            # Stress wrists with farther reaches + alternating orientation demands.
            rt.control_dict["ee_left_pos"][:] = np.array([0.40, 0.30, 0.10], dtype=np.float32)
            rt.control_dict["ee_right_pos"][:] = np.array([0.40, -0.30, 0.10], dtype=np.float32)
            q_a = np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32)
            q_b = np.array([0.7071068, 0.0, -0.7071068, 0.0], dtype=np.float32)
            if (i // 120) % 2 == 0:
                rt.control_dict["ee_left_quat"][:] = q_a
                rt.control_dict["ee_right_quat"][:] = q_b
            else:
                rt.control_dict["ee_left_quat"][:] = q_b
                rt.control_dict["ee_right_quat"][:] = q_a
        else:
            rt.control_dict["ee_left_pos"][:] = np.array([0.32, 0.25, 0.07], dtype=np.float32)
            rt.control_dict["ee_right_pos"][:] = np.array([0.32, -0.25, 0.07], dtype=np.float32)


def _yaw_only_quat_from_xmat(xm: np.ndarray) -> np.ndarray:
    r = np.asarray(xm, dtype=np.float64).reshape(3, 3)
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    h = 0.5 * yaw
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def _apply_manual_lock_ee_pose(rt: Any, *, lock_orient_mode: str) -> None:
    """Lock both world EE orientation and position without keyboard teleop listener.

    This mirrors "always face east" orientation lock while keeping position commands
    fixed at home torso-frame points (no live position retargeting).
    """
    with rt.cmd_lock:
        cd = rt.control_dict
        if "_ee_left_quat_world_anchor" not in cd:
            return
        ti = int(rt.torso_index)
        xm = np.asarray(rt.data.xmat[ti], dtype=np.float64).reshape(3, 3)
        xq = np.asarray(rt.data.xquat[ti], dtype=np.float64).reshape(4)
        qr_full = normalize_quat(rt._ee_decode_quat(xm, xq))
        qr = (
            _yaw_only_quat_from_xmat(xm)
            if lock_orient_mode == "yaw_only"
            else qr_full
        )
        ql_w = normalize_quat(np.asarray(cd["_ee_left_quat_world_anchor"], dtype=np.float64).reshape(4))
        qr_w = normalize_quat(np.asarray(cd["_ee_right_quat_world_anchor"], dtype=np.float64).reshape(4))
        # Keep world orientation fixed.
        cd["ee_left_quat"][:] = quat_mul(quat_conj(qr), ql_w).astype(np.float32)
        cd["ee_right_quat"][:] = quat_mul(quat_conj(qr), qr_w).astype(np.float32)
        # Lock position command updates to home torso-frame targets.
        if "_ee_left_pos_home" in cd:
            cd["ee_left_pos"][:] = np.asarray(cd["_ee_left_pos_home"], dtype=np.float32)
        if "_ee_right_pos_home" in cd:
            cd["ee_right_pos"][:] = np.asarray(cd["_ee_right_pos_home"], dtype=np.float32)


def _run_scripted_profile(
    rt: Any,
    steps: int,
    *,
    profile: str,
    force_lock_ee_pose: bool,
    lock_orient_mode: str,
) -> None:
    """Scripted locomotion profiles for stable baseline or aggressive bug hunt."""
    cmd_stand = np.array(rt.config["cmd_init"], dtype=np.float64).reshape(3)
    cmd_turn_right = np.array([0.0, 0.0, -0.55], dtype=np.float64)
    cmd_walk_forward = np.array([0.75, 0.0, 0.0], dtype=np.float64)
    cmd_walk_backward = np.array([-0.70, 0.0, 0.0], dtype=np.float64)
    cmd_turn_hard = np.array([0.0, 0.0, -0.80], dtype=np.float64)
    cmd_turn_left_hard = np.array([0.0, 0.0, 0.80], dtype=np.float64)
    cmd_fwd_turn = np.array([0.78, 0.0, -0.52], dtype=np.float64)
    cmd_back_turn = np.array([-0.75, 0.0, -0.45], dtype=np.float64)

    for i in range(steps):
        if profile == "aggressive_explode":
            if i < 200:
                cmd = cmd_stand
            elif i < 850:
                cmd = cmd_turn_hard
            elif i < 2300:
                cmd = cmd_walk_backward
            elif i < 3100:
                cmd = cmd_fwd_turn
            elif i < 3900:
                cmd = cmd_back_turn
            elif i < 4700:
                cmd = cmd_walk_forward
            else:
                cmd = cmd_stand
        elif profile == "forward_floor":
            if i < 250:
                cmd = cmd_stand
            elif i < 5200:
                cmd = cmd_walk_forward
            else:
                cmd = cmd_stand
        elif profile == "manual_like_long_walk":
            # Manual-like repro: forward long, backward longer, turn left, forward again.
            if i < 300:
                cmd = cmd_stand
            elif i < 2300:
                cmd = cmd_walk_forward
            elif i < 5100:
                cmd = cmd_walk_backward
            elif i < 5600:
                cmd = cmd_turn_left_hard
            elif i < 8200:
                cmd = cmd_walk_forward
            else:
                cmd = cmd_stand
        else:
            if i < 220:
                cmd = cmd_stand
            elif i < 620:
                cmd = cmd_turn_right
            elif i < 1820:
                cmd = cmd_walk_forward
            elif i < 2220:
                cmd = cmd_walk_backward
            else:
                cmd = cmd_stand
        with rt.cmd_lock:
            rt.control_dict["loco_cmd"][:] = cmd
        if force_lock_ee_pose:
            _apply_manual_lock_ee_pose(rt, lock_orient_mode=lock_orient_mode)
        else:
            _set_arm_targets(rt, i, profile)
        rt.step_physics()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def _extract_series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals: list[float] = []
    for r in rows:
        v = r.get(key, None)
        if isinstance(v, (float, int)):
            vals.append(float(v))
    if not vals:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(vals, dtype=np.float64)


def _extract_time(rows: list[dict[str, Any]]) -> np.ndarray:
    vals: list[float] = []
    for r in rows:
        t = r.get("sim_time", None)
        if isinstance(t, (float, int)):
            vals.append(float(t))
    if not vals:
        return np.zeros(0, dtype=np.float64)
    t = np.asarray(vals, dtype=np.float64)
    return t - t[0]


def _extract_pelvis_xy(rows: list[dict[str, Any]]) -> np.ndarray:
    pts: list[list[float]] = []
    for r in rows:
        p = r.get("pelvis_pos", None)
        if isinstance(p, list) and len(p) >= 2:
            pts.append([float(p[0]), float(p[1])])
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _save_plot(rows: list[dict[str, Any]], out_png: Path, *, profile: str, scene: str, lock_mode: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _extract_time(rows)
    e_l = _extract_series(rows, "err_meas_minus_ik_tgt_world_left_l2")
    e_r = _extract_series(rows, "err_meas_minus_ik_tgt_world_right_l2")
    e_q_l = _extract_series(rows, "err_quat_meas_minus_ik_tgt_world_left_deg")
    e_q_r = _extract_series(rows, "err_quat_meas_minus_ik_tgt_world_right_deg")
    clip_n = _extract_series(rows, "arm_tau_n_clip")
    xy = _extract_pelvis_xy(rows)

    n = min(len(t), len(e_l), len(e_r), len(e_q_l), len(e_q_r))
    if n == 0:
        raise RuntimeError("No teleop_ik rows found; cannot plot.")
    if len(clip_n) < n:
        clip_n = np.pad(clip_n, (0, n - len(clip_n)))
    t = t[:n]
    e_l = e_l[:n]
    e_r = e_r[:n]
    e_q_l = e_q_l[:n]
    e_q_r = e_q_r[:n]
    clip_n = clip_n[:n]

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(t, e_l, label="left pos err", color="tab:green")
    axes[0].plot(t, e_r, label="right pos err", color="tab:orange")
    axes[0].set_ylabel("Pos err (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, e_q_l, label="left quat err", color="tab:blue")
    axes[1].plot(t, e_q_r, label="right quat err", color="tab:red")
    axes[1].set_ylabel("Quat err (deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, clip_n, label="arm_tau_n_clip", color="tab:purple")
    axes[2].set_ylabel("Clipped joints")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    if xy.shape[0] > 0:
        axes[3].plot(xy[:, 0], xy[:, 1], color="tab:brown", label="pelvis XY path")
        axes[3].scatter([float(xy[0, 0])], [float(xy[0, 1])], color="tab:green", s=16, label="start")
        axes[3].scatter([float(xy[-1, 0])], [float(xy[-1, 1])], color="tab:red", s=16, label="end")
    axes[3].set_ylabel("Pelvis Y (m)")
    axes[3].set_xlabel("Pelvis X (m)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    axes[0].axhline(POS_ERR_BREAK_M, linestyle="--", color="0.45", linewidth=1.0, label=f"{POS_ERR_BREAK_M:.2f}m break")
    axes[0].axhline(POS_ERR_WARN_M, linestyle=":", color="0.60", linewidth=1.0, label=f"{POS_ERR_WARN_M:.2f}m warn")
    axes[1].axhline(QUAT_ERR_BREAK_DEG, linestyle="--", color="0.45", linewidth=1.0, label=f"{QUAT_ERR_BREAK_DEG:.0f}deg break")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")

    fig.suptitle(f"{scene} {profile} lock={lock_mode} teleop regression")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def summarize_teleop_rows(
    rows: list[dict[str, Any]],
    *,
    scene: str,
    profile: str,
    lock_mode: str,
) -> WalkRegressionSummary:
    teleop = [r for r in rows if r.get("kind") == "teleop_ik"]
    if not teleop:
        raise RuntimeError("Expected teleop_ik rows in trace log.")

    walk_steps = 0
    p0: np.ndarray | None = None
    p1: np.ndarray | None = None
    e_l: list[float] = []
    e_r: list[float] = []
    eq_l: list[float] = []
    eq_r: list[float] = []
    clip_n: list[int] = []
    clip_d: list[float] = []
    pos_over = 0
    quat_over = 0
    for r in teleop:
        lc = r.get("loco_cmd", None)
        if isinstance(lc, list) and len(lc) >= 3:
            norm = float(np.linalg.norm(np.asarray(lc[:3], dtype=np.float64)))
            if norm > 0.05:
                walk_steps += 1
        pelvis = r.get("pelvis_pos", None)
        if isinstance(pelvis, list) and len(pelvis) >= 2:
            p = np.asarray(pelvis[:2], dtype=np.float64)
            if p0 is None:
                p0 = p
            p1 = p
        if isinstance(r.get("err_meas_minus_ik_tgt_world_left_l2"), (float, int)):
            e_l.append(float(r["err_meas_minus_ik_tgt_world_left_l2"]))
        if isinstance(r.get("err_meas_minus_ik_tgt_world_right_l2"), (float, int)):
            e_r.append(float(r["err_meas_minus_ik_tgt_world_right_l2"]))
        el = float(r["err_meas_minus_ik_tgt_world_left_l2"]) if isinstance(r.get("err_meas_minus_ik_tgt_world_left_l2"), (float, int)) else 0.0
        er = float(r["err_meas_minus_ik_tgt_world_right_l2"]) if isinstance(r.get("err_meas_minus_ik_tgt_world_right_l2"), (float, int)) else 0.0
        if max(el, er) > POS_ERR_BREAK_M:
            pos_over += 1
        if isinstance(r.get("err_quat_meas_minus_ik_tgt_world_left_deg"), (float, int)):
            eq_l.append(float(r["err_quat_meas_minus_ik_tgt_world_left_deg"]))
        if isinstance(r.get("err_quat_meas_minus_ik_tgt_world_right_deg"), (float, int)):
            eq_r.append(float(r["err_quat_meas_minus_ik_tgt_world_right_deg"]))
        eql = float(r["err_quat_meas_minus_ik_tgt_world_left_deg"]) if isinstance(r.get("err_quat_meas_minus_ik_tgt_world_left_deg"), (float, int)) else 0.0
        eqr = float(r["err_quat_meas_minus_ik_tgt_world_right_deg"]) if isinstance(r.get("err_quat_meas_minus_ik_tgt_world_right_deg"), (float, int)) else 0.0
        if max(eql, eqr) > QUAT_ERR_BREAK_DEG:
            quat_over += 1
        if isinstance(r.get("arm_tau_n_clip"), (float, int)):
            clip_n.append(int(r["arm_tau_n_clip"]))
        if isinstance(r.get("arm_tau_max_abs_clip_delta"), (float, int)):
            clip_d.append(float(r["arm_tau_max_abs_clip_delta"]))

    steps_total = len(teleop)
    walk_fraction = float(walk_steps) / float(max(steps_total, 1))
    disp = float(np.linalg.norm(p1 - p0)) if p0 is not None and p1 is not None else 0.0

    def _max_or_zero(xs: list[float]) -> float:
        return float(max(xs)) if xs else 0.0

    def _p95_or_zero(xs: list[float]) -> float:
        return float(np.percentile(np.asarray(xs, dtype=np.float64), 95.0)) if xs else 0.0

    max_err_left = _max_or_zero(e_l)
    max_err_right = _max_or_zero(e_r)
    max_quat_left = _max_or_zero(eq_l)
    max_quat_right = _max_or_zero(eq_r)
    reproduced_bug = bool(
        (max(max_err_left, max_err_right) > 0.22 and max(max_quat_left, max_quat_right) > 60.0)
        or (pos_over >= 120 and quat_over >= 120)
    )

    return WalkRegressionSummary(
        scene=str(scene),
        profile=str(profile),
        lock_mode=str(lock_mode),
        steps_total=steps_total,
        walk_steps=walk_steps,
        walk_fraction=walk_fraction,
        pelvis_xy_displacement_m=disp,
        max_err_left_m=max_err_left,
        max_err_right_m=max_err_right,
        p95_err_left_m=_p95_or_zero(e_l),
        p95_err_right_m=_p95_or_zero(e_r),
        max_err_quat_left_deg=max_quat_left,
        max_err_quat_right_deg=max_quat_right,
        max_arm_tau_n_clip=max(clip_n) if clip_n else 0,
        max_arm_tau_clip_delta_nm=_max_or_zero(clip_d),
        pos_err_over_015m_steps=pos_over,
        quat_err_over_45deg_steps=quat_over,
        reproduced_bug=reproduced_bug,
    )


def run_table_box_walk_regression(
    *,
    out_dir: Path,
    steps: int = 2400,
    lock_ee_orient: bool = False,
    profile: str = "baseline",
    scene: str = "table_pnp",
    lock_orient_mode: str = "full",
) -> tuple[WalkRegressionSummary, Path, Path]:
    """Run scripted table_box walk and write ``jsonl`` + ``png`` outputs."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_tag = str(lock_orient_mode if lock_ee_orient else "off")
    trace_path = out_dir / f"{scene}_{profile}_{lock_tag}_trace.jsonl"
    plot_path = out_dir / f"{scene}_{profile}_{lock_tag}_errors.png"

    scene_from_task, config_yaml, _meta = resolve_task_scene("table_box")
    if scene not in ("floor", "table_pnp", "stylish_diner"):
        raise ValueError(f"unsupported scene {scene!r}")
    resolved_scene = str(scene_from_task if scene == "table_pnp" else scene)
    kwargs: dict[str, Any] = {
        "scene": resolved_scene,
        # Keep viewer/X11-independent in CI: use UDP source flag to enter the same
        # teleop IK path without starting pynput listener.
        "enable_teleop": False,
        "udp_teleop_bind": "127.0.0.1:55115",
        "log_path": str(trace_path),
        "log_hz": 0.0,  # every step for regression fidelity
        "lock_ee_orient": False,
    }
    if config_yaml is not None:
        kwargs["config_yaml"] = config_yaml

    env = G1GearWBCEnv(**kwargs)
    try:
        env.reset()
        rt = env.unwrapped._rt
        _run_scripted_profile(
            rt,
            int(steps),
            profile=profile,
            force_lock_ee_pose=bool(lock_ee_orient),
            lock_orient_mode=str(lock_orient_mode),
        )
    finally:
        env.close()

    rows = _read_jsonl(trace_path)
    summary = summarize_teleop_rows(
        rows,
        scene=resolved_scene,
        profile=profile,
        lock_mode=lock_tag,
    )
    _save_plot(
        [r for r in rows if r.get("kind") == "teleop_ik"],
        plot_path,
        profile=profile,
        scene=resolved_scene,
        lock_mode=lock_tag,
    )

    (out_dir / f"{resolved_scene}_{profile}_{lock_tag}_summary.json").write_text(
        json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary, trace_path, plot_path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument(
        "--profile",
        choices=("baseline", "aggressive_explode", "forward_floor", "manual_like_long_walk"),
        default="baseline",
        help="baseline=stable walk regression, aggressive_explode=stress wrists, "
        "forward_floor=long forward walk, manual_like_long_walk=forward/back/turn/forward sequence.",
    )
    parser.add_argument(
        "--scene",
        choices=("floor", "table_pnp", "stylish_diner"),
        default="table_pnp",
        help="Run scene for regression (requested bug triage uses floor).",
    )
    parser.add_argument("--lock-ee-orient", action="store_true")
    parser.add_argument(
        "--lock-orient-mode",
        choices=("full", "yaw_only"),
        default="full",
        help="When --lock-ee-orient: full torso quat inverse vs yaw-only inverse for command transform.",
    )
    args = parser.parse_args()
    summary, trace_path, plot_path = run_table_box_walk_regression(
        out_dir=args.out_dir,
        steps=args.steps,
        lock_ee_orient=args.lock_ee_orient,
        profile=args.profile,
        scene=args.scene,
        lock_orient_mode=args.lock_orient_mode,
    )
    print(
        f"teleop regression complete (scene={args.scene}, profile={args.profile}, "
        f"lock={args.lock_ee_orient}, mode={args.lock_orient_mode})"
    )
    print(f"trace: {trace_path}")
    print(f"plot : {plot_path}")
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
