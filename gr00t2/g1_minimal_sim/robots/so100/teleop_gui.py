"""Tkinter panel for SO-100 world-frame EE target + gripper (thread-safe; no MuJoCo in GUI thread)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import tkinter as tk
from tkinter import ttk

from ik_site import quat_wxyz_mul, quat_wxyz_normalize, quat_wxyz_from_axis_angle


@dataclass
class TeleopState:
    """Desired ``ee_site`` pose in **world** (m, wxyz) + Jaw actuator reference."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    p_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    q_world: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    gripper: float = 0.0
    request_fk_sync: bool = False

    def nudge_p_world(self, axis: int, sign: float, step: float) -> None:
        """Move target position along **world** X/Y/Z (axis 0/1/2)."""
        with self.lock:
            self.p_world[int(axis)] += float(sign) * float(step)

    def nudge_rot_world(self, axis: int, sign: float, dtheta: float) -> None:
        """Apply a small rotation about **world** X/Y/Z on the left: ``q ← q(Δ) ⊗ q``."""
        ax = np.zeros(3, dtype=np.float64)
        ax[int(axis)] = 1.0
        dq = quat_wxyz_from_axis_angle(ax, float(sign) * float(dtheta))
        with self.lock:
            self.q_world[:] = quat_wxyz_normalize(quat_wxyz_mul(dq, self.q_world))

    def nudge_gripper(self, delta: float, lo: float, hi: float) -> None:
        with self.lock:
            self.gripper = float(np.clip(self.gripper + float(delta), lo, hi))

    def set_gripper(self, value: float, lo: float, hi: float) -> None:
        with self.lock:
            self.gripper = float(np.clip(float(value), lo, hi))

    def apply_quat_world(self, w: float, x: float, y: float, z: float) -> None:
        q = quat_wxyz_normalize(np.array([w, x, y, z], dtype=np.float64))
        with self.lock:
            self.q_world[:] = q

    def request_sync(self) -> None:
        with self.lock:
            self.request_fk_sync = True

    def snapshot_targets(self) -> tuple[np.ndarray, np.ndarray, float]:
        with self.lock:
            return self.p_world.copy(), self.q_world.copy(), float(self.gripper)

    def take_fk_sync_request(self) -> bool:
        with self.lock:
            r = self.request_fk_sync
            self.request_fk_sync = False
            return r


def _fmt_vec3(v: np.ndarray) -> str:
    return f"{v[0]:+.4f}  {v[1]:+.4f}  {v[2]:+.4f}"


def _fmt_quat(q: np.ndarray) -> str:
    return f"w={q[0]:+.3f}  x={q[1]:+.3f}  y={q[2]:+.3f}  z={q[3]:+.3f}"


def _bind_press_hold_repeat(
    widget: tk.Widget,
    root: tk.Misc,
    tick: Callable[[], None],
    *,
    interval_ms: int = 70,
) -> None:
    """Hold button: ``tick()`` once on press, then every ``interval_ms`` until release."""
    job: list[str | None] = [None]

    def cancel() -> None:
        if job[0] is not None:
            try:
                root.after_cancel(job[0])
            except tk.TclError:
                pass
            job[0] = None

    def again() -> None:
        tick()
        job[0] = root.after(interval_ms, again)

    def on_press(_event: tk.Event) -> None:
        cancel()
        tick()
        job[0] = root.after(interval_ms, again)

    def on_release(_event: tk.Event) -> None:
        cancel()

    widget.bind("<ButtonPress-1>", on_press)
    widget.bind("<ButtonRelease-1>", on_release)


def run_teleop_window(
    state: TeleopState,
    *,
    trans_step: float = 0.01,
    rot_step: float = 0.06,
    gripper_lo: float = -0.174,
    gripper_hi: float = 1.75,
    gripper_delta: float = 0.04,
    hold_repeat_ms: int = 70,
) -> None:
    """Start Tk mainloop (call from a **daemon** thread)."""
    root = tk.Tk()
    root.title("SO-100 EE teleop (world frame)")

    frm = ttk.Frame(root, padding=8)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    pos_lbl = ttk.Label(frm, text="p_world: —", font=("TkFixedFont", 10))
    pos_lbl.grid(row=0, column=0, columnspan=6, sticky="w")
    quat_lbl = ttk.Label(frm, text="q_world: —", font=("TkFixedFont", 10))
    quat_lbl.grid(row=1, column=0, columnspan=6, sticky="w")
    grip_lbl = ttk.Label(frm, text="gripper (Jaw ctrl): —", font=("TkFixedFont", 10))
    grip_lbl.grid(row=2, column=0, columnspan=6, sticky="w", pady=(0, 6))

    def refresh_labels() -> None:
        p, q, g = state.snapshot_targets()
        pos_lbl.configure(text=f"p_world (m): {_fmt_vec3(p)}")
        quat_lbl.configure(text=f"q_world (wxyz): {_fmt_quat(q)}")
        grip_lbl.configure(text=f"gripper (Jaw ctrl): {g:+.4f}  (lim [{gripper_lo}, {gripper_hi}])")
        root.after(120, refresh_labels)

    refresh_labels()

    r = 3
    ttk.Label(frm, text="Translate along **world** X / Y / Z (m)").grid(
        row=r, column=0, columnspan=6, sticky="w"
    )
    r += 1
    axes = ("X", "Y", "Z")
    for i, name in enumerate(axes):
        bm = ttk.Button(frm, text=f"−{name}", width=5)
        bm.grid(row=r, column=i * 2, padx=2, pady=2)
        _bind_press_hold_repeat(
            bm,
            root,
            lambda ax=i: state.nudge_p_world(ax, -1.0, trans_step),
            interval_ms=hold_repeat_ms,
        )
        bp = ttk.Button(frm, text=f"+{name}", width=5)
        bp.grid(row=r, column=i * 2 + 1, padx=2, pady=2)
        _bind_press_hold_repeat(
            bp,
            root,
            lambda ax=i: state.nudge_p_world(ax, +1.0, trans_step),
            interval_ms=hold_repeat_ms,
        )
    r += 1

    ttk.Label(
        frm,
        text="Roll / Pitch / Yaw = small rotations about **world** X / Y / Z (not aerospace Euler)",
        wraplength=440,
    ).grid(row=r, column=0, columnspan=6, sticky="w", pady=(8, 0))
    r += 1
    for i, name in enumerate(("Roll (Wx)", "Pitch (Wy)", "Yaw (Wz)")):
        bm = ttk.Button(frm, text=f"−{name}", width=10)
        bm.grid(row=r, column=i * 2, padx=2, pady=2)
        _bind_press_hold_repeat(
            bm,
            root,
            lambda ax=i: state.nudge_rot_world(ax, -1.0, rot_step),
            interval_ms=hold_repeat_ms,
        )
        bp = ttk.Button(frm, text=f"+{name}", width=10)
        bp.grid(row=r, column=i * 2 + 1, padx=2, pady=2)
        _bind_press_hold_repeat(
            bp,
            root,
            lambda ax=i: state.nudge_rot_world(ax, +1.0, rot_step),
            interval_ms=hold_repeat_ms,
        )
    r += 1

    ttk.Label(frm, text="Quaternion wxyz (world, optional)").grid(
        row=r, column=0, columnspan=6, sticky="w", pady=(8, 0)
    )
    r += 1
    w_var = tk.StringVar(value="1")
    x_var = tk.StringVar(value="0")
    y_var = tk.StringVar(value="0")
    z_var = tk.StringVar(value="0")
    for j, (lab, var) in enumerate((("w", w_var), ("x", x_var), ("y", y_var), ("z", z_var))):
        ttk.Label(frm, text=lab).grid(row=r, column=j * 2, sticky="e")
        ttk.Entry(frm, textvariable=var, width=10).grid(row=r, column=j * 2 + 1, padx=2)
    r += 1

    def apply_quat() -> None:
        try:
            ww = float(w_var.get())
            xx = float(x_var.get())
            yy = float(y_var.get())
            zz = float(z_var.get())
        except ValueError:
            return
        state.apply_quat_world(ww, xx, yy, zz)

    ttk.Button(frm, text="Apply quat", command=apply_quat).grid(row=r, column=0, columnspan=2, pady=4)

    def reset_quat_identity() -> None:
        w_var.set("1")
        x_var.set("0")
        y_var.set("0")
        z_var.set("0")
        state.apply_quat_world(1.0, 0.0, 0.0, 0.0)

    ttk.Button(frm, text="Quat → identity", command=reset_quat_identity).grid(
        row=r, column=2, columnspan=2, pady=4
    )
    ttk.Button(frm, text="Sync from sim", command=state.request_sync).grid(
        row=r, column=4, columnspan=2, pady=4
    )
    r += 1

    ttk.Label(frm, text="Gripper (Jaw joint reference)").grid(
        row=r, column=0, columnspan=6, sticky="w", pady=(8, 0)
    )
    r += 1
    bo = ttk.Button(frm, text="Open +")
    bo.grid(row=r, column=0, columnspan=3, padx=2, pady=2)
    _bind_press_hold_repeat(
        bo,
        root,
        lambda: state.nudge_gripper(+gripper_delta, gripper_lo, gripper_hi),
        interval_ms=hold_repeat_ms,
    )
    bc = ttk.Button(frm, text="Close −")
    bc.grid(row=r, column=3, columnspan=3, padx=2, pady=2)
    _bind_press_hold_repeat(
        bc,
        root,
        lambda: state.nudge_gripper(-gripper_delta, gripper_lo, gripper_hi),
        interval_ms=hold_repeat_ms,
    )
    r += 1

    g_scale = ttk.Scale(
        frm,
        from_=gripper_lo,
        to=gripper_hi,
        orient=tk.HORIZONTAL,
        command=lambda v: state.set_gripper(float(v), gripper_lo, gripper_hi),
    )
    g_scale.grid(row=r, column=0, columnspan=6, sticky="ew", pady=4)
    _, _, g0 = state.snapshot_targets()
    g_scale.set(g0)
    r += 1

    ttk.Label(
        frm,
        text="Hold translate / rotate / grip buttons to repeat. Red arrow = target (arm may not fully reach).",
        wraplength=440,
    ).grid(row=r, column=0, columnspan=6, sticky="w")

    root.mainloop()
