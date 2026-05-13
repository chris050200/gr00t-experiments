"""One-command Tier-A vs Tier-B GR00T policy I/O A/B dump harness.

What this does, in order:

1. Verify the policy server is reachable (TCP probe on ``--policy-host:--policy-port``).
2. Run the **oracle** client (``Isaac-GR00T/gr00t/eval/rollout_policy.py``) with
   ``--policy-ab-dump <run-id>/oracle`` — captures the step-0 policy input dict
   + raw returned action chunks against a known-good RoboCasa env reset.
3. Run the **subject** client (``g1_minimal_sim/scripts/run_gr00t_inference.py``)
   with ``--policy-ab-dump <run-id>/subject`` — same step-0 capture against
   the minimal-sim env reset.
4. Load both dumps and print a per-key max-|diff| table for every shared
   ``state.*`` / ``annotation.*`` / ``video.*`` / ``action.*`` key. Exit
   non-zero if any ``state.*`` key has ``max|diff| > --state-atol`` (the
   hard Bug #8 regression gate) or if any ``action.*`` key has
   ``max|diff| > --action-atol`` (softer; helpful when iterating on the
   server config).

Run from ``g1_minimal_sim/`` (the script self-locates Isaac-GR00T as
``../Isaac-GR00T`` by default — override with ``--isaac-groot-root``):

.. code-block:: bash

    python scripts/run_policy_ab_dump.py \\
        --policy-host 127.0.0.1 --policy-port 2000 \\
        --run-id post4a_bis

Defaults are tuned for fast iteration: oracle runs only ``--max_episode_steps 200``
(dump captures step 0, so longer is wasted wall clock) and the subject's
``--save-video`` is intentionally NOT enabled by default. Pass
``--save-video PATH`` if you want the subject to also produce an MP4 for
side-by-side visual inspection.

Pinned by the same regression that built it: ``tests/test_policy_ab_dump.py``
covers the underlying ``policy_ab_dump.dump_policy_roundtrip_pack`` writer;
``tests/test_runtime_leg_default_pose.py`` covers the Bug #8 invariant this
harness measures end-to-end. Canonical protocol + interpretation matrix:
``memory-bank/gr00t_compatible_data_collection.md`` §6.7.

For **two existing** dump trees without re-running clients, use
``scripts/compare_policy_ab_dumps.py`` (``--exit-on never`` when XML/scenes differ;
``--exit-on strict`` matches this harness's exit gate).
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_MIN_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(_MIN_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIN_SIM_ROOT))

from policy_ab_compare import diff_pack_roots

# ANSI colour codes (graceful no-op when stdout is not a TTY).
_USE_COLOUR = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    if not _USE_COLOUR:
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def _green(s: str) -> str:
    return _c("32", s)


def _red(s: str) -> str:
    return _c("31", s)


def _yellow(s: str) -> str:
    return _c("33", s)


def _bold(s: str) -> str:
    return _c("1", s)


# --------------------------------------------------------------------------- #
# Server probe                                                                #
# --------------------------------------------------------------------------- #


def probe_server(host: str, port: int, *, timeout_s: float = 1.5) -> bool:
    """Return True iff ``host:port`` accepts a TCP connection within ``timeout_s``."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        return s.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# Client launchers                                                            #
# --------------------------------------------------------------------------- #


def _resolve_venv_python(isaac_groot_root: Path) -> Path:
    """Locate the WholeBodyControl uv venv python both clients require."""
    p = (
        isaac_groot_root
        / "gr00t"
        / "eval"
        / "sim"
        / "GR00T-WholeBodyControl"
        / "GR00T-WholeBodyControl_uv"
        / ".venv"
        / "bin"
        / "python"
    )
    if not p.is_file():
        raise FileNotFoundError(
            f"Could not find WholeBodyControl venv python at {p}. "
            "Pass --venv-python if your checkout puts it elsewhere, or run "
            "``bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh`` "
            "in the Isaac-GR00T repo first."
        )
    return p


def run_oracle(
    *,
    venv_python: Path,
    isaac_groot_root: Path,
    policy_host: str,
    policy_port: int,
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    n_action_steps: int,
    n_envs: int,
    dump_dir: Path,
    extra_args: list[str],
    timeout_s: float | None,
) -> int:
    """Run ``rollout_policy.py`` with ``--policy-ab-dump`` and return its exit code."""
    cmd = [
        str(venv_python),
        "gr00t/eval/rollout_policy.py",
        "--policy_client_host",
        policy_host,
        "--policy_client_port",
        str(policy_port),
        "--model_path",
        "",
        "--env_name",
        env_name,
        "--n_episodes",
        str(n_episodes),
        "--max_episode_steps",
        str(max_episode_steps),
        "--n_action_steps",
        str(n_action_steps),
        "--n_envs",
        str(n_envs),
        "--policy-ab-dump",
        str(dump_dir),
        *extra_args,
    ]
    print(_bold("[oracle] $ ") + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(isaac_groot_root), timeout=timeout_s)


def run_subject(
    *,
    venv_python: Path,
    minimal_sim_root: Path,
    policy_host: str,
    policy_port: int,
    scene: str,
    max_steps: int,
    n_action_steps: int,
    dump_dir: Path,
    save_video: Path | None,
    extra_args: list[str],
    timeout_s: float | None,
) -> int:
    """Run ``scripts/run_gr00t_inference.py`` with ``--policy-ab-dump``."""
    cmd = [
        str(venv_python),
        "scripts/run_gr00t_inference.py",
        "--scene",
        scene,
        "--policy-host",
        policy_host,
        "--policy-port",
        str(policy_port),
        "--max-steps",
        str(max_steps),
        "--n-action-steps",
        str(n_action_steps),
        "--policy-ab-dump",
        str(dump_dir),
    ]
    if save_video is not None:
        cmd.extend(["--save-video", str(save_video)])
    cmd.extend(extra_args)
    print(_bold("[subject] $ ") + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(minimal_sim_root), timeout=timeout_s)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=2000)
    p.add_argument(
        "--run-id",
        default=None,
        help="Run id for dump dirs (default: timestamp). "
        "Final layout: ``--out-root/<run-id>/{oracle,subject}/manifest.json``.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/tmp/gr00t_ab"),
        help="Parent directory under which ``<run-id>/{oracle,subject}/`` is created.",
    )
    p.add_argument(
        "--env-name",
        default="gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
        help="Oracle Gym env id (RoboCasa locomanip env).",
    )
    p.add_argument(
        "--scene",
        default="table_pnp_apple",
        help="Subject scene (``g1_gear_wbc_env.G1GearWBCEnv`` ``scene=`` kwarg).",
    )
    p.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Oracle ``--n_episodes`` (dump captures step 0 so 1 is enough).",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Oracle ``--max_episode_steps`` (dump captures step 0; default is small for fast iter).",
    )
    p.add_argument(
        "--subject-max-steps",
        type=int,
        default=200,
        help="Subject ``--max-steps`` (same dump-captures-step-0 logic).",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=30,
        help="Action chunk length (matches ``unitree_g1`` modality).",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Oracle ``--n_envs``. Should be 1 for clean A/B.",
    )
    p.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Optional path: also save the subject's MP4 (oracle saves its own under sim_eval_videos_*).",
    )
    p.add_argument(
        "--isaac-groot-root",
        type=Path,
        default=None,
        help="Isaac-GR00T checkout root. Default: ../Isaac-GR00T relative to this script.",
    )
    p.add_argument(
        "--venv-python",
        type=Path,
        default=None,
        help="Override path to the WholeBodyControl uv venv python.",
    )
    p.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip the oracle run (re-use an existing dump under ``--out-root/<run-id>/oracle``).",
    )
    p.add_argument(
        "--skip-subject",
        action="store_true",
        help="Skip the subject run (re-use an existing dump under ``--out-root/<run-id>/subject``).",
    )
    p.add_argument(
        "--diff-only",
        action="store_true",
        help="Skip BOTH client runs; only load + diff existing dumps. Implies --skip-oracle --skip-subject.",
    )
    p.add_argument(
        "--state-atol",
        type=float,
        default=1e-4,
        help="Hard tolerance for ``state.*`` keys (Bug #8 gate). Exceeding => exit non-zero.",
    )
    p.add_argument(
        "--action-atol",
        type=float,
        default=5e-2,
        help="Soft tolerance for ``action.*`` keys. Exceeding => exit non-zero.",
    )
    p.add_argument(
        "--video-atol",
        type=float,
        default=10.0,
        help="Soft tolerance on ``video.*`` max|diff| in RGB units. Used for the WARN/FAIL label only.",
    )
    p.add_argument(
        "--client-timeout-s",
        type=float,
        default=600.0,
        help="Subprocess timeout per client (oracle and subject share the same value).",
    )
    p.add_argument(
        "--oracle-arg",
        action="append",
        default=[],
        help="Pass-through extra argv to rollout_policy.py (repeatable).",
    )
    p.add_argument(
        "--subject-arg",
        action="append",
        default=[],
        help="Pass-through extra argv to run_gr00t_inference.py (repeatable).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    minimal_sim_root = Path(__file__).resolve().parents[1]
    if args.isaac_groot_root is None:
        isaac_groot_root = (minimal_sim_root.parent / "Isaac-GR00T").resolve()
    else:
        isaac_groot_root = args.isaac_groot_root.resolve()
    if not isaac_groot_root.is_dir():
        print(_red(f"Isaac-GR00T root not found: {isaac_groot_root}"), file=sys.stderr)
        return 2

    venv_python = (
        args.venv_python.resolve()
        if args.venv_python is not None
        else _resolve_venv_python(isaac_groot_root)
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.out_root / run_id
    oracle_dir = base / "oracle"
    subject_dir = base / "subject"
    base.mkdir(parents=True, exist_ok=True)

    print(_bold("=" * 78))
    print(_bold(f"  GR00T policy I/O A/B run  id={run_id}"))
    print(_bold("=" * 78))
    print(f"  isaac_groot_root : {isaac_groot_root}")
    print(f"  minimal_sim_root : {minimal_sim_root}")
    print(f"  venv_python      : {venv_python}")
    print(f"  base out         : {base}")
    print(f"  policy server    : {args.policy_host}:{args.policy_port}")

    skip_oracle = args.skip_oracle or args.diff_only
    skip_subject = args.skip_subject or args.diff_only

    # Server probe (only needed if we're going to actually run a client).
    if not (skip_oracle and skip_subject):
        if probe_server(args.policy_host, args.policy_port):
            print(_green(f"  server probe     : OK (TCP connect to {args.policy_host}:{args.policy_port})"))
        else:
            print(
                _red(
                    f"  server probe     : FAILED to connect to "
                    f"{args.policy_host}:{args.policy_port}.\n"
                    "  Start it first:\n"
                    f"    cd {isaac_groot_root}\n"
                    f"    {venv_python} gr00t/eval/run_gr00t_server.py \\\n"
                    "      --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper \\\n"
                    f"      --port {args.policy_port} --device cuda \\\n"
                    "      --model-path <your-checkpoint-or-HF-id>\n"
                )
            )
            return 3

    # Oracle
    if skip_oracle:
        print(_yellow("[oracle] skipped (--skip-oracle / --diff-only)"))
    else:
        t0 = time.monotonic()
        rc = run_oracle(
            venv_python=venv_python,
            isaac_groot_root=isaac_groot_root,
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            env_name=args.env_name,
            n_episodes=args.n_episodes,
            max_episode_steps=args.max_episode_steps,
            n_action_steps=args.n_action_steps,
            n_envs=args.n_envs,
            dump_dir=oracle_dir,
            extra_args=list(args.oracle_arg),
            timeout_s=args.client_timeout_s,
        )
        dt = time.monotonic() - t0
        if rc != 0:
            print(_red(f"[oracle] exit code {rc} after {dt:.1f}s"))
            return rc
        print(_green(f"[oracle] OK in {dt:.1f}s -> {oracle_dir}"))

    # Subject
    if skip_subject:
        print(_yellow("[subject] skipped (--skip-subject / --diff-only)"))
    else:
        t0 = time.monotonic()
        rc = run_subject(
            venv_python=venv_python,
            minimal_sim_root=minimal_sim_root,
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            scene=args.scene,
            max_steps=args.subject_max_steps,
            n_action_steps=args.n_action_steps,
            dump_dir=subject_dir,
            save_video=args.save_video,
            extra_args=list(args.subject_arg),
            timeout_s=args.client_timeout_s,
        )
        dt = time.monotonic() - t0
        if rc != 0:
            print(_red(f"[subject] exit code {rc} after {dt:.1f}s"))
            return rc
        print(_green(f"[subject] OK in {dt:.1f}s -> {subject_dir}"))

    # Diff
    if not oracle_dir.is_dir() or not (oracle_dir / "manifest.json").is_file():
        print(_red(f"oracle dump dir missing or has no manifest: {oracle_dir}"), file=sys.stderr)
        return 4
    if not subject_dir.is_dir() or not (subject_dir / "manifest.json").is_file():
        print(_red(f"subject dump dir missing or has no manifest: {subject_dir}"), file=sys.stderr)
        return 4

    return diff_pack_roots(
        oracle_dir,
        subject_dir,
        label_a="oracle",
        label_b="subject",
        state_atol=args.state_atol,
        action_atol=args.action_atol,
        video_atol=args.video_atol,
        exit_on="strict",
    )


if __name__ == "__main__":
    raise SystemExit(main())
