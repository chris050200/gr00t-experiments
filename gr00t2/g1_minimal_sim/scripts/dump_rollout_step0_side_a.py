#!/usr/bin/env python3
"""Tier-A oracle Phase-3 step-0 dumper for ``LMPnPAppleToPlateDC_G1_gear_wbc``.

Companion to ``run_gr00t_inference.py --dump-step0-dir`` (side-B): builds the
SAME env chain ``rollout_policy.py`` uses (``get_groot_locomanip_env_fn`` +
``MultiStepWrapper`` + ``SyncVectorEnv``), resets once, sends the resulting
observation to ``PolicyClient.get_action``, and saves the exact request payload,
the raw action chunk, the ego frame as PNG, and a ``metadata.json``. Pair with
``diff_step0_payloads.py`` to localise where minimal-sim and the upstream
eval diverge in the observation -> action contract.

Run from ``g1_minimal_sim/`` against the same policy server side-B uses::

    ../Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \\
        scripts/dump_rollout_step0_side_a.py \\
        --policy-host 127.0.0.1 --policy-port 2000 \\
        --out-dir data/ab_pnp_20260512/step0_side_a

Then run side-B's dumper into a sibling directory and diff the payloads.

The env name default mirrors the CLI the user has confirmed works with the
CloudWalk PnP checkpoint (see ``activeContext.md`` and the side-A video at
``data/ab_pnp_20260512/side_a_rollout/...``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _ensure_imports() -> None:
    # The script is intended to run inside the Isaac-GR00T WholeBodyControl
    # venv (sibling repo); we add its package roots to ``sys.path`` so users
    # don't have to ``pip install -e`` it. Mirrors ``gr00t_observation_builder``.
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # gr00t2/
    isaac_root = repo_root / "Isaac-GR00T"
    wbc_root = isaac_root / "external_dependencies" / "GR00T-WholeBodyControl"
    for r in (str(isaac_root), str(wbc_root)):
        if Path(r).is_dir() and r not in sys.path:
            sys.path.insert(0, r)


def _save_frame_png(path: Path, rgb: np.ndarray) -> None:
    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), np.asarray(rgb, dtype=np.uint8))
        print(f"saved frame: {path}")
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not save frame png ({exc})")


def _save_payload(
    out_dir: Path,
    *,
    policy_obs: dict[str, Any],
    action_raw: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_to_save: dict[str, np.ndarray] = {}
    for k, v in policy_obs.items():
        if isinstance(v, np.ndarray):
            obs_to_save[k] = v
        elif isinstance(v, (tuple, list)) and len(v) > 0 and isinstance(v[0], str):
            obs_to_save[k] = np.asarray(v, dtype=object)
        else:
            obs_to_save[k] = np.asarray(v)
    np.savez(out_dir / "obs_step0.npz", **obs_to_save)

    action_to_save: dict[str, np.ndarray] = {}
    for k, v in action_raw.items():
        action_to_save[k] = np.asarray(v, dtype=np.float32)
    np.savez(out_dir / "action_step0.npz", **action_to_save)

    ego_key = None
    for cand in ("video.ego_view", "video.ego_view_image"):
        if cand in policy_obs:
            ego_key = cand
            break
    if ego_key is not None:
        ego = np.asarray(policy_obs[ego_key])
        while ego.ndim > 3:
            ego = ego[0]
        _save_frame_png(out_dir / "ego_view_step0.png", ego)
    else:
        print("WARNING: no ego_view key in policy_obs; skipping PNG dump")

    with (out_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)
    print(f"dumped step-0 payload to {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-host", type=str, default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=2000)
    p.add_argument(
        "--env-name",
        type=str,
        default="gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
        help="Tier-A oracle env (matches the rollout_policy.py CLI the user confirmed works).",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=30,
        help="MultiStepWrapper n_action_steps. Default 30 matches CloudWalk PnP eval.",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=720,
        help="MultiStepWrapper max_episode_steps. Only affects env init, not what we dump.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--reset-seed",
        type=int,
        default=None,
        help="Optional seed for env.reset(seed=...). Match side-B by leaving unset for default.",
    )
    args = p.parse_args()

    _ensure_imports()

    import gymnasium as gym
    from gr00t.eval.rollout_policy import (
        MultiStepConfig,
        VideoConfig,
        WrapperConfigs,
        create_eval_env,
    )
    from gr00t.policy.server_client import PolicyClient

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=None,  # skip video recording wrapper to keep obs minimal
            max_episode_steps=args.max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=True,
        ),
    )

    print(f"creating Tier-A env: {args.env_name}")
    env_fns = [
        lambda: create_eval_env(
            env_name=args.env_name,
            env_idx=0,
            total_n_envs=1,
            wrapper_configs=wrapper_configs,
        )
    ]
    env = gym.vector.SyncVectorEnv(env_fns)

    print("env.reset(...) ...")
    reset_kwargs: dict[str, Any] = {}
    if args.reset_seed is not None:
        reset_kwargs["seed"] = int(args.reset_seed)
    observations, _info = env.reset(**reset_kwargs)

    print(f"connecting to PolicyClient at {args.policy_host}:{args.policy_port} ...")
    client = PolicyClient(host=args.policy_host, port=args.policy_port, strict=False)
    if not client.ping():
        raise RuntimeError(
            f"cannot reach policy server at {args.policy_host}:{args.policy_port}; "
            "start run_gr00t_server.py first"
        )
    client.reset()

    print("obs keys:")
    for k in sorted(observations.keys()):
        v = observations[k]
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v).__name__}")

    print("client.get_action(observations) ...")
    action_raw, _info_action = client.get_action(observations)

    print("action keys:")
    for k in sorted(action_raw.keys()):
        v = np.asarray(action_raw[k])
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} L2={float(np.linalg.norm(v)):.4f}")

    _save_payload(
        Path(args.out_dir),
        policy_obs=observations,
        action_raw=action_raw,
        metadata={
            "side": "A_rollout_oracle",
            "env_name": args.env_name,
            "n_action_steps": int(args.n_action_steps),
            "max_episode_steps": int(args.max_episode_steps),
            "reset_seed": (None if args.reset_seed is None else int(args.reset_seed)),
            "policy_host": str(args.policy_host),
            "policy_port": int(args.policy_port),
            "obs_keys": sorted(observations.keys()),
            "action_keys": sorted(action_raw.keys()),
        },
    )

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
