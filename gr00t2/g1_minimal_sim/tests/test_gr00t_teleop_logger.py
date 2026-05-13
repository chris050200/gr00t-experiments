"""P2 teleop logger: action extraction + NPZ roundtrip (needs WholeBodyControl + ONNX)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("gymnasium")

_WBC = (
    _ROOT.parent
    / "Isaac-GR00T"
    / "external_dependencies"
    / "GR00T-WholeBodyControl"
).resolve()


def test_gr00t_episode_logger_npz_roundtrip() -> None:
    if not _WBC.is_dir():
        pytest.skip(f"Missing GR00T-WholeBodyControl at {_WBC}")

    sys.path.insert(0, str(_WBC))
    pytest.importorskip("gr00t_wbc")

    import gymnasium as gym

    from g1_gear_wbc_env import ENV_ID, register_g1_gear_wbc_env
    from gr00t_observation_builder import Gr00tObservationBuilder
    from gr00t_teleop_logger import Gr00tTeleopEpisodeLogger

    register_g1_gear_wbc_env()
    env = gym.make(ENV_ID, scene="stylish_diner", use_hands=True)
    env.reset()
    rt = env.unwrapped._rt
    ob = Gr00tObservationBuilder.for_locomanip_default()
    log = Gr00tTeleopEpisodeLogger(
        rt,
        obs_builder=ob,
        task_description="pytest pick cube",
        include_video=False,
    )
    try:
        # Logger samples at 50 Hz against sim dt=0.005 (200 Hz): every 4 sim steps.
        for _ in range(20):
            env.step(np.zeros(1, dtype=np.float32))
            log.record_step()
        tmp = Path(tempfile.mkdtemp()) / "episode_000.npz"
        log.save_npz(tmp)
        z = np.load(tmp)
        assert int(z["T"]) == 5
        assert z["action_navigate_command"].shape == (5, 3)
        assert z["action_left_arm"].shape == (5, 7)
        assert z["state_left_arm"].shape == (5, 7)
        # Relative-arm default: action deltas should equal commanded-target minus current state.
        assert np.max(np.abs(z["action_left_arm"])) < 2.0
        meta_path = tmp.parent / "episode_000_metadata.json"
        assert meta_path.is_file()
    finally:
        ob.close()
        env.close()
