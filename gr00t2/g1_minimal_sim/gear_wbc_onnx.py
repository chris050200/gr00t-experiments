"""ONNX policy runner (CPU) for Gear WBC stand/walk."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_onnx_runner(path: str):
    import onnxruntime as ort

    if not Path(path).is_file():
        policy_dir = Path(path).parent
        raise FileNotFoundError(
            f"ONNX policy not found: {path}\n"
            f"Expected one of in {policy_dir}:\n"
            "  ft92.onnx or GR00T-WholeBodyControl-Balance.onnx (stand)\n"
            "  ft109.onnx or GR00T-WholeBodyControl-Walk.onnx (walk)\n"
            "From GR00T-WholeBodyControl repo root: git lfs pull\n"
            "Or run Isaac-GR00T: bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh"
        )
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    def run(inp: np.ndarray) -> np.ndarray:
        out = session.run(None, {in_name: inp.astype(np.float32)})[0]
        return np.asarray(out, dtype=np.float32).squeeze()

    return run
