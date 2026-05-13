#!/usr/bin/env bash
# Quick checks for "why does run_gr00t_server.py fail?" — especially flash_attn + venv mixups.
# Run from anywhere:
#   bash /path/to/g1_minimal_sim/scripts/diagnose_gr00t_policy_server.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G1_SIM="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC="$(cd "${G1_SIM}/../Isaac-GR00T" && pwd)"
ROOT_PY="${ISAAC}/.venv/bin/python"
WBC_PY="${ISAAC}/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"

check_py() {
  local label="$1" exe="$2"
  echo "======== ${label} ========"
  if [[ ! -f "${exe}" ]]; then
    echo "MISSING: ${exe}"
    echo
    return
  fi
  echo "executable: ${exe}"
  "${exe}" - <<'PY' 2>&1 || true
import sys
print("version", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
except Exception as e:
    print("torch", "ERR", e)
try:
    import flash_attn
    v = getattr(flash_attn, "__version__", "imported")
    print("flash_attn", v)
except Exception as e:
    print("flash_attn", "MISSING or ERR:", e)
try:
    import mujoco
    print("mujoco", mujoco.__version__)
except Exception as e:
    print("mujoco", "MISSING (ok for server-only)", e)
PY
  echo
}

echo "Isaac-GR00T root: ${ISAAC}"
echo
check_py "A) Root uv / project .venv (NVIDIA: use THIS for run_gr00t_server.py)" "${ROOT_PY}"
check_py "B) WholeBodyControl sim .venv (use for rollout_policy + run_gr00t_inference; often NO flash_attn)" "${WBC_PY}"

echo "======== Interpretation ========"
echo "  Policy server loads the VLA (Eagle + flash). It needs flash_attn in the SAME interpreter as run_gr00t_server.py."
echo "  Minimal sim client needs mujoco — usually the WholeBodyControl venv, not necessarily the root .venv."
echo
echo "======== Fix (preferred): run server with root .venv ========"
echo "cd '${ISAAC}'"
echo "'${ROOT_PY}' gr00t/eval/run_gr00t_server.py --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \\"
echo "  --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper --device cuda --host 0.0.0.0 --port 2000"
echo "# or: cd '${ISAAC}' && uv run python gr00t/eval/run_gr00t_server.py ... (same env if uv wired to .venv)"
echo
echo "======== Fix (alternative): install flash-attn into WholeBodyControl venv ========"
echo "# Only if you must use one Python for everything — build must match that venv's torch/CUDA."
echo "# '${WBC_PY}' -m pip install 'flash-attn' --no-build-isolation"
echo "# (often needs nvcc + matching CUDA; prebuilt wheels are torch-version-specific.)"
echo
