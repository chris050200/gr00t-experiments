#!/usr/bin/env bash
# Canonical PnP A/B: same checkpoint + server; Side A = Isaac-GR00T rollout (RoboCasa DC),
# Side B = g1_minimal_sim run_gr00t_inference (``--scene table_pnp_apple``). See memory-bank/techContext.md
# § "Canonical PnP A/B".
#
# IMPORTANT — two Python interpreters (NVIDIA README):
#   Policy server → Isaac-GR00T project root .venv (or `uv run` from repo root) — has flash_attn for Eagle.
#   Sim clients   → GR00T-WholeBodyControl_uv/.venv — MuJoCo + RoboCasa; often does NOT have flash_attn.
#
# Usage:
#   ./scripts/run_pnp_inference_ab.sh print     # default: echo all commands
#   ./scripts/run_pnp_inference_ab.sh server    # terminal 1 — uses GR00T_SERVER_PYTHON (root .venv by default)
#   ./scripts/run_pnp_inference_ab.sh side-a    # terminal 2 — RoboCasa (GR00T_CLIENT_PYTHON)
#   ./scripts/run_pnp_inference_ab.sh side-b    # terminal 2 — minimal sim client
#
# Environment (override any):
#   GR00T_SERVER_PYTHON — interpreter for run_gr00t_server.py (default: Isaac-GR00T/.venv/bin/python)
#   GR00T_CLIENT_PYTHON — interpreter for rollout + run_gr00t_inference (default: WholeBodyControl .venv)
#   GR00T_VENV          — legacy alias: if set and GR00T_CLIENT_PYTHON unset, used for clients only
#   MODEL_PATH, POLICY_PORT, DEVICE, PNP_PROMPT, SIDE_A_*, SIDE_B_MAX_STEPS — see below

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G1_SIM="$(cd "${SCRIPT_DIR}/.." && pwd)"
GR00T2_ROOT="$(cd "${G1_SIM}/.." && pwd)"
ISAAC_GR00T="${GR00T2_ROOT}/Isaac-GR00T"

DEFAULT_SERVER_PY="${ISAAC_GR00T}/.venv/bin/python"
DEFAULT_CLIENT_PY="${ISAAC_GR00T}/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"

# Clients: prefer explicit GR00T_CLIENT_PYTHON; fall back to legacy GR00T_VENV; then default sim venv.
if [[ -n "${GR00T_CLIENT_PYTHON:-}" ]]; then
  CLIENT_PYTHON="${GR00T_CLIENT_PYTHON}"
elif [[ -n "${GR00T_VENV:-}" ]]; then
  CLIENT_PYTHON="${GR00T_VENV}"
else
  CLIENT_PYTHON="${DEFAULT_CLIENT_PY}"
fi

SERVER_PYTHON="${GR00T_SERVER_PYTHON:-${DEFAULT_SERVER_PY}}"

MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.6-G1-PnPAppleToPlate}"
POLICY_PORT="${POLICY_PORT:-2000}"
DEVICE="${DEVICE:-cuda}"
PNP_PROMPT="${PNP_PROMPT:-pick up the apple and place it on the plate}"
SIDE_A_N_ENVS="${SIDE_A_N_ENVS:-1}"
SIDE_A_EPISODES="${SIDE_A_EPISODES:-2}"
SIDE_B_MAX_STEPS="${SIDE_B_MAX_STEPS:-5000}"

cmd_server() {
  exec "${SERVER_PYTHON}" "${ISAAC_GR00T}/gr00t/eval/run_gr00t_server.py" \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper \
    --device "${DEVICE}" \
    --host 0.0.0.0 \
    --port "${POLICY_PORT}"
}

cmd_side_a() {
  cd "${ISAAC_GR00T}"
  exec "${CLIENT_PYTHON}" gr00t/eval/rollout_policy.py \
    --policy_client_host 127.0.0.1 \
    --policy_client_port "${POLICY_PORT}" \
    --model_path "" \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --n_episodes "${SIDE_A_EPISODES}" \
    --max_episode_steps 5000 \
    --n_action_steps 30 \
    --n_envs "${SIDE_A_N_ENVS}"
}

cmd_side_b() {
  cd "${G1_SIM}"
  exec "${CLIENT_PYTHON}" scripts/run_gr00t_inference.py \
    --policy-host 127.0.0.1 \
    --policy-port "${POLICY_PORT}" \
    --scene table_pnp_apple \
    --prompt "${PNP_PROMPT}" \
    --max-steps "${SIDE_B_MAX_STEPS}" \
    --n-action-steps 30 \
    --plan-hz 50.0 \
    --arm-action-mode auto \
    --debug-policy-prints
}

print_block() {
  cat <<EOF

# --- Two interpreters (fixes "flash_attn not installed" if server used sim .venv by mistake) ---
#   Server:  ${SERVER_PYTHON}
#   Clients: ${CLIENT_PYTHON}
#
# Optional overrides:
#   export GR00T_SERVER_PYTHON='${DEFAULT_SERVER_PY}'
#   export GR00T_CLIENT_PYTHON='${DEFAULT_CLIENT_PY}'
#   export MODEL_PATH='${MODEL_PATH}'
#   export POLICY_PORT=${POLICY_PORT}
#   export DEVICE=cuda

# Diagnose both envs:
#   bash '${G1_SIM}/scripts/diagnose_gr00t_policy_server.sh'

# =============================================================================
# Terminal 1 — policy server (${DEVICE})  [use SERVER python = root .venv]
# =============================================================================
cd '${ISAAC_GR00T}'
'${SERVER_PYTHON}' gr00t/eval/run_gr00t_server.py \\
  --model-path '${MODEL_PATH}' \\
  --embodiment-tag UNITREE_G1 \\
  --use-sim-policy-wrapper \\
  --device '${DEVICE}' \\
  --host 0.0.0.0 \\
  --port ${POLICY_PORT}

# Same env via uv (from Isaac-GR00T root):
#   cd '${ISAAC_GR00T}' && uv run python gr00t/eval/run_gr00t_server.py \\
#     --model-path '${MODEL_PATH}' --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper \\
#     --device '${DEVICE}' --host 0.0.0.0 --port ${POLICY_PORT}

# =============================================================================
# Terminal 2a — Side A: RoboCasa client [WholeBodyControl / sim venv]
# =============================================================================
cd '${ISAAC_GR00T}'
'${CLIENT_PYTHON}' gr00t/eval/rollout_policy.py \\
  --policy_client_host 127.0.0.1 \\
  --policy_client_port ${POLICY_PORT} \\
  --model_path "" \\
  --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \\
  --n_episodes ${SIDE_A_EPISODES} \\
  --max_episode_steps 5000 \\
  --n_action_steps 30 \\
  --n_envs ${SIDE_A_N_ENVS}

# =============================================================================
# Terminal 2b — Side B: run_gr00t_inference.py (PolicyClient → same server)
# =============================================================================
cd '${G1_SIM}'
'${CLIENT_PYTHON}' scripts/run_gr00t_inference.py \\
  --policy-host 127.0.0.1 \\
  --policy-port ${POLICY_PORT} \\
  --scene table_pnp_apple \\
  --prompt '${PNP_PROMPT}' \\
  --max-steps ${SIDE_B_MAX_STEPS} \\
  --n-action-steps 30 \\
  --plan-hz 50.0 \\
  --arm-action-mode auto \\
  --debug-policy-prints

# Optional Side B video:
#   add:  --save-video /tmp/table_pnp_infer.mp4

EOF
}

usage() {
  echo "Usage: $0 {print|server|side-a|side-b}"
  echo "  print   — show copy-paste commands (default)"
  echo "  server  — run policy server (GR00T_SERVER_PYTHON, default: Isaac-GR00T/.venv)"
  echo "  side-a  — RoboCasa rollout_policy (GR00T_CLIENT_PYTHON, default: WholeBodyControl .venv)"
  echo "  side-b  — minimal sim run_gr00t_inference (same client python)"
  echo ""
  echo "G1_SIM=${G1_SIM}"
  echo "ISAAC_GR00T=${ISAAC_GR00T}"
  echo "SERVER_PYTHON=${SERVER_PYTHON}"
  echo "CLIENT_PYTHON=${CLIENT_PYTHON}"
  if [[ ! -f "${SERVER_PYTHON}" ]]; then
    echo "WARNING: server python missing: ${SERVER_PYTHON}"
  fi
  if [[ ! -f "${CLIENT_PYTHON}" ]]; then
    echo "WARNING: client python missing: ${CLIENT_PYTHON}"
  fi
}

SUB="${1:-print}"
case "${SUB}" in
  print)
    usage
    print_block
    echo "# How to read A vs B: A ok + B bad → tune minimal-sim obs/camera/prompt/cadence."
    echo "# A bad → fix install, checkpoint, RoboCasa setup, or ports before tuning B."
    ;;
  server) cmd_server ;;
  side-a) cmd_side_a ;;
  side-b) cmd_side_b ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 1
    ;;
esac
