#!/usr/bin/env bash
# Run inside WSL2 Ubuntu, after scripts/setup_asr_wsl.sh.
# Serves services/asr_service.py (Qwen3-ASR + vLLM) on 127.0.0.1:8000.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV="${HOME}/.venvs/tpdk-asr"

if [ ! -x "${VENV}/bin/python" ]; then
  echo "Missing ${VENV}. Run scripts/setup_asr_wsl.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Triton compiles a C launcher at engine init; the WSL image lacks python3.12-dev.
# Reuse the extracted headers when present (same as the benchmark launchers).
if [ -d "${HOME}/.cache/tpdk-pydeps/root/usr/include/python3.12" ]; then
  export CPATH="${HOME}/.cache/tpdk-pydeps/root/usr/include:${HOME}/.cache/tpdk-pydeps/root/usr/include/python3.12${CPATH:+:${CPATH}}"
fi

ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-0.6B}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.68}"   # 4 GB card, single client; lower = less KV cache
SWAP_SPACE="${SWAP_SPACE:-0}"           # GiB CPU KV swap; 0 for the 8 GB WSL VM
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}" # longest utterance + reply, not vLLM's 65536 default
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"      # one real-time stream
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

EAGER_ARGS=()
if [ "${ENFORCE_EAGER}" = "1" ]; then
  EAGER_ARGS+=(--enforce-eager)
fi

exec python "${REPO_DIR}/services/asr_service.py" \
  --asr-model-path "${ASR_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --swap-space "${SWAP_SPACE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${EAGER_ARGS[@]}" \
  --chunk-size-sec 1.0 \
  --unfixed-chunk-num 4 \
  --unfixed-token-num 5
