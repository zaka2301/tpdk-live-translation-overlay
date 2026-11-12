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

ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-0.6B}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"   # leave VRAM for NLLB on the same 4 GB GPU
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

exec python "${REPO_DIR}/services/asr_service.py" \
  --asr-model-path "${ASR_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --chunk-size-sec 1.0 \
  --unfixed-chunk-num 4 \
  --unfixed-token-num 5
