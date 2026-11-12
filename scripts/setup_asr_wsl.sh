#!/usr/bin/env bash
# Run inside WSL2 Ubuntu. Creates an isolated Linux environment for the ASR
# service; vLLM has no native Windows build.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

python3 -c 'import sys; assert sys.version_info >= (3, 10), f"need Python >= 3.10, got {sys.version}"'
nvidia-smi >/dev/null || {
  echo "nvidia-smi is not available in WSL. Enable GPU passthrough first." >&2
  exit 1
}

VENV="${HOME}/.venvs/tpdk-asr"
if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv "${VENV}" || {
    echo "python3 -m venv failed. Install it with: sudo apt install python3-venv" >&2
    exit 1
  }
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_DIR}/services/requirements-asr.txt"

python - <<'PY'
import vllm
import qwen_asr

print("qwen-asr ready; vllm", vllm.__version__)
PY
