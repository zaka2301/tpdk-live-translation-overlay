# TPdK Overlay — real-time lecture transcription, translation, and summary

Desktop overlay for exchange students in online lectures (TPdK Kelompok 2, ITB).
It captures lecture audio and a chosen screen region, transcribes and translates
locally, extracts slide text locally, and writes periodic and final summaries.

The selected architecture is the hybrid local–cloud alternative (Bab III §3.3.5):
Qwen3-ASR-0.6B (vLLM streaming), NLLB-200-distilled-600M, and PP-OCRv6_small run
on the machine; only the summarizer calls the cloud (Gemini API). Audio and
screen pixels never leave the device, and only text reaches the cloud.

When WSL2/vLLM is unavailable, `asr.backend: native` runs Qwen3-ASR with the
transformers backend inside the Windows backend process (final-only transcript,
no partials). See [ASR backends](#asr-backends).

Implementation status and the build order live in [AGENT.md](AGENT.md).

## Processes

| Process | Where | Command |
|---|---|---|
| ASR service (Qwen3-ASR + vLLM) | WSL2 Ubuntu | `bash scripts/serve_asr_wsl.sh` |
| Backend (capture, MT, OCR, summaries) | Windows | `.\.venv\Scripts\python.exe -m tpdk_overlay.server.app` |
| Overlay UI (PySide6) | Windows | not built yet — AGENT.md phase 4 |

The backend listens on `127.0.0.1:8765`: `GET /api/health` and WebSocket `/ws`.
The ASR service listens on `127.0.0.1:8000`.

## ASR backends

| `asr.backend` | Runs in | Partials | Requires |
|---|---|---|---|
| `http` (default) | WSL2 Ubuntu + vLLM service | yes, streaming | `scripts/setup_asr_wsl.sh`, service running |
| `native` | Windows backend process (`qwen-asr` + transformers) | no, final-only | `.[asr-native]` extra, ~1.3 GB weights |

Switch per run with `TPDK__ASR__BACKEND=native` or in `configs/default.yaml`.
Native mode needs no WSL and no second process, but it transcribes only after the
utterance ends and shares the 4 GB GPU with NLLB. If VRAM is tight, set
`asr.native.device: cpu` (much slower) or return to the HTTP backend.

```powershell
# Native fallback without WSL
$env:TPDK__ASR__BACKEND = "native"
.\.venv\Scripts\python.exe -m tpdk_overlay.server.app
```

## First-time setup

```powershell
# 1. Windows environment (Python 3.12 venv + CUDA torch + project extras)
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

# 2. ASR environment (inside WSL2 Ubuntu, once)
wsl -d Ubuntu -- bash /mnt/c/path/to/repo/scripts/setup_asr_wsl.sh
```

Copy `.env.example` to `.env` and fill `GEMINI_API_KEY`. Then verify:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe -m pytest
```

## Configuration

`configs/default.yaml` holds every knob. Override any key with an environment
variable using `TPDK__` and `__` nesting, e.g. `TPDK__SERVER__PORT=9000` or
`TPDK__OCR__ENABLED=false`. The schema is `src/tpdk_overlay/config.py`.

## Licenses

NLLB-200 is CC-BY-NC-4.0 (non-commercial); Gemini free tier may use inputs for
product improvement. This project targets the academic TPdK prototype only.
