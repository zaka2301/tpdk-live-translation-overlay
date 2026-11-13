# TPdK Overlay

Real-time lecture captions for exchange students. It listens to system audio (or
a microphone), transcribes and translates locally with GPUs, redraws slide text
in place, and writes checkpoint/final lecture summaries. Audio and screen
pixels never leave the machine; only transcript and OCR text reach Gemini.

Local stack: Qwen3-ASR-0.6B and PLaMo-3-Translate-0.8B GGUF behind two
`llama serve` processes, PP-OCRv6_small via PaddleOCR, a FastAPI backend on one
WebSocket, and PySide6 overlay windows.
## Requirements

- Windows 11 x64, PowerShell, Python 3.12 from python.org (the bootstrap script
  creates the venv with the `py -3.12` launcher).
- [uv](https://docs.astral.sh/uv/) on `PATH` (the bootstrap script installs with it).
- NVIDIA GPU and driver (tested on an RTX 3050 Laptop, 4 GB). ASR + translation
  + OCR fit in ~3.2 GB VRAM. There are CPU fallbacks for OCR and NLLB, but the
  default stack expects the GPU.
- The official llama.cpp CLI, for the two model servers:

  ```powershell
  irm https://llama.app/install.ps1 | iex
  llama --version
  ```

- Optional: a Gemini API key for summaries. Without it, transcription and
  translation still work; summaries degrade.
- WSL2 Ubuntu only if you want the optional `asr.backend: http` (vLLM) path.

## Setup from scratch

Clone the repo, then work from the repository root in PowerShell:

```powershell
git clone <repo-url> "tpdk translation overlay"
cd "tpdk translation overlay"

# 1. Python 3.12 venv, CUDA 13 torch, all extras, Paddle GPU, env check
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

# 2. Local settings
Copy-Item .env.example .env
notepad .env            # set GEMINI_API_KEY=...
```

`bootstrap.ps1` creates `.venv`, installs the CUDA 13.0 torch wheel before the
project extras (`ui,ml,ocr,summarize,ct2,dev`), installs `paddlepaddle-gpu`
plus the DLL hook that lets it share torch's CUDA libraries, then runs
`scripts\check_env.py`. Missing optional pieces are reported as `WARN`.

Verify the install:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe -m pytest
```

## Run

```powershell
python main.py
```

That one command starts and supervises the whole stack:

| Process | Endpoint |
|---|---|
| ASR server (`llama serve`, Qwen3-ASR GGUF) | `127.0.0.1:8081` |
| Translation server (`llama serve`, PLaMo GGUF) | `127.0.0.1:8082` |
| Backend (capture, VAD, MT, OCR, summaries) | `127.0.0.1:8765` (`/api/health`, `/ws`) |
| Overlay UI (PySide6) | desktop |

If `python` is not the project venv, the launcher re-executes itself with
`.venv\Scripts\python.exe`. It waits for each health endpoint before opening the
overlay, and reuses servers that are already running.

**First run downloads the model weights** (Qwen3-ASR ≈ 0.7 GB, PLaMo ≈ 0.55 GB)
and loads PaddleOCR, so expect several minutes before the first caption.
Later starts are fast.

Useful flags: `--no-asr`, `--no-mt`, `--no-backend` (reuse a server you started
yourself) and `--model <llama -hf spec>` (try a different ASR checkpoint).

Quit from the tray menu or press `Ctrl+C` in the launcher console. Only the
processes this launcher started are stopped.

### Start the pieces manually

Useful while developing:

```powershell
# Terminal 1 + 2: model servers (leave running)
powershell -ExecutionPolicy Bypass -File scripts\serve_asr_llama.ps1
powershell -ExecutionPolicy Bypass -File scripts\serve_mt_plamo.ps1

# Terminal 3: backend
powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1

# Terminal 4: overlay
powershell -ExecutionPolicy Bypass -File scripts\run_overlay.ps1
```

## Using the overlay

The caption window is frameless, translucent, always on top, and starts in
click-through mode, so clicks reach the lecture app beneath it.

- `Ctrl+Alt+L` or the tray menu toggles interactive mode: this shows the control
  bar and enables dragging and the bottom-right resize grip. If the hotkey is
  taken, the tray menu still works; if neither is available, the overlay starts
  interactive.
- **Start / Stop** controls the session (Pause/Resume are in the tray menu).
  `ui.auto_start` starts a session automatically when the overlay connects.
- **From / To** and the **Summary** interval apply to the current or next
  session.
- **Region** opens a rubber-band picker over the desktop; the OCR overlay then
  translates slide text inside that region.
- **History** keeps final source/translation pairs; **Export...** writes them to
  a Markdown file.

Source text is small, the translation is large. Partials update the current
segment in place and finals move to history. Painted finals and OCR frames are
acknowledged to the backend, which records their latency in `metrics.csv`.
Captions ellipsize when they overflow; the full text stays in History.

## Configuration

`configs/default.yaml` is the runtime source of truth (`src/tpdk_overlay/config.py`
is the schema). Override any key with `TPDK__` plus the YAML path in uppercase,
using `__` between levels:

```powershell
$env:TPDK__SERVER__PORT = "9000"
$env:TPDK__OCR__DEVICE = "cpu"
$env:TPDK__ASR__BACKEND = "http"
```

The knobs you are most likely to touch:

| Key | Default | Meaning |
|---|---|---|
| `audio.source` | `loopback` | `loopback` (system audio) or `microphone` |
| `audio.input_device` | `null` | device name; `null` = system default |
| `app.language.source/target` | `en`/`id` | initial caption direction |
| `summary.checkpoint_interval_s` | `300` | checkpoint summary period |
| `summary.enabled` | `true` | set `false` to run fully local |
| `ocr.enabled` / `screen.enabled` | `true` | turn off slide OCR |
| `mt.translate_partials` | `true` | translate live previews too |

Session artifacts go to `data/sessions/<session_id>/`: `session.json`,
`transcript.jsonl`, `slides.jsonl`, checkpoint and final summaries, and
`metrics.csv`.

## Alternatives

- **Translation without the GPU** — `TPDK__MT__PROVIDER=nllb_ct2` runs NLLB-200
  int8 on CPU (zero VRAM). Convert it once:
  `.\.venv\Scripts\ct2-transformers-converter.exe --model facebook/nllb-200-distilled-600M --output_dir data/models/nllb-ct2 --quantization int8`.
  The transformers GPU path is `TPDK__MT__PROVIDER=nllb` with `TPDK__MT__DEVICE=cuda`.
- **vLLM streaming ASR (report path)** — run `bash scripts/setup_asr_wsl.sh`
  once inside WSL2, start `bash scripts/serve_asr_wsl.sh`, then set
  `TPDK__ASR__BACKEND=http`. It shares the 4 GB card with the translation model.
- **CPU OCR** — `TPDK__OCR__DEVICE=cpu` (several seconds per frame instead of
  ~0.4 s on GPU).

## Troubleshooting

- **`ASR server did not become healthy`** — `llama` is not on `PATH`, or the
  first-run model download is still going. Run `llama --version`; the launcher
  waits up to 5 minutes per model server.
- **`backend offline - retrying`** — the backend process died; check its
  terminal. The overlay reconnects within 5 s of a restart.
- **Captions stay empty on loopback** — some WASAPI devices open but deliver
  silence. Set `audio.source: microphone` or pick another device via
  `audio.input_device`.
- **No summaries / `degraded` status** — `GEMINI_API_KEY` missing, or the cloud
  call failed. Transcription continues; check `scripts/smoke_summary.py`.
- **Out of VRAM** — use `nllb_ct2` for translation or `TPDK__OCR__DEVICE=cpu`,
  and make sure only one process is serving each `llama` model.
- **Unsupported language pair** — PLaMo supports 53 codes; the From/To menus
  only offer valid pairs for the active provider.

## Licenses

NLLB-200 is CC-BY-NC-4.0 (non-commercial). PLaMo-3-Translate uses the PLaMo
Community License, free for personal and academic use. The Gemini free tier may
use inputs for product improvement. This project is an academic TPdK prototype.
