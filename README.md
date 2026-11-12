# TPdK Overlay — real-time lecture transcription, translation, and summary

Desktop overlay for exchange students in online lectures (TPdK Kelompok 2, ITB).
It captures lecture audio and a chosen screen region, transcribes and translates
locally, extracts slide text locally, and writes periodic and final summaries.

The selected architecture is the hybrid local–cloud alternative (Bab III §3.3.5):
Qwen3-ASR-0.6B (vLLM streaming), NLLB-200-distilled-600M, and PP-OCRv6_small run
on the machine; only the summarizer calls the cloud (Gemini API). Audio and
screen pixels never leave the device, and only text reaches the cloud.

On this workstation the machine default is `asr.backend: llama_cpp`: Qwen3-ASR
runs as a GGUF model in a local `llama serve` process and captions get partial
previews every 5 s. `http` (vLLM in WSL2) remains the report path.
See [ASR backends](#asr-backends). Translation defaults to `nllb_ct2`
(CTranslate2 int8 on CPU, zero VRAM); see
[Translation](#translation-nllb-200).

Implementation status and the build order live in [AGENT.md](AGENT.md).

## Processes

| Process | Where | Command |
|---|---|---|
| ASR server (Qwen3-ASR GGUF + llama.cpp) | Windows, `127.0.0.1:8081` | `powershell -File scripts\serve_asr_llama.ps1` |
| ASR service (Qwen3-ASR + vLLM, optional) | WSL2 Ubuntu, `127.0.0.1:8000` | `bash scripts/serve_asr_wsl.sh` |
| Backend (capture, MT, OCR, summaries) | Windows | `.\.venv\Scripts\python.exe -m tpdk_overlay.server.app` |
| Overlay UI (PySide6) | Windows | not built yet — AGENT.md phase 4 |

The backend listens on `127.0.0.1:8765`: `GET /api/health` and WebSocket `/ws`.

## ASR backends

| `asr.backend` | Runs in | Partials | Requires |
|---|---|---|---|
| `llama_cpp` (default on this machine) | local `llama serve` process, GGUF, `127.0.0.1:8081` | yes, every 5 s over the last 4 s of audio | official llama CLI (`llama` on `PATH`; install with `irm https://llama.app/install.ps1 \| iex`), Qwen3-ASR-0.6B-GGUF (~1.8 GiB total measured) |
| `http` (report default) | WSL2 Ubuntu + vLLM service, `127.0.0.1:8000` | yes, streaming | `scripts/setup_asr_wsl.sh`, service running |

`llama_cpp` needs no torch or transformers for ASR: the backend posts each
utterance window as base64 WAV to llama serve's OpenAI-compatible
`/v1/chat/completions` endpoint and parses the Qwen reply (`language <NAME>`
plus the transcribed text). Start the server before the backend:

```powershell
# Wraps: llama serve -hf ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0 --host 127.0.0.1
#   --port 8081 -ngl all -c 4096 -np 1 -n 128 --cache-type-k q8_0 --cache-type-v q8_0
powershell -ExecutionPolicy Bypass -File scripts\serve_asr_llama.ps1
```

The model and its mmproj download on the first run. Install the official llama
CLI once (Windows PowerShell):

```powershell
irm https://llama.app/install.ps1 | iex
llama --version
```

The installer selects a prebuilt backend and places `llama.exe` on `PATH`.
Start the ASR server with the launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve_asr_llama.ps1
```

The launcher runs `llama serve` on `127.0.0.1:8081`; `-Llama`, `-Model`, and
`-Port` override the defaults. `-n 128` is the server-side backstop against
greedy repetition loops, and `asr.llama_cpp.max_new_tokens` stays at 128
client-side (partials use 64).

Language is always auto-detected by the server: `asr.language` is ignored by
this backend (the adapter logs a warning if it is set). Partials preview the
tail of the current utterance every `asr.llama_cpp.partial_interval_ms` (5000 ms
by default); the final pass transcribes the whole utterance. Windows quieter
than `asr.llama_cpp.min_energy_dbfs` (-40 dBFS) never reach the server, and
music/silence hallucinations plus repetition-loop outputs are dropped.
Measured here with the 0.6B Q8_0 server: repeated live backend smoke on
5.1 s, 16.2 s, and 25.6 s clips returned finals in 0.26 s, 0.62 s, and 0.96 s
after utterance end (across runs, ~0.3–1.4 s); previews land at 5/10/15/20/25 s
of audio. `nvidia-smi` showed ~1.8 GiB total with the server loaded. The first
preview still arrives near 5 s of audio, so this is a chunked preview, not
NFR-02's ≤1 s first-partial target.

`http` is the report's streaming service. It runs in WSL2 and shares the 4 GB
GPU with NLLB; if VRAM is tight, prefer `llama_cpp` or a smaller/quantized HTTP
model.

```powershell
# Optional override (default is llama_cpp in configs/default.yaml)
$env:TPDK__ASR__BACKEND = "http"     # vLLM service in WSL2
.\.venv\Scripts\python.exe -m tpdk_overlay.server.app
```

## Translation (NLLB-200)

`mt.provider` selects how the same `facebook/nllb-200-distilled-600M`
checkpoint runs. Only final transcripts and OCR blocks reach this stage; no
audio or pixels are involved.

| `mt.provider` | Runs as | Device | Dependency |
|---|---|---|---|
| `nllb_ct2` (default) | CTranslate2 int8 | CPU (zero VRAM), CUDA optional | `ct2` extra |
| `nllb` | transformers, fp16 | CUDA when available, CPU float32 otherwise | `ml` extra |

The default CT2 path keeps the 4 GB GPU free for ASR. The `ct2` extra and one
conversion are required on first use (the output directory is gitignored):

```powershell
.\.venv\Scripts\pip.exe install -e ".[ct2]"
.\.venv\Scripts\ct2-transformers-converter.exe `
  --model facebook/nllb-200-distilled-600M `
  --output_dir data/models/nllb-ct2 --quantization int8
```

If a conversion already exists (for example the old Live Subtitle prototype's
`Live Subtitle/models/nllb-ct2`), point `mt.ct2.model` at that directory instead
of re-converting.

Then the default runs with no overrides. To use the optional GPU path instead:

```powershell
$env:TPDK__MT__PROVIDER = "nllb"
$env:TPDK__MT__DEVICE = "cuda"
.\.venv\Scripts\python.exe -m tpdk_overlay.server.app
```

`mt.batch_size` caps both sentence and OCR-block batches; `mt.max_new_tokens`
bounds the translation length, not the input (`mt.max_input_tokens` truncates
long inputs).

Measured on this workstation (RTX 3050 4 GB, Ryzen 5 5600H, 8 lecture
sentences): the default `nllb_ct2` int8 on CPU has p50 ≈ 0.79–0.84 s/sentence
at 1.46 GB RAM and zero VRAM; `nllb` fp16 on CUDA has p50 ≈ 0.46–0.49 s/sentence
at 1.22 GB VRAM and 1.73 GB RAM; `nllb` fp32 on CPU is ≈ 2.3 s/sentence at
3.2 GB RAM. The CT2 default keeps the GPU with ASR; use `nllb` on CUDA when
there is VRAM headroom. See AGENT.md §7 for the full table.

## First-time setup

```powershell
# 1. Windows environment (Python 3.12 venv + CUDA torch + project extras)
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

# 2. ASR service environment (inside WSL2 Ubuntu, only for asr.backend: http)
wsl -d Ubuntu -- bash /mnt/c/path/to/repo/scripts/setup_asr_wsl.sh
```

Copy `.env.example` to `.env` and fill `GEMINI_API_KEY`. Then start the ASR
server and verify:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve_asr_llama.ps1
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe -m pytest
```

## Configuration

`configs/default.yaml` holds every knob. Override any key with an environment
variable using `TPDK__` and `__` nesting, e.g. `TPDK__SERVER__PORT=9000`,
`TPDK__ASR__LLAMA_CPP__BASE_URL=http://127.0.0.1:9001`, or
`TPDK__OCR__ENABLED=false`. The schema is `src/tpdk_overlay/config.py`.

## Licenses

NLLB-200 is CC-BY-NC-4.0 (non-commercial); Gemini free tier may use inputs for
product improvement. This project targets the academic TPdK prototype only.
