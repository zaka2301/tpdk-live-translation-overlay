# TPdK Overlay — real-time lecture transcription, translation, and summary

Desktop overlay for exchange students in online lectures (TPdK Kelompok 2, ITB).
It captures lecture audio and a chosen screen region, transcribes and translates
locally, extracts slide text locally, and writes periodic and final summaries.

The selected architecture is the hybrid local–cloud alternative (Bab III §3.3.5):
Qwen3-ASR-0.6B, PLaMo-3-Translate-0.8B (GGUF, GPU), and PP-OCRv6_small run on the
machine; NLLB-200-distilled-600M remains the CPU fallback. Only the summarizer
calls the cloud (Gemini API). Audio and screen pixels never leave the device,
and only text reaches the cloud.

On this workstation the machine default is `asr.backend: llama_cpp`: Qwen3-ASR
runs as a GGUF model in a local `llama serve` process and captions get partial
previews every 5 s. `http` (vLLM in WSL2) remains the report path.
See [ASR backends](#asr-backends). Translation defaults to `plamo_llama`:
PLaMo-3-Translate 0.8B GGUF on the GPU behind a second `llama serve` process;
see [Translation](#translation-plamo--nllb-200).

Implementation status and the build order live in [AGENTS.md](AGENTS.md).

## One command

`main.py` starts and supervises the four local processes:

```powershell
python main.py
```

If `python` is not the project venv, the launcher restarts itself with
`.venv\Scripts\python.exe`. It waits for the ASR, PLaMo, and backend health
endpoints before opening the overlay, reuses services that are already running,
and stops the ones it started when the overlay exits (tray **Quit** or Ctrl+C).
`--no-asr`, `--no-mt`, `--no-backend`, and `--model <llama -hf spec>` are
available for partial stacks and alternate checkpoints.

## Processes

| Process | Where | Command |
|---|---|---|
| ASR server (Qwen3-ASR GGUF + llama.cpp) | Windows, `127.0.0.1:8081` | `powershell -File scripts\serve_asr_llama.ps1` |
| PLaMo translation server (GGUF + llama.cpp) | Windows, `127.0.0.1:8082` | `powershell -File scripts\serve_mt_plamo.ps1` |
| ASR service (Qwen3-ASR + vLLM, optional) | WSL2 Ubuntu, `127.0.0.1:8000` | `bash scripts/serve_asr_wsl.sh` |
| Backend (capture, MT, OCR, summaries) | Windows | `.\.venv\Scripts\python.exe -m tpdk_overlay.server.app` |
| Overlay UI (PySide6) | Windows | `powershell -File scripts\run_overlay.ps1` |
| All of the above | One console | `python main.py` |

The backend listens on `127.0.0.1:8765`: `GET /api/health` and WebSocket `/ws`.

## Caption overlay

`python main.py` starts the ASR server, PLaMo server, backend, and overlay
together. To run the pieces manually instead, start the backend first and the
overlay in a second terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_overlay.ps1
```

The caption window is frameless, translucent, and always on top. It starts in
click-through mode, so clicks go to the lecture app beneath it. `Ctrl+Alt+L` or
the tray menu toggles interactive mode, which reveals the control bar and
enables dragging the window or resizing it from the bottom-right grip. If the
hotkey is already registered by another app, the tray menu is the fallback; when
neither the tray nor the hotkey is available, the overlay starts interactive so
it cannot lock itself out.

- **Start / Stop** sends `session.start` / `session.stop`; Pause and Resume are
  in the tray menu.
- **From / To** and the **Summary** interval send `settings.update` while a
  session is active (they set the values for the next session otherwise).
- **Region** opens a full-desktop rubber-band picker and sends `region.update`
  in physical pixels (monitor index plus left/top/width/height), which is what
  the mss capture adapter consumes.
- **History** lists final source/translation pairs in a scrollable window;
  **Export...** writes them to a markdown file.

Source text stays small and the translation is large. Partials replace the
current segment, finals accumulate in history, and the UI acknowledges every
painted final with `render.ack` so NFR-01/NFR-03 latency rows can be computed.
The backend starts capture without waiting for the translation model: source captions appear about two seconds after speech starts (the llama.cpp
preview cadence), the translation model finishes loading in the
background (the first segment may show `translating...` briefly), and the model
stays loaded for later sessions so subsequent Starts are sub-second.
The VAD caps one utterance at 10 s, and the caption ellipsizes anything that
still overflows (newest words for a partial, the sentence start for a final);
the full text is always available in History. A translated final stays on
screen for at least `ui.caption.final_hold_ms` (2.2 s by default); partials and
source-only previews are coalesced during that hold, while resolved finals
always paint so their latency measurement is not lost.
The WebSocket client runs on a background thread with 0.5 s -> 5 s capped
backoff, so a restarted backend is picked up within five seconds; the status
line shows `backend offline - retrying` in the meantime.

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
GPU with the translation model; if VRAM is tight, prefer `llama_cpp` or a
smaller/quantized HTTP model.

```powershell
# Optional override (default is llama_cpp in configs/default.yaml)
$env:TPDK__ASR__BACKEND = "http"     # vLLM service in WSL2
.\.venv\Scripts\python.exe -m tpdk_overlay.server.app
```

## Translation (PLaMo / NLLB-200)

`mt.provider` selects how final transcripts and OCR blocks are translated; no
audio or pixels reach this stage.

| `mt.provider` | Runs as | Device | Server / dependency |
|---|---|---|---|
| `plamo_llama` (default) | PLaMo-3-Translate 0.8B GGUF, Q4_K_M | GPU (`-ngl all`) | local `llama serve` on `127.0.0.1:8082` |
| `nllb_ct2` | CTranslate2 int8 | CPU (zero VRAM), CUDA optional | `ct2` extra + converted model |
| `nllb` | transformers, fp16 | CUDA when available, CPU float32 otherwise | `ml` extra |

PLaMo runs in its own llama.cpp process. `main.py` starts it with the rest of
the stack; to start it alone:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve_mt_plamo.ps1
```

The GGUF is read from `data/models/plamo-3-translate/` when it exists; otherwise
the launcher downloads `pfnet/plamo-3-translate-0.8b-GGUF:Q4_K_M` on first run.
PLaMo adds ~0.7 GiB VRAM next to the ASR server's ~1.7 GiB; both measured
together at ~2.6 GiB of the 4 GB card.

Measured on this workstation (RTX 3050, Ryzen 5 5600H, the eight lecture
sentences from `scripts/bench_mt.py`): PLaMo Q4_K_M on the GPU has p50 ≈ 0.16 s
en→id and ≈ 0.10 s id→en, 0.72 GiB VRAM, 0.96 GiB server RSS. On CPU (6
threads) it is p50 ≈ 1.13 s / 0.65 s, too slow for the caption path. NLLB CT2
int8 on CPU is p50 ≈ 0.63 s / 0.86 s at 1.46 GiB RAM and zero VRAM; NLLB fp16 on
CUDA is p50 ≈ 0.44 s / 0.46 s at 1.22 GiB VRAM. PLaMo needs a second `llama
serve`; NLLB runs in-process.

Batches (OCR blocks) are fanned out to `mt.plamo_llama.concurrency` parallel
requests; keep that at or below the server's `-np` slots. Supported languages
are the 53 PLaMo codes; `zh` maps to `zh-hans` and `fil` to `tl`. The overlay's
From/To menus are built from the active provider's set, so unsupported pairs
cannot be selected. Unsupported pairs fail per segment and leave the source
text visible. Always pass
`source_lang`/`target_lang`: omitting `target_lang` makes the model switch
between Japanese and English.

To use the CPU fallback instead:

```powershell
$env:TPDK__MT__PROVIDER = "nllb_ct2"
.\.venv\Scripts\python.exe -m tpdk_overlay.server.app
```

The `ct2` extra and one conversion are required on first use (the output
directory is gitignored):

```powershell
.\.venv\Scripts\pip.exe install -e ".[ct2]"
.\.venv\Scripts\ct2-transformers-converter.exe `
  --model facebook/nllb-200-distilled-600M `
  --output_dir data/models/nllb-ct2 --quantization int8
```

If a conversion already exists (for example the old Live Subtitle prototype's
`Live Subtitle/models/nllb-ct2`), point `mt.ct2.model` at that directory instead
of re-converting. The optional GPU path is `TPDK__MT__PROVIDER=nllb` with
`TPDK__MT__DEVICE=cuda`.

`mt.batch_size` caps both sentence and OCR-block batches; `mt.max_new_tokens`
bounds the translation length, not the input (`mt.max_input_tokens` truncates
long inputs for NLLB).

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

NLLB-200 is CC-BY-NC-4.0 (non-commercial). PLaMo-3-Translate is released under
the PLaMo Community License, which is free for personal and academic use.
Gemini free tier may use inputs for product improvement. This project targets
the academic TPdK prototype only.
