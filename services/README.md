# ASR service (Qwen3-ASR + vLLM)

vLLM has no supported native Windows build. The ASR service therefore runs in
WSL2 Ubuntu and exposes localhost HTTP; the Windows backend treats it as a
black-box adapter (`src/tpdk_overlay/asr/client.py`).

`services/asr_service.py` is a small vendored wrapper around
`qwen_asr.cli.demo_streaming`. It keeps the same streaming contract and adds
forced-language support (`/api/start?language=English`), which the upstream demo
does not expose. The Windows side is
`src/tpdk_overlay/asr/http_backend.py`.

## Setup (inside WSL2 Ubuntu)

```bash
bash scripts/setup_asr_wsl.sh
```

This creates `~/.venvs/tpdk-asr` and installs
`services/requirements-asr.txt` (`qwen-asr[vllm]`, Flask, NumPy). It requires
the NVIDIA Windows driver with WSL GPU passthrough (`nvidia-smi` must work
inside WSL).

## Run

```bash
bash scripts/serve_asr_wsl.sh
# overrides: ASR_MODEL=Qwen/Qwen3-ASR-1.7B GPU_MEM_UTIL=0.55 PORT=8000 \
#            bash scripts/serve_asr_wsl.sh
```

The script runs `services/asr_service.py` and binds `127.0.0.1:8000`. Windows
reaches it at `http://127.0.0.1:8000` (WSL localhost forwarding).

## HTTP contract used by the backend

| Method | Path | Input | Response |
|---|---|---|---|
| GET | `/` | — | service metadata, used by `scripts/check_env.py` |
| POST | `/api/start?language=<name>&context=<text>` | both optional | `{"session_id": "<hex>"}` |
| POST | `/api/chunk?session_id=...` | raw `float32` little-endian, 16 kHz mono | `{"language": str, "text": str}` |
| POST | `/api/finish?session_id=...` | — | `{"language": str, "text": str}` |

* `text` is **cumulative** for that stream: replace the displayed partial, do
  not append.
* Streaming mode returns no timestamps; the backend timestamps VAD utterances
  and chunks itself.
* One stream = one utterance. Call `finish` when VAD closes the utterance, then
  open a new stream for the next one.
* `language` accepts Qwen3-ASR language names such as `English` and
  `Indonesian`; omit it to auto-detect. Forcing the source language improves
  short-utterance stability and keeps ASR aligned with the MT direction.

## Language forcing

Upstream `qwen_asr.cli.demo_streaming` calls
`init_streaming_state(unfixed_chunk_num=..., unfixed_token_num=..., chunk_size_sec=...)`
and exposes no language parameter. `services/asr_service.py` passes
`context` and `language` through to `init_streaming_state`, so
`ports.AsrClient.open(language="English")` is honored. If you fall back to the
upstream demo, ASR auto-detects and only MT uses the selected source language.

## GPU budget (RTX 3050 Laptop, 4 GB)

vLLM preallocates `gpu_memory_utilization × VRAM`, so the default is `0.45`
(~1.8 GB). NLLB-200-distilled-600M in fp16 takes roughly 1.2 GB on the Windows
side, and OCR stays on CPU. Raise the ASR share only after re-measuring NFR-05
(total local model VRAM ≤ 4 GB).

## Verify

```bash
curl -s http://127.0.0.1:8000/            # {"service":"tpdk-asr",...}
nvidia-smi                                 # python process holds ~1.8 GB
```

## Native Windows fallback (no WSL)

When WSL2/vLLM is unavailable, set `asr.backend: native` (or
`TPDK__ASR__BACKEND=native`). The Windows backend then loads
`qwen-asr` with its transformers engine in-process
(`src/tpdk_overlay/asr/transformers_backend.py`); no HTTP service is used.

Trade-offs: the transformers engine has no streaming decode, so each VAD
utterance is buffered and transcribed once at `finish` (final-only, no partial
captions), and the model competes with NLLB for the same 4 GB of VRAM. Use it as
a backup for demos without WSL, not as the latency-critical default.
