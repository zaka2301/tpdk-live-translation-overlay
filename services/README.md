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
# single-client defaults (RTX 3050 4 GB):
#   GPU_MEM_UTIL=0.68 MAX_MODEL_LEN=1024 MAX_NUM_BATCHED_TOKENS=512
#   MAX_NUM_SEQS=1
# overrides: ASR_MODEL=Qwen/Qwen3-ASR-1.7B GPU_MEM_UTIL=0.55 PORT=8000 \
#            bash scripts/serve_asr_wsl.sh
```

The script runs `services/asr_service.py` and binds `127.0.0.1:8000`. Windows
reaches it at `http://127.0.0.1:8000` (WSL localhost forwarding).
`MAX_NUM_SEQS=1` is deliberate: this service is a single real-time stream, not a
batch server. `ENFORCE_EAGER=1` trades CUDA-graph memory for decode speed if you
need a lower `GPU_MEM_UTIL`.
`SWAP_SPACE=0` is deliberate: vLLM's default 4 GiB CPU KV swap does not fit the
8 GB WSL VM alongside the engine. There is no startup inference warmup; the
first utterance after a fresh start can be slower.

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

vLLM preallocates `gpu_memory_utilization × VRAM`; it is not demand-paged. The
single-client profile is `GPU_MEM_UTIL=0.68`, `MAX_MODEL_LEN=1024`,
`MAX_NUM_BATCHED_TOKENS=512`, `MAX_NUM_SEQS=1`, which measured ~3.3 GB VRAM
in the production benchmark (a warmup transient touched ~3.56 GB). The old
`0.45` default cannot boot with vLLM's
default `max_num_batched_tokens=8192` / `max_model_len=65536`; the reduced
profile is what fits. `0.60` with a 256-token batch crashed the engine on the
first chunk (`CUDA error: device not ready`); `0.60/512` was stable but the long
clip's finish latency rose to ~426 ms (vs ~92 ms at `0.68`), so `0.68` is the
default. NLLB-200-distilled-600M in fp16 takes roughly 1.2 GB on the Windows
side, so even this profile leaves too little for ASR + NLLB; use the Windows
`llama_cpp` backend (or a quantized/smaller HTTP model) when both must coexist.
OCR stays on CPU.
The WSL VM is capped at 8 GB (`.wslconfig`); keep `swap_space=0`. The default
4 GiB CPU swap plus the ~4.5 GB engine RSS left no headroom and likely
contributed to the host OOM during testing. `swap_space=0` is safe here because
`MAX_NUM_SEQS=1` means there is no preemption to swap.
Raise the ASR share only after re-measuring NFR-05 (total local model VRAM ≤ 4 GB).

## Verify

```bash
curl -s http://127.0.0.1:8000/            # {"service":"tpdk-asr",...}
nvidia-smi                                 # ~3.3 GB at 0.68
```

