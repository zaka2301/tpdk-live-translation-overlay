"""Latency/throughput/resource benchmark for the PLaMo-3-Translate GGUF server.

The model runs in a separate ``llama serve`` process, so this script talks to its
OpenAI-compatible endpoint and cannot time model loading itself; start the
server first (GPU and CPU variants differ only by ``-ngl``)::

    llama serve -m data/models/plamo-3-translate/plamo-3-translate-0.8b-Q4_K_M.gguf \
        -c 16384 -np 8 --port 8082 --alias plamo-translate --jinja \
        --no-prefill-assistant -ngl 99
    python scripts/bench_mt_plamo.py --server-pid <pid>

    # CPU-only comparison: restart with -ngl 0 -t <physical cores>
    python scripts/bench_mt_plamo.py --server-pid <pid> --label plamo-gguf-cpu-q4km

It reuses the exact eight lecture sentences and measurement shape of
``bench_mt.py`` (single-sentence latency, one eight-sentence pass, plus a
concurrent eight-request pass, which is the batch analogue for a one-text-per-
request server). A JSON report lands in
``data/benchmarks/mt_plamo_bench_<label>_<UTC stamp>.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from bench_mt import _UNSAFE_LABEL, EN, ID, _rss_mib, _stats  # noqa: E402

SCHEMA = "tpdk-mt-plamo-bench-v1"
MIB = 1024.0 * 1024.0

_DIRECTIONS: tuple[tuple[str, str, Sequence[str]], ...] = (
    ("en", "id", EN),
    ("id", "en", ID),
)


class ServerClient:
    """Minimal OpenAI-compatible chat client for one translation per request."""

    def __init__(self, base_url: str, model: str, max_tokens: int, timeout: float) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
        self._model = model
        self._max_tokens = max_tokens

    def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        response = self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": text}],
                "chat_template_kwargs": {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                },
                "temperature": 0,
                "max_tokens": self._max_tokens,
                "stop": ["<|plamo:tag|>", "<|plamo:bos|>"],
            },
        )
        response.raise_for_status()
        payload = response.json()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        content = payload["choices"][0]["message"]["content"].strip()
        timings = payload.get("timings") or {}
        return content, {
            "elapsed_ms": round(elapsed_ms, 3),
            "prompt_ms": round(float(timings.get("prompt_ms", 0.0)), 3),
            "predicted_ms": round(float(timings.get("predicted_ms", 0.0)), 3),
            "predicted_n": timings.get("predicted_n"),
        }

    def close(self) -> None:
        self._client.close()


class VramSampler:
    """Poll ``nvidia-smi`` in the background and keep the peak."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_mib = 0.0
        self.baseline_mib = 0.0

    @staticmethod
    def _used_mib() -> float:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return float(out.strip().splitlines()[0])

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.peak_mib = max(self.peak_mib, self._used_mib())
            except (OSError, ValueError, subprocess.SubprocessError):
                return

    def __enter__(self) -> VramSampler:
        try:
            self.baseline_mib = self._used_mib()
            self.peak_mib = self.baseline_mib
            self._thread.start()
        except (OSError, ValueError, subprocess.SubprocessError):
            self.baseline_mib = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)


def _measure_direction(
    client: ServerClient,
    texts: Sequence[str],
    *,
    source_lang: str,
    target_lang: str,
    reps: int,
) -> dict[str, Any]:
    single_ms: list[float] = []
    outputs: list[str] = []
    predicted_n: list[int] = []
    prompt_ms: list[float] = []
    for _ in range(reps):
        for text in texts:
            output, meta = client.translate(text, source_lang, target_lang)
            single_ms.append(meta["elapsed_ms"])
            prompt_ms.append(meta["prompt_ms"])
            if meta["predicted_n"] is not None:
                predicted_n.append(int(meta["predicted_n"]))
            outputs.append(output)

    sequential_pass_ms: list[float] = []
    for _ in range(reps):
        started = time.perf_counter()
        for text in texts:
            client.translate(text, source_lang, target_lang)
        sequential_pass_ms.append((time.perf_counter() - started) * 1000.0)

    concurrent_pass_ms: list[float] = []
    concurrent_request_ms: list[float] = []
    with ThreadPoolExecutor(max_workers=len(texts)) as pool:
        for _ in range(reps):
            started = time.perf_counter()
            results = list(
                pool.map(
                    lambda text: client.translate(text, source_lang, target_lang)[1]["elapsed_ms"],
                    texts,
                )
            )
            concurrent_pass_ms.append((time.perf_counter() - started) * 1000.0)
            concurrent_request_ms.extend(results)

    return {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "sentences": len(texts),
        "single_ms": _stats(single_ms),
        "single_ms_samples": [round(value, 3) for value in single_ms],
        "prompt_ms_median": round(statistics.median(prompt_ms), 3),
        "predicted_tokens_median": statistics.median(predicted_n) if predicted_n else None,
        "sequential_pass_ms": _stats(sequential_pass_ms),
        "sequential_sentences_per_second": round(
            len(texts) * 1000.0 / statistics.median(sequential_pass_ms), 3
        ),
        "concurrent_pass_ms": _stats(concurrent_pass_ms),
        "concurrent_request_ms": _stats(concurrent_request_ms),
        "outputs": outputs[-len(texts) :],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a running PLaMo-3-Translate llama-server instance.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--model", default="plamo-translate")
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument(
        "--vram-idle-mib",
        type=float,
        default=None,
        help="GPU memory used before the server started (for model_peak accounting)",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--label", default="plamo-gguf")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.reps < 1:
        raise ValueError("--reps must be >= 1")

    client = ServerClient(args.base_url, args.model, args.max_tokens, args.timeout)
    server_rss_before = (
        psutil.Process(args.server_pid).memory_info().rss / MIB
        if args.server_pid is not None
        else None
    )
    try:
        # Untimed warmup so the first measured call sees steady-state kernels.
        client.translate(EN[0], "en", "id")
        client.translate(ID[0], "id", "en")

        with VramSampler() as vram:
            directions = {
                f"{source}->{target}": _measure_direction(
                    client,
                    texts,
                    source_lang=source,
                    target_lang=target,
                    reps=args.reps,
                )
                for source, target, texts in _DIRECTIONS
            }
    finally:
        client.close()

    server_rss_after = (
        psutil.Process(args.server_pid).memory_info().rss / MIB
        if args.server_pid is not None
        else None
    )

    stamp = datetime.now(UTC)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "label": args.label,
        "timestamp": stamp.isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "reps": args.reps,
        "sentences": len(EN),
        "rss_client_mib": round(_rss_mib(), 3),
        "rss_server_mib": (
            {
                "before": round(server_rss_before, 3),
                "after": round(server_rss_after, 3),
            }
            if server_rss_before is not None and server_rss_after is not None
            else None
        ),
        "vram_mib": {
            "baseline": round(vram.baseline_mib, 3),
            "peak": round(vram.peak_mib, 3),
            "model_peak": round(
                max(0.0, vram.peak_mib - (args.vram_idle_mib or vram.baseline_mib)), 3
            ),
        },
        "directions": directions,
    }

    output = args.json_out or (
        PROJECT_ROOT
        / "data"
        / "benchmarks"
        / f"mt_plamo_bench_{_UNSAFE_LABEL.sub('-', args.label)}_"
        f"{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"label: {payload['label']}  base_url: {payload['base_url']}  model: {payload['model']}")
    print(f"reps: {payload['reps']}  sentences: {payload['sentences']}")
    if payload["rss_server_mib"]:
        print(
            f"server rss: {payload['rss_server_mib']['before']:.1f} -> "
            f"{payload['rss_server_mib']['after']:.1f} MiB"
        )
    print(
        f"vram: baseline {payload['vram_mib']['baseline']:.0f} MiB  "
        f"peak {payload['vram_mib']['peak']:.0f} MiB  "
        f"(model {payload['vram_mib']['model_peak']:.0f} MiB)"
    )
    for name, entry in payload["directions"].items():
        stats = entry["single_ms"]
        print(
            f"{name} single_ms p50 {stats['p50']:.1f} p95 {stats['p95']:.1f} "
            f"mean {stats['mean']:.1f} max {stats['max']:.1f} | "
            f"pred_tokens {entry['predicted_tokens_median']} | "
            f"seq8 {entry['sequential_pass_ms']['p50']:.0f} ms "
            f"({entry['sequential_sentences_per_second']:.1f} sent/s) | "
            f"concurrent8 wall {entry['concurrent_pass_ms']['p50']:.0f} ms "
            f"req p50 {entry['concurrent_request_ms']['p50']:.0f} ms"
        )
        for sample in entry["outputs"]:
            print(f"  {sample}")
    print(f"JSON: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
