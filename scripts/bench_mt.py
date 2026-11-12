"""Reproducible latency/resource benchmark for the NLLB-200 MT adapters.

One configuration per process, so GPU/RAM state from an earlier run never leaks
into the next measurement. Run the matrix as separate invocations::

    python scripts/bench_mt.py                     # project default: nllb_ct2 int8, CPU
    python scripts/bench_mt.py --provider nllb --device cuda --dtype float16
    python scripts/bench_mt.py --provider nllb --device cpu --dtype float32

Each run times ``warmup()`` (model load), then eight fixed lecture sentences
one at a time and as a single batch, in both directions (en to id, id to en). Model
work runs on the adapter's dedicated worker thread, so timings include the
thread handoff (microseconds). A JSON report lands in
``data/benchmarks/mt_bench_<label>_<UTC stamp>.json`` unless ``--json-out``
overrides it; load or translation failures exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast, get_args

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tpdk_overlay.config import MtConfig, MtCt2Config, load_config  # noqa: E402
from tpdk_overlay.mt import (  # noqa: E402
    TranslationError,
    WorkerTranslator,
    create_translator,
)

SCHEMA = "tpdk-mt-bench-v1"
MIB = 1024.0 * 1024.0

_DTYPES = get_args(MtConfig.model_fields["dtype"].annotation)
_COMPUTE_TYPES = get_args(MtCt2Config.model_fields["compute_type"].annotation)
_UNSAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")

EN = [
    "Today we will discuss the design of distributed systems and their trade-offs.",
    "A distributed system consists of independent computers that appear to users as a "
    "single system.",
    "Replication improves availability, but it also introduces the problem of consistency.",
    "The CAP theorem states that a network partition forces a choice between consistency and "
    "availability.",
    "We use a coordinator to order writes and detect conflicts between replicas.",
    "Latency matters more than throughput for interactive lecture applications.",
    "Please submit your assignment before the deadline on Friday.",
    "The next slide summarizes the key points from this section.",
]
ID = [
    "Hari ini kita akan membahas desain sistem terdistribusi dan trade-off-nya.",
    "Sistem terdistribusi terdiri dari komputer-komputer independen yang tampak sebagai "
    "satu sistem.",
    "Replikasi meningkatkan ketersediaan, tetapi juga menimbulkan masalah konsistensi.",
    "Teorema CAP menyatakan bahwa partisi jaringan memaksa pilihan antara konsistensi dan "
    "ketersediaan.",
    "Kita menggunakan koordinator untuk mengurutkan penulisan dan mendeteksi konflik "
    "antar-replika.",
    "Latensi lebih penting daripada throughput untuk aplikasi kuliah interaktif.",
    "Silakan kumpulkan tugas sebelum tenggat waktu hari Jumat.",
    "Slide berikutnya merangkum poin-poin penting dari bagian ini.",
]

_DIRECTIONS: tuple[tuple[str, str, Sequence[str]], ...] = (
    ("en", "id", EN),
    ("id", "en", ID),
)


# ------------------------------------------------------------------ measurement


def _rss_mib() -> float:
    """Resident set size of this process in MiB."""

    return psutil.Process().memory_info().rss / MIB


def _percentile(samples: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of a non-empty sample list."""

    ordered = sorted(samples)
    position = (len(ordered) - 1) * pct / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(samples: Sequence[float]) -> dict[str, float]:
    return {
        "p50": round(_percentile(samples, 50.0), 3),
        "p95": round(_percentile(samples, 95.0), 3),
        "mean": round(statistics.fmean(samples), 3),
        "max": round(max(samples), 3),
    }


def _timed_translate(
    translator: WorkerTranslator,
    texts: Sequence[str],
    *,
    source_lang: str,
    target_lang: str,
    sync: Callable[[], None] | None,
) -> tuple[float, list[str]]:
    """Translate once and return (elapsed ms, outputs)."""

    if sync is not None:
        sync()
    started = time.perf_counter()
    outputs = translator.translate(texts, source_lang=source_lang, target_lang=target_lang)
    if sync is not None:
        sync()
    return (time.perf_counter() - started) * 1000.0, outputs


def _measure_direction(
    translator: WorkerTranslator,
    texts: Sequence[str],
    *,
    source_lang: str,
    target_lang: str,
    reps: int,
    sync: Callable[[], None] | None,
) -> dict[str, Any]:
    """Time single sentences and one full batch for one language pair."""

    single_ms: list[float] = []
    for _ in range(reps):
        for text in texts:
            if not text.strip():
                continue
            elapsed, _ = _timed_translate(
                translator,
                [text],
                source_lang=source_lang,
                target_lang=target_lang,
                sync=sync,
            )
            single_ms.append(elapsed)

    batch_ms: list[float] = []
    outputs: list[str] = []
    for _ in range(reps):
        elapsed, outputs = _timed_translate(
            translator,
            list(texts),
            source_lang=source_lang,
            target_lang=target_lang,
            sync=sync,
        )
        batch_ms.append(elapsed)

    batch_median = statistics.median(batch_ms)
    return {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "sentences": len(texts),
        "single_ms": _stats(single_ms),
        "single_ms_samples": [round(value, 3) for value in single_ms],
        "batch8_ms": round(batch_median, 3),
        "batch8_ms_samples": [round(value, 3) for value in batch_ms],
        "sentences_per_second": round(len(texts) * 1000.0 / batch_median, 3),
        "samples": [{"source": texts[index], "output": outputs[index]} for index in range(3)],
    }


# -------------------------------------------------------------------- config


def _resolve_config(args: argparse.Namespace) -> MtConfig:
    """Apply CLI overrides to the loaded ``mt`` section; models stay frozen."""

    config = load_config().mt
    ct2_updates: dict[str, str] = {}
    if args.compute_type is not None:
        ct2_updates["compute_type"] = args.compute_type
    if args.ct2_model is not None:
        ct2_updates["model"] = args.ct2_model

    updates: dict[str, Any] = {}
    for name in ("provider", "device", "dtype"):
        value = getattr(args, name)
        if value is not None:
            updates[name] = value
    if ct2_updates:
        updates["ct2"] = config.ct2.model_copy(update=ct2_updates)
    return config.model_copy(update=updates)


def _default_label(config: MtConfig) -> str:
    variant = config.ct2.compute_type if config.provider == "nllb_ct2" else config.dtype
    return f"{config.provider}-{config.device}-{variant}"


# ------------------------------------------------------------------------ cli


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one NLLB-200 adapter configuration (one per process).",
    )
    parser.add_argument("--provider", choices=("nllb", "nllb_ct2"), default="nllb_ct2")
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for the nllb adapter, e.g. cuda, cuda:1, cpu (default: config)",
    )
    parser.add_argument("--dtype", choices=_DTYPES, default=None)
    parser.add_argument("--compute-type", choices=_COMPUTE_TYPES, default=None)
    parser.add_argument(
        "--ct2-model",
        default=None,
        help="CTranslate2 model directory (default: config.mt.ct2.model)",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--label", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def _print_summary(payload: dict[str, Any], output: Path) -> None:
    provider = payload["provider"]
    print(f"label: {payload['label']}  provider: {provider['name']}")
    print(
        f"device: {provider['device']}  dtype: {provider['dtype']}  "
        f"compute_type: {payload['ct2']['compute_type']}"
    )
    print(f"model: {provider['model']}  ct2_model: {payload['ct2']['model']}")
    print(f"reps: {payload['reps']}  sentences: {payload['sentences']}")
    print(f"load_ms: {payload['load_ms']:.1f}")
    print(
        f"rss: {payload['rss_before_mib']:.1f} -> {payload['rss_after_mib']:.1f} MiB "
        f"(+{payload['rss_after_mib'] - payload['rss_before_mib']:.1f})"
    )
    if payload["gpu_peak_mb"] is None:
        print("gpu: not measured (provider/device is not CUDA nllb)")
    else:
        print(
            f"gpu_peak_mb: {payload['gpu_peak_mb']:.1f}  "
            f"gpu_reserved_mb: {payload['gpu_reserved_mb']:.1f}"
        )
    for name, entry in payload["directions"].items():
        stats = entry["single_ms"]
        print(
            f"{name} single_ms p50 {stats['p50']:.1f} p95 {stats['p95']:.1f} "
            f"mean {stats['mean']:.1f} max {stats['max']:.1f} | "
            f"batch8 {entry['batch8_ms']:.1f} ms  "
            f"{entry['sentences_per_second']:.1f} sent/s"
        )
        for sample in entry["samples"]:
            print(f"  {sample['output']}")
    print(f"JSON: {output}")


# ----------------------------------------------------------------------- main


def _run(args: argparse.Namespace) -> int:
    if args.reps < 1:
        raise ValueError("--reps must be >= 1")

    config = _resolve_config(args)
    label = args.label or _default_label(config)

    rss_before = _rss_mib()
    translator = cast(WorkerTranslator, create_translator(config))
    try:
        load_started = time.perf_counter()
        try:
            translator.warmup()
        except TranslationError as exc:
            raise TranslationError(
                f"cannot load {config.provider} model {config.model!r}: {exc}"
            ) from exc
        load_ms = (time.perf_counter() - load_started) * 1000.0

        # Untimed warmup passes so the first measured call sees steady-state kernels.
        translator.translate(EN[:2], source_lang="en", target_lang="id")
        translator.translate(ID[:2], source_lang="id", target_lang="en")

        rss_after = _rss_mib()
        sync: Callable[[], None] | None = None
        torch_module: Any = None
        if config.provider == "nllb" and config.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
            sync = torch.cuda.synchronize
            torch_module = torch

        directions = {
            f"{source}->{target}": _measure_direction(
                translator,
                texts,
                source_lang=source,
                target_lang=target,
                reps=args.reps,
                sync=sync,
            )
            for source, target, texts in _DIRECTIONS
        }

        gpu_peak_mb: float | None = None
        gpu_reserved_mb: float | None = None
        if torch_module is not None:
            gpu_peak_mb = round(torch_module.cuda.max_memory_allocated() / MIB, 3)
            gpu_reserved_mb = round(torch_module.cuda.memory_reserved() / MIB, 3)
    finally:
        translator.close()

    stamp = datetime.now(UTC)
    output = args.json_out or (
        PROJECT_ROOT
        / "data"
        / "benchmarks"
        / f"mt_bench_{_UNSAFE_LABEL.sub('-', label)}_{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "label": label,
        "created_utc": stamp.isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "reps": args.reps,
        "sentences": len(EN),
        "provider": {
            "name": config.provider,
            "model": config.model,
            "device": config.device,
            "dtype": config.dtype,
            "max_input_tokens": config.max_input_tokens,
            "max_new_tokens": config.max_new_tokens,
            "batch_size": config.batch_size,
        },
        "ct2": {
            "model": config.ct2.model,
            "compute_type": config.ct2.compute_type,
            "max_new_tokens": config.ct2.max_new_tokens,
        },
        "load_ms": round(load_ms, 3),
        "rss_before_mib": round(rss_before, 3),
        "rss_after_mib": round(rss_after, 3),
        "gpu_peak_mb": gpu_peak_mb,
        "gpu_reserved_mb": gpu_reserved_mb,
        "directions": directions,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    _print_summary(payload, output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        print(f"benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
