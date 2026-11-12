r"""Reproducible latency/resource benchmark for the HTTP ASR service.

Measures the shipping HTTP adapter (``create_asr_client``) end to end: model
load, stream open, per-chunk push, post-utterance finish, and pipeline latency,
plus GPU/process resources sampled in the background.

Examples::

    python scripts/bench_asr.py --mode realtime --reps 5
    python scripts/bench_asr.py --vram-source wsl --wsl-pgrep asr_service.py

Realtime pacing: chunk ``k`` is pushed once it is complete, i.e. at
``t0 + (k+1)*chunk_size_sec`` (a shorter tail chunk at ``t0 + audio_duration``),
so ``first_partial_ms`` includes the chunk accumulation latency and ``finish``
starts at the audio end. Burst mode pushes without pacing.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import platform
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tpdk_overlay.asr import create_asr_client  # noqa: E402
from tpdk_overlay.config import load_config  # noqa: E402
from tpdk_overlay.ports import AsrClient  # noqa: E402
from tpdk_overlay.utils.audio import SAMPLE_RATE_16K, as_float32_mono, resample  # noqa: E402

DEFAULT_CLIPS_DIR = PROJECT_ROOT / "data" / "benchmarks" / "clips"
MIB = 1024.0 * 1024.0
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_METRICS = (
    "open_ms",
    "finish_ms",
    "first_partial_ms",
    "compute_ms",
    "rtf",
    "pipeline_ms",
    "schedule_lag_ms",
)

_WSL_PROC_SNAPSHOT_PY = """\
import os
import sys

root = int(sys.argv[1])
pids = [root]
seen = {root}
index = 0
while index < len(pids):
    current = pids[index]
    index += 1
    try:
        children = open(f"/proc/{current}/task/{current}/children").read().split()
    except OSError:
        children = []
    for child in children:
        child = int(child)
        if child not in seen:
            seen.add(child)
            pids.append(child)

rss_kb = utime = stime = 0
for process in pids:
    try:
        for line in open(f"/proc/{process}/status"):
            if line.startswith("VmRSS:"):
                rss_kb += int(line.split()[1])
        fields = open(f"/proc/{process}/stat").read().split()
        utime += int(fields[13])
        stime += int(fields[14])
    except (OSError, IndexError, ValueError):
        pass

print(f"pid={root}")
print(f"rss_kb={rss_kb}")
print(f"utime={utime}")
print(f"stime={stime}")
print(f"clk_tck={os.sysconf('SC_CLK_TCK')}")
print(f"npids={len(pids)}")
"""

# wsl.exe joins argv into one shell command line, so newlines/quotes in the
# script are mangled. Keep the wrapper on a single line and ship the Python
# helper as base64.
_WSL_SNAPSHOT_SCRIPT = (
    "pid=$(pgrep -f __PATTERN__ | head -n1) || true; "
    'if [ -z "$pid" ]; then echo missing=1; exit 0; fi; '
    f"echo '{base64.b64encode(_WSL_PROC_SNAPSHOT_PY.encode()).decode()}' "
    "| base64 -d > /tmp/tpdk_bench_snapshot.py; "
    'python3 /tmp/tpdk_bench_snapshot.py "$pid"'
)


# --------------------------------------------------------------------------- io


@dataclass(frozen=True, slots=True)
class Clip:
    path: Path
    audio: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(self.audio.size) / SAMPLE_RATE_16K


def _load_clip(path: Path) -> Clip:
    rate, data = wavfile.read(path)
    mono = as_float32_mono(data)
    audio = mono if int(rate) == SAMPLE_RATE_16K else resample(mono, int(rate), SAMPLE_RATE_16K)
    if audio.size == 0:
        raise ValueError(f"clip is empty: {path}")
    return Clip(path=path, audio=audio)


def _discover_clips(clips_dir: Path) -> list[Clip]:
    paths = sorted(clips_dir.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"no *.wav clips found in {clips_dir}")
    return [_load_clip(path) for path in paths]


# ------------------------------------------------------------------ measurement


@dataclass(frozen=True, slots=True)
class PassResult:
    open_ms: float
    finish_ms: float
    first_partial_ms: float | None
    compute_ms: float
    rtf: float
    pipeline_ms: float
    push_ms_total: float
    push_count: int
    text_updates: int
    schedule_lag_ms: float | None
    final_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_ms": round(self.open_ms, 3),
            "finish_ms": round(self.finish_ms, 3),
            "first_partial_ms": _round_opt(self.first_partial_ms),
            "compute_ms": round(self.compute_ms, 3),
            "rtf": round(self.rtf, 4),
            "pipeline_ms": round(self.pipeline_ms, 3),
            "push_ms_total": round(self.push_ms_total, 3),
            "push_count": self.push_count,
            "text_updates": self.text_updates,
            "schedule_lag_ms": _round_opt(self.schedule_lag_ms),
            "final_text": self.final_text,
            "final_text_chars": len(self.final_text),
        }


def _round_opt(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _run_pass(
    client: AsrClient,
    clip: Clip,
    *,
    chunk_samples: int,
    chunk_size_sec: float,
    language: str | None,
    realtime: bool,
) -> PassResult:
    open_start = time.perf_counter()
    stream = client.open(language=language)
    open_ms = (time.perf_counter() - open_start) * 1000.0

    push_ms: list[float] = []
    text_updates = 0
    max_lag_ms = 0.0
    first_partial_ms: float | None = None
    previous_text = ""
    audio = clip.audio
    slices = [
        audio[start : start + chunk_samples] for start in range(0, audio.size, chunk_samples)
    ]
    t0 = time.perf_counter()
    try:
        for index, chunk in enumerate(slices):
            if realtime:
                # Chunk k is only complete at its end: a full chunk is due at
                # t0 + (k+1)*cs, a shorter tail chunk at the audio duration.
                if chunk.size < chunk_samples:
                    due = t0 + clip.duration_s
                else:
                    due = t0 + (index + 1) * chunk_size_sec
                now = time.perf_counter()
                if now < due:
                    time.sleep(due - now)
                max_lag_ms = max(max_lag_ms, (time.perf_counter() - due) * 1000.0)
            push_start = time.perf_counter()
            update = stream.push(chunk)
            push_ms.append((time.perf_counter() - push_start) * 1000.0)
            if update.text != previous_text:
                previous_text = update.text
                text_updates += 1
                if update.text and first_partial_ms is None:
                    first_partial_ms = (time.perf_counter() - t0) * 1000.0
        finish_start = time.perf_counter()
        final = stream.finish()
        finish_ms = (time.perf_counter() - finish_start) * 1000.0
        final_ts = time.perf_counter()
    finally:
        stream.close()

    push_ms_total = sum(push_ms)
    compute_ms = push_ms_total + finish_ms
    duration_ms = clip.duration_s * 1000.0
    return PassResult(
        open_ms=open_ms,
        finish_ms=finish_ms,
        first_partial_ms=first_partial_ms,
        compute_ms=compute_ms,
        rtf=compute_ms / duration_ms,
        pipeline_ms=(final_ts - t0) * 1000.0,
        push_ms_total=push_ms_total,
        push_count=len(push_ms),
        text_updates=text_updates,
        schedule_lag_ms=max_lag_ms if realtime else None,
        final_text=final.text,
    )


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def _summarize(results: list[PassResult]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for name in _METRICS:
        values = [
            float(value) for value in (getattr(r, name) for r in results) if value is not None
        ]
        summary[name] = {
            "p50": _round_opt(_percentile(values, 50.0)),
            "p95": _round_opt(_percentile(values, 95.0)),
        }
    return summary


# -------------------------------------------------------------------- resources


def _run_nvidia_smi(query: str, *, vram_source: str, wsl_distro: str) -> list[str]:
    args = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    if vram_source == "wsl":
        args = ["wsl.exe", "-d", wsl_distro, "--", *args]
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"warning: nvidia-smi failed ({exc})", file=sys.stderr)
        return []
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        print(f"warning: nvidia-smi failed ({detail})", file=sys.stderr)
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.strip())
    except ValueError:
        return None


class ResourceSampler:
    """Samples GPU + process resources in a background thread per window."""

    def __init__(
        self,
        interval_s: float,
        *,
        vram_source: str,
        wsl_distro: str,
    ) -> None:
        self.interval_s = max(interval_s, 1.0) if vram_source == "wsl" else interval_s
        self.vram_source = vram_source
        self.wsl_distro = wsl_distro
        self.process = psutil.Process()
        self.windows: list[list[dict[str, Any]]] = []
        self._active: list[dict[str, Any]] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_warned = False
        self.process.cpu_percent(None)
        psutil.cpu_percent(None)

    def begin(self) -> None:
        if self._active is not None:
            raise RuntimeError("resource sampler is already running")
        self._active = [self._sample()]
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="bench-resources", daemon=True)
        self._thread.start()

    def end(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5.0)
            self._thread = None
        if self._active is not None:
            self._active.append(self._sample())
            self.windows.append(self._active)
            self._active = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            sample = self._sample()
            if self._active is not None:
                self._active.append(sample)

    def _sample(self) -> dict[str, Any]:
        vram_mib: float | None = None
        gpu_util: float | None = None
        rows = _run_nvidia_smi(
            "memory.used,utilization.gpu",
            vram_source=self.vram_source,
            wsl_distro=self.wsl_distro,
        )
        if rows:
            parts = [part.strip() for part in rows[0].split(",")]
            if len(parts) >= 2:
                vram_mib = _to_float(parts[0])
                gpu_util = _to_float(parts[1])
        elif not self._gpu_warned:
            self._gpu_warned = True
            print("warning: GPU metrics unavailable; fields will be null", file=sys.stderr)
        times = self.process.cpu_times()
        return {
            "ts": time.perf_counter(),
            "vram_mib": vram_mib,
            "gpu_util_pct": gpu_util,
            "rss_mib": self.process.memory_info().rss / MIB,
            "proc_cpu_s": times.user + times.system,
            "system_cpu_pct": psutil.cpu_percent(None),
        }

    def snapshot(self) -> dict[str, Any]:
        """One-off resource reading outside a sampled window (baseline/idle)."""

        sample = self._sample()
        return {
            "vram_mib": _round_opt(sample["vram_mib"]),
            "gpu_util_pct": _round_opt(sample["gpu_util_pct"]),
            "rss_mib": _round_opt(sample["rss_mib"]),
        }

    def aggregate(self) -> dict[str, Any]:
        samples = [sample for window in self.windows for sample in window]
        vram = [float(s["vram_mib"]) for s in samples if s["vram_mib"] is not None]
        util = [float(s["gpu_util_pct"]) for s in samples if s["gpu_util_pct"] is not None]
        rss = [float(s["rss_mib"]) for s in samples]
        system_cpu = [float(s["system_cpu_pct"]) for s in samples]
        elapsed = sum(w[-1]["ts"] - w[0]["ts"] for w in self.windows if len(w) > 1)
        cpu_delta = sum(
            w[-1]["proc_cpu_s"] - w[0]["proc_cpu_s"] for w in self.windows if len(w) > 1
        )
        return {
            "vram_peak_mib": _round_opt(max(vram)) if vram else None,
            "vram_mean_mib": _round_opt(sum(vram) / len(vram)) if vram else None,
            "gpu_util_peak_pct": _round_opt(max(util)) if util else None,
            "gpu_util_mean_pct": _round_opt(sum(util) / len(util)) if util else None,
            "rss_peak_mib": _round_opt(max(rss)) if rss else None,
            "rss_end_mib": _round_opt(rss[-1]) if rss else None,
            "proc_cpu_avg_pct": _round_opt(cpu_delta / elapsed * 100.0) if elapsed > 0 else None,
            "system_cpu_avg_pct": (
                _round_opt(sum(system_cpu) / len(system_cpu)) if system_cpu else None
            ),
        }


def _wsl_service_snapshot(*, distro: str, pattern: str) -> dict[str, Any] | None:
    args = [
        "wsl.exe",
        "-d",
        distro,
        "-e",
        "bash",
        "-lc",
        _WSL_SNAPSHOT_SCRIPT.replace("__PATTERN__", shlex.quote(pattern)),
    ]
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"warning: WSL service snapshot failed ({exc})", file=sys.stderr)
        return None
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        print(f"warning: WSL service snapshot failed ({detail})", file=sys.stderr)
        return None
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip()
    if "missing" in values:
        print(f"warning: WSL service process not found (pattern {pattern!r})", file=sys.stderr)
        return None
    try:
        pid = int(values["pid"])
        rss_kb = float(values["rss_kb"])
        cpu_s = (float(values["utime"]) + float(values["stime"])) / float(values["clk_tck"])
    except (KeyError, ValueError) as exc:
        print(f"warning: WSL service snapshot malformed ({exc})", file=sys.stderr)
        return None
    return {"pid": pid, "rss_mib": rss_kb / 1024.0, "cpu_s": cpu_s, "ts": time.perf_counter()}


# ------------------------------------------------------------------ environment


def _dist_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment(vram_source: str, wsl_distro: str) -> dict[str, Any]:
    rows = _run_nvidia_smi("name", vram_source=vram_source, wsl_distro=wsl_distro)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": rows[0] if rows else None,
        "torch": _dist_version("torch"),
        "transformers": _dist_version("transformers"),
    }


# ------------------------------------------------------------------------ cli


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the HTTP ASR adapter (services/asr_service.py in WSL2).",
    )
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument(
        "--language",
        default="English",
        help='verbatim language code for the adapter, or "none" to disable hints',
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--mode", choices=("realtime", "burst"), default="realtime")
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gpu-sample-ms", type=int, default=200)
    parser.add_argument("--vram-source", choices=("host", "wsl"), default="host")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument("--wsl-pgrep", default="asr_service.py")
    return parser


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pair(summary: dict[str, dict[str, float | None]], metric: str, digits: int = 1) -> str:
    entry = summary[metric]
    return f"{_fmt(entry['p50'], digits)} / {_fmt(entry['p95'], digits)}"


def _print_table(clips: list[dict[str, Any]]) -> None:
    header = (
        "| clip | dur_s | open_ms | finish_ms | first_partial_ms | compute_ms "
        "| rtf | pipeline_ms | chars |"
    )
    print(header)
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for entry in clips:
        summary = entry["summary"]
        print(
            f"| {entry['name']} | {entry['duration_s']:.2f} | "
            f"{_pair(summary, 'open_ms')} | {_pair(summary, 'finish_ms')} | "
            f"{_pair(summary, 'first_partial_ms')} | {_pair(summary, 'compute_ms')} | "
            f"{_pair(summary, 'rtf', digits=3)} | {_pair(summary, 'pipeline_ms')} | "
            f"{entry['final_text_chars']} |"
        )


def _print_resources(
    during: dict[str, Any],
    *,
    vram_source: str,
    wsl: dict[str, Any] | None,
    wsl_elapsed_s: float | None,
) -> None:
    print("\nResources (measured window)")
    print(
        f"- GPU VRAM peak/mean: {_fmt(during['vram_peak_mib'])} / "
        f"{_fmt(during['vram_mean_mib'])} MiB (source: {vram_source})"
    )
    print(
        f"- GPU util peak/mean: {_fmt(during['gpu_util_peak_pct'])} / "
        f"{_fmt(during['gpu_util_mean_pct'])} %"
    )
    print(
        f"- Process RSS peak/end: {_fmt(during['rss_peak_mib'])} / "
        f"{_fmt(during['rss_end_mib'])} MiB"
    )
    print(
        f"- CPU avg: process {_fmt(during['proc_cpu_avg_pct'])} %, "
        f"system {_fmt(during['system_cpu_avg_pct'])} %"
    )
    if wsl:
        elapsed = wsl_elapsed_s or 0.0
        print(
            f"- WSL service pid {wsl['pid']}: rss {_fmt(wsl['rss_mib'])} MiB, "
            f"cpu avg {_fmt(wsl['cpu_avg_pct'])} % over {elapsed:.1f} s"
        )
    else:
        print("- WSL service: not sampled")


# ----------------------------------------------------------------------- main


def _run(args: argparse.Namespace) -> int:
    if args.reps < 1:
        raise ValueError("--reps must be >= 1")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0")
    if args.chunk_size_sec <= 0:
        raise ValueError("--chunk-size-sec must be > 0")
    if args.gpu_sample_ms < 1:
        raise ValueError("--gpu-sample-ms must be >= 1")

    language = None if args.language.lower() == "none" else args.language
    clips = _discover_clips(args.clips_dir)
    asr_config = load_config().asr.model_copy(
        update={
            "backend": "http",
            "base_url": args.base_url,
            "language": language,
            "chunk_size_sec": args.chunk_size_sec,
        },
    )
    client = create_asr_client(asr_config)
    sampler = ResourceSampler(
        args.gpu_sample_ms / 1000.0,
        vram_source=args.vram_source,
        wsl_distro=args.wsl_distro,
    )

    baseline = sampler.snapshot()
    load_start = time.perf_counter()
    client.warmup()
    model_load_ms = (time.perf_counter() - load_start) * 1000.0
    idle = sampler.snapshot()

    wsl_start: dict[str, Any] | None = None
    wsl_end: dict[str, Any] | None = None
    if args.wsl_pgrep:
        wsl_start = _wsl_service_snapshot(distro=args.wsl_distro, pattern=args.wsl_pgrep)

    chunk_samples = max(1, int(round(args.chunk_size_sec * SAMPLE_RATE_16K)))
    realtime = args.mode == "realtime"
    clip_entries: list[dict[str, Any]] = []

    for clip in clips:
        for _ in range(args.warmups):
            _run_pass(
                client,
                clip,
                chunk_samples=chunk_samples,
                chunk_size_sec=args.chunk_size_sec,
                language=language,
                realtime=realtime,
            )
        sampler.begin()
        measured = [
            _run_pass(
                client,
                clip,
                chunk_samples=chunk_samples,
                chunk_size_sec=args.chunk_size_sec,
                language=language,
                realtime=realtime,
            )
            for _ in range(args.reps)
        ]
        sampler.end()
        clip_entries.append(
            {
                "name": clip.path.name,
                "duration_s": round(clip.duration_s, 3),
                "reps": [result.to_dict() for result in measured],
                "summary": _summarize(measured),
                "final_text_chars": len(measured[-1].final_text),
            }
        )

    client.close()

    if args.wsl_pgrep:
        wsl_end = _wsl_service_snapshot(distro=args.wsl_distro, pattern=args.wsl_pgrep)

    during = sampler.aggregate()

    wsl_summary: dict[str, Any] | None = None
    wsl_elapsed_s: float | None = None
    if wsl_start is not None and wsl_end is not None and wsl_end["ts"] > wsl_start["ts"]:
        wsl_elapsed_s = wsl_end["ts"] - wsl_start["ts"]
        wsl_summary = {
            "rss_mib": round(wsl_end["rss_mib"], 3),
            "cpu_avg_pct": round(
                (wsl_end["cpu_s"] - wsl_start["cpu_s"]) / wsl_elapsed_s * 100.0,
                3,
            ),
            "pid": wsl_end["pid"],
        }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        PROJECT_ROOT / "data" / "benchmarks" / f"asr_bench_http_{stamp}.json"
    )

    payload = {
        "schema": "tpdk-asr-bench-v1",
        "backend": "http",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment(args.vram_source, args.wsl_distro),
        "settings": {
            "clips_dir": str(args.clips_dir),
            "language": language,
            "reps": args.reps,
            "warmups": args.warmups,
            "mode": args.mode,
            "chunk_size_sec": args.chunk_size_sec,
            "base_url": args.base_url,
            "gpu_sample_ms": args.gpu_sample_ms,
            "vram_source": args.vram_source,
            "wsl_distro": args.wsl_distro,
            "wsl_pgrep": args.wsl_pgrep,
        },
        "model_load_ms": round(model_load_ms, 3),
        "resources": {
            "baseline": baseline,
            "idle": idle,
            "during": during,
            "wsl": wsl_summary,
        },
        "clips": [
            {
                "name": entry["name"],
                "duration_s": entry["duration_s"],
                "reps": entry["reps"],
                "summary": entry["summary"],
            }
            for entry in clip_entries
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"backend: http  mode: {args.mode}  language: {language}  ")
    print(f"warmup: {args.warmups}  reps: {args.reps}  chunk: {args.chunk_size_sec}s  ")
    print(f"model_load_ms: {model_load_ms:.1f}")
    print(f"baseline: {baseline}")
    print(f"idle: {idle}\n")
    _print_table(clip_entries)
    _print_resources(
        during,
        vram_source=args.vram_source,
        wsl=wsl_summary,
        wsl_elapsed_s=wsl_elapsed_s,
    )
    print(f"\nJSON: {output}")
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
