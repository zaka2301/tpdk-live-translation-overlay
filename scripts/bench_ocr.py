"""Benchmark PaddleOCR model sizes on CPU and GPU (FR-05, NFR-03/05).

Runs one configuration per process so VRAM/RAM is attributable:

    .\\.venv\\Scripts\\python.exe scripts\\bench_ocr.py --device cpu --size small
    .\\.venv\\Scripts\\python.exe scripts\\bench_ocr.py --device gpu --size tiny

The script measures model load+warmup, the first real frame, steady-state
latency (mean/p50/p95), process RSS peak, and NVIDIA VRAM peak (GPU runs). It
writes JSON under ``data/benchmarks/`` for the report and prints one summary line.

The synthetic slide is deterministic and saved next to the JSON unless
``--image`` points at a real captured slide.

On this workstation the venv itself runs ``paddlepaddle-gpu==3.3.1`` (cu130);
``register_dll_dirs`` adds the CUDA/torch DLL directories. CPU fallback runs on
the same build with oneDNN disabled (``--enable-mkldnn`` is opt-in and only
works on Paddle 3.2.x, because 3.3.x oneDNN fails on PP-OCRv6).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from tpdk_overlay.config import OcrConfig
from tpdk_overlay.ocr.paddle import PaddleOcrEngine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "benchmarks"
SLIDE_SIZE = (1280, 720)
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"


def _font(*names: str, size: int) -> Any:
    from PIL import ImageFont

    for name in names:
        path = FONT_DIR / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def build_slide() -> np.ndarray:
    """A 1280x720 two-column slide with a title and nine body lines."""

    from PIL import Image, ImageDraw

    width, height = SLIDE_SIZE
    image = Image.new("RGB", (width, height), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    title = _font("segoeuib.ttf", "arialbd.ttf", "arial.ttf", size=40)
    body = _font("segoeui.ttf", "arial.ttf", size=24)
    small = _font("segoeui.ttf", "arial.ttf", size=20)

    draw.text((60, 45), "Machine Learning Fundamentals", font=title, fill=(18, 22, 28))
    draw.line((60, 105, 1220, 105), fill=(180, 186, 194), width=2)

    left = [
        "Supervised learning maps labelled inputs to outputs.",
        "Unsupervised learning finds structure without labels.",
        "Reinforcement learning optimizes a policy from rewards.",
        "Generalization measures performance on unseen data.",
        "Regularization trades training fit for robustness.",
    ]
    right = [
        "Bias and variance describe two failure modes.",
        "Cross-validation estimates generalization error.",
        "Feature engineering encodes domain knowledge.",
        "Gradient descent minimizes the training objective.",
    ]
    y = 145
    for line in left:
        draw.text((60, y), f"- {line}", font=body, fill=(35, 40, 48))
        y += 52
    y = 145
    for line in right:
        draw.text((660, y), f"- {line}", font=body, fill=(35, 40, 48))
        y += 52
    draw.text(
        (60, 640),
        "Lecture 4 - supervised and unsupervised learning",
        font=small,
        fill=(110, 116, 124),
    )

    rgb = np.asarray(image)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _nvidia_query(fields: str) -> list[str] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in output.splitlines() if line.strip()]


def gpu_info() -> dict[str, Any] | None:
    rows = _nvidia_query("name,memory.used,memory.total")
    if not rows:
        return None
    name, used, total = (part.strip() for part in rows[0].split(","))
    return {"name": name, "used_mb": int(used), "total_mb": int(total)}


# Keep the add_dll_directory handles alive for the whole process.
_DLL_HANDLES: list[Any] = []


def register_dll_dirs(extra: list[Path]) -> list[str]:
    """Make CUDA/cuDNN DLLs visible to Paddle on Windows.

    Paddle's GPU wheels do not bundle CUDA. This workstation has CUDA 13.1 and
    the torch cu130 wheel supplies the cuBLAS/cuDNN 9 DLLs; adding both keeps
    the benchmark reproducible without changing the project's CPU install.
    """

    if os.name != "nt":
        return []
    import sysconfig

    candidates = list(extra)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    candidates.append(Path(sysconfig.get_paths()["purelib"]) / "torch" / "lib")

    registered: list[str] = []
    for path in candidates:
        if not path.is_dir():
            continue
        try:
            _DLL_HANDLES.append(os.add_dll_directory(str(path)))
        except OSError:
            continue
        registered.append(str(path))
    return registered


class ResourceSampler(threading.Thread):
    """Poll process RSS and (on GPU runs) nvidia-smi until stopped."""

    def __init__(self, *, sample_vram: bool) -> None:
        super().__init__(name="ocr-bench-sampler", daemon=True)
        self._sample_vram = sample_vram
        self._stop_event = threading.Event()
        self.peak_rss = 0
        self.peak_vram = 0
        self.vram_samples: list[int] = []
        try:
            import psutil

            self._proc = psutil.Process()
        except Exception:  # pragma: no cover - psutil is a dev dependency here
            self._proc = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._proc is not None:
                self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
            if self._sample_vram:
                info = gpu_info()
                if info is not None:
                    self.peak_vram = max(self.peak_vram, info["used_mb"])
                    self.vram_samples.append(info["used_mb"])
            self._stop_event.wait(0.05)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100.0 * len(ordered)) - 1)
    return ordered[index]


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover
        return None


def run_benchmark(
    *,
    device: str,
    size: str,
    repeats: int,
    image: np.ndarray,
    dll_dirs: list[str] | None = None,
    enable_mkldnn: bool = True,
) -> dict[str, Any]:
    det_model = f"PP-OCRv6_{size}_det"
    rec_model = f"PP-OCRv6_{size}_rec"
    config = OcrConfig(
        device=device,  # type: ignore[arg-type]
        det_model=det_model,
        rec_model=rec_model,
        resize_max_side=1280,
        enable_mkldnn=enable_mkldnn,
    )
    gpu = gpu_info() if device == "gpu" else None
    sampler = ResourceSampler(sample_vram=device == "gpu")
    sampler.start()
    rss_before = _rss_mb()
    vram_baseline = gpu["used_mb"] if gpu is not None else None

    engine = PaddleOcrEngine(config)
    load_started = time.perf_counter()
    engine.warmup()
    load_s = time.perf_counter() - load_started
    rss_loaded = _rss_mb()
    loaded_info = gpu_info() if gpu is not None else None
    vram_loaded = loaded_info["used_mb"] if loaded_info is not None else None

    first_started = time.perf_counter()
    first_lines = engine.recognize(image)
    first_s = time.perf_counter() - first_started

    runs: list[float] = []
    line_counts: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter()
        lines = engine.recognize(image)
        runs.append((time.perf_counter() - started) * 1000.0)
        line_counts.append(len(lines))

    rss_after = _rss_mb()
    after_info = gpu_info() if gpu is not None else None
    vram_after = after_info["used_mb"] if after_info is not None else None
    sampler.stop()
    engine.close()

    import paddle

    return {
        "device": device,
        "paddle_version": paddle.__version__,
        "paddle_cuda_build": paddle.device.is_compiled_with_cuda(),
        "size": size,
        "det_model": det_model,
        "rec_model": rec_model,
        "image_size": list(SLIDE_SIZE),
        "lines": len(first_lines),
        "line_counts": line_counts,
        "texts": [line.text for line in first_lines],
        "load_warmup_s": round(load_s, 3),
        "first_frame_ms": round(first_s * 1000.0, 1),
        "steady_ms": [round(value, 1) for value in runs],
        "steady_mean_ms": round(statistics.fmean(runs), 1) if runs else None,
        "steady_p50_ms": round(statistics.median(runs), 1) if runs else None,
        "steady_p95_ms": round(_percentile(runs, 95), 1) if runs else None,
        "rss_before_mb": round(rss_before, 1) if rss_before else None,
        "rss_loaded_mb": round(rss_loaded, 1) if rss_loaded else None,
        "rss_after_mb": round(rss_after, 1) if rss_after else None,
        "rss_peak_mb": round(sampler.peak_rss / (1024 * 1024), 1)
        if sampler.peak_rss
        else None,
        "vram_baseline_mb": vram_baseline,
        "vram_loaded_mb": vram_loaded,
        "vram_after_mb": vram_after,
        "vram_peak_mb": sampler.peak_vram or None,
        "vram_total_mb": gpu["total_mb"] if gpu is not None else None,
        "gpu_name": gpu["name"] if gpu is not None else None,
        "dll_dirs": dll_dirs or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--size", choices=("tiny", "small", "medium"), default="small")
    parser.add_argument("--repeats", type=int, default=5, help="timed warm passes")
    parser.add_argument("--image", type=Path, help="use a captured slide image")
    parser.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="enable oneDNN on CPU (Paddle 3.2.x only; 3.3.x fails on PP-OCRv6)",
    )
    parser.add_argument("--output", type=Path, help="JSON output path")
    parser.add_argument(
        "--dll-dir",
        type=Path,
        action="append",
        default=[],
        help="extra directory with CUDA/cuDNN DLLs (repeatable)",
    )
    args = parser.parse_args()
    dll_dirs = register_dll_dirs(args.dll_dir)

    if args.image is not None:
        import cv2

        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            parser.error(f"could not read image: {args.image}")
    else:
        image = build_slide()
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import cv2

            cv2.imwrite(str(DEFAULT_OUTPUT_DIR / "ocr_slide_1280x720.png"), image)
        except Exception:
            pass

    result = run_benchmark(
        device=args.device,
        size=args.size,
        repeats=args.repeats,
        image=image,
        dll_dirs=dll_dirs,
        enable_mkldnn=args.enable_mkldnn,
    )
    suffix = "" if args.enable_mkldnn else "_nomkldnn"
    output = args.output or (
        DEFAULT_OUTPUT_DIR / f"ocr_{args.device}_{args.size}{suffix}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{args.device:>3} {args.size:<6} "
        f"load={result['load_warmup_s']:.1f}s "
        f"first={result['first_frame_ms']:.0f}ms "
        f"p50={result['steady_p50_ms'] or 0:.0f}ms "
        f"p95={result['steady_p95_ms'] or 0:.0f}ms "
        f"rss={result['rss_peak_mb'] or 0:.0f}MB "
        f"vram={result['vram_peak_mb'] or 0}MB "
        f"-> {output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
