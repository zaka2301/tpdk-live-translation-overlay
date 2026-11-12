r"""Environment doctor: verify the venv, dependencies, devices, and services.

Run after bootstrap and before a recording session:

    .\.venv\Scripts\python.exe scripts\check_env.py

Exit code 0 means core requirements are present. Optional ML/OCR/UI pieces are
reported as WARN; they are needed for the full pipeline but not for the backend
smoke test.
"""

from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if TYPE_CHECKING:
    from tpdk_overlay.config import AppConfig

CORE_MODULES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "yaml",
    "numpy",
    "httpx",
    "dotenv",
    "websockets",
    "mss",
    "soundcard",
    "scipy",
    "tpdk_overlay",
]

OPTIONAL_MODULES = {
    "torch": "NLLB transformers path (optional GPU)",
    "transformers": "NLLB tokenizer / transformers path",
    "sentencepiece": "NLLB tokenizer",
    "ctranslate2": "NLLB CT2 fallback (int8)",
    "silero_vad": "voice activity detection",
    "paddle": "PP-OCRv6 runtime",
    "paddleocr": "slide OCR",
    "PySide6": "overlay UI",
    "google.genai": "Gemini summaries",
}

failures: list[str] = []
warnings: list[str] = []


def _line(status: str, message: str) -> None:
    print(f"[{status:<4}] {message}")


def _register_cuda_dlls() -> None:
    """Load the CUDA/cuDNN DLL dirs before Paddle is imported on Windows."""

    try:
        from tpdk_overlay.ocr.paddle import register_cuda_dlls

        register_cuda_dlls()
    except Exception:  # noqa: BLE001 - reported by the import checks
        pass


def _check_imports() -> None:
    _register_cuda_dlls()
    for name in CORE_MODULES:
        try:
            importlib.import_module(name)
            _line("OK", f"import {name}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            failures.append(name)
            _line("FAIL", f"import {name}: {exc}")

    for name, purpose in OPTIONAL_MODULES.items():
        try:
            importlib.import_module(name)
            _line("OK", f"import {name} ({purpose})")
        except Exception as exc:  # noqa: BLE001
            warnings.append(name)
            _line("WARN", f"import {name} missing ({purpose}): {exc}")


def _check_python() -> None:
    version = sys.version_info
    if version < (3, 12):
        failures.append("python>=3.12")
        _line("FAIL", f"Python {version.major}.{version.minor}; need 3.12")
    else:
        _line("OK", f"Python {version.major}.{version.minor}.{version.micro}")


def _check_config() -> AppConfig | None:
    try:
        from tpdk_overlay.config import load_config

        config = load_config()
        config.data_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        _line(
            "OK",
            f"config: {config.server.host}:{config.server.port} "
            f"{config.app.language.source}->{config.app.language.target} "
            f"asr={config.asr.backend}",
        )
        key = config.summary.api_key()
        _line(
            "OK" if key else "WARN",
            f"GEMINI_API_KEY {'set' if key else 'missing (summaries disabled)'}",
        )
        if not key:
            warnings.append("GEMINI_API_KEY")
        return config
    except Exception as exc:  # noqa: BLE001
        failures.append("config")
        _line("FAIL", f"config: {exc}")
        return None


def _check_torch(config: AppConfig | None) -> None:
    try:
        import torch
    except Exception:  # noqa: BLE001 - already reported by import check
        return
    cuda = torch.cuda.is_available()
    wants_cuda = (
        config is not None
        and config.mt.provider == "nllb"
        and config.mt.device.startswith("cuda")
    )
    if cuda:
        _line("OK", f"torch {torch.__version__} cuda=True")
        _line("OK", f"GPU: {torch.cuda.get_device_name(0)}")
        return
    if wants_cuda:
        failures.append("cuda")
        _line("FAIL", f"torch {torch.__version__} has no CUDA but mt.device=cuda")
        _line(
            "FAIL",
            "install the CUDA wheel: "
            "uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130",
        )
    else:
        provider = config.mt.provider if config is not None else "unknown"
        _line(
            "OK",
            f"torch {torch.__version__} cuda=False "
            f"(not required for mt.provider={provider})",
        )


def _check_ocr(config: AppConfig | None) -> None:
    if config is None or not config.ocr.enabled:
        if config is not None:
            _line("OK", "OCR disabled")
        return
    try:
        import paddle
    except Exception as exc:  # noqa: BLE001 - reported by the import check
        failures.append("paddle")
        _line("FAIL", f"OCR runtime import failed: {exc}")
        return

    models = f"{config.ocr.det_model}/{config.ocr.rec_model}"
    if config.ocr.device == "cpu":
        _line("OK", f"OCR CPU: {models} (paddle {paddle.__version__})")
        return
    if not paddle.device.is_compiled_with_cuda():
        failures.append("ocr-gpu")
        _line(
            "FAIL",
            f"ocr.device=gpu but paddle {paddle.__version__} is a CPU build",
        )
        return
    try:
        devices = paddle.device.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        devices = 0
        _line("FAIL", f"OCR CUDA device probe failed: {exc}")
    if devices < 1:
        failures.append("ocr-gpu")
        _line("FAIL", "ocr.device=gpu but no CUDA device is available")
        return
    _line(
        "OK",
        f"OCR GPU: {models} (paddle {paddle.__version__}, {devices} device)",
    )


def _check_audio() -> None:
    try:
        import soundcard as sc

        speaker = sc.default_speaker()
        loopback = sc.get_microphone(speaker.name, include_loopback=True)
        _line("OK", f"default speaker: {speaker.name}")
        _line("OK", f"loopback capture: {loopback.name}")
    except Exception as exc:  # noqa: BLE001
        warnings.append("audio-device")
        _line("WARN", f"audio device probe failed: {exc}")


def _check_port(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        in_use = sock.connect_ex((host, port)) == 0
    _line(
        "WARN" if in_use else "OK",
        f"backend port {host}:{port} {'in use' if in_use else 'free'}",
    )
    if in_use:
        warnings.append("backend-port")


def _check_llama_asr(base_url: str) -> None:
    try:
        import httpx

        response = httpx.get(base_url.rstrip("/") + "/health", timeout=2.0)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status") if isinstance(payload, dict) else None
        if status in (None, "ok"):
            _line("OK", f"llama.cpp ASR server ready at {base_url}")
            return
        warnings.append("llama-asr-not-ready")
        _line("WARN", f"llama.cpp ASR server at {base_url} not ready: {payload}")
    except Exception as exc:  # noqa: BLE001
        warnings.append("llama-asr")
        _line(
            "WARN",
            f"llama.cpp ASR server not reachable at {base_url}: {exc} "
            "(start scripts/serve_asr_llama.ps1)",
        )


def _check_asr(base_url: str) -> None:
    try:
        import httpx

        response = httpx.get(base_url.rstrip("/") + "/", timeout=2.0)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("service") == "tpdk-asr":
            _line("OK", f"ASR service ready (model={payload.get('model', '?')})")
            return
        warnings.append("asr-service-contract")
        _line("WARN", "ASR endpoint is alive but is not services/asr_service.py")
    except Exception as exc:  # noqa: BLE001
        warnings.append("asr-service")
        _line("WARN", f"ASR service not reachable at {base_url}: {exc}")


def _check_mt(config: AppConfig | None) -> None:
    if config is None:
        return
    if config.mt.provider == "plamo_llama":
        base_url = config.mt.plamo_llama.base_url.rstrip("/")
        try:
            import httpx

            response = httpx.get(f"{base_url}/health", timeout=2.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            warnings.append("plamo-server")
            _line(
                "WARN",
                f"PLaMo translation server not reachable at {base_url}: {exc}; "
                "start scripts\\serve_mt_plamo.ps1",
            )
            return
        if isinstance(payload, dict) and payload.get("status") in (None, "ok", "ready"):
            _line("OK", f"PLaMo translation server ready at {base_url}")
        else:
            warnings.append("plamo-server")
            _line("WARN", f"PLaMo server at {base_url} is not ready: {payload}")
        return
    if config.mt.provider != "nllb_ct2":
        return
    try:
        import ctranslate2  # noqa: F401 - presence is reported by the import check
    except Exception:  # noqa: BLE001
        return
    model_dir = config.resolve(Path(config.mt.ct2.model))
    model_bin = model_dir / "model.bin"
    if model_bin.is_file():
        _line("OK", f"CTranslate2 NLLB model: {model_dir}")
    else:
        warnings.append("ct2-model")
        _line(
            "WARN",
            f"CTranslate2 NLLB model missing at {model_bin}; convert it with "
            "ct2-transformers-converter --model facebook/nllb-200-distilled-600M "
            "--output_dir data/models/nllb-ct2 --quantization int8",
        )


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    _check_python()
    _check_imports()
    config = _check_config()
    _check_torch(config)
    _check_ocr(config)
    _check_mt(config)
    _check_audio()
    if config is not None:
        _check_port(config.server.host, config.server.port)
        if config.asr.backend == "http":
            _check_asr(config.asr.base_url)
        elif config.asr.backend == "llama_cpp":
            _check_llama_asr(config.asr.llama_cpp.base_url)
        else:
            warnings.append("asr-backend")
            _line("WARN", f"unknown asr.backend {config.asr.backend!r}; check configs/default.yaml")

    if failures:
        print(f"\n{len(failures)} core check(s) failed: {', '.join(failures)}")
        return 1
    if warnings:
        print(f"\nReady with {len(warnings)} warning(s): {', '.join(warnings)}")
    else:
        print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
