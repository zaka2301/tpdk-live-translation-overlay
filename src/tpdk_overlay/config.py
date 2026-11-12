"""Typed application configuration.

Defaults live in ``configs/default.yaml``. Environment variables prefixed with
``TPDK__`` override any key using ``__`` as the nesting separator, for example
``TPDK__SERVER__PORT=9000`` or ``TPDK__OCR__ENABLED=false``. Values are parsed
as YAML scalars, so booleans, null, and numbers work without quotes.

``.env`` in the project root is loaded before overrides are applied. Secrets are
never stored in the config tree; ``SummaryConfig.api_key()`` reads the
environment variable named by ``summary.api_key_env`` at call time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tpdk_overlay.contracts import ScreenRegion

ENV_PREFIX = "TPDK__"


def project_root() -> Path:
    """Repository root: the directory containing ``pyproject.toml``."""

    return Path(__file__).resolve().parents[2]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LanguageConfig(_Section):
    source: str = "en"
    target: str = "id"


class AppSection(_Section):
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    language: LanguageConfig = LanguageConfig()

    @field_validator("log_level")
    @classmethod
    def _normalize_level(cls, value: str) -> str:
        return value.upper()


class ServerConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)


class VadConfig(_Section):
    engine: Literal["silero", "webrtc"] = "silero"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_ms: int = Field(default=250, ge=0)
    min_silence_ms: int = Field(default=450, ge=0)
    max_utterance_ms: int = Field(default=12_000, gt=0)
    pre_roll_ms: int = Field(default=300, ge=0)


class AudioConfig(_Section):
    source: Literal["loopback", "microphone"] = "loopback"
    input_device: str | None = None
    sample_rate: int = Field(default=48_000, gt=0)
    block_ms: int = Field(default=30, gt=0)
    vad: VadConfig = VadConfig()


class LlamaCppAsrConfig(_Section):
    """Qwen3-ASR GGUF served by a separate local ``llama serve`` process."""

    base_url: str = "http://127.0.0.1:8081"
    max_new_tokens: int = Field(default=128, gt=0)
    partial_interval_ms: int = Field(default=3000, gt=0)
    partial_tail_sec: float = Field(default=0.0, ge=0.0)
    min_energy_dbfs: float = -40.0
    system_prompt: str = ""


class AsrConfig(_Section):
    backend: Literal["http", "llama_cpp"] = "llama_cpp"
    base_url: str = "http://127.0.0.1:8000"
    model: str = "Qwen/Qwen3-ASR-0.6B"
    language: str | None = None
    chunk_size_sec: float = Field(default=1.0, gt=0)
    unfixed_chunk_num: int = Field(default=4, ge=0)
    unfixed_token_num: int = Field(default=5, ge=0)
    timeout_s: float = Field(default=30.0, gt=0)
    llama_cpp: LlamaCppAsrConfig = LlamaCppAsrConfig()


class MtCt2Config(_Section):
    """CTranslate2 conversion of the same NLLB checkpoint (int8 by default)."""

    model: str = "data/models/nllb-ct2"
    compute_type: Literal[
        "int8",
        "int8_float32",
        "int8_float16",
        "int8_bfloat16",
        "int16",
        "float16",
        "bfloat16",
        "float32",
    ] = "int8"
    max_new_tokens: int = Field(default=256, gt=0)


class PlamoMtConfig(_Section):
    """PLaMo-3-Translate GGUF served by a separate local ``llama serve`` process."""

    base_url: str = "http://127.0.0.1:8082"
    model: str = "plamo-translate"  # llama serve --alias, also the API model name
    model_path: str = (
        "data/models/plamo-3-translate/plamo-3-translate-0.8b-Q4_K_M.gguf"
    )
    hf_model: str = "pfnet/plamo-3-translate-0.8b-GGUF:Q4_K_M"
    context_size: int = Field(default=16384, gt=0)
    max_new_tokens: int = Field(default=256, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)
    concurrency: int = Field(default=8, gt=0)  # keep <= llama serve -np


class MtConfig(_Section):
    """Translation defaults to PLaMo GGUF on the GPU (``plamo_llama``).

    PLaMo runs out of process in a local ``llama serve`` instance; start it
    with ``scripts\\serve_mt_plamo.ps1``. Set ``provider: nllb_ct2`` for the
    CTranslate2 int8 CPU path (zero VRAM), or ``provider: nllb`` and
    ``device: cuda`` for the optional transformers fp16 path.
    """

    provider: Literal["plamo_llama", "nllb", "nllb_ct2"] = "plamo_llama"
    model: str = "facebook/nllb-200-distilled-600M"
    device: str = "cpu"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    max_input_tokens: int = Field(default=512, gt=0)
    max_new_tokens: int = Field(default=256, gt=0)
    batch_size: int = Field(default=8, gt=0)
    # Translate each source preview as it arrives, not only the final. The MT
    # worker translates previews only while the final queue is idle and keeps
    # just the newest preview per segment, so finals always take priority.
    translate_partials: bool = True
    ct2: MtCt2Config = MtCt2Config()
    plamo_llama: PlamoMtConfig = PlamoMtConfig()


class ScreenConfig(_Section):
    enabled: bool = True
    monitor: int = Field(default=1, ge=1)
    region: ScreenRegion | None = None
    capture_interval_ms: int = Field(default=1000, ge=100)
    change_threshold: float = Field(default=0.02, ge=0.0, le=1.0)


class OcrConfig(_Section):
    enabled: bool = True
    device: Literal["cpu", "gpu"] = "cpu"
    det_model: str = "PP-OCRv6_small_det"
    rec_model: str = "PP-OCRv6_small_rec"
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_blocks: int = Field(default=80, gt=0)
    block_gap_px: int = Field(default=18, ge=0)
    resize_max_side: int = Field(default=1280, ge=320)


class SummaryConfig(_Section):
    enabled: bool = True
    provider: Literal["gemini"] = "gemini"
    model: str = "gemini-3-flash-lite"
    checkpoint_interval_s: int = Field(default=300, gt=0)
    api_key_env: str = "GEMINI_API_KEY"
    max_output_tokens: int = Field(default=2048, gt=0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout_s: float = Field(default=60.0, gt=0)
    min_request_interval_s: float = Field(default=4.5, ge=0.0)

    def api_key(self) -> str | None:
        raw = os.getenv(self.api_key_env)
        if not raw:
            return None
        return raw.strip() or None


class CaptionConfig(_Section):
    width: int = Field(default=1100, gt=0)
    height: int = Field(default=260, gt=0)
    anchor: Literal["bottom-center", "bottom-left", "bottom-right", "top-center"] = "bottom-center"
    opacity: float = Field(default=0.85, ge=0.0, le=1.0)
    font_family: str = "Segoe UI"
    font_size: int = Field(default=20, gt=0)
    translation_font_size: int = Field(default=22, gt=0)
    # Minimum time a translated final remains before the next segment may
    # replace it; the caption coalesces events while it is held.
    final_hold_ms: int = Field(default=2200, ge=0, le=10_000)


class OcrOverlayConfig(_Section):
    enabled: bool = True
    background: str = "#101418"
    opacity: float = Field(default=0.92, ge=0.0, le=1.0)
    font_family: str = "Segoe UI"
    font_size: int = Field(default=14, gt=0)


class UiConfig(_Section):
    caption: CaptionConfig = CaptionConfig()
    ocr_overlay: OcrOverlayConfig = OcrOverlayConfig()
    click_through: bool = True


class AppConfig(_Section):
    """Complete configuration tree.

    ``root_dir`` is injected by :func:`load_config`; it is excluded from
    serialization so it cannot leak into a config dump.
    """

    app: AppSection
    server: ServerConfig
    audio: AudioConfig
    asr: AsrConfig
    mt: MtConfig
    screen: ScreenConfig
    ocr: OcrConfig
    summary: SummaryConfig
    ui: UiConfig
    root_dir: Path = Field(default_factory=project_root, exclude=True)

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root_dir / path

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.app.data_dir)

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"


def _parse_env_value(raw: str) -> Any:
    if raw.strip() == "":
        return ""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        current = base.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            _deep_merge(current, value)
        else:
            base[key] = value


def _env_overrides() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX) :].lower().split("__")
        if not all(parts):
            continue
        cursor = result
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                break
            cursor = child
        else:
            cursor[parts[-1]] = _parse_env_value(raw)
    return result


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML defaults, apply ``TPDK__*`` overrides, and validate."""

    load_dotenv(project_root() / ".env", override=False)
    config_path = (
        Path(path).expanduser()
        if path is not None
        else project_root() / "configs" / "default.yaml"
    )
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    _deep_merge(data, _env_overrides())
    return AppConfig.model_validate(data)


__all__ = [
    "AppConfig",
    "AppSection",
    "AsrConfig",
    "AudioConfig",
    "CaptionConfig",
    "ENV_PREFIX",
    "LanguageConfig",
    "LlamaCppAsrConfig",
    "MtConfig",
    "MtCt2Config",
    "OcrConfig",
    "OcrOverlayConfig",
    "ScreenConfig",
    "ServerConfig",
    "SummaryConfig",
    "UiConfig",
    "VadConfig",
    "load_config",
    "project_root",
]
