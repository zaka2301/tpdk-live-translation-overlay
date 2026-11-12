"""Translation adapters (FR-03).

``create_translator`` selects the implementation from ``mt.provider`` without
importing torch, CTranslate2, or opening an HTTP client until a model is
actually requested:

* ``plamo_llama`` — PLaMo-3-Translate 0.8B GGUF behind a local ``llama serve``
  process on the GPU (default; the server must be running).
* ``nllb_ct2`` — CTranslate2 int8 on CPU (zero VRAM, keeps the 4 GB GPU for
  ASR).
* ``nllb`` — transformers, fp16 on CUDA when available (optional GPU path).

All adapters lazily load on a dedicated worker thread and preserve batch order;
see :mod:`tpdk_overlay.mt.base`.
"""

from __future__ import annotations

from tpdk_overlay.config import MtConfig
from tpdk_overlay.mt.base import TranslationError, WorkerTranslator
from tpdk_overlay.ports import Translator


def create_translator(config: MtConfig) -> Translator:
    """Build the configured translator; model loading stays lazy."""

    if config.provider == "plamo_llama":
        from tpdk_overlay.mt.plamo import PlamoTranslator

        return PlamoTranslator(config)
    if config.provider == "nllb":
        from tpdk_overlay.mt.nllb import NllbTranslator

        return NllbTranslator(config)
    if config.provider == "nllb_ct2":
        from tpdk_overlay.mt.nllb_ct2 import NllbCt2Translator

        return NllbCt2Translator(config)
    raise ValueError(f"unsupported mt.provider: {config.provider!r}")


__all__ = ["TranslationError", "WorkerTranslator", "create_translator"]
