"""ASR adapters (FR-02, NFR-02).

Two interchangeable backends behind :class:`tpdk_overlay.ports.AsrClient`:

* ``llama_cpp`` (default) - Qwen3-ASR GGUF behind a separate local
  ``llama serve`` process (``asr/llama_cpp_backend.py``). No torch in this
  process; partial previews over the recent audio tail; auto-detects language.
* ``http`` (optional) - the WSL2 Qwen3-ASR/vLLM service in
  ``services/asr_service.py``. Streaming partials, lowest latency; use it for
  report/vLLM runs.

Select with ``asr.backend`` (``llama_cpp`` | ``http``).
"""

from __future__ import annotations

from tpdk_overlay.asr.errors import AsrTimeout, AsrUnavailable
from tpdk_overlay.config import AsrConfig
from tpdk_overlay.ports import AsrClient


def create_asr_client(config: AsrConfig) -> AsrClient:
    """Build the ASR adapter selected by ``config.backend``."""

    if config.backend == "llama_cpp":
        from tpdk_overlay.asr.llama_cpp_backend import LlamaCppAsrClient

        return LlamaCppAsrClient(config)
    if config.backend == "http":
        from tpdk_overlay.asr.http_backend import HttpAsrClient

        return HttpAsrClient(config)
    raise ValueError(f"unknown ASR backend: {config.backend!r}")


__all__ = ["AsrTimeout", "AsrUnavailable", "create_asr_client"]
