"""ASR adapters (FR-02, NFR-02).

Two interchangeable backends behind :class:`tpdk_overlay.ports.AsrClient`:

* ``http`` (default) - the WSL2 Qwen3-ASR/vLLM service in
  ``services/asr_service.py``. Streaming partials, lowest latency.
* ``native`` - ``qwen_asr`` transformers inference directly in the Windows
  process (``asr/transformers_backend.py``). Backup when WSL/vLLM is
  unavailable; buffers each utterance and produces only a final transcript.

Select with ``asr.backend`` (``http`` | ``native``).
"""

from __future__ import annotations

from tpdk_overlay.asr.errors import AsrTimeout, AsrUnavailable
from tpdk_overlay.config import AsrConfig
from tpdk_overlay.ports import AsrClient


def create_asr_client(config: AsrConfig) -> AsrClient:
    """Build the ASR adapter selected by ``config.backend``."""

    if config.backend == "http":
        from tpdk_overlay.asr.http_backend import HttpAsrClient

        return HttpAsrClient(config)
    if config.backend == "native":
        from tpdk_overlay.asr.transformers_backend import TransformersAsrClient

        return TransformersAsrClient(config)
    raise ValueError(f"unknown ASR backend: {config.backend!r}")


__all__ = ["AsrTimeout", "AsrUnavailable", "create_asr_client"]
