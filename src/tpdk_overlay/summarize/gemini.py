"""Gemini adapter for checkpoint and final summaries (FR-07, FR-08, NFR-06).

Only text leaves the machine: the prompts assembled in
:mod:`tpdk_overlay.summarize.checkpoints` carry the transcript and OCR text,
never audio or screenshot pixels (trust boundary, AGENTS.md §1).

The SDK is imported lazily so a transcript-only install can import this module
(and the backend's guarded factory) without the ``summarize`` extra. Failures
surface as :class:`~tpdk_overlay.ports.SummarizerUnavailable`: the orchestrator
marks the checkpoint ``degraded`` and keeps transcribing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tpdk_overlay.config import SummaryConfig
from tpdk_overlay.ports import (
    CheckpointSummaryRequest,
    FinalSummaryRequest,
    SummarizerUnavailable,
)
from tpdk_overlay.summarize.checkpoints import (
    RequestPacer,
    checkpoint_prompt,
    final_prompt,
    is_retryable,
    retry_delay,
)

logger = logging.getLogger(__name__)


class GeminiSummarizer:
    """``ports.Summarizer`` backed by the Gemini Developer API.

    ``client`` is injectable so unit tests exercise retries and prompt wiring
    without a network call; production leaves it ``None`` and the SDK client is
    created on first use with the key from ``SummaryConfig.api_key()``.
    """

    def __init__(self, config: SummaryConfig, *, client: Any | None = None) -> None:
        self._config = config
        self._client = client
        self._pacer = RequestPacer(config.min_request_interval_s)

    async def checkpoint(self, request: CheckpointSummaryRequest) -> str:
        return await self._generate(*checkpoint_prompt(request))

    async def final(self, request: FinalSummaryRequest) -> str:
        return await self._generate(*final_prompt(request))

    # -- internals ---------------------------------------------------------

    def _api_key(self) -> str:
        key = self._config.api_key()
        if not key:
            raise SummarizerUnavailable(
                f"{self._config.api_key_env} is not set; summaries are disabled"
            )
        return key

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=self._api_key(),
            http_options=types.HttpOptions(
                timeout=int(self._config.timeout_s * 1000)
            ),
        )
        return self._client

    async def _generate(self, system_instruction: str, user_prompt: str) -> str:
        from google.genai import types

        client = self._ensure_client()
        attempts = self._config.max_attempts
        for attempt in range(1, attempts + 1):
            await self._pacer.wait()
            try:
                response = await client.aio.models.generate_content(
                    model=self._config.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self._config.temperature,
                        max_output_tokens=self._config.max_output_tokens,
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - mapped to the port's error
                if attempt >= attempts or not is_retryable(exc):
                    raise SummarizerUnavailable(
                        f"Gemini request failed: {exc}"
                    ) from exc
                delay = retry_delay(
                    attempt,
                    base_s=self._config.retry_base_s,
                    max_s=self._config.retry_max_s,
                )
                logger.warning(
                    "Gemini request failed (%s); retry %d/%d in %.2f s",
                    exc,
                    attempt,
                    attempts - 1,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            text = (response.text or "").strip()
            if not text:
                raise SummarizerUnavailable("Gemini returned an empty summary")
            return text
        raise SummarizerUnavailable("Gemini request failed")  # pragma: no cover


__all__ = ["GeminiSummarizer"]
