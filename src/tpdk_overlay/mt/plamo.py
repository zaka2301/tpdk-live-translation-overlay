"""PLaMo-3-Translate GGUF through a local ``llama serve`` process.

This is the default MT path on this workstation: translation runs on the GPU
inside a separate llama.cpp server (``mt.plamo_llama.base_url``), while the
backend only sends OpenAI-compatible chat requests. The model card's chat
template accepts ``source_lang``/``target_lang`` through
``chat_template_kwargs``; generation stops at the PFML3 boundary tokens.

One request carries one text (the chat endpoint has no batch input), so batches
are fanned out to ``mt.plamo_llama.concurrency`` parallel requests, matching the
server's ``-np`` slot count, and results are returned in input order.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from tpdk_overlay.config import MtConfig
from tpdk_overlay.mt.base import TranslationError, WorkerTranslator
from tpdk_overlay.mt.languages import to_plamo

logger = logging.getLogger(__name__)

_WARMUP_TIMEOUT_S = 5.0
_STOP_TOKENS = ("<|plamo:tag|>", "<|plamo:bos|>")
_READY_STATUSES = frozenset({None, "ok", "ready"})


class PlamoTranslator(WorkerTranslator):
    """Translate through a local PLaMo llama.cpp server."""

    def __init__(
        self,
        config: MtConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__()
        self._plamo = config.plamo_llama
        self._transport = transport
        self._client: httpx.Client | None = None
        self._pool: ThreadPoolExecutor | None = None

    def _language_codes(self, source_lang: str, target_lang: str) -> tuple[str, str]:
        return to_plamo(source_lang), to_plamo(target_lang)

    # -- worker-thread hooks -------------------------------------------------

    def _load(self) -> None:
        base_url = self._plamo.base_url.rstrip("/")
        client = httpx.Client(
            base_url=base_url,
            timeout=self._plamo.timeout_s,
            transport=self._transport,
        )
        try:
            response = client.get("/health", timeout=_WARMUP_TIMEOUT_S)
            response.raise_for_status()
            payload = response.json() if response.content.strip() else {}
        except httpx.TimeoutException as exc:
            client.close()
            raise TranslationError(f"PLaMo server warmup timed out at {base_url}") from exc
        except httpx.HTTPError as exc:
            client.close()
            raise TranslationError(
                f"PLaMo server is not reachable at {base_url} ({exc}). Start it with "
                "scripts\\serve_mt_plamo.ps1 or: llama serve "
                "-m data/models/plamo-3-translate/plamo-3-translate-0.8b-Q4_K_M.gguf "
                "--port 8082 --jinja --no-prefill-assistant"
            ) from exc
        except ValueError as exc:
            client.close()
            raise TranslationError("PLaMo server returned non-JSON for /health") from exc

        if isinstance(payload, dict) and payload.get("status") not in _READY_STATUSES:
            client.close()
            raise TranslationError(f"PLaMo server at {base_url} is not ready: {payload}")

        self._client = client
        self._pool = ThreadPoolExecutor(
            max_workers=self._plamo.concurrency,
            thread_name_prefix="plamo-mt",
        )
        logger.info(
            "PLaMo translation server reachable at %s (model=%s, concurrency=%d)",
            base_url,
            self._plamo.model,
            self._plamo.concurrency,
        )

    def _translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        client, pool = self._client, self._pool
        if client is None or pool is None:
            raise TranslationError("PLaMo translator is not loaded")

        def translate_one(text: str) -> str:
            return self._translate_one(client, text, source_lang, target_lang)

        return list(pool.map(translate_one, texts))

    def _release(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- HTTP ----------------------------------------------------------------

    def _translate_one(
        self,
        client: httpx.Client,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._plamo.model,
            "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {
                "source_lang": source_lang,
                "target_lang": target_lang,
            },
            "temperature": 0,
            "max_tokens": self._plamo.max_new_tokens,
            "stop": list(_STOP_TOKENS),
        }
        try:
            response = client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise TranslationError(
                f"PLaMo translation timed out after {self._plamo.timeout_s:.0f}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise TranslationError(f"PLaMo server request failed: {exc}") from exc
        except ValueError as exc:
            raise TranslationError("PLaMo server returned non-JSON") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TranslationError("PLaMo server returned an unexpected payload") from exc
        if not isinstance(content, str):
            raise TranslationError("PLaMo server returned non-text content")
        return content.strip()


__all__ = ["PlamoTranslator"]
