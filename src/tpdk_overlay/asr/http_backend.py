"""HTTP adapter for the Qwen3-ASR streaming service (WSL2 + vLLM).

This is the default backend. It speaks the contract documented in
``services/README.md``: one stream per utterance, cumulative text on every
chunk, no timestamps. Streaming partials arrive while the lecturer speaks.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import numpy as np

from tpdk_overlay.asr.errors import AsrTimeout, AsrUnavailable
from tpdk_overlay.config import AsrConfig
from tpdk_overlay.ports import AsrStream, AsrUpdate
from tpdk_overlay.utils.audio import SAMPLE_RATE_16K

logger = logging.getLogger(__name__)


def _request_json(client: httpx.Client, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = client.request(method, path, **kwargs)
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise AsrTimeout(f"ASR request timed out: {path}") from exc
    except httpx.HTTPError as exc:
        raise AsrUnavailable(f"ASR request failed: {path}: {exc}") from exc
    except ValueError as exc:
        raise AsrUnavailable(f"ASR returned non-JSON for {path}") from exc
    if not isinstance(payload, dict):
        raise AsrUnavailable(f"ASR returned an unexpected payload for {path}")
    return payload


def _update(payload: dict[str, Any]) -> AsrUpdate:
    return AsrUpdate(
        language=str(payload.get("language") or ""),
        text=str(payload.get("text") or ""),
    )


class HttpAsrClient:
    def __init__(
        self,
        config: AsrConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_s,
            transport=transport,
        )

    def warmup(self) -> None:
        """Fail fast when the service is down; accepts the upstream demo's HTML."""

        try:
            self._client.get("/").raise_for_status()
        except httpx.TimeoutException as exc:
            raise AsrTimeout("ASR warmup timed out") from exc
        except httpx.HTTPError as exc:
            raise AsrUnavailable(f"ASR warmup failed: {exc}") from exc

    def open(self, *, language: str | None = None) -> AsrStream:
        params = {"language": language} if language else None
        payload = _request_json(self._client, "POST", "/api/start", params=params)
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise AsrUnavailable("ASR service did not return a session_id")
        return _HttpAsrStream(self._client, session_id, self._config.chunk_size_sec)

    def close(self) -> None:
        self._client.close()


class _HttpAsrStream:
    def __init__(self, client: httpx.Client, session_id: str, chunk_size_sec: float) -> None:
        self._client = client
        self._session_id = session_id
        self._chunk_samples = max(1, int(round(chunk_size_sec * SAMPLE_RATE_16K)))
        self._buffer = np.zeros(0, dtype=np.float32)
        self._latest = AsrUpdate(language="", text="")
        self._final: AsrUpdate | None = None
        self._closed = False

    def push(self, pcm16k: np.ndarray) -> AsrUpdate:
        if self._closed:
            raise AsrUnavailable("ASR stream is closed")
        pcm = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        if pcm.size:
            self._buffer = np.concatenate((self._buffer, pcm))
        while self._buffer.size >= self._chunk_samples:
            chunk = self._buffer[: self._chunk_samples]
            self._buffer = self._buffer[self._chunk_samples :]
            self._latest = self._post_chunk(chunk)
        return self._latest

    def finish(self) -> AsrUpdate:
        if self._final is not None:
            return self._final
        if self._closed:
            raise AsrUnavailable("ASR stream is closed")
        try:
            if self._buffer.size:
                self._post_chunk(self._buffer)
                self._buffer = np.zeros(0, dtype=np.float32)
            payload = _request_json(
                self._client,
                "POST",
                "/api/finish",
                params={"session_id": self._session_id},
            )
            self._final = _update(payload)
        except Exception:
            self._closed = True
            raise
        self._closed = True
        return self._final

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer = np.zeros(0, dtype=np.float32)
        try:
            self._client.post("/api/finish", params={"session_id": self._session_id})
        except Exception:
            logger.debug("ASR stream cleanup failed", exc_info=True)

    def _post_chunk(self, chunk: np.ndarray) -> AsrUpdate:
        payload = _request_json(
            self._client,
            "POST",
            "/api/chunk",
            params={"session_id": self._session_id},
            content=np.asarray(chunk, dtype="<f4").tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        return _update(payload)
