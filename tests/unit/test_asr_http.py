from __future__ import annotations

import httpx
import numpy as np
import pytest

from tpdk_overlay.asr import create_asr_client
from tpdk_overlay.asr.errors import AsrTimeout, AsrUnavailable
from tpdk_overlay.asr.http_backend import HttpAsrClient
from tpdk_overlay.config import AsrConfig


def _client(handler, **config_kwargs) -> HttpAsrClient:
    config = AsrConfig(chunk_size_sec=1.0, **config_kwargs)
    return HttpAsrClient(config, transport=httpx.MockTransport(handler))


def test_open_passes_language_and_streams_cumulative_text() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/start":
            assert request.url.params.get("language") == "English"
            return httpx.Response(200, json={"session_id": "s1"})
        if request.url.path == "/api/chunk":
            assert request.url.params["session_id"] == "s1"
            assert request.headers["content-type"] == "application/octet-stream"
            return httpx.Response(200, json={"language": "English", "text": "partial"})
        if request.url.path == "/api/finish":
            return httpx.Response(200, json={"language": "English", "text": "final"})
        raise AssertionError(f"unexpected path {request.url}")

    client = _client(handler)
    stream = client.open(language="English")

    update = stream.push(np.zeros(16_000, dtype=np.float32))
    assert update.language == "English"
    assert update.text == "partial"

    # 8k samples stay buffered until finish, which must flush them first.
    assert stream.push(np.zeros(8_000, dtype=np.float32)).text == "partial"
    final = stream.finish()
    assert final.text == "final"
    assert len([call for call in calls if call.url.path == "/api/chunk"]) == 2
    client.close()


def test_finish_is_idempotent_and_close_is_safe() -> None:
    finish_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal finish_calls
        if request.url.path == "/api/start":
            return httpx.Response(200, json={"session_id": "s1"})
        if request.url.path == "/api/chunk":
            return httpx.Response(200, json={"language": "English", "text": "x"})
        if request.url.path == "/api/finish":
            finish_calls += 1
            return httpx.Response(200, json={"language": "English", "text": "final"})
        raise AssertionError(f"unexpected path {request.url}")

    client = _client(handler)
    stream = client.open()
    stream.push(np.zeros(16_000, dtype=np.float32))

    assert stream.finish().text == "final"
    assert stream.finish().text == "final"
    stream.close()
    assert finish_calls == 1
    client.close()


def test_service_error_becomes_asr_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    client = _client(handler)
    with pytest.raises(AsrUnavailable):
        client.open()
    client.close()


def test_timeout_becomes_asr_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow service", request=request)

    client = _client(handler)
    with pytest.raises(AsrTimeout):
        client.open()
    client.close()


def test_invalid_start_payload_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = _client(handler)
    with pytest.raises(AsrUnavailable):
        client.open()
    client.close()


def test_warmup_accepts_upstream_demo_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/"
        return httpx.Response(200, text="<html>demo</html>")

    client = _client(handler)
    client.warmup()
    client.close()


def test_factory_selects_http() -> None:
    client = create_asr_client(AsrConfig(backend="http"))
    assert isinstance(client, HttpAsrClient)
    client.close()
