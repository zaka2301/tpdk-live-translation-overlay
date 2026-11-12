from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx
import pytest

from tpdk_overlay.config import MtConfig, PlamoMtConfig
from tpdk_overlay.mt.base import TranslationError
from tpdk_overlay.mt.languages import to_plamo
from tpdk_overlay.mt.plamo import PlamoTranslator


class _FakePlamoServer:
    """MockTransport handler for /health and /v1/chat/completions."""

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.payloads: list[dict[str, Any]] = []
        self.health_calls = 0
        self._active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            self.health_calls += 1
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/v1/chat/completions", request.url.path
        payload = json.loads(request.content)
        self.payloads.append(payload)
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            text = payload["messages"][-1]["content"]
            target = payload["chat_template_kwargs"]["target_lang"]
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": f"[{target}] {text}"}}]},
            )
        finally:
            with self._lock:
                self._active -= 1


def _translator(
    server: _FakePlamoServer,
    **plamo_kwargs: Any,
) -> PlamoTranslator:
    config = MtConfig(
        provider="plamo_llama",
        plamo_llama=PlamoMtConfig(**plamo_kwargs),
    )
    return PlamoTranslator(config, transport=httpx.MockTransport(server))


def test_to_plamo_maps_aliases_and_validates() -> None:
    assert to_plamo("EN") == "en"
    assert to_plamo("zh") == "zh-hans"
    assert to_plamo("zh_Hant") == "zh-hant"
    assert to_plamo("fil") == "tl"
    with pytest.raises(ValueError, match="language code is empty"):
        to_plamo("  ")
    with pytest.raises(ValueError, match="unsupported PLaMo language"):
        to_plamo("pt")


def test_translate_preserves_order_and_skips_blanks() -> None:
    server = _FakePlamoServer()
    translator = _translator(server)
    try:
        outputs = translator.translate(
            ["one", "  ", "two", "three"],
            source_lang="en",
            target_lang="id",
        )
    finally:
        translator.close()

    assert outputs == ["[id] one", "", "[id] two", "[id] three"]
    assert [payload["messages"][-1]["content"] for payload in server.payloads] == [
        "one",
        "two",
        "three",
    ]


def test_request_uses_chat_template_options() -> None:
    server = _FakePlamoServer()
    translator = _translator(server, max_new_tokens=77)
    try:
        translator.translate(["hello"], source_lang="en", target_lang="id")
    finally:
        translator.close()

    payload = server.payloads[0]
    assert payload["model"] == "plamo-translate"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["chat_template_kwargs"] == {
        "source_lang": "en",
        "target_lang": "id",
    }
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 77
    assert payload["stop"] == ["<|plamo:tag|>", "<|plamo:bos|>"]


def test_warmup_checks_health_once() -> None:
    server = _FakePlamoServer()
    translator = _translator(server)
    try:
        translator.warmup()
        translator.warmup()
        translator.translate(["hello"], source_lang="en", target_lang="id")
    finally:
        translator.close()

    assert server.health_calls == 1


def test_batch_requests_run_concurrently() -> None:
    server = _FakePlamoServer(delay_s=0.05)
    translator = _translator(server, concurrency=4)
    try:
        outputs = translator.translate(
            [f"sentence {index}" for index in range(8)],
            source_lang="en",
            target_lang="id",
        )
    finally:
        translator.close()

    assert outputs == [f"[id] sentence {index}" for index in range(8)]
    assert server.max_active >= 2


def test_http_error_becomes_translation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(500, json={"error": "boom"})

    translator = PlamoTranslator(
        MtConfig(provider="plamo_llama"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(TranslationError, match="PLaMo server request failed"):
            translator.translate(["hello"], source_lang="en", target_lang="id")
    finally:
        translator.close()


def test_unreachable_server_fails_warmup() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    translator = PlamoTranslator(
        MtConfig(provider="plamo_llama"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(TranslationError, match="not reachable"):
            translator.warmup()
    finally:
        translator.close()
