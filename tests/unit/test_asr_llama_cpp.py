from __future__ import annotations

import base64
import io
import json
import logging
import wave
from typing import Any

import httpx
import numpy as np
import pytest

from tpdk_overlay.asr import create_asr_client
from tpdk_overlay.asr.errors import AsrTimeout, AsrUnavailable
from tpdk_overlay.asr.llama_cpp_backend import LlamaCppAsrClient
from tpdk_overlay.config import AsrConfig, LlamaCppAsrConfig


class _FakeLlamaServer:
    """MockTransport handler that records chat payloads and replays replies."""

    def __init__(self, *contents: str, finish_reason: str = "stop") -> None:
        self.contents = list(contents)
        self.finish_reason = finish_reason
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions", request.url.path
        self.payloads.append(json.loads(request.content))
        content = (
            self.contents.pop(0) if self.contents else "language English<asr_text>Hello"
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": content}, "finish_reason": self.finish_reason}
                ]
            },
        )


def _client(server: httpx.MockTransport, **llama_kwargs: Any) -> LlamaCppAsrClient:
    config = AsrConfig(
        backend="llama_cpp",
        llama_cpp=LlamaCppAsrConfig(**llama_kwargs),
    )
    return LlamaCppAsrClient(config, transport=server)


def _loud(count: int = 16_000) -> np.ndarray:
    return np.full(count, 0.25, dtype=np.float32)


def _wav_info(encoded: str) -> tuple[int, int, int, int]:
    with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as handle:
        return (
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getframerate(),
            handle.getnframes(),
        )


def _payload_audio(payload: dict[str, Any]) -> str:
    return payload["messages"][0]["content"][0]["input_audio"]["data"]


@pytest.mark.parametrize("body", [{"status": "ok"}, {}])
def test_warmup_accepts_ok_or_missing_status(body: dict[str, Any]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    client = _client(httpx.MockTransport(handler))
    client.warmup()
    assert [request.url.path for request in seen] == ["/health"]
    assert seen[0].extensions["timeout"]["read"] == 5.0
    client.close()


def test_warmup_rejects_not_ready_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "loading model"})

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(AsrUnavailable):
        client.warmup()
    client.close()


def test_warmup_maps_connection_error_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(AsrUnavailable):
        client.warmup()
    client.close()


def test_warmup_maps_timeout_to_asr_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow server", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(AsrTimeout):
        client.warmup()
    client.close()


def test_warmup_warns_when_language_is_configured(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    client = LlamaCppAsrClient(
        AsrConfig(
            backend="llama_cpp",
            language="English",
            llama_cpp=LlamaCppAsrConfig(),
        ),
        transport=httpx.MockTransport(handler),
    )
    with caplog.at_level(logging.WARNING, logger="tpdk_overlay.asr.llama_cpp_backend"):
        client.warmup()
    assert any("auto-detects" in record.getMessage() for record in caplog.records)
    client.close()


def test_partial_preview_then_final_uses_whole_utterance() -> None:
    server = _FakeLlamaServer()
    client = _client(
        httpx.MockTransport(server),
        partial_interval_ms=1000,
        partial_tail_sec=4.0,
        max_new_tokens=128,
    )
    stream = client.open(language="Indonesian")

    # 0.5 s: below the 1 s partial interval.
    update = stream.push(_loud(8_000))
    assert update.text == ""
    assert update.language == "id"
    assert server.payloads == []

    # Reaching 1 s triggers a partial over the last partial_tail_sec.
    update = stream.push(_loud(8_000))
    assert update.language == "en"
    assert update.text == "Hello"
    assert len(server.payloads) == 1
    partial = server.payloads[0]
    assert partial["max_tokens"] == 64
    assert partial["temperature"] == 0.0
    assert partial["repeat_penalty"] == 1.1
    assert partial["cache_prompt"] is False
    assert partial["stream"] is False
    assert partial["messages"][0]["role"] == "user"
    audio = partial["messages"][0]["content"][0]["input_audio"]
    assert audio["format"] == "wav"
    assert _wav_info(audio["data"]) == (1, 2, 16_000, 16_000)

    # 0.5 s after the partial: still below the interval.
    assert stream.push(_loud(8_000)).text == "Hello"
    assert len(server.payloads) == 1

    final = stream.finish()
    assert final.language == "en"
    assert final.text == "Hello"
    assert len(server.payloads) == 2
    full = server.payloads[1]
    assert full["max_tokens"] == 128
    frames = full["messages"][0]["content"][0]["input_audio"]["data"]
    assert _wav_info(frames) == (1, 2, 16_000, 24_000)
    client.close()


def test_zero_tail_previews_the_whole_utterance() -> None:
    server = _FakeLlamaServer()
    client = _client(
        httpx.MockTransport(server),
        partial_interval_ms=1000,
        partial_tail_sec=0.0,
    )
    stream = client.open()

    stream.push(_loud(8_000))
    assert server.payloads == []

    stream.push(_loud(8_000))
    assert len(server.payloads) == 1
    assert _wav_info(_payload_audio(server.payloads[0])) == (1, 2, 16_000, 16_000)

    stream.push(_loud(16_000))
    assert len(server.payloads) == 2
    assert _wav_info(_payload_audio(server.payloads[1])) == (1, 2, 16_000, 32_000)
    client.close()


def test_partial_budget_never_exceeds_configured_max_tokens() -> None:
    server = _FakeLlamaServer()
    client = _client(
        httpx.MockTransport(server), partial_interval_ms=1000, max_new_tokens=32
    )
    stream = client.open()

    stream.push(_loud(16_000))
    assert server.payloads[0]["max_tokens"] == 32
    stream.finish()
    assert server.payloads[1]["max_tokens"] == 32
    client.close()


def test_system_prompt_is_sent_with_the_audio() -> None:
    server = _FakeLlamaServer()
    client = _client(
        httpx.MockTransport(server),
        partial_interval_ms=1000,
        system_prompt="Transcribe only human speech.",
    )
    stream = client.open()
    stream.push(_loud(16_000))

    messages = server.payloads[0]["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "Transcribe only human speech.",
    }
    assert messages[1]["role"] == "user"
    client.close()


def test_no_partial_before_interval_and_finish_flushes() -> None:
    server = _FakeLlamaServer("language English<asr_text>Full utterance")
    client = _client(httpx.MockTransport(server))
    stream = client.open()

    assert stream.push(_loud(16_000)).text == ""
    assert server.payloads == []

    final = stream.finish()
    assert final.text == "Full utterance"
    assert len(server.payloads) == 1
    assert server.payloads[0]["max_tokens"] == 128
    frames = server.payloads[0]["messages"][0]["content"][0]["input_audio"]["data"]
    assert _wav_info(frames) == (1, 2, 16_000, 16_000)
    client.close()


def test_near_silence_is_not_posted() -> None:
    server = _FakeLlamaServer()
    client = _client(httpx.MockTransport(server), partial_interval_ms=1000)
    stream = client.open()

    # -60 dBFS room tone: audible to the mic, but below min_energy_dbfs (-40).
    room_tone = np.full(16_000, 0.001, dtype=np.float32)
    assert stream.push(room_tone).text == ""
    assert server.payloads == []

    final = stream.finish()
    assert final.text == ""
    assert server.payloads == []
    client.close()


def test_window_shorter_than_minimum_is_not_posted() -> None:
    server = _FakeLlamaServer()
    client = _client(httpx.MockTransport(server), partial_interval_ms=100)
    stream = client.open()

    # 0.1 s is below the 0.2 s minimum even though it clears the partial interval.
    assert stream.push(_loud(1_600)).text == ""
    assert server.payloads == []
    assert stream.finish().text == ""
    assert server.payloads == []
    client.close()


@pytest.mark.parametrize(
    ("content", "expected_language", "expected_text"),
    [
        ("language English<asr_text>Hello", "en", "Hello"),
        ("language Indonesian<asr_text>Halo dunia", "id", "Halo dunia"),
        ("language en-US<asr_text>Hi", "en", "Hi"),
        ("language Japanese<asr_text>こんにちは", "ja", "こんにちは"),
        ("language Klingon<asr_text>Qapla", "klingon", "Qapla"),
    ],
)
def test_language_prefix_is_normalized(
    content: str, expected_language: str, expected_text: str
) -> None:
    server = _FakeLlamaServer(content)
    client = _client(httpx.MockTransport(server), partial_interval_ms=1000)
    stream = client.open()

    update = stream.push(_loud(16_000))
    assert update.language == expected_language
    assert update.text == expected_text
    client.close()


def test_plain_reply_without_prefix_uses_open_language() -> None:
    server = _FakeLlamaServer("Hello there")
    client = _client(httpx.MockTransport(server), partial_interval_ms=1000)
    stream = client.open(language="Indonesian")

    update = stream.push(_loud(16_000))
    assert update.language == "id"
    assert update.text == "Hello there"
    client.close()


def test_music_hallucination_drops_and_keeps_previous_preview() -> None:
    server = _FakeLlamaServer(
        "language English<asr_text>Hello",
        "language English<asr_text>Thank you for watching",
        "language English<asr_text>Thanks for watching!",
    )
    client = _client(httpx.MockTransport(server), partial_interval_ms=1000)
    stream = client.open()

    assert stream.push(_loud(16_000)).text == "Hello"
    # Dropped hallucination: the old preview survives instead of clearing.
    assert stream.push(_loud(16_000)).text == "Hello"

    final = stream.finish()
    assert final.text == ""
    assert final.language == "en"
    assert len(server.payloads) == 3
    client.close()


def test_repetition_loop_is_dropped() -> None:
    server = _FakeLlamaServer(
        "language English<asr_text>" + "ab" * 100,
        "language English<asr_text>" + "ab" * 100,
    )
    client = _client(httpx.MockTransport(server), partial_interval_ms=1000)
    stream = client.open()

    assert stream.push(_loud(16_000)).text == ""
    final = stream.finish()
    assert final.text == ""
    assert len(server.payloads) == 2
    client.close()


def test_length_finish_reason_warns(caplog: pytest.LogCaptureFixture) -> None:
    server = _FakeLlamaServer("language English<asr_text>Hello", finish_reason="length")
    client = _client(httpx.MockTransport(server))
    stream = client.open()
    stream.push(_loud(16_000))

    with caplog.at_level(logging.WARNING, logger="tpdk_overlay.asr.llama_cpp_backend"):
        assert stream.finish().text == "Hello"
    assert any("max_tokens" in record.getMessage() for record in caplog.records)
    client.close()


def test_finish_is_idempotent_and_closed_stream_rejects_use() -> None:
    server = _FakeLlamaServer("language English<asr_text>Final")
    client = _client(httpx.MockTransport(server))
    stream = client.open()
    stream.push(_loud(16_000))

    first = stream.finish()
    second = stream.finish()
    assert first == second
    assert len(server.payloads) == 1

    with pytest.raises(AsrUnavailable):
        stream.push(_loud(100))

    stream.close()
    stream.close()
    client.close()


def test_close_clears_the_buffer() -> None:
    server = _FakeLlamaServer()
    client = _client(httpx.MockTransport(server))
    stream = client.open()
    stream.push(_loud(16_000))

    stream.close()
    with pytest.raises(AsrUnavailable):
        stream.push(_loud(100))
    with pytest.raises(AsrUnavailable):
        stream.finish()
    # Closing discards the utterance: no request was made.
    assert server.payloads == []
    client.close()


def test_final_request_failure_clears_and_closes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("server crashed", request=request)

    client = _client(httpx.MockTransport(handler))
    stream = client.open()
    stream.push(_loud(16_000))

    with pytest.raises(AsrUnavailable):
        stream.finish()
    with pytest.raises(AsrUnavailable):
        stream.finish()
    with pytest.raises(AsrUnavailable):
        stream.push(_loud(100))
    client.close()


def test_factory_selects_llama_cpp() -> None:
    assert AsrConfig().backend == "llama_cpp"
    client = create_asr_client(AsrConfig())
    assert isinstance(client, LlamaCppAsrClient)
    client.close()
