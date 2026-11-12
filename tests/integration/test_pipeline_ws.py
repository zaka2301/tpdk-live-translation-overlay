"""Integration: real orchestrator + WebSocket behind fake ports (AGENT.md §7 phase 3).

No GPU, no ASR service, no audio device: the fake audio source plays a scripted
utterance, a fake segmenter emits ``SpeechStart``/``Utterance``, and a fake ASR
stream returns a partial plus a final. The tests assert the observable contract:
idle -> running -> partial/final -> stopped, persisted artifacts, and that an ASR
failure only produces a notice while the session keeps running.
"""

from __future__ import annotations

import csv
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from tpdk_overlay.asr.errors import AsrUnavailable
from tpdk_overlay.audio.vad import VAD_FRAME_MS, VAD_FRAME_SAMPLES, SpeechStart, Utterance
from tpdk_overlay.config import AppConfig, load_config
from tpdk_overlay.pipeline.orchestrator import PipelinePorts
from tpdk_overlay.ports import AsrUpdate
from tpdk_overlay.server.app import create_app

FRAME = VAD_FRAME_SAMPLES


class FakeAudioSource:
    """Play a fixed frame script, then idle; paced so partials can be painted."""

    def __init__(self, script: list[np.ndarray], pace_s: float = 0.01) -> None:
        self._script = deque(script)
        self._pace_s = pace_s
        self._stopped = threading.Event()

    def start(self) -> None:
        pass

    def read(self, frames: int) -> np.ndarray:
        assert frames == FRAME
        if self._script:
            time.sleep(self._pace_s)
            return self._script.popleft()
        self._stopped.wait(0.02)
        return np.zeros(FRAME, dtype=np.float32)

    def stop(self) -> None:
        self._stopped.set()


class FakeSegmenter:
    """First loud frame starts an utterance; six quiet frames close it."""

    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._in_speech = False
        self._start_index = 0
        self._silence = 0

    def feed(self, frame: np.ndarray) -> Any:
        loud = float(np.max(np.abs(frame))) > 0.1
        position = len(self._frames)
        self._frames.append(frame)
        if not self._in_speech:
            if not loud:
                return None
            self._in_speech = True
            self._start_index = position
            self._silence = 0
            return SpeechStart(
                audio=frame.copy(), t_start_ms=position * VAD_FRAME_MS
            )
        if loud:
            self._silence = 0
            return None
        self._silence += 1
        if self._silence < 6:
            return None
        self._in_speech = False
        end_index = position + 1 - self._silence
        audio = np.concatenate(self._frames[self._start_index : end_index])
        return Utterance(
            audio=audio,
            t_start_ms=self._start_index * VAD_FRAME_MS,
            t_end_ms=end_index * VAD_FRAME_MS,
            reason="silence",
        )

    def reset(self) -> None:
        self._frames.clear()
        self._in_speech = False
        self._start_index = 0
        self._silence = 0


class FakeAsrStream:
    def __init__(self) -> None:
        self._samples = 0
        self._closed = False

    def push(self, pcm16k: np.ndarray) -> AsrUpdate:
        if self._closed:
            raise AsrUnavailable("stream closed")
        self._samples += int(pcm16k.size)
        if self._samples >= FRAME * 6:
            return AsrUpdate(language="en", text="hello wor")
        return AsrUpdate(language="en", text="")

    def finish(self) -> AsrUpdate:
        if self._closed:
            raise AsrUnavailable("stream closed")
        self._closed = True
        return AsrUpdate(language="en", text="hello world")

    def close(self) -> None:
        self._closed = True


class FailingAsrStream(FakeAsrStream):
    """Dies mid-utterance, after the stream opened successfully."""

    def push(self, pcm16k: np.ndarray) -> AsrUpdate:
        if self._samples >= FRAME * 2:
            raise AsrUnavailable("fake ASR was killed")
        return super().push(pcm16k)


class FakeAsrClient:
    def __init__(self, stream_factory: Any = FakeAsrStream) -> None:
        self._stream_factory = stream_factory

    def warmup(self) -> None:
        pass

    def open(self, *, language: str | None = None) -> FakeAsrStream:
        return self._stream_factory()

    def close(self) -> None:
        pass


class FirstFailsStream(FakeAsrStream):
    def finish(self) -> AsrUpdate:
        raise AsrUnavailable("simulated ASR failure")


class FirstFailsClient(FakeAsrClient):
    """Fail the first stream after a partial, then behave normally."""

    def __init__(self) -> None:
        self._opened = 0

    def open(self, *, language: str | None = None) -> FakeAsrStream:
        self._opened += 1
        return FirstFailsStream() if self._opened == 1 else FakeAsrStream()


class FakeTranslator:
    def translate(
        self,
        texts: list[str],
        *,
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        return [f"[{target_lang}] {text}" for text in texts]

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        pass


def _config(tmp_path: Path) -> AppConfig:
    config = load_config().model_copy(update={"root_dir": tmp_path})
    return config.model_copy(
        update={
            "screen": config.screen.model_copy(update={"enabled": False}),
            "ocr": config.ocr.model_copy(update={"enabled": False}),
            "summary": config.summary.model_copy(update={"enabled": False}),
            "mt": config.mt.model_copy(update={"translate_partials": False}),
        }
    )


def _ports(
    asr_client: FakeAsrClient, translator: Any | None = None
) -> PipelinePorts:
    script = [np.full(FRAME, 0.4, dtype=np.float32)] * 10 + [
        np.zeros(FRAME, dtype=np.float32)
    ] * 10
    segmenter = FakeSegmenter()
    return PipelinePorts(
        audio_source=FakeAudioSource(script),
        asr_client=asr_client,
        translator=translator or FakeTranslator(),
        segmenter_factory=lambda: segmenter,
    )


def _recv(ws: Any, timeout: float = 5.0) -> dict[str, Any]:
    """Receive one message with a timeout (TestClient has no receive timeout)."""

    box: queue.Queue[tuple[str, Any]] = queue.Queue()

    def read() -> None:
        try:
            box.put(("ok", ws.receive_text()))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the test thread
            box.put(("error", exc))

    threading.Thread(target=read, daemon=True).start()
    kind, value = box.get(timeout=timeout)
    if kind == "error":
        raise value
    return json.loads(value)


def _wait_for(
    ws: Any,
    predicate: Any,
    timeout: float = 10.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout
    seen: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        message = _recv(ws, timeout=max(0.1, deadline - time.monotonic()))
        seen.append(message)
        if predicate(message):
            return message, seen
    raise AssertionError(f"timed out; saw {[message.get('type') for message in seen]}")


def test_session_pipeline_persists_artifacts(tmp_path: Path) -> None:
    captured: dict[str, str | None] = {}

    def factory(config: AppConfig) -> PipelinePorts:
        captured["input_device"] = config.audio.input_device
        return _ports(FakeAsrClient())

    app = create_app(_config(tmp_path), ports_factory=factory)

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello = _recv(ws)
        assert hello["type"] == "session.state"
        assert hello["state"] == "idle"

        ws.send_text(
            json.dumps(
                {
                    "type": "session.start",
                    "source_lang": "en",
                    "target_lang": "id",
                    "audio_device": "Fake Mic",
                }
            )
        )
        partial, _ = _wait_for(
            ws,
            lambda message: message["type"] == "transcript.segment"
            and message["status"] == "partial"
            and bool(message.get("text")),
        )
        translated, _ = _wait_for(
            ws,
            lambda message: message["type"] == "transcript.segment"
            and message["status"] == "final"
            and message.get("translation_status") == "ok",
        )
        assert partial["segment_id"] == translated["segment_id"]
        assert partial["translation_status"] == "skipped"
        assert translated["text"] == "hello world"
        assert translated["translation"] == "[id] hello world"
        assert translated["audio_end_ts_ms"] is not None
        assert translated["mt_latency_ms"] is not None
        assert captured["input_device"] == "Fake Mic"

        ws.send_text(
            json.dumps(
                {
                    "type": "render.ack",
                    "event_id": translated["event_id"],
                    "painted_ts_ms": translated["ts_ms"] + 150,
                }
            )
        )
        ws.send_text(json.dumps({"type": "session.stop"}))
        _wait_for(
            ws,
            lambda message: message["type"] == "session.state"
            and message["state"] == "stopped",
        )
        assert client.get("/api/health").json()["state"] == "stopped"

    session_dirs = list((tmp_path / "data" / "sessions").iterdir())
    assert len(session_dirs) == 1
    session_dir = session_dirs[0]

    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert metadata["state"] == "stopped"
    assert metadata["source_lang"] == "en"
    assert metadata["target_lang"] == "id"
    assert metadata["ended_at_ms"] is not None
    assert [entry["state"] for entry in metadata["state_log"]] == [
        "starting",
        "running",
        "stopping",
        "stopped",
    ]

    transcript = (session_dir / "transcript.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in transcript.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["segment_id"] == translated["segment_id"]
    assert rows[0]["status"] == "final"
    assert rows[0]["translation_status"] == "ok"
    assert rows[0]["text"] == "hello world"
    assert rows[0]["translation"] == "[id] hello world"
    assert (session_dir / "slides.jsonl").exists()

    with (session_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 1
    assert metric_rows[0]["kind"] == "transcript"
    assert metric_rows[0]["segment_id"] == str(translated["segment_id"])
    assert metric_rows[0]["painted_ts_ms"] == str(translated["ts_ms"] + 150)


def test_partial_source_preview_is_translated_in_place(tmp_path: Path) -> None:
    script = [np.full(FRAME, 0.4, dtype=np.float32)] * 80 + [
        np.zeros(FRAME, dtype=np.float32)
    ] * 10
    translator = FakeTranslator()

    def factory(_config: AppConfig) -> PipelinePorts:
        return PipelinePorts(
            audio_source=FakeAudioSource(script),
            asr_client=FakeAsrClient(),
            translator=translator,
            segmenter_factory=FakeSegmenter,
        )

    config = _config(tmp_path)
    config = config.model_copy(
        update={"mt": config.mt.model_copy(update={"translate_partials": True})}
    )
    app = create_app(config, ports_factory=factory)

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        assert _recv(ws)["state"] == "idle"
        ws.send_text(
            json.dumps(
                {"type": "session.start", "source_lang": "en", "target_lang": "id"}
            )
        )
        translated, _ = _wait_for(
            ws,
            lambda message: message["type"] == "transcript.segment"
            and message["status"] == "partial"
            and message.get("translation_status") == "ok",
            timeout=10.0,
        )
        assert translated["text"] == "hello wor"
        assert translated["translation"] == "[id] hello wor"
        ws.send_text(json.dumps({"type": "session.stop"}))
        _wait_for(
            ws,
            lambda message: message["type"] == "session.state"
            and message["state"] == "stopped",
        )


def test_stale_preview_translation_is_not_painted_after_next_segment(
    tmp_path: Path,
) -> None:
    loud = np.full(FRAME, 0.4, dtype=np.float32)
    quiet = np.zeros(FRAME, dtype=np.float32)
    script = [loud] * 160 + [quiet] * 8 + [loud] * 80 + [quiet] * 10
    translator = BlockingTranslator()

    def factory(_config: AppConfig) -> PipelinePorts:
        return PipelinePorts(
            audio_source=FakeAudioSource(script),
            asr_client=FirstFailsClient(),
            translator=translator,
            segmenter_factory=FakeSegmenter,
        )

    config = _config(tmp_path)
    config = config.model_copy(
        update={"mt": config.mt.model_copy(update={"translate_partials": True})}
    )
    app = create_app(config, ports_factory=factory)

    seen: list[dict[str, Any]] = []
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        assert _recv(ws)["state"] == "idle"
        ws.send_text(
            json.dumps(
                {"type": "session.start", "source_lang": "en", "target_lang": "id"}
            )
        )
        assert translator.started.wait(10.0), "the preview translation never started"

        second_partial = None
        failed = False
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            message = _recv(ws)
            seen.append(message)
            if message["type"] == "notice" and message["code"] == "asr_unavailable":
                failed = True
            if (
                message["type"] == "transcript.segment"
                and message["status"] == "partial"
                and message["segment_id"] != 1
            ):
                second_partial = message
                break
        assert failed, "the first ASR stream did not fail"
        assert second_partial is not None, "the second segment never started"

        translator.release.set()
        final, after = _wait_for(
            ws,
            lambda message: message["type"] == "transcript.segment"
            and message["status"] == "final"
            and message.get("translation_status") == "ok"
            and message["segment_id"] == second_partial["segment_id"],
            timeout=10.0,
        )
        seen.extend(after)
        stale = [
            message
            for message in seen
            if message["type"] == "transcript.segment"
            and message.get("translation_status") == "ok"
            and message["segment_id"] == 1
        ]
        assert stale == [], "the old preview translation repainted segment 1"
        assert final["segment_id"] == second_partial["segment_id"]
        assert translator.calls >= 2

        ws.send_text(json.dumps({"type": "session.stop"}))
        _wait_for(
            ws,
            lambda message: message["type"] == "session.state"
            and message["state"] == "stopped",
        )


def test_asr_failure_keeps_session_running(tmp_path: Path) -> None:
    app = create_app(
        _config(tmp_path),
        ports_factory=lambda _: _ports(FakeAsrClient(FailingAsrStream)),
    )

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        assert _recv(ws)["state"] == "idle"
        ws.send_text(
            json.dumps(
                {"type": "session.start", "source_lang": "en", "target_lang": "id"}
            )
        )
        notice, _ = _wait_for(
            ws,
            lambda message: message["type"] == "notice"
            and message["code"] == "asr_unavailable",
        )
        assert notice["level"] == "warning"

        deadline = time.monotonic() + 5.0
        state = None
        while time.monotonic() < deadline:
            state = client.get("/api/health").json()["state"]
            if state == "running":
                break
            time.sleep(0.05)
        assert state == "running"

        ws.send_text(json.dumps({"type": "session.stop"}))
        _wait_for(
            ws,
            lambda message: message["type"] == "session.state"
            and message["state"] == "stopped",
        )


class SlowTranslator(FakeTranslator):
    """First call outlives the 3 s stop budget; later calls must never run."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0

    def translate(
        self,
        texts: list[str],
        *,
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        self.calls += 1
        if self.calls == 1:
            time.sleep(self.delay_s)
        return super().translate(
            texts, source_lang=source_lang, target_lang=target_lang
        )


class BlockingTranslator(FakeTranslator):
    """Hold the first translation until the test releases it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def translate(
        self,
        texts: list[str],
        *,
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            assert self.release.wait(10.0)
        return super().translate(
            texts, source_lang=source_lang, target_lang=target_lang
        )


def test_stop_does_not_touch_ports_held_by_a_slow_mt_thread(tmp_path: Path) -> None:
    """Regression: stop() used to null/close ports while MT was still translating.

    The MT thread then raised ``'NoneType' object has no attribute 'translator'``
    and the overlay received a spurious ``mt_unavailable`` notice after stop.
    """

    translator = SlowTranslator(4.0)
    loud = np.full(FRAME, 0.4, dtype=np.float32)
    quiet = np.zeros(FRAME, dtype=np.float32)
    script = ([loud] * 8 + [quiet] * 8) * 3

    def factory(_config: AppConfig) -> PipelinePorts:
        return PipelinePorts(
            audio_source=FakeAudioSource(script),
            asr_client=FakeAsrClient(),
            translator=translator,
            segmenter_factory=FakeSegmenter,
        )

    app = create_app(_config(tmp_path), ports_factory=factory)

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        assert _recv(ws)["state"] == "idle"
        ws.send_text(
            json.dumps(
                {"type": "session.start", "source_lang": "en", "target_lang": "id"}
            )
        )
        started, _ = _wait_for(
            ws,
            lambda message: message["type"] == "session.state"
            and message["state"] == "starting",
            timeout=10.0,
        )
        old_session_id = started["session_id"]
        _wait_for(
            ws,
            lambda message: message["type"] == "transcript.segment"
            and message["status"] == "final"
            and message.get("translation_status") == "pending",
            timeout=10.0,
        )
        time.sleep(0.2)  # let the MT thread enter the slow translation

        started = time.monotonic()
        ws.send_text(json.dumps({"type": "session.stop"}))
        _wait_for(
            ws,
            lambda message: message["type"] == "session.state"
            and message["state"] == "stopped",
            timeout=6.0,
        )
        assert time.monotonic() - started < 5.5

        # Start again immediately. start() must wait for the old MT worker, so
        # the late translation is still tagged with the old session id and can
        # never consume the new session's queue.
        ws.send_text(
            json.dumps(
                {"type": "session.start", "source_lang": "en", "target_lang": "id"}
            )
        )
        codes: list[str] = []
        new_session_id = None
        old_final_seen = False
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            message = _recv(ws, timeout=max(0.1, deadline - time.monotonic()))
            if message["type"] == "notice":
                codes.append(message["code"])
            elif message["type"] == "session.state":
                if message["state"] == "starting":
                    new_session_id = message["session_id"]
                elif message["state"] == "running":
                    break
            elif (
                message["type"] == "transcript.segment"
                and message["status"] == "final"
                and message.get("translation_status") == "ok"
                and message["session_id"] == old_session_id
            ):
                old_final_seen = True

        assert old_final_seen, "the in-flight translation was lost"
        assert new_session_id is not None and new_session_id != old_session_id
        assert "mt_unavailable" not in codes

        ws.send_text(json.dumps({"type": "session.stop"}))
        _wait_for(
            ws,
            lambda message: message["type"] == "session.state"
            and message["state"] == "stopped",
            timeout=6.0,
        )


class SlowWarmupTranslator(FakeTranslator):
    """Model load takes 2 s; capture startup must not wait for it."""

    def warmup(self) -> None:
        time.sleep(2.0)


def test_start_does_not_wait_for_translator_warmup(tmp_path: Path) -> None:
    def factory(_config: AppConfig) -> PipelinePorts:
        return _ports(FakeAsrClient(), translator=SlowWarmupTranslator())

    app = create_app(_config(tmp_path), ports_factory=factory)

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        assert _recv(ws)["state"] == "idle"
        started = time.monotonic()
        ws.send_text(
            json.dumps(
                {"type": "session.start", "source_lang": "en", "target_lang": "id"}
            )
        )
        _wait_for(
            ws,
            lambda message: message["type"] == "session.state"
            and message["state"] == "running",
            timeout=2.0,
        )
        assert time.monotonic() - started < 1.0

        ws.send_text(json.dumps({"type": "session.stop"}))
        _wait_for(
            ws,
            lambda message: message["type"] == "session.state"
            and message["state"] == "stopped",
            timeout=6.0,
        )
