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
        }
    )


def _ports(asr_client: FakeAsrClient) -> PipelinePorts:
    script = [np.full(FRAME, 0.4, dtype=np.float32)] * 10 + [
        np.zeros(FRAME, dtype=np.float32)
    ] * 10
    segmenter = FakeSegmenter()
    return PipelinePorts(
        audio_source=FakeAudioSource(script),
        asr_client=asr_client,
        translator=FakeTranslator(),
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
