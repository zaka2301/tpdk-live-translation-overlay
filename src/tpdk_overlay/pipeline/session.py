"""Session lifecycle and on-disk artifacts (FR-02/03/08, NFR-06).

A :class:`SessionStore` owns one ``data/sessions/<session_id>/`` directory and
the append-only files the report and evaluation scripts read back:

``session.json``
    Config snapshot, languages, state log, start/end timestamps.
``transcript.jsonl``
    One final :class:`~tpdk_overlay.contracts.TranscriptSegmentEvent` per line.
``slides.jsonl``
    One :class:`~tpdk_overlay.contracts.OcrFrameEvent` per OCR pass.
``summary_checkpoint_NN.md`` / ``summary_final.md``
    Summary markdown.
``metrics.csv``
    Written by :mod:`tpdk_overlay.pipeline.metrics` from render acks.

Writes arrive from the event loop (state transitions) and worker threads
(transcript/slide finals); a lock serializes them and a closed store ignores
late appends after shutdown.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from tpdk_overlay.contracts import OcrFrameEvent, TranscriptSegmentEvent
from tpdk_overlay.utils.clock import monotonic_ms, now_ms


class SessionClock:
    """Session-relative monotonic clock; only differences are meaningful."""

    def __init__(self) -> None:
        self._start_ms = monotonic_ms()

    def elapsed_ms(self) -> int:
        return max(0, monotonic_ms() - self._start_ms)


def _new_session_id(started_at_ms: int) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(started_at_ms / 1000))
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


class SessionStore:
    """Persist one session's artifacts and state transitions."""

    def __init__(
        self,
        sessions_dir: Path,
        *,
        source_lang: str,
        target_lang: str,
        checkpoint_interval_s: int,
        config_snapshot: dict[str, Any],
        started_at_ms: int,
    ) -> None:
        self.session_id = _new_session_id(started_at_ms)
        self.dir = sessions_dir / self.session_id
        self.dir.mkdir(parents=True, exist_ok=False)

        self._transcript_path = self.dir / "transcript.jsonl"
        self._slides_path = self.dir / "slides.jsonl"
        self._session_path = self.dir / "session.json"

        self._lock = threading.Lock()
        self._closed = False
        self._state = "idle"
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._checkpoint_interval_s = checkpoint_interval_s
        self._config_snapshot = config_snapshot
        self._started_at_ms = started_at_ms
        self._ended_at_ms: int | None = None
        self._state_log: list[dict[str, Any]] = []

        # The layout promises these files exist even when a stage produced no
        # rows (transcript-only sessions still leave an empty slides.jsonl).
        self._transcript_path.touch()
        self._slides_path.touch()
        self._write_session()

    # -- state -------------------------------------------------------------

    def record_state(self, state: str, detail: str | None = None) -> None:
        """Append a lifecycle transition and rewrite ``session.json``."""

        with self._lock:
            if self._closed:
                return
            self._state = state
            self._state_log.append({"ts_ms": now_ms(), "state": state, "detail": detail})
            self._write_session_locked()

    def set_languages(self, source_lang: str, target_lang: str) -> None:
        """Persist a mid-session language change."""

        with self._lock:
            if self._closed:
                return
            self._source_lang = source_lang
            self._target_lang = target_lang
            self._write_session_locked()

    def finalize(self, state: str, *, ended_at_ms: int) -> None:
        """Close the store and record the final state/end timestamp."""

        with self._lock:
            if self._closed:
                return
            self._state = state
            self._ended_at_ms = ended_at_ms
            self._closed = True
            self._write_session_locked()

    # -- artifacts ---------------------------------------------------------

    def append_transcript(self, event: TranscriptSegmentEvent) -> None:
        """Append one authoritative final segment."""

        if event.status.value != "final":
            return
        line = event.model_dump_json()
        with self._lock:
            if self._closed:
                return
            with self._transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def append_slide(self, event: OcrFrameEvent) -> None:
        """Append one OCR frame with its blocks."""

        line = event.model_dump_json()
        with self._lock:
            if self._closed:
                return
            with self._slides_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def write_checkpoint(self, checkpoint_id: int, markdown: str) -> Path:
        path = self.dir / f"summary_checkpoint_{checkpoint_id:02d}.md"
        path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        return path

    def write_final(self, markdown: str) -> Path:
        path = self.dir / "summary_final.md"
        path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        return path

    # -- internals ---------------------------------------------------------

    def _write_session(self) -> None:
        with self._lock:
            self._write_session_locked()

    def _write_session_locked(self) -> None:
        payload = {
            "session_id": self.session_id,
            "state": self._state,
            "started_at_ms": self._started_at_ms,
            "ended_at_ms": self._ended_at_ms,
            "source_lang": self._source_lang,
            "target_lang": self._target_lang,
            "checkpoint_interval_s": self._checkpoint_interval_s,
            "config": self._config_snapshot,
            "state_log": self._state_log,
        }
        temporary = self._session_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self._session_path)


__all__ = ["SessionClock", "SessionStore"]
