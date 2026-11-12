"""Latency rows for ``metrics.csv`` (NFR-01/NFR-03, phase 7).

Every resolved final transcript segment and OCR frame is noted when it is
emitted; when the UI returns a ``render.ack`` for that event, the painted
timestamp is joined back in and one row is appended. A source-only final
(``translation_status == pending``) is skipped: the orchestrator emits it just
before MT so the caption can appear early, and the resolved final that follows
is the user-facing paint that NFR-01 measures. That keeps one row per segment
instead of two. Partial revisions are not measured either.

The CSV columns match AGENT.md section 9; ``eval_latency.py`` computes
``painted_ts_ms - audio_end_ts_ms`` (or ``change_ts_ms``) for the p95.
"""

from __future__ import annotations

import csv
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from tpdk_overlay.contracts import (
    OcrFrameEvent,
    SegmentStatus,
    TranscriptSegmentEvent,
    TranslationStatus,
)

logger = logging.getLogger(__name__)

FIELDS = (
    "kind",
    "segment_id",
    "frame_id",
    "audio_end_ts_ms",
    "change_ts_ms",
    "emit_ts_ms",
    "painted_ts_ms",
    "asr_ms",
    "mt_ms",
    "ocr_ms",
)

_MAX_PENDING = 512


class MetricsRecorder:
    """Join emitted events with render acks and append ``metrics.csv`` rows."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._closed = False

    def note(self, event: TranscriptSegmentEvent | OcrFrameEvent) -> None:
        """Remember an emitted final event until its render ack arrives."""

        if isinstance(event, TranscriptSegmentEvent):
            if event.status is not SegmentStatus.FINAL:
                return
            if event.translation_status is TranslationStatus.PENDING:
                # Source preview paint; the resolved final carries the row.
                return
            row: dict[str, Any] = {
                "kind": "transcript",
                "segment_id": event.segment_id,
                "frame_id": "",
                "audio_end_ts_ms": event.audio_end_ts_ms,
                "change_ts_ms": "",
                "emit_ts_ms": event.ts_ms,
                "painted_ts_ms": "",
                "asr_ms": event.asr_latency_ms,
                "mt_ms": event.mt_latency_ms,
                "ocr_ms": "",
            }
        elif isinstance(event, OcrFrameEvent):
            row = {
                "kind": "ocr",
                "segment_id": "",
                "frame_id": event.frame_id,
                "audio_end_ts_ms": "",
                "change_ts_ms": event.change_ts_ms,
                "emit_ts_ms": event.ts_ms,
                "painted_ts_ms": "",
                "asr_ms": "",
                "mt_ms": event.mt_latency_ms,
                "ocr_ms": event.ocr_latency_ms,
            }
        else:
            return

        with self._lock:
            if self._closed:
                return
            self._pending[event.event_id] = row
            while len(self._pending) > _MAX_PENDING:
                self._pending.popitem(last=False)

    def ack(self, event_id: str, painted_ts_ms: int) -> None:
        """Write the measured row for ``event_id`` once the UI painted it."""

        with self._lock:
            if self._closed:
                return
            row = self._pending.pop(event_id, None)
            if row is None:
                return
            row["painted_ts_ms"] = painted_ts_ms
            self._append_locked(row)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending.clear()

    def _append_locked(self, row: dict[str, Any]) -> None:
        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        with self._path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {field: ("" if row.get(field) is None else row[field]) for field in FIELDS}
            )


__all__ = ["FIELDS", "MetricsRecorder"]
