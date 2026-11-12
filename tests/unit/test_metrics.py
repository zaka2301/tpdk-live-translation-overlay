"""Metrics rows: one resolved final per segment (NFR-01)."""

from __future__ import annotations

import csv

from tpdk_overlay.contracts import (
    SegmentStatus,
    TranscriptSegmentEvent,
    TranslationStatus,
)
from tpdk_overlay.pipeline.metrics import MetricsRecorder


def _final(
    *,
    revision: int,
    translation_status: TranslationStatus,
    translation: str | None,
) -> TranscriptSegmentEvent:
    return TranscriptSegmentEvent(
        segment_id=4,
        revision=revision,
        status=SegmentStatus.FINAL,
        t_start_ms=1_000,
        t_end_ms=2_000,
        source_lang="en",
        target_lang="id",
        text="The derivative is a limit",
        translation=translation,
        translation_status=translation_status,
        asr_latency_ms=120,
        mt_latency_ms=700,
        audio_end_ts_ms=10_000,
    )


def test_pending_source_final_does_not_add_a_second_row(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    recorder = MetricsRecorder(path)
    source = _final(
        revision=2,
        translation_status=TranslationStatus.PENDING,
        translation=None,
    )
    resolved = _final(
        revision=3,
        translation_status=TranslationStatus.OK,
        translation="Turunan adalah limit",
    )

    recorder.note(source)
    recorder.note(resolved)
    recorder.ack(source.event_id, 10_400)
    recorder.ack(resolved.event_id, 11_250)
    recorder.close()

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["kind"] == "transcript"
    assert rows[0]["segment_id"] == "4"
    assert rows[0]["painted_ts_ms"] == "11250"
    assert rows[0]["audio_end_ts_ms"] == "10000"
    assert rows[0]["mt_ms"] == "700"
