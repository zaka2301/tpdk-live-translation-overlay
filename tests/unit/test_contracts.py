from __future__ import annotations

import pytest
from pydantic import ValidationError

from tpdk_overlay.contracts import (
    BBox,
    NoticeEvent,
    OcrBlock,
    OcrFrameEvent,
    OverlayVisibilityEvent,
    PingCommand,
    PongEvent,
    ScreenRegion,
    TranscriptSegmentEvent,
    parse_client_command,
    parse_server_event,
)


def test_transcript_event_roundtrip() -> None:
    event = TranscriptSegmentEvent(
        segment_id=3,
        revision=1,
        status="final",
        t_start_ms=1000,
        t_end_ms=2500,
        source_lang="en",
        target_lang="id",
        text="hello class",
        translation="halo kelas",
        translation_status="ok",
        asr_latency_ms=420,
    )
    parsed = parse_server_event(event.model_dump_json())
    assert parsed == event
    assert isinstance(parsed, TranscriptSegmentEvent)
    assert parsed.status.value == "final"


def test_transcript_time_order_is_validated() -> None:
    with pytest.raises(ValidationError):
        TranscriptSegmentEvent(
            segment_id=1,
            status="partial",
            t_start_ms=5000,
            t_end_ms=4000,
            source_lang="en",
            target_lang="id",
            text="x",
        )


def test_ocr_frame_roundtrip() -> None:
    region = ScreenRegion(left=-1920, top=0, width=1280, height=720, monitor=2)
    block = OcrBlock(
        block_id="b1",
        bbox=BBox(x=10, y=20, w=300, h=40),
        text="Photosynthesis",
        translation="Fotosintesis",
        translation_status="ok",
        confidence=0.93,
    )
    event = OcrFrameEvent(frame_id=7, region=region, t_capture_ms=12_000, blocks=(block,))
    parsed = parse_server_event(event.model_dump_json())
    assert parsed == event
    assert parsed.region.left == -1920
    assert parsed.blocks[0].bbox.w == 300


def test_overlay_visibility_roundtrip() -> None:
    event = OverlayVisibilityEvent(target="ocr", visible=False)

    parsed = parse_server_event(event.model_dump_json())

    assert isinstance(parsed, OverlayVisibilityEvent)
    assert parsed.target == "ocr"
    assert parsed.visible is False
    assert parsed.reason == "capture"


def test_bbox_rejects_zero_size() -> None:
    with pytest.raises(ValidationError):
        BBox(x=0, y=0, w=0, h=10)


def test_unknown_event_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_server_event('{"type": "does.not.exist"}')


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_server_event('{"type": "pong", "echo_ts_ms": 1, "surprise": true}')


def test_pong_defaults_and_command_parsing() -> None:
    pong = parse_server_event('{"type": "pong", "echo_ts_ms": 123}')
    assert isinstance(pong, PongEvent)
    assert pong.ts_ms > 0

    command = parse_client_command(
        '{"type": "session.start", "source_lang": "en", "target_lang": "id"}'
    )
    assert command.type == "session.start"
    assert command.source_lang == "en"

    ping = parse_client_command('{"type": "ping"}')
    assert isinstance(ping, PingCommand)


def test_notice_level_enum() -> None:
    notice = NoticeEvent(level="warning", code="asr_unavailable", message="service down")
    assert parse_server_event(notice.model_dump_json()).level.value == "warning"
