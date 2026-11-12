"""OCR line grouping: paragraph merge, reading order, ids (phase 5)."""

from __future__ import annotations

from tpdk_overlay.contracts import BBox
from tpdk_overlay.ocr.grouping import block_id, group_lines
from tpdk_overlay.ports import OcrLine


def _line(
    text: str, x: int, y: int, w: int, h: int = 20, confidence: float = 0.9
) -> OcrLine:
    return OcrLine(text=text, bbox=BBox(x=x, y=y, w=w, h=h), confidence=confidence)


def test_wrapped_lines_merge_and_columns_stay_separate() -> None:
    lines = [
        _line("A slide title", 10, 10, 300, 24),
        _line("The first line of body", 10, 50, 500),
        _line("continues here", 12, 74, 480),
        _line("Right column", 700, 50, 200),
    ]

    blocks = group_lines(lines, gap_px=10, max_blocks=80)

    assert [block.text for block in blocks] == [
        "A slide title",
        "The first line of body continues here",
        "Right column",
    ]
    assert blocks[1].line_count == 2
    assert blocks[1].bbox == BBox(x=10, y=50, w=500, h=44)
    assert blocks[1].confidence == 0.9


def test_boxes_on_the_same_row_do_not_merge_without_overlap() -> None:
    lines = [
        _line("Left column line", 10, 50, 200),
        _line("Right column line", 500, 50, 200),
    ]

    blocks = group_lines(lines, gap_px=50)

    assert len(blocks) == 2
    assert blocks[0].bbox.x == 10
    assert blocks[1].bbox.x == 500


def test_large_vertical_gap_starts_a_new_block() -> None:
    lines = [
        _line("Heading", 10, 10, 300, 30),
        _line("Body", 10, 100, 300),
    ]

    blocks = group_lines(lines, gap_px=18)

    assert [block.text for block in blocks] == ["Heading", "Body"]


def test_max_blocks_keeps_reading_order_prefix() -> None:
    lines = [_line(f"block {index}", 10, index * 60, 200) for index in range(5)]

    blocks = group_lines(lines, gap_px=10, max_blocks=3)

    assert [block.text for block in blocks] == ["block 0", "block 1", "block 2"]


def test_confidence_is_the_line_average_and_clamped() -> None:
    lines = [
        _line("one", 10, 10, 100, confidence=0.5),
        _line("two", 10, 30, 100, confidence=0.9),
    ]

    blocks = group_lines(lines, gap_px=18)

    assert len(blocks) == 1
    assert blocks[0].confidence == 0.7


def test_block_id_is_stable_and_text_sensitive() -> None:
    bbox = BBox(x=10, y=20, w=100, h=30)

    assert block_id(bbox, "same text") == block_id(bbox, "same text")
    assert block_id(bbox, "same text") != block_id(bbox, "other text")
    assert block_id(bbox, "same text") != block_id(
        BBox(x=200, y=20, w=100, h=30), "same text"
    )


def test_empty_input_yields_no_blocks() -> None:
    assert group_lines([], gap_px=18, max_blocks=10) == []
