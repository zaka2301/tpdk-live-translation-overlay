"""Merge OCR lines into reading-order blocks (FR-05, FR-06).

PaddleOCR returns one entry per text line. Slides often split a sentence or a
bullet across two or three lines, so translating each line separately produces
fragmented, context-free output. This module merges lines whose vertical gap and
horizontal overlap place them in the same visual paragraph, then sorts the
resulting blocks top-to-bottom and left-to-right.

Block ids hash the geometry (quantized to a small grid so a one-pixel detection
jitter does not change the id) plus the exact text. The UI uses the id to keep
per-block visibility across frames.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from tpdk_overlay.contracts import BBox, OcrBlock
from tpdk_overlay.ports import OcrLine

# Minimum share of the narrower line that must sit under the block. Lines that
# are merely stacked (two columns, same row) have no overlap and stay separate.
_MIN_OVERLAP_FRACTION = 0.3
# Geometry is rounded to this grid before hashing; detection polys jitter by a
# pixel or two between frames and the id must survive that.
_ID_GRID_PX = 4


def _union(boxes: Sequence[BBox]) -> BBox:
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.w for box in boxes)
    bottom = max(box.y + box.h for box in boxes)
    return BBox(x=left, y=top, w=max(1, right - left), h=max(1, bottom - top))


def block_id(bbox: BBox, text: str) -> str:
    """Stable 16-hex id for a block's quantized geometry and exact text."""

    def quantize(value: int) -> int:
        return int(round(value / _ID_GRID_PX)) * _ID_GRID_PX

    payload = (
        f"{quantize(bbox.x)}:{quantize(bbox.y)}:"
        f"{quantize(bbox.w)}:{quantize(bbox.h)}:{text}"
    ).encode()
    return hashlib.blake2s(payload, digest_size=8).hexdigest()


def _joins(block: BBox, line: BBox, gap_px: int) -> bool:
    """True when ``line`` continues the paragraph represented by ``block``."""

    vertical_gap = line.y - (block.y + block.h)
    if vertical_gap < -2 or vertical_gap > gap_px:
        return False
    overlap = min(block.x + block.w, line.x + line.w) - max(block.x, line.x)
    if overlap <= 0:
        return False
    return overlap / min(block.w, line.w) >= _MIN_OVERLAP_FRACTION


@dataclass(slots=True)
class _Builder:
    lines: list[OcrLine] = field(default_factory=list)
    bbox: BBox | None = None

    def add(self, line: OcrLine) -> None:
        self.lines.append(line)
        self.bbox = line.bbox if self.bbox is None else _union((self.bbox, line.bbox))

    def to_block(self) -> OcrBlock:
        assert self.bbox is not None
        text = " ".join(" ".join(line.text.split()) for line in self.lines)
        confidence = sum(line.confidence for line in self.lines) / len(self.lines)
        return OcrBlock(
            block_id=block_id(self.bbox, text),
            bbox=self.bbox,
            text=text,
            confidence=min(1.0, max(0.0, confidence)),
            line_count=len(self.lines),
        )


def group_lines(
    lines: Sequence[OcrLine],
    *,
    gap_px: int = 18,
    max_blocks: int = 80,
) -> list[OcrBlock]:
    """Group ``lines`` into paragraph blocks in reading order.

    ``gap_px`` is the largest vertical distance between line boxes that still
    counts as the same paragraph. ``max_blocks`` caps the result after sorting,
    keeping the first blocks a reader would see.
    """

    ordered = sorted(lines, key=lambda line: (line.bbox.y, line.bbox.x))
    builders: list[_Builder] = []
    for line in ordered:
        target: _Builder | None = None
        # Newest block first: the previous text row belongs to the most recent
        # paragraph in the same column.
        for builder in reversed(builders):
            assert builder.bbox is not None
            if _joins(builder.bbox, line.bbox, gap_px):
                target = builder
                break
        if target is None:
            target = _Builder()
            builders.append(target)
        target.add(line)

    blocks = [builder.to_block() for builder in builders]
    blocks.sort(key=lambda block: (block.bbox.y, block.bbox.x))
    return blocks[:max_blocks]


__all__ = ["block_id", "group_lines"]
