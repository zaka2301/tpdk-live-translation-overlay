"""Slide-change detection that ignores cursor/caret noise (FR-05, NFR-03).

Consecutive frames are reduced to a small grayscale thumbnail before they are
compared. Comparing sampled pixels (instead of a full-resolution diff) makes the
score robust against antialiasing noise and keeps a 1920x1080 frame at ~160x90
values per pass. A cursor or text caret changes one or two thumbnail pixels and
stays below any sane ``screen.change_threshold``.

The first frame after construction (or ``reset``) reports ``1.0`` so the OCR
loop always recognizes the initial slide.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tpdk_overlay.config import ScreenConfig
from tpdk_overlay.contracts import BBox
from tpdk_overlay.ports import ImageArray

# Longest thumbnail side. 160 keeps a 16:9 slide near 90 rows, which renders a
# cursor as at most a couple of pixels after the stride is applied.
_THUMBNAIL_LONG_SIDE = 160
# Per-pixel luma delta that counts as changed (out of 255). Lower values make
# compression/antialiasing noise count as a slide change.
_PIXEL_DELTA = 16


def _thumbnail(image: ImageArray) -> tuple[np.ndarray, int]:
    """Sampled luma thumbnail in ``int16`` (BGR input, physical pixels).

    Returns the thumbnail and its sampling stride so callers can map region
    boxes (physical pixels) onto thumbnail cells.
    """

    if image.ndim == 2:
        gray_source = image.astype(np.int32)
    elif image.ndim == 3 and image.shape[2] >= 3:
        # int32: 255 * 150 overflows int16 before the shift.
        bgr = image[:, :, :3].astype(np.int32)
        # ITU-R BT.601 luma with integer weights (B=.114, G=.587, R=.299).
        gray_source = (bgr[:, :, 2] * 77 + bgr[:, :, 1] * 150 + bgr[:, :, 0] * 29) >> 8
    else:
        raise ValueError(f"unexpected image shape: {image.shape!r}")

    height, width = gray_source.shape[:2]
    stride = max(1, int(np.ceil(max(height, width) / _THUMBNAIL_LONG_SIDE)))
    return gray_source[::stride, ::stride].astype(np.int16, copy=False), stride


class SlideChangeDetector:
    """Fraction of thumbnail pixels that changed since the previous frame."""

    def __init__(self, config: ScreenConfig) -> None:
        self._config = config
        self._previous: np.ndarray | None = None

    def score(self, image: ImageArray, *, mask: Sequence[BBox] = ()) -> float:
        """Return ``0.0..1.0``; ``1.0`` when there is no comparable previous frame.

        ``mask`` lists region-relative boxes to ignore (the overlay panels the
        OCR loop painted itself). They are excluded from both the changed count
        and the denominator, so an overlay repaint never looks like a slide
        change while a real change elsewhere still trips the threshold.
        """

        current, stride = _thumbnail(image)
        previous = self._previous
        self._previous = current
        if previous is None or previous.shape != current.shape:
            return 1.0
        changed = np.abs(current - previous) > _PIXEL_DELTA
        if not mask:
            return float(np.count_nonzero(changed)) / changed.size

        keep = np.ones(current.shape, dtype=bool)
        height, width = keep.shape
        for bbox in mask:
            x0 = max(0, bbox.x // stride - 1)
            y0 = max(0, bbox.y // stride - 1)
            x1 = min(width, (bbox.x + bbox.w) // stride + 2)
            y1 = min(height, (bbox.y + bbox.h) // stride + 2)
            keep[y0:y1, x0:x1] = False
        total = int(keep.sum())
        if total == 0:
            return 0.0
        return float(np.count_nonzero(changed & keep)) / total

    def remember(self, image: ImageArray) -> None:
        """Make ``image`` the reference for the next comparison.

        The recording-mode OCR loop uses this after a clean grab: the frame
        before it still showed the previous overlay, which must not be diffed
        against the new slide.
        """

        self._previous, _ = _thumbnail(image)

    def reset(self) -> None:
        """Forget the previous frame (the next capture counts as a change)."""

        self._previous = None


__all__ = ["SlideChangeDetector"]
