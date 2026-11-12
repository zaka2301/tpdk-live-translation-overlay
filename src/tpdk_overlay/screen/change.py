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

import numpy as np

from tpdk_overlay.config import ScreenConfig
from tpdk_overlay.ports import ImageArray

# Longest thumbnail side. 160 keeps a 16:9 slide near 90 rows, which renders a
# cursor as at most a couple of pixels after the stride is applied.
_THUMBNAIL_LONG_SIDE = 160
# Per-pixel luma delta that counts as changed (out of 255). Lower values make
# compression/antialiasing noise count as a slide change.
_PIXEL_DELTA = 16


def _thumbnail(image: ImageArray) -> np.ndarray:
    """Sampled luma thumbnail in ``int16`` (BGR input, physical pixels)."""

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
    return gray_source[::stride, ::stride].astype(np.int16, copy=False)


class SlideChangeDetector:
    """Fraction of thumbnail pixels that changed since the previous frame."""

    def __init__(self, config: ScreenConfig) -> None:
        self._config = config
        self._previous: np.ndarray | None = None

    def score(self, image: ImageArray) -> float:
        """Return ``0.0..1.0``; ``1.0`` when there is no comparable previous frame."""

        current = _thumbnail(image)
        previous = self._previous
        self._previous = current
        if previous is None or previous.shape != current.shape:
            return 1.0
        delta = np.abs(current - previous)
        changed = int(np.count_nonzero(delta > _PIXEL_DELTA))
        return changed / float(delta.size)

    def reset(self) -> None:
        """Forget the previous frame (the next capture counts as a change)."""

        self._previous = None


__all__ = ["SlideChangeDetector"]
