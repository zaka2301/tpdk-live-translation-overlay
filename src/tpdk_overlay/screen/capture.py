"""``mss``-backed screen captures for the slide OCR loop (FR-05).

The capture adapter is deliberately thin: ``mss`` already returns physical
pixels on the virtual desktop, which is exactly the coordinate space of
:class:`~tpdk_overlay.contracts.ScreenRegion`. The only conversion here is
BGRA -> BGR (OpenCV/numpy order) and a contiguous array so PaddleOCR can read
the buffer without a copy per channel.

``mss`` instances are not thread-safe, so each thread gets its own capture
context. The OCR loop is a single thread, but tests and future callers may move
the adapter; a thread-local context makes that safe without a global lock.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from tpdk_overlay.config import ScreenConfig
from tpdk_overlay.contracts import ScreenRegion
from tpdk_overlay.ports import ImageArray

CaptureFactory = Callable[[], Any]


def _default_capture() -> Any:
    # Imported lazily so the backend starts without a display adapter when OCR
    # is disabled, and so tests can inject a fake factory.
    import mss

    factory = getattr(mss, "MSS", mss.mss)
    return factory()


class MssScreenSource:
    """Capture rectangular regions from the Windows desktop via ``mss``."""

    def __init__(
        self, config: ScreenConfig, *, factory: CaptureFactory | None = None
    ) -> None:
        self._config = config
        self._factory = factory or _default_capture
        self._local = threading.local()

    def capture(self, region: ScreenRegion) -> ImageArray:
        """Grab ``region`` and return a contiguous BGR ``uint8`` array."""

        capture = getattr(self._local, "capture", None)
        if capture is None:
            capture = self._factory()
            self._local.capture = capture
        shot = capture.grab(
            {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            }
        )
        frame = np.asarray(shot)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"unexpected mss frame shape: {frame.shape!r}")
        # mss returns BGRA; PaddleOCR expects BGR. A view would leave the
        # stride at 4 bytes per pixel, so copy into a compact buffer here.
        return np.ascontiguousarray(frame[:, :, :3])

    def close(self) -> None:
        """Release this thread's capture context; safe to call twice."""

        capture = getattr(self._local, "capture", None)
        if capture is None:
            return
        self._local.capture = None
        close = getattr(capture, "close", None)
        if callable(close):
            close()


__all__ = ["CaptureFactory", "MssScreenSource"]
