"""Screen source: BGRA->BGR conversion and thread-local capture reuse."""

from __future__ import annotations

import numpy as np

from tpdk_overlay.config import ScreenConfig
from tpdk_overlay.contracts import ScreenRegion
from tpdk_overlay.screen.capture import MssScreenSource


class _FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.regions: list[dict[str, int]] = []
        self.closed = False

    def grab(self, region: dict[str, int]) -> np.ndarray:
        self.regions.append(dict(region))
        return self.frame

    def close(self) -> None:
        self.closed = True


def test_capture_returns_contiguous_bgr_and_passes_region() -> None:
    frame = np.zeros((90, 160, 4), dtype=np.uint8)
    frame[:, :, 0] = 10  # B
    frame[:, :, 1] = 20  # G
    frame[:, :, 2] = 30  # R
    frame[:, :, 3] = 255  # A
    capture = _FakeCapture(frame)
    source = MssScreenSource(ScreenConfig(), factory=lambda: capture)
    region = ScreenRegion(left=-100, top=20, width=160, height=90, monitor=1)

    image = source.capture(region)

    assert image.shape == (90, 160, 3)
    assert image.flags["C_CONTIGUOUS"]
    assert int(image[0, 0, 0]) == 10
    assert int(image[0, 0, 1]) == 20
    assert int(image[0, 0, 2]) == 30
    assert capture.regions == [
        {"left": -100, "top": 20, "width": 160, "height": 90}
    ]


def test_capture_context_is_reused_and_closed() -> None:
    capture = _FakeCapture(np.zeros((10, 10, 4), dtype=np.uint8))
    created = 0

    def factory() -> _FakeCapture:
        nonlocal created
        created += 1
        return capture

    source = MssScreenSource(ScreenConfig(), factory=factory)
    region = ScreenRegion(left=0, top=0, width=10, height=10, monitor=1)
    source.capture(region)
    source.capture(region)
    source.close()

    assert created == 1
    assert capture.closed is True
    source.close()  # idempotent
