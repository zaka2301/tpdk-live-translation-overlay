"""Overlay windows must stay out of the slide capture (phase 5).

The OCR loop grabs the region with ``mss``, which composites always-on-top
windows; without the display-affinity guard the translation overlay is read back
as slide text and every repaint looks like a slide change.
"""

from __future__ import annotations

import logging

import pytest
from PySide6.QtWidgets import QApplication

from tpdk_overlay.config import CaptionConfig, OcrOverlayConfig, UiConfig
from tpdk_overlay.contracts import BBox, OcrBlock, OcrFrameEvent, ScreenRegion
from tpdk_overlay.ui import capture_guard
from tpdk_overlay.ui.caption import CaptionWindow, RegionPicker
from tpdk_overlay.ui.capture_guard import exclude_from_capture
from tpdk_overlay.ui.ocr_overlay import OcrOverlayWindow


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    yield app


class _Handle:
    """Minimal widget stand-in: only ``winId`` and attribute storage are used."""

    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd

    def winId(self) -> int:  # noqa: N802 - Qt naming
        return self.hwnd


def _region() -> ScreenRegion:
    return ScreenRegion(left=0, top=0, width=1280, height=720)


def _frame() -> OcrFrameEvent:
    block = OcrBlock(
        block_id="b1",
        bbox=BBox(x=10, y=20, w=200, h=40),
        text="hello",
        confidence=0.9,
    )
    return OcrFrameEvent(
        session_id="s1",
        frame_id=1,
        region=_region(),
        t_capture_ms=1000,
        blocks=(block,),
    )


def test_every_show_reapplies_exclusion() -> None:
    calls: list[int] = []
    widget = _Handle(4242)

    def setter(hwnd: int) -> bool:
        calls.append(hwnd)
        return True

    assert exclude_from_capture(widget, setter=setter) is True
    assert exclude_from_capture(widget, setter=setter) is True
    # Idempotent: the affinity may be reset while the handle stays the same,
    # so a second show must re-assert it instead of trusting a cache.
    assert calls == [4242, 4242]


def test_changed_handle_is_also_excluded() -> None:
    calls: list[int] = []
    widget = _Handle(4242)

    def setter(hwnd: int) -> bool:
        calls.append(hwnd)
        return True

    exclude_from_capture(widget, setter=setter)
    widget.hwnd = 5150  # a re-created native window
    assert exclude_from_capture(widget, setter=setter) is True
    assert calls == [4242, 5150]


def test_missing_handle_is_skipped() -> None:
    calls: list[int] = []

    def setter(hwnd: int) -> bool:
        calls.append(hwnd)
        return True

    assert exclude_from_capture(_Handle(0), setter=setter) is False
    assert calls == []


def test_failure_is_retried_but_warned_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts: list[int] = []
    widget = _Handle(4242)

    def setter(hwnd: int) -> bool:
        attempts.append(hwnd)
        return len(attempts) > 2

    with caplog.at_level(logging.WARNING, logger=capture_guard.__name__):
        assert exclude_from_capture(widget, setter=setter) is False
        assert exclude_from_capture(widget, setter=setter) is False
        assert exclude_from_capture(widget, setter=setter) is True

    assert len(attempts) == 3
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert widget._capture_exclusion_warned is False


def test_off_windows_platform_is_left_untouched(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The suite runs under the offscreen platform; the real user32 call must not
    # be attempted with the fake handle the offscreen plug-in returns.
    def fail(_hwnd: int) -> bool:
        raise AssertionError("user32 must not be called off the windows platform")

    monkeypatch.setattr(capture_guard, "_windows_affinity", fail)
    widget = _Handle(1)
    assert exclude_from_capture(widget) is False
    assert not hasattr(widget, "_capture_exclusion_warned")


def test_windows_are_excluded_when_shown(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def spy(widget: object) -> bool:
        seen.append(type(widget).__name__)
        return True

    monkeypatch.setattr("tpdk_overlay.ui.caption.exclude_from_capture", spy)
    monkeypatch.setattr("tpdk_overlay.ui.ocr_overlay.exclude_from_capture", spy)

    caption = CaptionWindow(
        UiConfig(), source_lang="en", target_lang="id", checkpoint_interval_s=300
    )
    caption.show()
    qt_app.processEvents()
    caption.close()

    overlay = OcrOverlayWindow(OcrOverlayConfig(font_size=10))
    overlay.show_frame(_frame())
    qt_app.processEvents()
    overlay.close()

    picker = RegionPicker()
    picker.show()
    qt_app.processEvents()
    picker.close()

    assert seen == ["CaptionWindow", "OcrOverlayWindow", "RegionPicker"]


def test_caption_can_be_kept_in_recordings(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def spy(widget: object) -> bool:
        seen.append(type(widget).__name__)
        return True

    monkeypatch.setattr("tpdk_overlay.ui.caption.exclude_from_capture", spy)
    caption = CaptionWindow(
        UiConfig(caption=CaptionConfig(exclude_from_capture=False)),
        source_lang="en",
        target_lang="id",
        checkpoint_interval_s=300,
    )
    caption.show()
    qt_app.processEvents()
    caption.close()

    assert seen == []
