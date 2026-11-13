"""Overlay windows must stay out of the slide capture (phase 5).

The OCR loop grabs the region with ``mss``, which composites always-on-top
windows; without the display-affinity guard the translation overlay is read back
as slide text and every repaint looks like a slide change.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from tpdk_overlay.config import CaptionConfig, OcrOverlayConfig, UiConfig
from tpdk_overlay.contracts import (
    BBox,
    OcrBlock,
    OcrFrameEvent,
    OverlayVisibilityEvent,
    RenderAckCommand,
    ScreenRegion,
)
from tpdk_overlay.pipeline.orchestrator import Orchestrator
from tpdk_overlay.ui import capture_guard
from tpdk_overlay.ui.app import OverlayApp
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


class _VisibilityHost:
    """Binds the app's capture-visibility handler to a real overlay window."""

    _on_overlay_visibility = OverlayApp._on_overlay_visibility

    def __init__(self, ocr: OcrOverlayWindow | None) -> None:
        self._ocr = ocr
        self._ocr_capture_hidden = False
        self.sent: list[object] = []

    def _send(self, command: object) -> None:
        self.sent.append(command)


def test_capture_hide_acks_then_restores_the_overlay(qt_app: QApplication) -> None:
    overlay = OcrOverlayWindow(OcrOverlayConfig(font_size=10), capture_excluded=False)
    overlay.show_frame(_frame())
    qt_app.processEvents()
    assert overlay.isVisible()
    host = _VisibilityHost(overlay)

    hide = OverlayVisibilityEvent(target="ocr", visible=False)
    host._on_overlay_visibility(hide)
    assert not overlay.isVisible()
    assert len(host.sent) == 1
    ack = host.sent[0]
    assert isinstance(ack, RenderAckCommand)
    assert ack.event_id == hide.event_id

    host._on_overlay_visibility(OverlayVisibilityEvent(target="ocr", visible=True))
    assert overlay.isVisible()
    overlay.close()


def test_capture_show_keeps_a_user_hidden_overlay_hidden(
    qt_app: QApplication,
) -> None:
    overlay = OcrOverlayWindow(OcrOverlayConfig(font_size=10), capture_excluded=False)
    overlay.show_frame(_frame())
    overlay.set_overlay_visible(False)
    host = _VisibilityHost(overlay)

    host._on_overlay_visibility(
        OverlayVisibilityEvent(target="ocr", visible=False)
    )
    host._on_overlay_visibility(OverlayVisibilityEvent(target="ocr", visible=True))

    assert not overlay.isVisible()
    overlay.close()


def test_recording_mode_leaves_the_overlay_capturable(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        "tpdk_overlay.ui.ocr_overlay.exclude_from_capture",
        lambda widget: seen.append(type(widget).__name__) or True,
    )
    overlay = OcrOverlayWindow(OcrOverlayConfig(font_size=10), capture_excluded=False)
    overlay.show_frame(_frame())
    qt_app.processEvents()
    overlay.close()

    assert seen == []


def _orchestrator_stub() -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._capture_ack_lock = threading.Lock()
    orchestrator._capture_ack_waiters = {}
    orchestrator._metrics = None
    orchestrator._stop_event = threading.Event()
    return orchestrator


def test_capture_ack_wakes_the_ocr_thread() -> None:
    orchestrator = _orchestrator_stub()
    event = OverlayVisibilityEvent(target="ocr", visible=False)
    published: list[OverlayVisibilityEvent] = []
    orchestrator._publish = published.append

    def ack() -> None:
        while not published:
            time.sleep(0.001)
        orchestrator.record_render_ack(event.event_id, 123)

    thread = threading.Thread(target=ack)
    thread.start()
    try:
        assert orchestrator._await_render_ack(event, 1.0) is True
    finally:
        thread.join()


def test_capture_ack_times_out_without_a_client() -> None:
    orchestrator = _orchestrator_stub()
    orchestrator._publish = lambda event: None
    event = OverlayVisibilityEvent(target="ocr", visible=False)

    assert orchestrator._await_render_ack(event, 0.05) is False


def test_hide_request_is_acked_without_an_ocr_window() -> None:
    host = _VisibilityHost(None)
    hide = OverlayVisibilityEvent(target="ocr", visible=False)

    host._on_overlay_visibility(hide)

    assert len(host.sent) == 1
    assert host.sent[0].event_id == hide.event_id


def test_capture_ack_aborts_on_stop() -> None:
    orchestrator = _orchestrator_stub()
    orchestrator._publish = lambda event: None
    orchestrator._stop_event.set()
    event = OverlayVisibilityEvent(target="ocr", visible=False)

    started = time.monotonic()
    assert orchestrator._await_render_ack(event, 5.0) is False

    assert time.monotonic() - started < 0.5
