"""OCR overlay geometry, block hit-testing, and visibility toggles (phase 5)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QApplication

from tpdk_overlay.config import OcrOverlayConfig
from tpdk_overlay.contracts import (
    BBox,
    OcrBlock,
    OcrFrameEvent,
    ScreenRegion,
    TranslationStatus,
)
from tpdk_overlay.ui.ocr_overlay import OcrOverlayWindow, region_logical_geometry


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    yield app


def _block(block_id: str, x: int, y: int, w: int, h: int) -> OcrBlock:
    return OcrBlock(
        block_id=block_id,
        bbox=BBox(x=x, y=y, w=w, h=h),
        text=f"source {block_id}",
        translation=f"translated {block_id}",
        translation_status=TranslationStatus.OK,
        confidence=0.9,
    )


def _event(blocks: tuple[OcrBlock, ...]) -> OcrFrameEvent:
    return OcrFrameEvent(
        session_id="session",
        frame_id=1,
        region=ScreenRegion(left=0, top=0, width=640, height=360, monitor=1),
        t_capture_ms=0,
        blocks=blocks,
        ocr_latency_ms=40,
        mt_latency_ms=30,
        change_ts_ms=100,
    )


def test_region_maps_to_screen_logical_geometry(qt_app: QApplication) -> None:
    screen = QGuiApplication.primaryScreen()
    assert screen is not None
    dpr = screen.devicePixelRatio() or 1.0
    geometry = screen.geometry()
    region = ScreenRegion(
        left=round(40 * dpr),
        top=round(20 * dpr),
        width=round(320 * dpr),
        height=round(180 * dpr),
        monitor=1,
    )

    rect = region_logical_geometry(region)

    assert rect.width() == 320
    assert rect.height() == 180
    # The screen origin is subtracted through the physical origin; both moves
    # must cancel out for a region measured from the primary screen corner.
    assert rect.x() == geometry.x() + 40
    assert rect.y() == geometry.y() + 20


def test_show_frame_positions_window_and_hit_tests_blocks(qt_app: QApplication) -> None:
    first = _block("first", 10, 20, 100, 40)
    second = _block("second", 300, 200, 120, 40)
    window = OcrOverlayWindow(OcrOverlayConfig(font_size=10))
    try:
        window.show_frame(_event((first, second)))

        assert window.isVisible()
        assert window.width() == 640
        assert window.height() == 360
        assert window.block_at(QPoint(20, 30)) is not None
        assert window.block_at(QPoint(20, 30)).block_id == "first"
        assert window.block_at(QPoint(320, 210)).block_id == "second"
        assert window.block_at(QPoint(5, 5)) is None
        assert window.has_content() is True
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_block_at_scales_with_widget_size(qt_app: QApplication) -> None:
    window = OcrOverlayWindow(OcrOverlayConfig(font_size=10))
    try:
        window.show_frame(_event((_block("scaled", 100, 100, 100, 100),)))
        # Half the region's logical size: widget (5,5) is physical (10,10).
        window.resize(320, 180)

        assert window.block_at(QPoint(52, 52)).block_id == "scaled"
        assert window.block_at(QPoint(48, 48)) is None
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_block_toggle_and_clear(qt_app: QApplication) -> None:
    first = _block("first", 10, 20, 100, 40)
    second = _block("second", 300, 200, 120, 40)
    window = OcrOverlayWindow(OcrOverlayConfig(font_size=10))
    try:
        window.show_frame(_event((first, second)))
        assert window.toggle_block("first") is False
        assert window.is_block_visible("first") is False
        assert window.has_content() is True
        assert window.toggle_block("first") is True
        assert window.is_block_visible("first") is True

        # A re-captured frame keeps a hidden id only while it still exists.
        window.toggle_block("first")
        window.show_frame(_event((second,)))
        assert window.is_block_visible("first") is True

        window.clear()
        assert window.isVisible() is False
        assert window.has_content() is False
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_click_through_gates_block_toggling(qt_app: QApplication) -> None:
    block = _block("first", 10, 20, 100, 40)
    window = OcrOverlayWindow(OcrOverlayConfig(font_size=10), click_through=True)
    try:
        window.show_frame(_event((block,)))
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(20, 30),
            QPointF(20, 30),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

        window.mousePressEvent(press)
        assert window.is_block_visible("first") is True

        window.set_click_through(False)
        window.mousePressEvent(press)
        assert window.is_block_visible("first") is False
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_empty_frame_clears_previous_boxes(qt_app: QApplication) -> None:
    window = OcrOverlayWindow(OcrOverlayConfig(font_size=10))
    try:
        window.show_frame(_event((_block("first", 10, 20, 100, 40),)))
        assert window.isVisible() is True
        assert window.has_content() is True

        window.show_frame(
            OcrFrameEvent(
                session_id="session",
                frame_id=2,
                region=ScreenRegion(left=0, top=0, width=640, height=360, monitor=1),
                t_capture_ms=1000,
                blocks=(),
                ocr_latency_ms=40,
                change_ts_ms=1000,
            )
        )

        assert window.isVisible() is False
        assert window.has_content() is False
        assert window.blocks() == ()
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_global_visibility_survives_frames(qt_app: QApplication) -> None:
    window = OcrOverlayWindow(OcrOverlayConfig(font_size=10))
    try:
        window.show_frame(_event((_block("first", 10, 20, 100, 40),)))
        window.set_overlay_visible(False)
        assert window.isVisible() is False

        window.show_frame(_event((_block("second", 10, 20, 100, 40),)))
        assert window.isVisible() is False

        window.set_overlay_visible(True)
        assert window.isVisible() is True
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_painting_produces_a_pixmap(qt_app: QApplication) -> None:
    window = OcrOverlayWindow(
        OcrOverlayConfig(background="#101418", opacity=0.5, font_size=10)
    )
    try:
        window.show_frame(_event((_block("first", 10, 20, 100, 40),)))
        window.paint_now()
        pixmap = window.grab()
        assert pixmap.isNull() is False
        assert pixmap.width() == window.width()
        assert pixmap.height() == window.height()
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()
