"""OCR overlay geometry, block hit-testing, and visibility toggles (phase 5)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
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
from tpdk_overlay.ui.ocr_overlay import (
    OcrOverlayWindow,
    block_right_limits,
    layout_block,
    region_logical_geometry,
    stabilize_line_heights,
)


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
    window = OcrOverlayWindow(OcrOverlayConfig())
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
    window = OcrOverlayWindow(OcrOverlayConfig())
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
    window = OcrOverlayWindow(OcrOverlayConfig())
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
    window = OcrOverlayWindow(OcrOverlayConfig(), click_through=True)
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
    window = OcrOverlayWindow(OcrOverlayConfig())
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
    window = OcrOverlayWindow(OcrOverlayConfig())
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
        OcrOverlayConfig(background="#101418", opacity=0.5)
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


def _layout(
    *,
    translation: str = "terjemahan",
    width: float = 400,
    height: float = 33,
    line_count: int = 1,
    window_height: float = 360,
    right_limit: float | None = None,
    config: OcrOverlayConfig | None = None,
):
    block = OcrBlock(
        block_id="layout",
        bbox=BBox(x=10, y=10, w=int(width), h=int(height)),
        text="source text",
        translation=translation,
        translation_status=TranslationStatus.OK,
        confidence=0.9,
        line_count=line_count,
    )
    return layout_block(
        block,
        QRectF(10, 10, width, height),
        config=config or OcrOverlayConfig(),
        window_height=window_height,
        window_width=1280,
        right_limit=right_limit,
    )


def test_block_font_matches_the_source_line_height(qt_app: QApplication) -> None:
    layout = _layout()

    # 33 px source polygon / 1.10 = a 30 px em, the size the slide used.
    assert layout.font.pixelSize() == 30
    assert layout.panel.height() >= 33
    # A translation that fits the source width keeps the source panel width.
    assert layout.panel.width() == 400


def test_multiline_block_uses_the_per_line_height(qt_app: QApplication) -> None:
    single = _layout()
    merged = _layout(height=66, line_count=2)

    assert merged.font.pixelSize() == single.font.pixelSize()


def test_long_translation_grows_the_panel_not_the_font(
    qt_app: QApplication,
) -> None:
    short = _layout()
    long = _layout(translation="terjemahan yang jauh lebih panjang " * 4)

    assert long.font.pixelSize() == short.font.pixelSize()
    assert long.panel.height() > short.panel.height()


def test_long_translation_shrinks_only_to_stay_inside_the_window(
    qt_app: QApplication,
) -> None:
    layout = _layout(
        translation="terjemahan yang jauh lebih panjang " * 8,
        width=200,
        window_height=120,
    )

    assert layout.font.pixelSize() < 30
    assert layout.panel.height() <= 120
    assert layout.panel.top() >= 0.0
    assert layout.panel.bottom() <= 120


def test_translation_widens_before_growing_taller(
    qt_app: QApplication,
) -> None:
    layout = _layout(
        translation="Regularisasi mengurangi overfitting secara signifikan",
        width=300,
    )

    assert layout.panel.width() > 300
    # One line at the matched size: the panel widened instead of stacking.
    assert layout.panel.height() < 2 * 40


def test_line_heights_are_stabilized_per_size_cluster(
    qt_app: QApplication,
) -> None:
    heights = stabilize_line_heights([54.0, 36.0, 34.0, 29.0])

    # The title keeps its own size; the body lines share one median even though
    # PaddleOCR reported different box heights for them.
    assert heights[0] == 54.0
    assert heights[1] == heights[2] == heights[3] == 34.0


def test_columns_bound_each_others_panel_width(
    qt_app: QApplication,
) -> None:
    left = QRectF(10, 10, 200, 36)
    right = QRectF(420, 10, 200, 36)

    limits = block_right_limits([left, right], window_width=1280)

    assert limits[0] == 420 - 4
    assert limits[1] == 1280


def test_panel_stops_at_the_next_column(qt_app: QApplication) -> None:
    layout = _layout(
        translation="Regularisasi mengurangi overfitting secara signifikan",
        width=200,
        right_limit=420,
    )

    assert layout.panel.width() > 200
    assert layout.panel.right() <= 420
