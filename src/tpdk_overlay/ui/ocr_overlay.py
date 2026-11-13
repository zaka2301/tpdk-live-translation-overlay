"""Transparent window that draws translated slide blocks over the source (FR-06).

One frameless, always-on-top window covers exactly the captured region. Each
OCR block is painted at its original bounding box (physical pixels scaled to Qt
logical pixels), with the translation on a translucent panel. The window is
click-through with the caption by default; in interactive mode a click toggles
one block so the underlying slide stays readable.

Coordinates: ``ScreenRegion``/``BBox`` are physical pixels. Qt geometry is
logical pixels, so the window's geometry is derived from the matching screen's
``devicePixelRatio`` and block boxes are scaled by ``widget size / region size``
at paint time (which also absorbs integer rounding at fractional DPIs).
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from tpdk_overlay.config import OcrOverlayConfig
from tpdk_overlay.contracts import OcrBlock, OcrFrameEvent, ScreenRegion, TranslationStatus
from tpdk_overlay.ui import theme
from tpdk_overlay.ui.caption import _physical_origin
from tpdk_overlay.ui.capture_guard import exclude_from_capture

_PADDING_PX = 3
_MIN_FONT_PT = 6
_TEXT_FLAGS = (
    Qt.AlignmentFlag.AlignLeft
    | Qt.AlignmentFlag.AlignVCenter
    | Qt.TextFlag.TextWordWrap
)


def region_logical_geometry(region: ScreenRegion) -> QRect:
    """Map a physical ``ScreenRegion`` onto Qt's global logical coordinates."""

    screens = QGuiApplication.screens()
    screen = None
    if 1 <= region.monitor <= len(screens):
        screen = screens[region.monitor - 1]
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    if screen is None:
        return QRect(region.left, region.top, region.width, region.height)

    dpr = screen.devicePixelRatio() or 1.0
    geometry = screen.geometry()
    origin = _physical_origin(screen)
    if origin is None:
        origin = (round(geometry.x() * dpr), round(geometry.y() * dpr))
    x = geometry.x() + round((region.left - origin[0]) / dpr)
    y = geometry.y() + round((region.top - origin[1]) / dpr)
    width = max(1, round(region.width / dpr))
    height = max(1, round(region.height / dpr))
    return QRect(x, y, width, height)


class OcrOverlayWindow(QWidget):
    """Paint translated blocks over the captured slide region."""

    def __init__(
        self,
        config: OcrOverlayConfig,
        *,
        click_through: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self._config = config
        self._click_through = bool(click_through)
        self._visible = True
        self._frame: OcrFrameEvent | None = None
        self._region: ScreenRegion | None = None
        self._hidden: set[str] = set()

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setWindowTitle("TPdK slide translations")
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, self._click_through)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, self._click_through)
        self.hide()

    # -- state -------------------------------------------------------------

    @property
    def region(self) -> ScreenRegion | None:
        return self._region

    @property
    def click_through(self) -> bool:
        return self._click_through

    def show_frame(self, event: OcrFrameEvent) -> None:
        """Replace the painted frame and reposition over its region."""

        self._region = event.region
        if not event.blocks:
            # The slide changed to one with no recognized text; stale boxes
            # must not stay over a blank slide.
            self._frame = None
            self._hidden.clear()
            self.hide()
            return
        self._frame = event
        current = {block.block_id for block in event.blocks}
        # Forget ids from earlier slides; a returning id starts visible again.
        self._hidden &= current
        self.setGeometry(region_logical_geometry(event.region))
        if self._visible:
            self.show()
        self.update()

    def set_region(self, region: ScreenRegion) -> None:
        """Follow a region-picker change; the old boxes no longer apply."""

        self._region = region
        self._frame = None
        self._hidden.clear()
        self.setGeometry(region_logical_geometry(region))
        # Hide until the next OCR pass paints boxes for the new origin; an
        # empty interactive window would otherwise swallow clicks on the slide.
        self.hide()

    def set_overlay_visible(self, visible: bool) -> None:
        """Global show/hide that survives incoming frames."""

        self._visible = bool(visible)
        if self._visible and self._frame is not None:
            self.show()
        else:
            self.hide()

    def clear(self) -> None:
        """Drop the frame and hide so an interactive window blocks no clicks."""

        self._frame = None
        self._hidden.clear()
        self.hide()

    def paint_now(self) -> None:
        """Force a synchronous repaint (before reporting ``render.ack``)."""

        self.repaint()

    def has_content(self) -> bool:
        """True when at least one block would paint."""

        if self._frame is None:
            return False
        return any(block.block_id not in self._hidden for block in self._frame.blocks)

    def blocks(self) -> tuple[OcrBlock, ...]:
        return tuple(self._frame.blocks) if self._frame is not None else ()

    # -- click-through -----------------------------------------------------

    def set_click_through(self, enabled: bool) -> None:
        enabled = bool(enabled)
        was_visible = self.isVisible()
        geometry = self.geometry()
        self._click_through = enabled
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enabled)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, enabled)
        if was_visible:
            self.show()
            self.setGeometry(geometry)
        self.update()

    # -- per-block visibility ---------------------------------------------

    def block_at(self, position: QPoint) -> OcrBlock | None:
        """Return the block under a widget-local point, if any."""

        region = self._region
        if region is None or self._frame is None or self.width() <= 0 or self.height() <= 0:
            return None
        physical_x = position.x() * region.width / self.width()
        physical_y = position.y() * region.height / self.height()
        for block in reversed(self._frame.blocks):
            bbox = block.bbox
            if (
                bbox.x <= physical_x < bbox.x + bbox.w
                and bbox.y <= physical_y < bbox.y + bbox.h
            ):
                return block
        return None

    def toggle_block(self, block_id: str) -> bool:
        """Flip one block's visibility; return the new visible state."""

        if block_id in self._hidden:
            self._hidden.discard(block_id)
            visible = True
        else:
            self._hidden.add(block_id)
            visible = False
        self.update()
        return visible

    def is_block_visible(self, block_id: str) -> bool:
        return block_id not in self._hidden

    def showEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        # The overlay covers the captured region. Exclude it from the grab so
        # the OCR loop reads the slide, not its own translations. Re-assert on
        # every show: a hidden/show cycle or a re-created handle would
        # otherwise leave the fresh window capturable.
        exclude_from_capture(self)

    def mousePressEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if self._click_through or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        block = self.block_at(event.position().toPoint())
        if block is not None:
            self.toggle_block(block.block_id)
        event.accept()

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        frame = self._frame
        region = self._region
        if frame is None or region is None or self.width() <= 0 or self.height() <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        scale_x = self.width() / region.width
        scale_y = self.height() / region.height
        background = QColor(self._config.background)
        background.setAlphaF(self._config.opacity)
        border = QColor(theme.BORDER)
        border.setAlpha(150)
        accent = QColor(theme.ACCENT)
        accent.setAlpha(120)

        for block in frame.blocks:
            bbox = block.bbox
            rect = QRectF(
                bbox.x * scale_x,
                bbox.y * scale_y,
                max(1.0, bbox.w * scale_x),
                max(1.0, bbox.h * scale_y),
            )
            if block.block_id in self._hidden:
                if not self._click_through:
                    # An invisible block must stay findable to toggle it back.
                    pen = QPen(accent)
                    pen.setWidth(1)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)
                continue

            painter.setPen(QPen(border, 1))
            painter.setBrush(background)
            painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)

            text = block.translation or block.text
            if not text:
                continue
            inner = rect.adjusted(_PADDING_PX, 1, -_PADDING_PX, -1)
            painter.setFont(self._fit_font(text, inner))
            if block.translation_status is TranslationStatus.FAILED:
                painter.setPen(QColor(theme.WARNING))
            else:
                painter.setPen(QColor(theme.TRANSLATION_TEXT))
            painter.drawText(inner, _TEXT_FLAGS, text)

    def _fit_font(self, text: str, rect: QRectF) -> QFont:
        """Largest font (down to the minimum) whose wrapped text fits the box."""

        font = theme.font(self._config.font_family, self._config.font_size)
        max_width = max(1, int(rect.width()))
        max_height = max(1, int(rect.height()))
        while font.pointSize() > _MIN_FONT_PT:
            metrics = QFontMetrics(font)
            bounds = metrics.boundingRect(
                QRect(0, 0, max_width, 10_000), _TEXT_FLAGS, text
            )
            if bounds.height() <= max_height:
                break
            font.setPointSize(font.pointSize() - 1)
        return font


__all__ = ["OcrOverlayWindow", "region_logical_geometry"]
