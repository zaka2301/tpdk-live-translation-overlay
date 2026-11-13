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

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from tpdk_overlay.config import OcrOverlayConfig
from tpdk_overlay.contracts import OcrBlock, OcrFrameEvent, ScreenRegion, TranslationStatus
from tpdk_overlay.ui import theme
from tpdk_overlay.ui.caption import _physical_origin
from tpdk_overlay.ui.capture_guard import exclude_from_capture

# PaddleOCR boxes hug the text line: the detected polygon is ~1.1x the source em
# tall (measured on rendered Segoe UI at 48 px: 1.02-1.21 across sample lines).
# Dividing the per-line box height by this reproduces the source text size.
_POLYGON_PER_EM = 1.10
_MIN_FONT_PX = 10
_PAD_X = 4
_PAD_Y = 3
_TEXT_MEASURE_HEIGHT = 100_000
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


@dataclass(frozen=True)
class BlockLayout:
    """Panel geometry and font for one translated block."""

    panel: QRectF
    font: QFont
    text: str


def _pixel_font(family: str, px: int) -> QFont:
    font = QFont(family)
    font.setPixelSize(max(1, int(px)))
    return font


def stabilize_line_heights(
    values: Sequence[float], *, ratio: float = 1.25
) -> list[float]:
    """Return cluster medians so descender-less lines do not render smaller.

    PaddleOCR boxes follow the glyphs present: a line with ``g``/``y``/``p``
    reports a taller box than one without, for the same source font. Blocks
    whose heights are within ``ratio`` share the cluster median, while
    genuinely different sizes (a title over body text) stay separate.
    """

    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cluster: list[int] = []

    def flush() -> None:
        if not cluster:
            return
        cluster_values = sorted(values[index] for index in cluster)
        median = cluster_values[len(cluster_values) // 2]
        for index in cluster:
            result[index] = median

    for index in order:
        if cluster and values[index] > values[cluster[-1]] * ratio:
            flush()
            cluster.clear()
        cluster.append(index)
    flush()
    return result


def block_right_limits(
    rects: Sequence[QRectF], *, window_width: float
) -> list[float]:
    """Right-edge limit per block: the window edge or the next block beside it.

    A panel may widen past its source box to keep a translation on one line, but
    it must stop at a block in another column at the same height instead of
    running over that column's own translation.
    """

    limits: list[float] = []
    for index, rect in enumerate(rects):
        limit = window_width
        for other_index, other in enumerate(rects):
            if other_index == index or other.x() <= rect.x():
                continue
            if other.top() < rect.bottom() and other.bottom() > rect.top():
                limit = min(limit, other.x() - _PAD_X)
        limits.append(limit)
    return limits


def layout_block(
    block: OcrBlock,
    rect: QRectF,
    *,
    config: OcrOverlayConfig,
    window_height: float,
    window_width: float,
    line_height: float | None = None,
    right_limit: float | None = None,
) -> BlockLayout:
    """Size a translation like the source line and fit it without shrinking.

    ``rect`` is the source block box in widget coordinates. The source em is
    recovered from the per-line box height (``bbox.h / line_count``), so a body
    line on a 1280x720 slide renders at the size the lecturer used instead of a
    fixed point size. The panel first widens to the right (slides usually have
    free space there) so the text keeps the source line count, and only grows
    taller when the window edge is reached. The font shrinks only when the
    panel could no longer fit inside the window.
    """

    text = block.translation or block.text
    per_block = rect.height() / max(1, block.line_count)
    line_height = max(1.0, line_height if line_height is not None else per_block)
    base_px = round(line_height / _POLYGON_PER_EM * config.font_scale)
    base_px = max(_MIN_FONT_PX, min(int(config.max_font_px), int(base_px)))
    source_inner = max(1, int(rect.width()) - 2 * _PAD_X)
    edge = window_width if right_limit is None else min(window_width, right_limit)
    window_inner = max(source_inner, int(edge - rect.x()) - 2 * _PAD_X)

    def wrapped_height(font: QFont, width: int) -> int:
        bounds = QFontMetrics(font).boundingRect(
            QRect(0, 0, max(1, width), _TEXT_MEASURE_HEIGHT), _TEXT_FLAGS, text
        )
        return bounds.height() + 2 * _PAD_Y

    def choose(px: int) -> tuple[QFont, int, int]:
        font = _pixel_font(config.font_family, px)
        # A translation may use as many lines as the source block had without
        # growing the panel.
        target = max(1, block.line_count) * QFontMetrics(font).height() + 2 * _PAD_Y
        width = source_inner
        needed = wrapped_height(font, width)
        if needed > target:
            low, high = source_inner, window_inner
            best = -1
            while low <= high:
                mid = (low + high) // 2
                if wrapped_height(font, mid) <= target:
                    best = mid
                    high = mid - 1
                else:
                    low = mid + 1
            width = best if best > 0 else window_inner
            needed = wrapped_height(font, width)
        return font, width, needed

    px = base_px
    font, inner, needed = choose(px)
    # A panel taller than the window would be clipped; shrink only then.
    while needed > window_height and px > _MIN_FONT_PX:
        px = max(_MIN_FONT_PX, int(px * 0.9))
        font, inner, needed = choose(px)

    height = max(rect.height(), float(needed))
    top = rect.center().y() - height / 2.0
    top = min(max(0.0, top), max(0.0, window_height - height))
    panel = QRectF(rect.x(), top, float(inner + 2 * _PAD_X), height)
    return BlockLayout(panel=panel, font=font, text=text)


class OcrOverlayWindow(QWidget):
    """Paint translated blocks over the captured slide region."""

    def __init__(
        self,
        config: OcrOverlayConfig,
        *,
        click_through: bool = True,
        capture_excluded: bool = True,
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
        self._capture_excluded = bool(capture_excluded)
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
        # the OCR loop reads the slide, not its own translations. Recording
        # mode (capture_excluded=False) leaves it capturable and the OCR loop
        # hides it around each grab instead.
        if self._capture_excluded:
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
        border.setAlpha(200)
        accent = QColor(theme.ACCENT)
        accent.setAlpha(140)

        visible: list[tuple[OcrBlock, QRectF]] = []
        hidden_rects: list[QRectF] = []
        for block in frame.blocks:
            bbox = block.bbox
            rect = QRectF(
                bbox.x * scale_x,
                bbox.y * scale_y,
                max(1.0, bbox.w * scale_x),
                max(1.0, bbox.h * scale_y),
            )
            if block.block_id in self._hidden:
                hidden_rects.append(rect)
                continue
            visible.append((block, rect))

        # One size per text-size cluster: PaddleOCR box heights wobble with the
        # glyphs present, and neighbouring lines must not alternate sizes.
        heights = stabilize_line_heights(
            [rect.height() / max(1, block.line_count) for block, rect in visible]
        )
        limits = block_right_limits(
            [rect for _block, rect in visible], window_width=float(self.width())
        )
        layouts = [
            (
                block,
                layout_block(
                    block,
                    rect,
                    config=self._config,
                    window_height=float(self.height()),
                    window_width=float(self.width()),
                    line_height=line_height,
                    right_limit=limit,
                ),
            )
            for (block, rect), line_height, limit in zip(
                visible, heights, limits, strict=True
            )
        ]

        # Panels first, text last: a panel that grew past its source box must
        # not hide a neighbouring block's translation.
        for _block, layout in layouts:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 45))
            painter.drawRoundedRect(layout.panel.adjusted(-1, 1, 1, 3), 5, 5)
            painter.setPen(QPen(border, 1))
            painter.setBrush(background)
            painter.drawRoundedRect(layout.panel.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

        if not self._click_through:
            # An invisible block must stay findable to toggle it back; draw the
            # outline over the panels so a widened neighbour cannot hide it.
            pen = QPen(accent)
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for rect in hidden_rects:
                painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)

        for block, layout in layouts:
            painter.setFont(layout.font)
            if block.translation_status is TranslationStatus.FAILED:
                painter.setPen(QColor(theme.WARNING))
            else:
                painter.setPen(QColor(theme.TRANSLATION_TEXT))
            painter.drawText(
                layout.panel.adjusted(_PAD_X, _PAD_Y, -_PAD_X, -_PAD_Y),
                _TEXT_FLAGS,
                layout.text,
            )


__all__ = [
    "BlockLayout",
    "OcrOverlayWindow",
    "block_right_limits",
    "layout_block",
    "region_logical_geometry",
    "stabilize_line_heights",
]
