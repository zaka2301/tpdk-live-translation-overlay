"""Caption surface, history window, and screen-region picker (FR-04, FR-09).

The caption window is frameless, translucent, always on top, and click-through
by default. It shows the source line small and the translation large; partials
replace the same ``segment_id`` while finals accumulate in the history window.
Controls (start/stop, languages, checkpoint interval, region, history) are only
reachable while click-through is off; the tray/hotkey in ``ui/app.py`` toggles
that mode.

The region picker spans the virtual desktop, reports the drag rectangle in
physical pixels, and matches the monitor to mss's physical coordinates so the
phase 5 capture adapter can use it directly.
"""

from __future__ import annotations

import html
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QRectF, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from tpdk_overlay.config import UiConfig
from tpdk_overlay.contracts import (
    NoticeLevel,
    ScreenRegion,
    SegmentStatus,
    SessionState,
    TranscriptSegmentEvent,
    TranslationStatus,
)
from tpdk_overlay.mt.languages import ISO_TO_NLLB
from tpdk_overlay.ui import theme

logger = logging.getLogger(__name__)

MIN_WIDTH = 480
MIN_HEIGHT = 150
PANEL_RADIUS = 16.0
GRIP_SIZE = 18
SCREEN_MARGIN = 48

LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    "zh-hans": "Chinese (Simplified)",
    "zh-hant": "Chinese (Traditional)",
}


def language_label(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code.upper())


def format_timestamp(ms: int) -> str:
    """Session-relative milliseconds as ``mm:ss`` (or ``h:mm:ss`` past an hour)."""

    seconds = max(0, ms) // 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class _Grip(QWidget):
    """Bottom-right resize handle; only active while click-through is off."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFixedSize(GRIP_SIZE, GRIP_SIZE)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setToolTip("Drag to resize the caption")
        self._press_global: QPoint | None = None
        self._press_size = QSize()

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(theme.STATUS_TEXT))
        pen.setWidthF(1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        width, height = self.width(), self.height()
        for offset in (5, 9, 13):
            painter.drawLine(width - offset, height - 2, width - 2, height - offset)

    def mousePressEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_global = event.globalPosition().toPoint()
            self._press_size = self.window().size()
            event.accept()

    def mouseMoveEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if self._press_global is None:
            return
        delta = event.globalPosition().toPoint() - self._press_global
        self.window().resize(
            max(MIN_WIDTH, self._press_size.width() + delta.x()),
            max(MIN_HEIGHT, self._press_size.height() + delta.y()),
        )
        event.accept()

    def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        self._press_global = None
        event.accept()


class CaptionWindow(QWidget):
    """Frameless, translucent, always-on-top caption bar with controls."""

    startRequested = Signal()
    stopRequested = Signal()
    historyRequested = Signal()
    regionRequested = Signal()
    languagesChanged = Signal(str, str)
    intervalChanged = Signal(int)
    finalPainted = Signal(object)

    def __init__(
        self,
        ui: UiConfig,
        *,
        source_lang: str,
        target_lang: str,
        checkpoint_interval_s: int,
        languages: Sequence[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self._caption = ui.caption
        self._languages = tuple(languages) if languages is not None else tuple(ISO_TO_NLLB)
        self._click_through = bool(ui.click_through)
        self._state = SessionState.IDLE
        self._connected = False
        self._region: ScreenRegion | None = None
        self._drag_offset: QPoint | None = None
        self._notice_text = ""
        self._notice_color = theme.STATUS_TEXT
        self._notice_until = 0.0
        self._full_source = ""
        self._full_translation = ""
        self._segment_final = False
        self._revisions: dict[int, int] = {}
        self._pending: TranscriptSegmentEvent | None = None
        self._final_hold_ms = int(ui.caption.final_hold_ms)
        self._hold_until = 0.0
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        # PreciseTimer never fires before _hold_until; _release_pending still
        # re-arms for any remaining time as a backstop.
        self._hold_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._hold_timer.timeout.connect(self._release_pending)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("TPdK Caption")
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        self._build_ui(ui)
        self._place_on_screen()
        self.set_source_language(source_lang)
        self.set_target_language(target_lang)
        self.set_checkpoint_interval(checkpoint_interval_s)
        self.set_click_through(self._click_through)

    # -- construction ------------------------------------------------------

    def _build_ui(self, ui: UiConfig) -> None:
        caption = ui.caption
        family = caption.font_family
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 16, 26, 14)
        layout.setSpacing(6)

        self._controls = QWidget(self)
        self._controls.setObjectName("controlBar")
        self._controls.setStyleSheet(theme.controls_qss(family, caption.font_size))
        row = QHBoxLayout(self._controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._start_button = QPushButton("Start", self._controls)
        self._start_button.setObjectName("primaryButton")
        self._start_button.setToolTip("Start or stop the capture/translation session")
        self._start_button.clicked.connect(self._on_start_clicked)
        row.addWidget(self._start_button)

        row.addWidget(self._control_label("From"))
        self._source_combo = QComboBox(self._controls)
        self._source_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._source_combo.setMaxVisibleItems(20)
        self._source_combo.setToolTip("Language the lecturer speaks")
        self._source_combo.currentIndexChanged.connect(self._emit_languages)
        row.addWidget(self._source_combo)

        row.addWidget(self._control_label("To"))
        self._target_combo = QComboBox(self._controls)
        self._target_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._target_combo.setMaxVisibleItems(20)
        self._target_combo.setToolTip("Caption translation language")
        self._target_combo.currentIndexChanged.connect(self._emit_languages)
        row.addWidget(self._target_combo)

        for code in sorted(self._languages, key=language_label):
            label = language_label(code)
            self._source_combo.addItem(label, code)
            self._target_combo.addItem(label, code)

        row.addWidget(self._control_label("Summary"))
        self._interval_spin = QSpinBox(self._controls)
        self._interval_spin.setRange(30, 3600)
        self._interval_spin.setSingleStep(30)
        self._interval_spin.setSuffix(" s")
        self._interval_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._interval_spin.setToolTip("Checkpoint summary interval")
        self._interval_spin.valueChanged.connect(self._on_interval_changed)
        row.addWidget(self._interval_spin)

        self._region_button = QPushButton("Region not set", self._controls)
        self._region_button.setToolTip("Select the screen region that contains the slides")
        self._region_button.clicked.connect(self.regionRequested.emit)
        row.addWidget(self._region_button)

        self._history_button = QPushButton("History", self._controls)
        self._history_button.setToolTip("Show final captions and export them")
        self._history_button.clicked.connect(self.historyRequested.emit)
        row.addWidget(self._history_button)

        row.addStretch(1)
        layout.addWidget(self._controls)

        self._source_label = QLabel("", self)
        self._source_label.setTextFormat(Qt.TextFormat.PlainText)
        self._source_label.setWordWrap(True)
        self._source_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        source_font = theme.font(
            family, caption.font_size, weight=QFont.Weight.Medium
        )
        self._source_label.setFont(source_font)
        self._source_label.setStyleSheet(f"color: {theme.SOURCE_TEXT};")
        self._source_label.setMaximumHeight(
            QFontMetrics(source_font).lineSpacing() * 2 + 8
        )
        layout.addWidget(self._source_label)

        self._translation_label = QLabel("", self)
        self._translation_label.setTextFormat(Qt.TextFormat.PlainText)
        self._translation_label.setWordWrap(True)
        self._translation_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._translation_label.setFont(
            theme.font(
                family,
                caption.translation_font_size,
                weight=QFont.Weight.DemiBold,
            )
        )
        self._translation_label.setStyleSheet(f"color: {theme.TRANSLATION_TEXT};")
        layout.addWidget(self._translation_label, 1)

        self._status_label = QLabel("", self)
        self._status_label.setTextFormat(Qt.TextFormat.PlainText)
        self._status_label.setFont(theme.font(family, max(9, caption.font_size - 10)))
        self._status_label.setStyleSheet(f"color: {theme.STATUS_TEXT};")
        layout.addWidget(self._status_label)

        self._grip = _Grip(self)
        self._refresh_status()

    @staticmethod
    def _control_label(text: str) -> QLabel:
        label = QLabel(text)
        return label

    # -- public state ------------------------------------------------------

    @property
    def is_click_through(self) -> bool:
        return self._click_through

    def current_text(self) -> tuple[str, str]:
        """Currently displayed (source, translation); used by smoke tests."""

        return self._source_label.text(), self._translation_label.text()

    def set_source_language(self, code: str) -> None:
        self._select_language(self._source_combo, code)

    def set_target_language(self, code: str) -> None:
        self._select_language(self._target_combo, code)

    def set_checkpoint_interval(self, seconds: int) -> None:
        with QSignalBlocker(self._interval_spin):
            self._interval_spin.setValue(seconds)

    def set_region(self, region: ScreenRegion | None) -> None:
        self._region = region
        if region is None:
            self._region_button.setText("Region not set")
            self._region_button.setToolTip(
                "Select the screen region that contains the slides"
            )
        else:
            self._region_button.setText(f"Region {region.width}x{region.height}")
            self._region_button.setToolTip(
                f"left={region.left}, top={region.top}, "
                f"{region.width} x {region.height} px (monitor {region.monitor})"
            )
        self._refresh_status()

    def set_connection(self, connected: bool) -> None:
        self._connected = connected
        self._refresh_status()

    def set_session_state(self, state: SessionState) -> None:
        self._state = state
        active = state in (
            SessionState.STARTING,
            SessionState.RUNNING,
            SessionState.PAUSED,
        )
        self._start_button.setText("Stop" if active else "Start")
        self._start_button.setEnabled(state is not SessionState.STOPPING)
        if state is SessionState.STARTING:
            self.clear_caption()
        self._refresh_status()

    def clear_caption(self) -> None:
        self._hold_timer.stop()
        self._hold_until = 0.0
        self._pending = None
        self._revisions.clear()
        self._full_source = ""
        self._full_translation = ""
        self._segment_final = False
        self._source_label.clear()
        self._translation_label.clear()

    def show_segment(self, event: TranscriptSegmentEvent) -> None:
        """Apply a partial/final, or queue it while a final is being read.

        A translated final starts a short reading hold; partials and source-only
        pending finals that arrive during it are coalesced and shown when it
        expires. Resolved finals always paint, so a slow translation is neither
        overwritten by the next utterance nor lost from NFR-01 metrics.
        """

        last = self._revisions.get(event.segment_id)
        if last is not None:
            if event.revision < last:
                return
            if event.revision == last and not self._is_partial_enrichment(event):
                # A source preview owns its revision; only the translation of
                # that same preview may update it in place.
                return
        self._revisions[event.segment_id] = event.revision
        if len(self._revisions) > 64:
            cutoff = event.segment_id - 32
            self._revisions = {
                segment_id: revision
                for segment_id, revision in self._revisions.items()
                if segment_id >= cutoff
            }

        now = time.monotonic()
        if self._is_resolved_final(event):
            # A resolved final must always paint: render.ack drives NFR-01,
            # and skipping one would silently drop its metrics row.
            self._hold_timer.stop()
            self._hold_until = 0.0
            self._pending = None
            self._show_event(event)
            return
        if self._hold_until and now >= self._hold_until:
            # The hold expired but its timer callback has not run yet. Fold
            # this preview into the pending slot instead of painting it and
            # letting the stale callback overwrite it with an older event.
            self._release_expired_hold(event)
            return
        if self._hold_until > now:
            self._buffer_event(event)
            return
        self._show_event(event)

    def _show_event(self, event: TranscriptSegmentEvent) -> None:
        """Render the event immediately; a translated final starts a hold."""

        final = event.status is SegmentStatus.FINAL
        self._segment_final = final
        source_font = self._source_label.font()
        source_font.setItalic(not final)
        self._source_label.setFont(source_font)
        self._full_source = event.text
        self._full_translation = self._translation_text(event)
        self._apply_fitted_text()
        self._translation_label.setStyleSheet(
            f"color: {self._translation_color(event)};"
        )
        if final:
            self.finalPainted.emit(event)
            if self._is_translated_final(event):
                self._start_hold()

    def _start_hold(self) -> None:
        if self._final_hold_ms <= 0:
            return
        self._hold_until = time.monotonic() + self._final_hold_ms / 1000
        self._hold_timer.start(self._final_hold_ms)

    def _release_pending(self) -> None:
        now = time.monotonic()
        if now < self._hold_until:
            # An early/stale timeout; re-arm for the remaining hold instead of
            # leaving the pending preview with no timer to release it.
            remaining = int((self._hold_until - now) * 1000) + 1
            self._hold_timer.start(max(1, remaining))
            return
        self._release_expired_hold()

    def _release_expired_hold(
        self, event: TranscriptSegmentEvent | None = None
    ) -> None:
        """Show the newest pending preview after the hold has elapsed."""

        self._hold_timer.stop()
        self._hold_until = 0.0
        if event is not None:
            self._buffer_event(event)
        pending = self._pending
        self._pending = None
        if pending is not None:
            self._show_event(pending)

    def _buffer_event(self, event: TranscriptSegmentEvent) -> None:
        # Only previews are buffered: partials and source-only finals. A newer
        # segment wins; resolved finals bypass the buffer entirely.
        pending = self._pending
        if pending is None or event.segment_id >= pending.segment_id:
            self._pending = event

    @staticmethod
    def _is_partial_enrichment(event: TranscriptSegmentEvent) -> bool:
        return (
            event.status is SegmentStatus.PARTIAL
            and event.translation_status is TranslationStatus.OK
            and bool(event.translation)
        )

    @staticmethod
    def _is_resolved_final(event: TranscriptSegmentEvent) -> bool:
        return (
            event.status is SegmentStatus.FINAL
            and event.translation_status is not TranslationStatus.PENDING
        )

    @staticmethod
    def _is_translated_final(event: TranscriptSegmentEvent) -> bool:
        return (
            event.status is SegmentStatus.FINAL
            and event.translation_status is TranslationStatus.OK
            and bool(event.translation)
        )

    def show_notice(self, level: NoticeLevel, message: str) -> None:
        self._notice_text = message
        self._notice_color = theme.notice_color(level.value)
        self._notice_until = time.monotonic() + 8.0
        self._refresh_status()
        QTimer.singleShot(8500, self._refresh_status)

    def _apply_fitted_text(self) -> None:
        """Show as much as fits; the full text still lives in history."""

        if self.layout() is not None:
            self.layout().activate()
        self._fit_text(
            self._source_label,
            self._full_source,
            keep_tail=not self._segment_final,
            max_height=self._source_label.maximumHeight(),
        )
        # The source text just changed; let the layout shrink the translation
        # label before deciding how many lines of it fit.
        if self.layout() is not None:
            self.layout().invalidate()
            self.layout().activate()
        self._fit_text(
            self._translation_label,
            self._full_translation,
            keep_tail=not self._segment_final,
            max_height=max(0, self._translation_label.height() - 4),
        )

    @staticmethod
    def _fit_text(
        label: QLabel,
        text: str,
        *,
        keep_tail: bool,
        max_height: int,
    ) -> None:
        if not text:
            label.clear()
            return
        width = label.width()
        height = max_height if 0 < max_height < 10_000 else label.height()
        if width <= 0 or height <= 0:
            label.setText(text)
            return

        metrics = QFontMetrics(label.font())
        flags = Qt.TextFlag.TextWordWrap

        def measure(candidate: str) -> int:
            return metrics.boundingRect(
                QRect(0, 0, width, 10_000), flags, candidate
            ).height()

        if measure(text) <= height:
            label.setText(text)
            return

        words = text.split()
        prefix = "… " if keep_tail else ""
        suffix = "" if keep_tail else " …"
        low, high = 0, len(words)
        while low < high:
            middle = (low + high + 1) // 2
            subset = words[-middle:] if keep_tail else words[:middle]
            candidate = prefix + " ".join(subset) + suffix
            if measure(candidate) <= height:
                low = middle
            else:
                high = middle - 1
        if low <= 0:
            label.setText("…")
            return
        subset = words[-low:] if keep_tail else words[:low]
        label.setText(prefix + " ".join(subset) + suffix)

    def paint_now(self) -> None:
        """Force an immediate paint so a render ack carries a real timestamp."""

        self._source_label.repaint()
        self._translation_label.repaint()
        self._status_label.repaint()
        self.repaint()

    # -- click-through, drag, resize --------------------------------------

    def set_click_through(self, enabled: bool) -> None:
        enabled = bool(enabled)
        # Changing window flags hides the widget, so remember whether it was on
        # screen and restore it afterwards; otherwise the caption disappears.
        was_visible = self.isVisible()
        geometry = self.geometry()
        self._click_through = enabled
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enabled)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, enabled)
        self._controls.setVisible(not enabled)
        self._grip.setVisible(not enabled)
        if was_visible:
            self.show()
            self.setGeometry(geometry)
            self.raise_()
        self._refresh_status()

    def mousePressEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if self._click_through or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        event.accept()

    def mouseMoveEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if self._drag_offset is not None and not self._click_through:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._grip.move(self.width() - GRIP_SIZE - 2, self.height() - GRIP_SIZE - 2)
        if hasattr(self, "_source_label"):
            self._apply_fitted_text()

    def showEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        # Re-fit after a hidden window is shown; show() is not guaranteed to
        # emit resizeEvent when the geometry did not change.
        self._apply_fitted_text()

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0),
            PANEL_RADIUS,
            PANEL_RADIUS,
        )
        panel = QColor(theme.BACKGROUND)
        panel.setAlphaF(max(0.0, min(1.0, self._caption.opacity)))
        painter.fillPath(path, panel)
        pen = QPen(QColor(theme.BORDER))
        pen.setWidthF(1.0)
        painter.strokePath(path, pen)

    # -- internals ---------------------------------------------------------

    def _on_start_clicked(self, _checked: bool = False) -> None:
        if self._state in (
            SessionState.STARTING,
            SessionState.RUNNING,
            SessionState.PAUSED,
        ):
            self.stopRequested.emit()
        else:
            self.startRequested.emit()

    def _on_interval_changed(self, value: int) -> None:
        self.intervalChanged.emit(value)

    def _emit_languages(self, _index: int = -1) -> None:
        source = self._source_combo.currentData()
        target = self._target_combo.currentData()
        if source and target:
            self.languagesChanged.emit(str(source), str(target))

    def language_codes(self) -> list[str]:
        """Codes currently offered in the From/To menus."""

        return [
            str(self._source_combo.itemData(index))
            for index in range(self._source_combo.count())
        ]

    def _select_language(self, combo: QComboBox, code: str) -> None:
        index = combo.findData(code)
        if index < 0:
            logger.warning(
                "language %r is not offered by the configured translator; "
                "the menu keeps its current selection",
                code,
            )
            return
        with QSignalBlocker(combo):
            combo.setCurrentIndex(index)

    def _translation_text(self, event: TranscriptSegmentEvent) -> str:
        if event.translation:
            return event.translation
        if event.status is SegmentStatus.PARTIAL:
            return ""
        if event.translation_status is TranslationStatus.PENDING:
            return "translating..."
        if event.translation_status is TranslationStatus.FAILED:
            return "(translation unavailable)"
        return ""

    def _translation_color(self, event: TranscriptSegmentEvent) -> str:
        if event.translation:
            return theme.TRANSLATION_TEXT
        if event.translation_status is TranslationStatus.FAILED:
            return theme.WARNING
        return theme.STATUS_TEXT

    def _refresh_status(self) -> None:
        now = time.monotonic()
        if self._notice_text and now < self._notice_until:
            text = self._notice_text
            color = self._notice_color
        else:
            self._notice_text = ""
            connection = "connected" if self._connected else "backend offline - retrying"
            parts = [self._state.value, connection]
            if self._click_through:
                parts.append("Ctrl+Alt+L unlocks controls")
            text = " | ".join(parts)
            color = theme.STATUS_TEXT
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")

    def _place_on_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        width = min(self._caption.width, area.width())
        height = min(self._caption.height, area.height())
        anchor = self._caption.anchor
        if anchor == "bottom-center":
            x = area.x() + (area.width() - width) // 2
            y = area.y() + area.height() - height - SCREEN_MARGIN
        elif anchor == "bottom-left":
            x = area.x() + SCREEN_MARGIN
            y = area.y() + area.height() - height - SCREEN_MARGIN
        elif anchor == "bottom-right":
            x = area.x() + area.width() - width - SCREEN_MARGIN
            y = area.y() + area.height() - height - SCREEN_MARGIN
        else:  # top-center
            x = area.x() + (area.width() - width) // 2
            y = area.y() + SCREEN_MARGIN
        self.setGeometry(x, y, width, height)


# --------------------------------------------------------------------------- #
# Caption history
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class HistoryEntry:
    segment_id: int
    t_start_ms: int
    source: str
    translation: str | None = None


class HistoryWindow(QWidget):
    """Scrollable list of final caption pairs with a markdown export."""

    def __init__(self, ui: UiConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._session_id = ""
        self._entries: dict[int, HistoryEntry] = {}
        self.setWindowTitle("TPdK caption history")
        self.resize(780, 520)
        self.setStyleSheet(
            theme.history_window_qss(ui.caption.font_family, ui.caption.font_size)
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Session captions", self)
        title_font = theme.font(
            ui.caption.font_family, ui.caption.font_size, weight=QFont.Weight.DemiBold
        )
        title.setFont(title_font)
        header.addWidget(title)
        header.addStretch(1)
        self._export_button = QPushButton("Export...", self)
        self._export_button.setToolTip("Write the captions to a markdown file")
        self._export_button.clicked.connect(self.export)
        header.addWidget(self._export_button)
        layout.addLayout(header)

        self._browser = QTextBrowser(self)
        self._browser.setOpenExternalLinks(False)
        self._browser.setOpenLinks(False)
        layout.addWidget(self._browser, 1)

        self._footer = QLabel("", self)
        self._footer.setObjectName("historyFooter")
        self._footer.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self._footer)

        self._render()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def set_session(self, session_id: str | None) -> None:
        if not session_id or session_id == self._session_id:
            return
        self._session_id = session_id
        self._entries.clear()
        self._footer.clear()
        self._render()

    def update_final(
        self,
        segment_id: int,
        t_start_ms: int,
        source: str,
        translation: str | None,
    ) -> None:
        """Insert or update one final segment; translated finals replace the pair."""

        entry = self._entries.get(segment_id)
        if entry is None:
            self._entries[segment_id] = HistoryEntry(
                segment_id=segment_id,
                t_start_ms=t_start_ms,
                source=source,
                translation=translation,
            )
        else:
            entry.t_start_ms = t_start_ms
            entry.source = source
            if translation:
                entry.translation = translation
        self._render()

    def export_markdown(self) -> str:
        lines = [
            f"# Caption history - session {self._session_id}"
            if self._session_id
            else "# Caption history",
            "",
        ]
        for entry in self._entries.values():
            lines.append(f"**[{format_timestamp(entry.t_start_ms)}]** {entry.source}")
            if entry.translation:
                lines.append("")
                lines.append(entry.translation)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def export(self) -> None:
        if not self._entries:
            QMessageBox.information(
                self, "Export captions", "There are no final captions to export yet."
            )
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = str(Path.home() / f"tpdk_captions_{stamp}.md")
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export captions",
            default,
            "Markdown (*.md);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self.export_markdown(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._footer.setText(f"Exported to {path}")

    def _render(self) -> None:
        family = self._browser.font().family()
        if not self._entries:
            body = (
                f'<p style="color:{theme.HISTORY_MUTED};">'
                "Final captions appear here as the lecture runs."
                "</p>"
            )
        else:
            blocks: list[str] = []
            for entry in self._entries.values():
                stamp = format_timestamp(entry.t_start_ms)
                blocks.append(
                    f'<p style="margin:0 0 2px 0;color:{theme.HISTORY_MUTED};'
                    f'font-size:9pt;">{stamp}</p>'
                    f'<p style="margin:0 0 4px 0;color:{theme.SOURCE_TEXT};'
                    f'font-size:11pt;">{html.escape(entry.source)}</p>'
                )
                if entry.translation:
                    blocks.append(
                        f'<p style="margin:0 0 14px 0;color:{theme.HISTORY_TEXT};'
                        f'font-size:12pt;">{html.escape(entry.translation)}</p>'
                    )
                else:
                    blocks.append('<div style="margin:0 0 14px 0;"></div>')
            body = "".join(blocks)

        scroll = self._browser.verticalScrollBar()
        at_bottom = scroll.value() >= scroll.maximum() - 4
        self._browser.setHtml(
            "<html><body style=\"font-family:'"
            f"{html.escape(family)}'\">"
            f"{body}</body></html>"
        )
        if at_bottom:
            scroll.setValue(scroll.maximum())


# --------------------------------------------------------------------------- #
# Region picker
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _mss_monitors() -> tuple[dict[str, int], ...]:
    """Physical monitor rectangles from mss, or an empty tuple when unavailable."""

    try:
        import mss

        factory = getattr(mss, "MSS", mss.mss)
        with factory() as capture:
            return tuple(dict(monitor) for monitor in capture.monitors[1:])
    except Exception:  # noqa: BLE001 - the picker works without the hint
        logger.debug("mss monitor enumeration failed", exc_info=True)
        return ()


def _physical_origin(screen: object) -> tuple[int, int] | None:
    """Physical origin of a Qt screen, matched by size against mss monitors."""

    dpr = screen.devicePixelRatio() or 1.0
    geometry = screen.geometry()
    width = round(geometry.width() * dpr)
    height = round(geometry.height() * dpr)
    candidates = [
        monitor
        for monitor in _mss_monitors()
        if abs(monitor["width"] - width) <= 2 and abs(monitor["height"] - height) <= 2
    ]
    if not candidates:
        return None
    # Same-size monitors are common; pick the one nearest the logical origin
    # scaled by this screen's ratio instead of the first size match.
    target_x = geometry.x() * dpr
    target_y = geometry.y() * dpr
    best = min(
        candidates,
        key=lambda monitor: (monitor["left"] - target_x) ** 2
        + (monitor["top"] - target_y) ** 2,
    )
    return int(best["left"]), int(best["top"])


class RegionPicker(QWidget):
    """Full-desktop rubber-band picker returning physical-pixel coordinates."""

    regionSelected = Signal(object)
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._screens = list(QGuiApplication.screens())
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowTitle("Select slide region")

        union = QRect()
        for screen in self._screens:
            union = union.united(screen.geometry())
        if union.isNull():
            union = QRect(0, 0, 1920, 1080)
        self.setGeometry(union)

    def start(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    # -- input -------------------------------------------------------------

    def mousePressEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self._cancel()

    def mouseMoveEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if event.button() != Qt.MouseButton.LeftButton or self._origin is None:
            return
        self._current = event.position().toPoint()
        rect = QRect(self._origin, self._current).normalized().intersected(self.rect())
        self._origin = None
        self._current = None
        if rect.width() >= 8 and rect.height() >= 8:
            self.regionSelected.emit(self._region_for(rect))
        else:
            self.cancelled.emit()
        self.close()

    def keyPressEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
        else:
            super().keyPressEvent(event)

    def _cancel(self) -> None:
        self._origin = None
        self._current = None
        self.cancelled.emit()
        self.close()

    def showEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        self.activateWindow()
        self.setFocus()

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        rect = self._selection_rect()
        if rect is not None and not rect.isNull():
            fill = QColor(theme.ACCENT)
            fill.setAlpha(40)
            painter.fillRect(rect, fill)
            pen = QPen(QColor(theme.ACCENT))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)

            label = f"{rect.width()} x {rect.height()} px"
            metrics = painter.fontMetrics()
            label_rect = metrics.boundingRect(label).adjusted(-8, -5, 8, 5)
            if rect.top() > label_rect.height() + 12:
                label_rect.moveBottomLeft(QPoint(rect.left() + 4, rect.top() - 4))
            else:
                label_rect.moveTopLeft(QPoint(rect.left() + 4, rect.bottom() + 4))
            painter.fillRect(label_rect, QColor(theme.BACKGROUND))
            painter.setPen(QColor(theme.TRANSLATION_TEXT))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

        painter.setPen(QColor(theme.TRANSLATION_TEXT))
        painter.drawText(
            QRect(0, 24, self.width(), 30),
            Qt.AlignmentFlag.AlignCenter,
            "Drag over the slide area - Esc or right click cancels",
        )

    def _selection_rect(self) -> QRect | None:
        if self._origin is None or self._current is None:
            return None
        return QRect(self._origin, self._current).normalized().intersected(self.rect())

    # -- coordinate conversion --------------------------------------------

    def _region_for(self, rect: QRect) -> ScreenRegion:
        global_top_left = self.mapToGlobal(rect.topLeft())
        center = global_top_left + QPoint(rect.width() // 2, rect.height() // 2)
        screen = QGuiApplication.screenAt(center) or QGuiApplication.screenAt(
            global_top_left
        )
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        try:
            monitor = self._screens.index(screen) + 1 if screen else 1
        except ValueError:
            monitor = 1
        dpr = (screen.devicePixelRatio() if screen else 1.0) or 1.0
        if screen is not None:
            local = global_top_left - screen.geometry().topLeft()
            origin = _physical_origin(screen)
            if origin is not None:
                left = origin[0] + round(local.x() * dpr)
                top = origin[1] + round(local.y() * dpr)
            else:
                left = round((screen.geometry().x() + local.x()) * dpr)
                top = round((screen.geometry().y() + local.y()) * dpr)
        else:
            left = round(global_top_left.x() * dpr)
            top = round(global_top_left.y() * dpr)
        return ScreenRegion(
            left=left,
            top=top,
            width=max(1, round(rect.width() * dpr)),
            height=max(1, round(rect.height() * dpr)),
            monitor=monitor,
        )


__all__ = [
    "CaptionWindow",
    "HistoryEntry",
    "HistoryWindow",
    "RegionPicker",
    "format_timestamp",
    "language_label",
]
