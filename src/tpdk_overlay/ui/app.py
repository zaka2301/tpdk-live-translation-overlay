"""Overlay entry point: DPI awareness, tray icon, global hotkey, and wiring.

Run with ``scripts\\run_overlay.ps1`` (or ``python -m tpdk_overlay.ui.app``).
The window starts click-through so it never blocks the lecture app; the tray
menu and ``Ctrl+Alt+L`` toggle interactive mode, where the control bar, drag,
and resize handle work.

DPI: per-monitor-v2 awareness is enabled before ``QApplication`` is created, so
Qt reports logical pixels consistently. Capture math stays in physical pixels
(see :class:`~tpdk_overlay.ui.caption.RegionPicker`), and the only conversion is
``devicePixelRatio`` at the Qt boundary.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import signal
import sys
import threading
import time
from ctypes import wintypes
from typing import Any

from PySide6.QtCore import QObject, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from tpdk_overlay.config import AppConfig, load_config
from tpdk_overlay.contracts import (
    NoticeEvent,
    NoticeLevel,
    OcrFrameEvent,
    OverlayToggleCommand,
    OverlayVisibilityEvent,
    RegionUpdateCommand,
    RenderAckCommand,
    ScreenRegion,
    SegmentStatus,
    SessionPauseCommand,
    SessionResumeCommand,
    SessionStartCommand,
    SessionState,
    SessionStateEvent,
    SessionStopCommand,
    SettingsPatch,
    SettingsUpdateCommand,
    SummaryCheckpointEvent,
    SummaryFinalEvent,
    TranscriptSegmentEvent,
    TranslationStatus,
)
from tpdk_overlay.mt.languages import supported_languages
from tpdk_overlay.ui import theme
from tpdk_overlay.ui.caption import CaptionWindow, HistoryWindow, RegionPicker
from tpdk_overlay.ui.client import OverlayClient
from tpdk_overlay.ui.ocr_overlay import OcrOverlayWindow
from tpdk_overlay.utils.clock import now_ms
from tpdk_overlay.utils.logging import setup_logging

logger = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
VK_L = 0x4C
HOTKEY_ID = 1
HOTKEY_LABEL = "Ctrl+Alt+L"


# --------------------------------------------------------------------------- #
# DPI awareness
# --------------------------------------------------------------------------- #


def enable_dpi_awareness() -> None:
    """Enable per-monitor-v2 DPI awareness; must run before ``QApplication``."""

    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        set_context = user32.SetProcessDpiAwarenessContext
        set_context.argtypes = [ctypes.c_void_p]
        set_context.restype = wintypes.BOOL
        if set_context(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        shcore = ctypes.windll.shcore
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness.restype = ctypes.c_long
        shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return
    except (AttributeError, OSError):
        pass
    try:
        user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        logger.debug("DPI awareness could not be set", exc_info=True)


# --------------------------------------------------------------------------- #
# Global hotkey (Windows)
# --------------------------------------------------------------------------- #


class GlobalHotkey(QObject):
    """``Ctrl+Alt+L`` via ``RegisterHotKey`` on a small message-loop thread.

    The hotkey thread owns the WM_HOTKEY queue (``hWnd=None`` posts to the
    calling thread); ``triggered`` is emitted from that thread and Qt queues it
    back to the GUI thread. ``register()`` returns ``False`` when the chord is
    unavailable, so the caller can keep the caption interactive instead.
    """

    triggered = Signal()

    def __init__(
        self,
        *,
        key: int = VK_L,
        modifiers: int = MOD_CONTROL | MOD_ALT,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._key = key
        self._modifiers = modifiers
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._available = False
        self._ready = threading.Event()

    @property
    def available(self) -> bool:
        return self._available

    def register(self) -> bool:
        if sys.platform != "win32" or self._thread is not None:
            return self._available
        self._thread = threading.Thread(
            target=self._run, name="tpdk-hotkey", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=2.0)
        if not self._available:
            logger.warning("%s is unavailable; use the tray menu", HOTKEY_LABEL)
        return self._available

    def release(self) -> None:
        user32 = ctypes.windll.user32 if sys.platform == "win32" else None
        if (
            user32 is not None
            and self._thread_id
            and self._thread is not None
            and self._thread.is_alive()
        ):
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(timeout=1.0)
        self._available = False

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(
            None, HOTKEY_ID, self._modifiers | MOD_NOREPEAT, self._key
        ):
            self._ready.set()
            return
        self._available = True
        self._ready.set()
        message = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                if message.message == WM_HOTKEY and message.wParam == HOTKEY_ID:
                    self.triggered.emit()
        finally:
            user32.UnregisterHotKey(None, HOTKEY_ID)


# --------------------------------------------------------------------------- #
# Overlay application
# --------------------------------------------------------------------------- #


def _app_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(theme.ACCENT))
    painter.drawRoundedRect(4, 4, 56, 56, 8, 8)
    painter.setBrush(QColor(theme.ACCENT_TEXT))
    painter.drawRoundedRect(14, 22, 36, 7, 2, 2)
    painter.drawRoundedRect(14, 35, 26, 7, 2, 2)
    painter.end()
    return QIcon(pixmap)


def _supported_language(code: str, languages: list[str], preferred: str) -> str:
    """Coerce a configured language onto the active provider's supported set."""

    if code in languages:
        return code
    fallback = preferred if preferred in languages else languages[0]
    logger.warning(
        "configured language %r is not supported by the MT provider; using %r",
        code,
        fallback,
    )
    return fallback


class OverlayApp(QObject):
    """Owns the overlay windows, tray menu, hotkey, and WebSocket client."""

    def __init__(self, app: QApplication, config: AppConfig) -> None:
        super().__init__()
        self._app = app
        self._config = config
        self._languages = supported_languages(config.mt.provider)
        self._source_lang = _supported_language(
            config.app.language.source, self._languages, "en"
        )
        self._target_lang = _supported_language(
            config.app.language.target, self._languages, "id"
        )
        self._checkpoint_s = config.summary.checkpoint_interval_s
        self._region = config.screen.region
        self._state = SessionState.IDLE
        self._session_id: str | None = None
        self._click_through = bool(config.ui.click_through)
        self._auto_start_pending = bool(config.ui.auto_start)
        self._picker: RegionPicker | None = None

        self._caption = CaptionWindow(
            config.ui,
            source_lang=self._source_lang,
            target_lang=self._target_lang,
            checkpoint_interval_s=self._checkpoint_s,
            languages=self._languages,
        )
        self._caption.set_region(self._region)
        self._history = HistoryWindow(config.ui)
        self._history.hide()
        self._ocr = (
            OcrOverlayWindow(
                config.ui.ocr_overlay,
                click_through=self._click_through,
                capture_excluded=not config.screen.hide_overlay_during_capture,
            )
            if config.ui.ocr_overlay.enabled
            and config.screen.enabled
            and config.ocr.enabled
            else None
        )
        self._ocr_capture_hidden = False
        self._ocr_visible = True

        self._client = OverlayClient(
            f"ws://{config.server.host}:{config.server.port}/ws", parent=self
        )
        self._hotkey = GlobalHotkey(parent=self)
        self._hotkey.triggered.connect(self._toggle_click_through)
        self._tray: QSystemTrayIcon | None = None
        self._action_ocr: QAction | None = None
        self._tray = self._build_tray()
        self._update_actions()

        self._caption.startRequested.connect(self._toggle_session)
        self._caption.stopRequested.connect(self._stop_session)
        self._caption.historyRequested.connect(self._show_history)
        self._caption.regionRequested.connect(self._pick_region)
        self._caption.languagesChanged.connect(self._on_languages_changed)
        self._caption.intervalChanged.connect(self._on_interval_changed)
        self._caption.finalPainted.connect(self._ack)
        self._client.eventReceived.connect(self._on_event)
        self._client.connected.connect(self._on_connected)
        self._client.disconnected.connect(self._on_disconnected)
        self._client.protocolError.connect(self._on_protocol_error)
        self._app.aboutToQuit.connect(self.shutdown)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._caption.show()
        if self._tray is not None:
            self._tray.show()
        self._client.start()
        hotkey_available = self._hotkey.register()
        if not hotkey_available:
            if self._tray is None and self._click_through:
                # Without either toggle the caption would be locked forever.
                self._set_click_through(False)
            self._caption.show_notice(
                NoticeLevel.WARNING,
                f"{HOTKEY_LABEL} is unavailable; use the tray menu to unlock controls",
            )

    def shutdown(self) -> None:
        self._stop_session_for_exit()
        self._hotkey.release()
        self._client.stop()
        if self._tray is not None:
            self._tray.hide()
        self._caption.close()
        self._history.close()
        if self._ocr is not None:
            self._ocr.close()

    def _stop_session_for_exit(self) -> None:
        """Ask the backend to stop and give it a bounded moment to persist."""

        if not self._session_active() or not self._client.is_connected:
            return
        self._client.send(SessionStopCommand())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            self._app.processEvents()
            if self._state in (
                SessionState.IDLE,
                SessionState.STOPPED,
                SessionState.ERROR,
            ):
                return
            time.sleep(0.05)

    # -- session commands --------------------------------------------------

    def _session_active(self) -> bool:
        return self._state in (
            SessionState.STARTING,
            SessionState.RUNNING,
            SessionState.PAUSED,
        )

    def _toggle_session(self) -> None:
        if self._session_active():
            self._send(SessionStopCommand())
            return
        self._start_session()

    def _start_session(self) -> None:
        self._auto_start_pending = False
        self._send(
            SessionStartCommand(
                source_lang=self._source_lang,
                target_lang=self._target_lang,
                checkpoint_interval_s=self._checkpoint_s,
                region=self._region,
            )
        )

    def _stop_session(self) -> None:
        if self._session_active():
            self._send(SessionStopCommand())

    def _toggle_pause(self) -> None:
        if self._state is SessionState.PAUSED:
            self._send(SessionResumeCommand())
        elif self._state is SessionState.RUNNING:
            self._send(SessionPauseCommand())

    def _send(self, command: Any) -> None:
        if not self._client.send(command):
            self._caption.show_notice(
                NoticeLevel.WARNING,
                "Backend is not connected yet; the command was not sent",
            )

    def _on_languages_changed(self, source: str, target: str) -> None:
        self._source_lang = source
        self._target_lang = target
        if self._session_active():
            self._send(
                SettingsUpdateCommand(
                    patch=SettingsPatch(source_lang=source, target_lang=target)
                )
            )

    def _on_interval_changed(self, seconds: int) -> None:
        self._checkpoint_s = seconds
        if self._session_active():
            self._send(
                SettingsUpdateCommand(
                    patch=SettingsPatch(checkpoint_interval_s=seconds)
                )
            )

    # -- event handling ----------------------------------------------------

    def _on_event(self, event: object) -> None:
        if isinstance(event, SessionStateEvent):
            self._on_session_state(event)
        elif isinstance(event, TranscriptSegmentEvent):
            self._caption.show_segment(event)
            if event.status is SegmentStatus.FINAL:
                self._history.update_final(
                    segment_id=event.segment_id,
                    t_start_ms=event.t_start_ms,
                    source=event.text,
                    translation=(
                        event.translation
                        if event.translation_status is TranslationStatus.OK
                        else None
                    ),
                )
        elif isinstance(event, NoticeEvent):
            self._caption.show_notice(event.level, event.message)
            if event.level is NoticeLevel.ERROR and self._tray is not None:
                self._tray.showMessage(
                    "TPdK Overlay",
                    event.message,
                    QSystemTrayIcon.MessageIcon.Warning,
                    5000,
                )
        elif isinstance(event, SummaryFinalEvent):
            self._caption.show_notice(
                NoticeLevel.INFO, f"Final summary written to {event.path}"
            )
        elif isinstance(event, SummaryCheckpointEvent):
            if event.status == "degraded":
                self._caption.show_notice(
                    NoticeLevel.WARNING,
                    f"Checkpoint {event.checkpoint_id} summary degraded; "
                    "transcription continues",
                )
        elif isinstance(event, OcrFrameEvent):
            self._on_ocr_frame(event)
        elif isinstance(event, OverlayVisibilityEvent):
            self._on_overlay_visibility(event)

    def _on_ocr_frame(self, event: OcrFrameEvent) -> None:
        """Paint translated slide blocks and report NFR-03's paint time."""

        if self._ocr is None:
            return
        self._ocr.show_frame(event)
        self._update_actions()
        if not self._ocr.has_content() or not self._ocr.isVisible():
            return
        self._ocr.paint_now()
        self._send(
            RenderAckCommand(event_id=event.event_id, painted_ts_ms=now_ms())
        )

    def _on_overlay_visibility(self, event: OverlayVisibilityEvent) -> None:
        """Hide/show the OCR overlay around the backend's screen grab.

        Recording mode keeps the overlay out of the display affinity so OBS and
        screen shares capture it; the OCR loop therefore hides it for the few
        frames it grabs. The ack tells the backend the window is off screen,
        and the UI restores the user's own visibility state afterwards.
        """

        if event.reason != "capture":
            return
        if event.visible:
            if self._ocr is not None and self._ocr_capture_hidden:
                self._ocr_capture_hidden = False
                if self._ocr.has_content():
                    self._ocr.show()
            return
        if self._ocr is not None:
            self._ocr_capture_hidden = self._ocr.isVisible()
            if self._ocr_capture_hidden:
                self._ocr.hide()
        # Ack even when there was nothing to hide (no OCR window configured):
        # the backend waits for it before grabbing a frame.
        self._send(
            RenderAckCommand(event_id=event.event_id, painted_ts_ms=now_ms())
        )

    def _on_session_state(self, event: SessionStateEvent) -> None:
        self._state = event.state
        if event.session_id and event.session_id != self._session_id:
            self._session_id = event.session_id
            self._history.set_session(event.session_id)
        if event.state in (
            SessionState.IDLE,
            SessionState.STOPPED,
            SessionState.ERROR,
        ):
            ocr = getattr(self, "_ocr", None)
            if ocr is not None:
                ocr.clear()
            self._ocr_capture_hidden = False
        self._caption.set_session_state(event.state)
        self._update_actions()
        self._maybe_auto_start(event.state)

    def _maybe_auto_start(self, state: SessionState) -> None:
        """Start once on launch if the backend has no active session."""

        if not self._auto_start_pending:
            return
        self._auto_start_pending = False
        if state in (SessionState.IDLE, SessionState.STOPPED):
            logger.info("auto-starting session after backend connection")
            self._start_session()

    def _ack(self, event: TranscriptSegmentEvent) -> None:
        """Report the paint time for NFR-01 when a final was actually shown."""

        if not self._caption.isVisible():
            return
        self._caption.paint_now()
        self._send(
            RenderAckCommand(event_id=event.event_id, painted_ts_ms=now_ms())
        )

    def _on_connected(self) -> None:
        self._caption.set_connection(True)
        if self._tray is not None:
            self._tray.setToolTip("TPdK Overlay - backend connected")

    def _on_disconnected(self, reason: str) -> None:
        logger.info("backend disconnected: %s", reason)
        self._caption.set_connection(False)
        if self._tray is not None:
            self._tray.setToolTip("TPdK Overlay - backend offline, retrying")

    def _on_protocol_error(self, message: str) -> None:
        logger.warning("protocol error from backend: %s", message)

    # -- overlay toggles ---------------------------------------------------

    def _set_caption_visible(self, visible: bool) -> None:
        self._caption.setVisible(visible)
        if visible:
            self._caption.raise_()
        self._update_actions()
        if self._client.is_connected:
            self._send(OverlayToggleCommand(target="caption", visible=visible))

    def _set_ocr_visible(self, visible: bool) -> None:
        self._ocr_visible = bool(visible)
        if self._ocr is not None:
            self._ocr.set_overlay_visible(self._ocr_visible)
        self._update_actions()
        if self._client.is_connected:
            self._send(OverlayToggleCommand(target="ocr", visible=self._ocr_visible))

    def _set_click_through(self, enabled: bool) -> None:
        self._click_through = bool(enabled)
        self._caption.set_click_through(self._click_through)
        if self._ocr is not None:
            self._ocr.set_click_through(self._click_through)
        if self._tray is not None:
            with QSignalBlocker(self._action_click_through):
                self._action_click_through.setChecked(self._click_through)

    def _toggle_click_through(self) -> None:
        self._set_click_through(not self._click_through)

    def _show_history(self) -> None:
        self._history.show()
        self._history.raise_()
        self._history.activateWindow()

    def _pick_region(self) -> None:
        if self._picker is not None:
            self._picker.close()
        picker = RegionPicker()
        picker.regionSelected.connect(self._on_region_selected)
        picker.cancelled.connect(self._on_region_cancelled)
        self._picker = picker
        picker.start()

    def _on_region_selected(self, region: ScreenRegion) -> None:
        self._picker = None
        self._region = region
        self._caption.set_region(region)
        if self._ocr is not None:
            self._ocr.set_region(region)
        self._send(RegionUpdateCommand(region=region))

    def _on_region_cancelled(self) -> None:
        self._picker = None

    # -- tray --------------------------------------------------------------

    def _build_tray(self) -> QSystemTrayIcon | None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.info("system tray unavailable; hotkey controls remain active")
            return None

        tray = QSystemTrayIcon(_app_icon(), self)
        tray.setToolTip("TPdK Overlay")
        menu = QMenu()

        self._action_visible = QAction("Caption visible", menu)
        self._action_visible.setCheckable(True)
        self._action_visible.setChecked(True)
        self._action_visible.toggled.connect(self._set_caption_visible)
        menu.addAction(self._action_visible)

        self._action_ocr = None
        if self._ocr is not None:
            self._action_ocr = QAction("Slide translations visible", menu)
            self._action_ocr.setCheckable(True)
            self._action_ocr.setChecked(True)
            self._action_ocr.toggled.connect(self._set_ocr_visible)
            menu.addAction(self._action_ocr)

        self._action_click_through = QAction(
            f"Click-through ({HOTKEY_LABEL})", menu
        )
        self._action_click_through.setCheckable(True)
        self._action_click_through.setChecked(self._click_through)
        self._action_click_through.toggled.connect(self._set_click_through)
        menu.addAction(self._action_click_through)

        menu.addSeparator()
        self._action_session = QAction("Start session", menu)
        self._action_session.triggered.connect(lambda: self._toggle_session())
        menu.addAction(self._action_session)

        self._action_pause = QAction("Pause", menu)
        self._action_pause.setEnabled(False)
        self._action_pause.triggered.connect(lambda: self._toggle_pause())
        menu.addAction(self._action_pause)

        menu.addSeparator()
        history_action = QAction("Caption history...", menu)
        history_action.triggered.connect(lambda: self._show_history())
        menu.addAction(history_action)

        region_action = QAction("Select screen region...", menu)
        region_action.triggered.connect(lambda: self._pick_region())
        menu.addAction(region_action)

        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._app.quit)
        menu.addAction(quit_action)

        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)
        return tray

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason is QSystemTrayIcon.ActivationReason.DoubleClick:
            self._set_caption_visible(not self._caption.isVisible())

    def _update_actions(self) -> None:
        if self._tray is None:
            return
        active = self._session_active()
        self._action_session.setText("Stop session" if active else "Start session")
        self._action_session.setEnabled(self._state is not SessionState.STOPPING)
        self._action_pause.setEnabled(
            self._state in (SessionState.RUNNING, SessionState.PAUSED)
        )
        self._action_pause.setText(
            "Resume" if self._state is SessionState.PAUSED else "Pause"
        )
        with QSignalBlocker(self._action_visible):
            self._action_visible.setChecked(self._caption.isVisible())
        if self._action_ocr is not None:
            with QSignalBlocker(self._action_ocr):
                # Intent, not window visibility: the capture handshake hides
                # the window for a few frames without changing the user's
                # preference.
                self._action_ocr.setChecked(
                    self._ocr is not None and self._ocr_visible
                )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _install_quit_signals(app: QApplication) -> QTimer:
    """Make Ctrl+C quit the Qt loop (handlers run between bytecodes)."""

    def _quit(*_args: object) -> None:
        app.quit()

    for name in ("SIGINT", "SIGBREAK"):
        value = getattr(signal, name, None)
        if value is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            signal.signal(value, _quit)

    timer = QTimer(app)
    timer.setInterval(250)
    timer.timeout.connect(lambda: None)
    timer.start()
    return timer


def main(argv: list[str] | None = None) -> int:
    enable_dpi_awareness()
    # Qt 6 enables high-DPI scaling by default; pass fractional factors through
    # instead of rounding, so the logical geometry matches the physical one.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    config = load_config()
    setup_logging(config.app.log_level, config.logs_dir)

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("TPdK Overlay")
    app.setApplicationDisplayName("TPdK Overlay")
    app.setQuitOnLastWindowClosed(False)

    signal_timer = _install_quit_signals(app)
    overlay = OverlayApp(app, config)
    overlay.start()
    logger.info(
        "overlay ready; backend ws://%s:%s/ws",
        config.server.host,
        config.server.port,
    )
    try:
        return app.exec()
    finally:
        signal_timer.stop()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GlobalHotkey", "OverlayApp", "enable_dpi_awareness", "main"]
