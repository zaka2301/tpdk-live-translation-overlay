"""Unit tests for the overlay client and caption surfaces (phase 4).

The transport tests run a real ``websockets`` server on localhost: no backend,
no GPU. They cover command delivery, event parsing, and the reconnect path used
when the backend restarts (NFR-06). Qt widgets are created with the offscreen
platform so the suite stays headless.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QSize, Qt, Slot
from PySide6.QtGui import QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QApplication
from websockets.sync.server import ServerConnection, serve

from tpdk_overlay.config import CaptionConfig, UiConfig
from tpdk_overlay.contracts import (
    PingCommand,
    PongEvent,
    SegmentStatus,
    SessionState,
    SessionStateEvent,
    TranscriptSegmentEvent,
    TranslationStatus,
)
from tpdk_overlay.mt.languages import supported_languages
from tpdk_overlay.ui.caption import CaptionWindow, HistoryWindow, RegionPicker
from tpdk_overlay.ui.client import Backoff, OverlayClient


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    yield app


class _Collector(QObject):
    """Receiver in the GUI thread; queued signals land here on processEvents."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []
        self.connected = 0
        self.disconnected: list[str] = []

    @Slot()
    def on_connected(self) -> None:
        self.connected += 1

    @Slot(str)
    def on_disconnected(self, reason: str) -> None:
        self.disconnected.append(reason)

    @Slot(object)
    def on_event(self, event: object) -> None:
        self.events.append(event)


class _Server:
    """One sync websockets server on a fixed port; start/stop/reuse it."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.connections = 0
        self.received: list[str] = []
        self._server: object | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        ready = threading.Event()

        def handler(websocket: ServerConnection) -> None:
            self.connections += 1
            try:
                websocket.send(
                    SessionStateEvent(state=SessionState.IDLE).model_dump_json()
                )
                for raw in websocket:
                    self.received.append(raw)
                    try:
                        command = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if command.get("type") == "ping":
                        websocket.send(
                            PongEvent(
                                echo_ts_ms=int(command.get("client_ts_ms", 0))
                            ).model_dump_json()
                        )
            except Exception:  # noqa: BLE001 - shutdown closes live connections
                return

        def run() -> None:
            with serve(handler, "127.0.0.1", self.port) as server:
                self._server = server
                ready.set()
                server.serve_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert ready.wait(5.0), "websocket server did not start"

    def stop(self) -> None:
        server = self._server
        if server is not None:
            server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(qt_app: QApplication, predicate, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qt_app.processEvents()
    return bool(predicate())


def test_backoff_caps_at_five_seconds() -> None:
    backoff = Backoff()
    assert [backoff.next() for _ in range(6)] == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0]
    backoff.reset()
    assert backoff.next() == 0.5


def test_client_delivers_events_and_recovers(qt_app: QApplication) -> None:
    port = _free_port()
    server = _Server(port)
    server.start()

    collector = _Collector()
    client = OverlayClient(
        f"ws://127.0.0.1:{port}/ws", connect_timeout_s=1.0
    )
    client.connected.connect(collector.on_connected)
    client.disconnected.connect(collector.on_disconnected)
    client.eventReceived.connect(collector.on_event)

    restarted: _Server | None = None
    client.start()
    try:
        assert _wait(qt_app, lambda: collector.connected == 1)
        assert _wait(
            qt_app,
            lambda: any(isinstance(item, SessionStateEvent) for item in collector.events),
        )

        assert client.send(PingCommand())
        assert _wait(
            qt_app,
            lambda: any(isinstance(item, PongEvent) for item in collector.events),
        )

        # Backend goes away: the client reports it and retries with backoff.
        server.stop()
        assert _wait(qt_app, lambda: bool(collector.disconnected))
        assert not client.is_connected

        restarted = _Server(port)
        restarted.start()
        assert _wait(qt_app, lambda: collector.connected >= 2, timeout=7.0)
        assert _wait(
            qt_app,
            lambda: sum(
                isinstance(item, SessionStateEvent) for item in collector.events
            )
            >= 2,
        )
    finally:
        client.stop()
        server.stop()
        if restarted is not None:
            restarted.stop()


def _segment(
    *,
    revision: int,
    status: SegmentStatus,
    text: str,
    translation: str | None = None,
    translation_status: TranslationStatus = TranslationStatus.PENDING,
) -> TranscriptSegmentEvent:
    return TranscriptSegmentEvent(
        segment_id=7,
        revision=revision,
        status=status,
        t_start_ms=1_000,
        t_end_ms=2_000,
        source_lang="en",
        target_lang="id",
        text=text,
        translation=translation,
        translation_status=translation_status,
    )


def test_caption_language_menu_matches_provider(qt_app: QApplication) -> None:
    window = CaptionWindow(
        UiConfig(),
        source_lang="en",
        target_lang="id",
        checkpoint_interval_s=300,
        languages=supported_languages("plamo_llama"),
    )
    try:
        codes = set(window.language_codes())
    finally:
        window.close()
    assert {"en", "id", "zh-hans", "zh-hant", "tl"} <= codes
    assert {"pt", "el", "tr", "yue"}.isdisjoint(codes)


def test_caption_ignores_stale_revisions(qt_app: QApplication) -> None:
    window = CaptionWindow(
        UiConfig(), source_lang="en", target_lang="id", checkpoint_interval_s=300
    )
    try:
        window.show_segment(
            _segment(
                revision=3,
                status=SegmentStatus.FINAL,
                text="The derivative is a limit",
            )
        )
        assert window.current_text() == ("The derivative is a limit", "translating...")

        stale = _segment(
            revision=2, status=SegmentStatus.PARTIAL, text="the deriv"
        )
        window.show_segment(stale)
        assert window.current_text()[0] == "The derivative is a limit"

        window.show_segment(
            _segment(
                revision=4,
                status=SegmentStatus.FINAL,
                text="The derivative is a limit",
                translation="Turunan adalah limit",
                translation_status=TranslationStatus.OK,
            )
        )
        assert window.current_text() == (
            "The derivative is a limit",
            "Turunan adalah limit",
        )
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_history_replaces_translation_for_same_segment(qt_app: QApplication) -> None:
    history = HistoryWindow(UiConfig())
    try:
        history.update_final(7, 1_000, "The derivative is a limit", None)
        history.update_final(7, 1_000, "The derivative is a limit", "Turunan adalah limit")
        assert history.entry_count == 1
        exported = history.export_markdown()
        assert exported.count("The derivative is a limit") == 1
        assert "Turunan adalah limit" in exported
        assert "[00:01]" in exported
    finally:
        history.close()
        history.deleteLater()
        qt_app.processEvents()


def test_drag_and_resize_only_when_click_through_is_off(
    qt_app: QApplication,
) -> None:
    window = CaptionWindow(
        UiConfig(), source_lang="en", target_lang="id", checkpoint_interval_s=300
    )
    try:
        window.move(200, 200)
        window.resize(800, 220)
        start = window.pos()

        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(30, 30),
            QPointF(300, 300),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(40, 40),
            QPointF(330, 320),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

        window.set_click_through(True)
        window.mousePressEvent(press)
        window.mouseMoveEvent(move)
        assert window.pos() == start, "click-through must pass mouse input through"

        window.set_click_through(False)
        window.mousePressEvent(press)
        window.mouseMoveEvent(move)
        window.mouseReleaseEvent(move)
        assert window.pos() == start + QPoint(30, 20)
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_resize_grip_grows_the_window(qt_app: QApplication) -> None:
    window = CaptionWindow(
        UiConfig(), source_lang="en", target_lang="id", checkpoint_interval_s=300
    )
    try:
        window.set_click_through(False)
        window.resize(800, 220)
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(5, 5),
            QPointF(1000, 1000),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(15, 15),
            QPointF(1030, 1060),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window._grip.mousePressEvent(press)
        window._grip.mouseMoveEvent(move)
        assert window.size() == QSize(830, 280)
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_region_picker_reports_physical_pixels(qt_app: QApplication) -> None:
    picker = RegionPicker()
    try:
        screen = QGuiApplication.primaryScreen()
        dpr = (screen.devicePixelRatio() if screen else 1.0) or 1.0
        region = picker._region_for(QRect(10, 20, 100, 50))
        assert region.monitor >= 1
        assert region.width == round(100 * dpr)
        assert region.height == round(50 * dpr)
        assert region.left >= 0 and region.top >= 0
    finally:
        picker.close()
        picker.deleteLater()
        qt_app.processEvents()


def test_caption_trims_overflow_but_keeps_full_text_in_history(
    qt_app: QApplication,
) -> None:
    window = CaptionWindow(
        UiConfig(caption=CaptionConfig(final_hold_ms=0)),
        source_lang="en",
        target_lang="id",
        checkpoint_interval_s=300,
    )
    window.show()
    qt_app.processEvents()
    try:
        long_text = " ".join(f"word{index}" for index in range(200))
        window.show_segment(
            TranscriptSegmentEvent(
                segment_id=3,
                revision=2,
                status=SegmentStatus.FINAL,
                t_start_ms=0,
                t_end_ms=10_000,
                source_lang="en",
                target_lang="id",
                text=long_text,
                translation=long_text,
                translation_status=TranslationStatus.OK,
            )
        )
        qt_app.processEvents()
        source, translation = window.current_text()
        assert len(source) < len(long_text) and source.endswith("…")
        assert len(translation) < len(long_text) and translation.endswith("…")

        window.show_segment(
            TranscriptSegmentEvent(
                segment_id=4,
                revision=2,
                status=SegmentStatus.PARTIAL,
                t_start_ms=10_000,
                t_end_ms=20_000,
                source_lang="en",
                target_lang="id",
                text=long_text,
            )
        )
        qt_app.processEvents()
        assert window.current_text()[0].startswith("… ")
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_final_translation_holds_before_next_partial(
    qt_app: QApplication,
) -> None:
    window = CaptionWindow(
        UiConfig(caption=CaptionConfig(final_hold_ms=120)),
        source_lang="en",
        target_lang="id",
        checkpoint_interval_s=300,
    )
    painted: list[TranscriptSegmentEvent] = []
    window.finalPainted.connect(painted.append)

    def event(
        segment_id: int,
        revision: int,
        status: SegmentStatus,
        text: str,
        translation: str | None = None,
        translation_status: TranslationStatus = TranslationStatus.PENDING,
    ) -> TranscriptSegmentEvent:
        return TranscriptSegmentEvent(
            segment_id=segment_id,
            revision=revision,
            status=status,
            t_start_ms=0,
            t_end_ms=2_000,
            source_lang="en",
            target_lang="id",
            text=text,
            translation=translation,
            translation_status=translation_status,
        )

    try:
        window.show_segment(
            event(
                1,
                2,
                SegmentStatus.FINAL,
                "First sentence",
                "Kalimat pertama",
                TranslationStatus.OK,
            )
        )
        assert window.current_text() == ("First sentence", "Kalimat pertama")
        assert len(painted) == 1

        # The next utterance arrives while the translated final is still held.
        window.show_segment(
            event(2, 1, SegmentStatus.PARTIAL, "Second sentence")
        )
        assert window.current_text() == ("First sentence", "Kalimat pertama")
        assert len(painted) == 1

        time.sleep(0.15)
        qt_app.processEvents()
        assert window.current_text() == ("Second sentence", "")
        assert len(painted) == 1
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_new_resolved_final_stops_the_old_hold(
    qt_app: QApplication,
) -> None:
    window = CaptionWindow(
        UiConfig(caption=CaptionConfig(final_hold_ms=60)),
        source_lang="en",
        target_lang="id",
        checkpoint_interval_s=300,
    )

    def event(
        segment_id: int,
        revision: int,
        status: SegmentStatus,
        text: str,
        translation: str | None = None,
        translation_status: TranslationStatus = TranslationStatus.PENDING,
    ) -> TranscriptSegmentEvent:
        return TranscriptSegmentEvent(
            segment_id=segment_id,
            revision=revision,
            status=status,
            t_start_ms=0,
            t_end_ms=2_000,
            source_lang="en",
            target_lang="id",
            text=text,
            translation=translation,
            translation_status=translation_status,
        )

    try:
        window.show_segment(
            event(
                1,
                2,
                SegmentStatus.FINAL,
                "First",
                "Satu",
                TranslationStatus.OK,
            )
        )
        # Let the first hold expire in time, but do not run the event loop yet:
        # the second final arrives before the stale timer callback is delivered.
        time.sleep(0.09)
        window.show_segment(
            event(
                2,
                2,
                SegmentStatus.FINAL,
                "Second",
                "Dua",
                TranslationStatus.OK,
            )
        )
        assert window.current_text() == ("Second", "Dua")

        qt_app.processEvents()  # deliver the stale first timer
        window.show_segment(
            event(3, 1, SegmentStatus.PARTIAL, "Third")
        )
        assert window.current_text() == ("Second", "Dua")
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_resolved_final_is_painted_during_hold(
    qt_app: QApplication,
) -> None:
    window = CaptionWindow(
        UiConfig(caption=CaptionConfig(final_hold_ms=5_000)),
        source_lang="en",
        target_lang="id",
        checkpoint_interval_s=300,
    )
    painted: list[TranscriptSegmentEvent] = []
    window.finalPainted.connect(painted.append)

    def event(
        segment_id: int,
        revision: int,
        status: SegmentStatus,
        text: str,
        translation: str | None = None,
        translation_status: TranslationStatus = TranslationStatus.PENDING,
    ) -> TranscriptSegmentEvent:
        return TranscriptSegmentEvent(
            segment_id=segment_id,
            revision=revision,
            status=status,
            t_start_ms=0,
            t_end_ms=2_000,
            source_lang="en",
            target_lang="id",
            text=text,
            translation=translation,
            translation_status=translation_status,
        )

    try:
        window.show_segment(
            event(
                1,
                2,
                SegmentStatus.FINAL,
                "First",
                "Satu",
                TranslationStatus.OK,
            )
        )
        # A resolved final must paint even though the previous hold is active:
        # otherwise its render.ack would never reach metrics.csv.
        window.show_segment(
            event(
                2,
                2,
                SegmentStatus.FINAL,
                "Second",
                "Dua",
                TranslationStatus.OK,
            )
        )
        assert window.current_text() == ("Second", "Dua")
        assert len(painted) == 2

        window.show_segment(event(3, 1, SegmentStatus.PARTIAL, "Third"))
        assert window.current_text() == ("Second", "Dua")
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_early_hold_timeout_rearms_instead_of_dropping_pending(
    qt_app: QApplication,
) -> None:
    window = CaptionWindow(
        UiConfig(caption=CaptionConfig(final_hold_ms=5_000)),
        source_lang="en",
        target_lang="id",
        checkpoint_interval_s=300,
    )

    def event(
        segment_id: int,
        revision: int,
        status: SegmentStatus,
        text: str,
        translation: str | None = None,
        translation_status: TranslationStatus = TranslationStatus.PENDING,
    ) -> TranscriptSegmentEvent:
        return TranscriptSegmentEvent(
            segment_id=segment_id,
            revision=revision,
            status=status,
            t_start_ms=0,
            t_end_ms=2_000,
            source_lang="en",
            target_lang="id",
            text=text,
            translation=translation,
            translation_status=translation_status,
        )

    try:
        window.show_segment(
            event(
                1,
                2,
                SegmentStatus.FINAL,
                "First",
                "Satu",
                TranslationStatus.OK,
            )
        )
        window.show_segment(event(2, 1, SegmentStatus.PARTIAL, "Second"))

        # Simulate a CoarseTimer firing early: the pending preview must stay
        # buffered and the hold timer must be re-armed for the remaining time.
        window._hold_timer.timeout.emit()
        assert window.current_text() == ("First", "Satu")
        assert window._hold_timer.isActive()
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_expired_hold_uses_the_newest_preview(
    qt_app: QApplication,
) -> None:
    window = CaptionWindow(
        UiConfig(caption=CaptionConfig(final_hold_ms=60)),
        source_lang="en",
        target_lang="id",
        checkpoint_interval_s=300,
    )

    def event(
        segment_id: int,
        revision: int,
        status: SegmentStatus,
        text: str,
        translation: str | None = None,
        translation_status: TranslationStatus = TranslationStatus.PENDING,
    ) -> TranscriptSegmentEvent:
        return TranscriptSegmentEvent(
            segment_id=segment_id,
            revision=revision,
            status=status,
            t_start_ms=0,
            t_end_ms=2_000,
            source_lang="en",
            target_lang="id",
            text=text,
            translation=translation,
            translation_status=translation_status,
        )

    try:
        window.show_segment(
            event(
                1,
                2,
                SegmentStatus.FINAL,
                "First",
                "Satu",
                TranslationStatus.OK,
            )
        )
        window.show_segment(event(2, 1, SegmentStatus.PARTIAL, "Second"))

        # Let the hold expire in time without delivering the timer callback,
        # then paint a newer preview before the stale callback can run.
        time.sleep(0.09)
        window.show_segment(event(3, 1, SegmentStatus.PARTIAL, "Third"))
        assert window.current_text() == ("Third", "")

        qt_app.processEvents()
        assert window.current_text() == ("Third", "")
        assert not window._hold_timer.isActive()
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_caption_fits_long_text_while_hidden_and_after_show(
    qt_app: QApplication,
) -> None:
    window = CaptionWindow(
        UiConfig(), source_lang="en", target_lang="id", checkpoint_interval_s=300
    )
    try:
        long_text = " ".join(f"word{index}" for index in range(200))
        window.show_segment(
            TranscriptSegmentEvent(
                segment_id=5,
                revision=2,
                status=SegmentStatus.FINAL,
                t_start_ms=0,
                t_end_ms=10_000,
                source_lang="en",
                target_lang="id",
                text=long_text,
                translation=long_text,
                translation_status=TranslationStatus.OK,
            )
        )
        hidden_translation = window.current_text()[1]
        assert hidden_translation.endswith("…")

        window.show()
        qt_app.processEvents()
        shown_translation = window.current_text()[1]
        assert shown_translation.endswith("…")
        assert len(shown_translation) <= len(long_text)
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()
