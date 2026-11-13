"""Overlay lifecycle wiring: auto-start once on the first backend state."""

from __future__ import annotations

from tpdk_overlay.config import UiConfig
from tpdk_overlay.contracts import (
    OverlayToggleCommand,
    SessionStartCommand,
    SessionState,
    SessionStateEvent,
)
from tpdk_overlay.ui.app import OverlayApp


class _Caption:
    def __init__(self) -> None:
        self.state: SessionState | None = None

    def set_session_state(self, state: SessionState) -> None:
        self.state = state


class _History:
    def __init__(self) -> None:
        self.session_id: str | None = None

    def set_session(self, session_id: str) -> None:
        self.session_id = session_id

    def show(self) -> None:
        pass


class _FakeOverlay:
    _on_session_state = OverlayApp._on_session_state
    _maybe_auto_start = OverlayApp._maybe_auto_start

    def __init__(self, *, pending: bool) -> None:
        self._auto_start_pending = pending
        self._state = SessionState.IDLE
        self._session_id: str | None = None
        self._caption = _Caption()
        self._capture_hidden: set[str] = set()
        self._history = _History()
        self.started = 0

    def _update_actions(self) -> None:
        pass

    def _start_session(self) -> None:
        self.started += 1


def test_auto_start_defaults_on() -> None:
    assert UiConfig().auto_start is True


def test_initial_idle_state_auto_starts_once() -> None:
    overlay = _FakeOverlay(pending=True)
    state = SessionStateEvent(state=SessionState.IDLE, session_id="session-1")

    overlay._on_session_state(state)
    assert overlay.started == 1
    assert overlay._caption.state is SessionState.IDLE
    assert overlay._history.session_id == "session-1"

    overlay._on_session_state(
        SessionStateEvent(state=SessionState.IDLE, session_id="session-1")
    )
    assert overlay.started == 1


def test_initial_running_state_does_not_auto_start() -> None:
    overlay = _FakeOverlay(pending=True)

    overlay._on_session_state(SessionStateEvent(state=SessionState.RUNNING))
    assert overlay.started == 0
    assert overlay._auto_start_pending is False


def test_auto_start_disabled_sends_nothing() -> None:
    overlay = _FakeOverlay(pending=False)

    overlay._on_session_state(SessionStateEvent(state=SessionState.IDLE))
    assert overlay.started == 0


def test_start_session_sends_configured_command() -> None:
    class _Sender:
        _start_session = OverlayApp._start_session
        _source_lang = "en"
        _target_lang = "id"
        _checkpoint_s = 300
        _region = None
        _auto_start_pending = True

        def __init__(self) -> None:
            self.sent: list[SessionStartCommand] = []

        def _send(self, command: SessionStartCommand) -> None:
            self.sent.append(command)

    sender = _Sender()
    sender._start_session()

    assert sender._auto_start_pending is False
    assert len(sender.sent) == 1
    command = sender.sent[0]
    assert command.source_lang == "en"
    assert command.target_lang == "id"
    assert command.checkpoint_interval_s == 300
    assert command.region is None


class _Client:
    is_connected = True


class _OcrWindow:
    def __init__(self) -> None:
        self.visible: bool | None = None

    def set_overlay_visible(self, visible: bool) -> None:
        self.visible = visible


class _VisibilityState:
    _set_ocr_visible = OverlayApp._set_ocr_visible

    def __init__(self) -> None:
        self._ocr = _OcrWindow()
        self._ocr_visible = True
        self._client = _Client()
        self.sent: list[object] = []

    def _update_actions(self) -> None:
        pass

    def _send(self, command: object) -> None:
        self.sent.append(command)


def test_ocr_visibility_toggle_tracks_intent() -> None:
    overlay = _VisibilityState()

    overlay._set_ocr_visible(False)

    assert overlay._ocr_visible is False
    assert overlay._ocr.visible is False
    command = overlay.sent[-1]
    assert isinstance(command, OverlayToggleCommand)
    assert command.target == "ocr"
    assert command.visible is False
