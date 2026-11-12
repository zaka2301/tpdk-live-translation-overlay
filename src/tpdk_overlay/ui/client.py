"""WebSocket client for the overlay UI (FR-04, FR-09).

The client owns a private asyncio loop on a daemon thread, so connecting,
receiving, and reconnecting never block Qt's GUI thread. Qt signals are the only
crossing point: :attr:`OverlayClient.eventReceived` carries parsed contract
models, while ``connected``/``disconnected`` drive the status line.

Reconnects use exponential backoff from 0.5 s up to 5 s, so a restarted backend
is picked up within the NFR-06 recovery budget. Commands sent from the GUI are
handed to the loop with ``call_soon_threadsafe`` and written by a dedicated
sender task; a bounded outbox drops the oldest command instead of blocking.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import websockets
from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from tpdk_overlay.contracts import ClientCommand, parse_server_event

logger = logging.getLogger(__name__)

_DEFAULT_OUTBOX_SIZE = 256


@dataclass(slots=True)
class Backoff:
    """Exponential reconnect delay capped at ``maximum_s`` (default 5 s)."""

    initial_s: float = 0.5
    factor: float = 2.0
    maximum_s: float = 5.0
    _current: float = field(init=False)

    def __post_init__(self) -> None:
        self._current = self.initial_s

    def reset(self) -> None:
        self._current = self.initial_s

    def next(self) -> float:
        delay = self._current
        self._current = min(self._current * self.factor, self.maximum_s)
        return delay


def _describe(exc: BaseException) -> str:
    if isinstance(exc, websockets.exceptions.WebSocketException):
        return str(exc) or type(exc).__name__
    return f"{type(exc).__name__}: {exc}"


class OverlayClient(QObject):
    """Threaded WebSocket client speaking protocol v1.0.

    ``start()`` spawns the loop thread; ``send()`` is safe from the GUI thread;
    ``stop()`` signals the loop and joins for at most three seconds.
    """

    connected = Signal()
    disconnected = Signal(str)
    eventReceived = Signal(object)
    protocolError = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        url: str,
        *,
        parent: QObject | None = None,
        outbox_size: int = _DEFAULT_OUTBOX_SIZE,
        connect_timeout_s: float = 4.0,
        backoff: Backoff | None = None,
    ) -> None:
        super().__init__(parent)
        self._url = url
        self._outbox_size = max(1, outbox_size)
        self._connect_timeout_s = connect_timeout_s
        self._backoff = backoff or Backoff()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._outbox: asyncio.Queue[str] | None = None
        self._stop_event: asyncio.Event | None = None
        self._stop_requested = threading.Event()
        self._connected = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._thread_main, name="tpdk-overlay-ws", daemon=True
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 3.0) -> None:
        self._stop_requested.set()
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._set_stop_event)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=join_timeout_s)

    # -- sending -----------------------------------------------------------

    def send(self, command: ClientCommand) -> bool:
        """Queue one command; returns ``False`` before the loop is running."""

        loop = self._loop
        if loop is None:
            return False
        payload = command.model_dump_json()
        try:
            loop.call_soon_threadsafe(self._enqueue, payload)
        except RuntimeError:
            return False
        return True

    def _enqueue(self, payload: str) -> None:
        outbox = self._outbox
        if outbox is None:
            return
        if outbox.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                outbox.get_nowait()
        outbox.put_nowait(payload)

    # -- worker thread -----------------------------------------------------

    def _set_stop_event(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._outbox = asyncio.Queue(maxsize=self._outbox_size)
        self._stop_event = asyncio.Event()
        self._loop = loop
        if self._stop_requested.is_set():
            self._stop_event.set()
        try:
            loop.run_until_complete(self._main())
        except Exception:  # noqa: BLE001 - the thread must not die silently
            logger.exception("overlay WebSocket thread crashed")
        finally:
            self._loop = None
            self._outbox = None
            self._connected = False
            with contextlib.suppress(Exception):
                loop.close()
            self.stopped.emit()

    async def _main(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            reason = "connection closed"
            try:
                async with websockets.connect(
                    self._url,
                    open_timeout=self._connect_timeout_s,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=2,
                    max_queue=32,
                ) as websocket:
                    self._connected = True
                    self._backoff.reset()
                    self.connected.emit()
                    await self._pump(websocket)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any failure means reconnect
                reason = _describe(exc)
                logger.info("overlay WebSocket unavailable: %s", reason)
            finally:
                self._connected = False

            if self._stop_event.is_set():
                break
            self.disconnected.emit(reason)
            delay = self._backoff.next()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)

    async def _pump(self, websocket: Any) -> None:
        """Run reader/writer until the socket closes or stop is requested."""

        assert self._stop_event is not None
        reader = asyncio.create_task(self._read_loop(websocket), name="tpdk-ws-read")
        writer = asyncio.create_task(self._write_loop(websocket), name="tpdk-ws-write")
        stopper = asyncio.create_task(self._stop_event.wait(), name="tpdk-ws-stop")
        try:
            done, _ = await asyncio.wait(
                {reader, writer, stopper}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in (reader, writer, stopper):
                task.cancel()
            await asyncio.gather(reader, writer, stopper, return_exceptions=True)

        for task in done:
            if task is stopper or task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                raise error

    async def _read_loop(self, websocket: Any) -> None:
        async for raw in websocket:
            try:
                event = parse_server_event(raw)
            except ValidationError as exc:
                errors = exc.errors()
                reason = str(errors[0].get("msg", exc)) if errors else str(exc)
                logger.warning("dropping invalid server event: %s", reason)
                self.protocolError.emit(reason)
                continue
            self.eventReceived.emit(event)

    async def _write_loop(self, websocket: Any) -> None:
        outbox = self._outbox
        if outbox is None:
            return
        while True:
            payload = await outbox.get()
            await websocket.send(payload)


__all__ = ["Backoff", "OverlayClient"]
