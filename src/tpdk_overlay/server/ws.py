"""WebSocket connection manager and command dispatch (FR-09).

One WebSocket carries commands from the overlay and every pipeline event back.
Each connection gets its own bounded subscription to the
:class:`~tpdk_overlay.pipeline.events.EventBus`; a slow client is disconnected
instead of blocking producers. All protocol messages go through
``parse_client_command`` / the pydantic models in ``contracts.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from tpdk_overlay.contracts import (
    ClientCommand,
    NoticeEvent,
    NoticeLevel,
    OverlayToggleCommand,
    PingCommand,
    PongEvent,
    RegionUpdateCommand,
    RenderAckCommand,
    SessionPauseCommand,
    SessionResumeCommand,
    SessionStartCommand,
    SessionStopCommand,
    SettingsUpdateCommand,
    SummaryFinalRequestCommand,
    parse_client_command,
)
from tpdk_overlay.pipeline.events import EventBus, Subscription, SubscriptionClosed
from tpdk_overlay.pipeline.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

SendText = Callable[[str], Awaitable[None]]


class ConnectionManager:
    """Serve the ``/ws`` endpoint: event fan-out plus command dispatch."""

    def __init__(self, bus: EventBus, orchestrator: Orchestrator) -> None:
        self._bus = bus
        self._orchestrator = orchestrator

    async def serve(self, socket: WebSocket) -> None:
        await socket.accept()
        subscription = self._bus.subscribe()
        send_lock = asyncio.Lock()

        async def send_text(payload: str) -> None:
            async with send_lock:
                await socket.send_text(payload)

        try:
            await send_text(self._orchestrator.session_state_event().model_dump_json())
            receiver = asyncio.create_task(
                self._receive_loop(socket, send_text), name="tpdk-ws-receive"
            )
            sender = asyncio.create_task(
                self._forward_events(subscription, send_text), name="tpdk-ws-send"
            )
            done, pending = await asyncio.wait(
                {receiver, sender}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                with contextlib.suppress(WebSocketDisconnect):
                    task.result()
        except WebSocketDisconnect:
            pass
        except SubscriptionClosed as exc:
            logger.warning("dropping WebSocket client: %s", exc)
        finally:
            self._bus.unsubscribe(subscription)

    async def _receive_loop(self, socket: WebSocket, send_text: SendText) -> None:
        while True:
            raw = await socket.receive_text()
            try:
                command = parse_client_command(raw)
            except ValidationError as exc:
                first = exc.errors()[0].get("msg", "invalid command")
                await self._send(
                    send_text,
                    NoticeEvent(
                        level=NoticeLevel.WARNING,
                        code="invalid_command",
                        message=str(first),
                    ),
                )
                continue
            if isinstance(command, PingCommand):
                await self._send(send_text, PongEvent(echo_ts_ms=command.client_ts_ms))
                continue
            try:
                await self._dispatch(command)
            except WebSocketDisconnect:
                raise
            except Exception as exc:  # noqa: BLE001 - a command failure is a notice
                logger.exception("command failed: %s", command.type)
                await self._send(
                    send_text,
                    NoticeEvent(
                        level=NoticeLevel.ERROR,
                        code="command_failed",
                        message=f"{command.type} failed: {exc}",
                    ),
                )

    async def _forward_events(
        self, subscription: Subscription, send_text: SendText
    ) -> None:
        while True:
            event = await subscription.get()
            await send_text(event.model_dump_json())

    async def _dispatch(self, command: ClientCommand) -> None:
        orchestrator = self._orchestrator
        if isinstance(command, SessionStartCommand):
            await orchestrator.start(command)
        elif isinstance(command, SessionStopCommand):
            await orchestrator.stop()
        elif isinstance(command, SessionPauseCommand):
            orchestrator.pause()
        elif isinstance(command, SessionResumeCommand):
            orchestrator.resume()
        elif isinstance(command, SettingsUpdateCommand):
            orchestrator.update_settings(command.patch)
        elif isinstance(command, RegionUpdateCommand):
            orchestrator.update_region(command.region)
        elif isinstance(command, OverlayToggleCommand):
            orchestrator.set_overlay_visibility(command.target, command.visible)
        elif isinstance(command, SummaryFinalRequestCommand):
            orchestrator.request_final_summary()
        elif isinstance(command, RenderAckCommand):
            orchestrator.record_render_ack(
                command.event_id,
                command.painted_ts_ms,
                command.caption_rect,
            )
        else:  # pragma: no cover - contract union is exhaustive
            logger.warning("unhandled command: %s", command.type)

    @staticmethod
    async def _send(send_text: SendText, event: NoticeEvent | PongEvent) -> None:
        await send_text(event.model_dump_json())


__all__ = ["ConnectionManager"]
