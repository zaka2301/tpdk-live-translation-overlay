"""Backend surface: REST health check plus the WebSocket command/event channel.

The session orchestrator is not wired yet. This module fixes the transport and
protocol behavior (connect -> idle state, ping -> pong, unknown command ->
notice) so the UI and adapters can be developed against a stable surface.
AGENT.md phase 3 replaces :func:`_dispatch` with the real orchestrator.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from tpdk_overlay import __version__
from tpdk_overlay.config import AppConfig, load_config
from tpdk_overlay.contracts import (
    PROTOCOL_VERSION,
    ClientCommand,
    NoticeEvent,
    PingCommand,
    PongEvent,
    SessionState,
    SessionStateEvent,
    parse_client_command,
)
from tpdk_overlay.utils.logging import setup_logging

logger = logging.getLogger(__name__)


async def _dispatch(socket: WebSocket, command: ClientCommand) -> None:
    if isinstance(command, PingCommand):
        await socket.send_text(PongEvent(echo_ts_ms=command.client_ts_ms).model_dump_json())
        return
    logger.warning("command not wired: %s", command.type)
    await socket.send_text(
        NoticeEvent(
            level="info",
            code="command_unavailable",
            message=(
                f"{command.type} is not wired yet; the pipeline lands in AGENT.md phase 3"
            ),
        ).model_dump_json()
    )


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="TPdK Overlay Backend", version=__version__)
    app.state.config = cfg

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "protocol": PROTOCOL_VERSION,
            "state": SessionState.IDLE.value,
        }

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        await socket.send_text(SessionStateEvent(state=SessionState.IDLE).model_dump_json())
        try:
            while True:
                raw = await socket.receive_text()
                try:
                    command = parse_client_command(raw)
                except ValidationError as exc:
                    first = exc.errors()[0].get("msg", "invalid command")
                    await socket.send_text(
                        NoticeEvent(
                            level="warning",
                            code="invalid_command",
                            message=str(first),
                        ).model_dump_json()
                    )
                    continue
                await _dispatch(socket, command)
        except WebSocketDisconnect:
            return

    return app


app = create_app()


def main() -> None:
    import uvicorn

    config = load_config()
    setup_logging(config.app.log_level, config.logs_dir)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(config), host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
