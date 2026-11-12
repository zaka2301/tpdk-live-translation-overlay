"""Backend surface: REST health check plus the WebSocket command/event channel.

The transport is fixed here (one ``/ws`` for commands and events, REST only for
health); :class:`~tpdk_overlay.pipeline.orchestrator.Orchestrator` owns the
pipeline. ``create_app`` accepts a ``ports_factory`` so integration tests inject
fake adapters without a GPU or a running ASR service.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket

from tpdk_overlay import __version__
from tpdk_overlay.config import AppConfig, load_config
from tpdk_overlay.contracts import PROTOCOL_VERSION
from tpdk_overlay.pipeline.events import EventBus
from tpdk_overlay.pipeline.orchestrator import Orchestrator, PipelinePorts
from tpdk_overlay.server.ws import ConnectionManager
from tpdk_overlay.utils.logging import setup_logging

logger = logging.getLogger(__name__)

PortsFactory = Callable[[AppConfig], PipelinePorts]


def create_app(
    config: AppConfig | None = None,
    *,
    ports_factory: PortsFactory | None = None,
) -> FastAPI:
    cfg = config or load_config()
    bus = EventBus()
    orchestrator = Orchestrator(cfg, bus=bus, ports_factory=ports_factory)
    connections = ConnectionManager(bus, orchestrator)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await orchestrator.shutdown()

    app = FastAPI(title="TPdK Overlay Backend", version=__version__, lifespan=lifespan)
    app.state.config = cfg
    app.state.bus = bus
    app.state.orchestrator = orchestrator
    app.state.connections = connections

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "protocol": PROTOCOL_VERSION,
            "state": orchestrator.state.value,
            "session_id": orchestrator.session_id,
        }

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await connections.serve(socket)

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


__all__ = ["ConnectionManager", "create_app", "main"]
