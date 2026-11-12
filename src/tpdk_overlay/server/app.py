"""Backend surface: REST health check plus the WebSocket command/event channel.

The transport is fixed here (one ``/ws`` for commands and events, REST only for
health); :class:`~tpdk_overlay.pipeline.orchestrator.Orchestrator` owns the
pipeline. ``create_app`` accepts a ``ports_factory`` so integration tests inject
fake adapters without a GPU or a running ASR service.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket

from tpdk_overlay import __version__
from tpdk_overlay.config import AppConfig, load_config
from tpdk_overlay.contracts import PROTOCOL_VERSION
from tpdk_overlay.pipeline.events import EventBus
from tpdk_overlay.pipeline.orchestrator import (
    Orchestrator,
    PipelinePorts,
    build_default_ports,
)
from tpdk_overlay.server.ws import ConnectionManager
from tpdk_overlay.utils.logging import setup_logging

logger = logging.getLogger(__name__)

PortsFactory = Callable[[AppConfig], PipelinePorts]


class _SharedTranslator:
    """Session-scoped translator handle backed by one process-wide model.

    ``Orchestrator._release_ports`` calls ``close()`` when a session stops.
    For the shared model that must be a no-op; the backend closes the real
    translator in the app lifespan so the next Start is fast.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def warmup(self) -> None:
        warmup = getattr(self._inner, "warmup", None)
        if callable(warmup):
            warmup()

    def translate(
        self, texts: Any, *, source_lang: str, target_lang: str
    ) -> list[str]:
        return self._inner.translate(
            texts, source_lang=source_lang, target_lang=target_lang
        )

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _SharedOcrEngine:
    """Session-scoped OCR handle backed by one process-wide PaddleOCR.

    Loading PP-OCRv6 takes ~18 s; recreating it for every Start would delay the
    first slide and churn memory. ``close`` is a no-op per session; the model
    cache closes the real engine at backend shutdown.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def warmup(self) -> None:
        warmup = getattr(self._inner, "warmup", None)
        if callable(warmup):
            warmup()

    def recognize(self, image: Any) -> Any:
        return self._inner.recognize(image)

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ModelCache:
    """Load expensive local models once per backend process, not per session."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._inner: Any | None = None
        self._key: str | None = None
        self._ocr_inner: Any | None = None
        self._ocr_key: str | None = None

    def _inner_for(self, config: AppConfig) -> Any:
        try:
            key = config.mt.model_dump_json()
        except AttributeError:  # pragma: no cover - config is pydantic today
            key = repr(config.mt)
        with self._lock:
            if self._inner is None or self._key != key:
                if self._inner is not None:
                    threading.Thread(
                        target=self._close_inner,
                        args=(self._inner,),
                        name="tpdk-mt-close",
                        daemon=True,
                    ).start()
                from tpdk_overlay.mt import create_translator

                self._inner = create_translator(config.mt)
                self._key = key
            return self._inner

    def _ocr_for(self, config: AppConfig) -> Any:
        try:
            key = config.ocr.model_dump_json()
        except AttributeError:  # pragma: no cover - config is pydantic today
            key = repr(config.ocr)
        with self._lock:
            if self._ocr_inner is None or self._ocr_key != key:
                if self._ocr_inner is not None:
                    threading.Thread(
                        target=self._close_inner,
                        args=(self._ocr_inner,),
                        name="tpdk-ocr-close",
                        daemon=True,
                    ).start()
                from tpdk_overlay.ocr.paddle import PaddleOcrEngine

                self._ocr_inner = PaddleOcrEngine(config.ocr)
                self._ocr_key = key
            return self._ocr_inner

    def ports_factory(self, config: AppConfig) -> PipelinePorts:
        ocr_engine = (
            _SharedOcrEngine(self._ocr_for(config))
            if config.screen.enabled and config.ocr.enabled
            else None
        )
        return build_default_ports(
            config,
            translator=_SharedTranslator(self._inner_for(config)),
            ocr_engine=ocr_engine,
        )

    def preload(self) -> None:
        """Warm the models while the overlay is coming up, off the event loop."""

        translator = _SharedTranslator(self._inner_for(self._config))
        threading.Thread(
            target=self._preload,
            args=("translator", translator),
            name="tpdk-mt-preload",
            daemon=True,
        ).start()
        if self._config.screen.enabled and self._config.ocr.enabled:
            ocr = _SharedOcrEngine(self._ocr_for(self._config))
            threading.Thread(
                target=self._preload,
                args=("OCR", ocr),
                name="tpdk-ocr-preload",
                daemon=True,
            ).start()

    @staticmethod
    def _preload(name: str, adapter: Any) -> None:
        try:
            adapter.warmup()
        except Exception:  # noqa: BLE001 - the first real pass retries and reports
            logger.exception("%s preload failed", name)

    def close(self) -> None:
        with self._lock:
            inner = self._inner
            ocr_inner = self._ocr_inner
            self._inner = None
            self._key = None
            self._ocr_inner = None
            self._ocr_key = None
        for name, model in (("tpdk-mt-close", inner), ("tpdk-ocr-close", ocr_inner)):
            if model is not None:
                threading.Thread(
                    target=self._close_inner,
                    args=(model,),
                    name=name,
                    daemon=True,
                ).start()

    @staticmethod
    def _close_inner(inner: Any) -> None:
        closer = getattr(inner, "close", None)
        if callable(closer):
            with contextlib.suppress(Exception):
                closer()


def create_app(
    config: AppConfig | None = None,
    *,
    ports_factory: PortsFactory | None = None,
) -> FastAPI:
    cfg = config or load_config()
    bus = EventBus()
    model_cache: _ModelCache | None = None
    if ports_factory is None:
        model_cache = _ModelCache(cfg)
        ports_factory = model_cache.ports_factory
    orchestrator = Orchestrator(cfg, bus=bus, ports_factory=ports_factory)
    connections = ConnectionManager(bus, orchestrator)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if model_cache is not None:
            model_cache.preload()
        yield
        await orchestrator.shutdown()
        if model_cache is not None:
            model_cache.close()

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
