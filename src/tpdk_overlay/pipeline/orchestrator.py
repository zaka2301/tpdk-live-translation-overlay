"""Session orchestration: capture -> VAD -> ASR -> MT, OCR, checkpoints.

The orchestrator owns every thread and background task of a session. It runs on
the uvicorn event loop, starts the adapter threads, and publishes every client
event through the :class:`~tpdk_overlay.pipeline.events.EventBus`:

* capture thread: reads 16 kHz frames, feeds Silero VAD, and streams one ASR
  stream per utterance (opened at :class:`SpeechStart`);
* ASR thread: calls ``push``/``finish`` on the ASR client, publishes partial
  previews (coalesced per segment) and the authoritative final;
* MT thread: translates final segments and updates the same segment revision;
* OCR thread: captures the configured region, skips unchanged slides, OCRs,
  translates blocks, and publishes :class:`OcrFrameEvent`;
* async checkpoint task: summarizes each interval with the injected summarizer
  and degrades instead of stopping when the cloud is unavailable.

Adapters are injected as :class:`PipelinePorts`, so integration tests replace
them with fakes and later build phases (screen/OCR/summarizer) only need to
extend :func:`build_default_ports`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np

from tpdk_overlay.asr.errors import AsrTimeout, AsrUnavailable
from tpdk_overlay.audio.vad import VAD_FRAME_MS, VAD_FRAME_SAMPLES, SpeechStart, Utterance
from tpdk_overlay.config import AppConfig
from tpdk_overlay.contracts import (
    BBox,
    NoticeEvent,
    NoticeLevel,
    OcrBlock,
    OcrFrameEvent,
    ScreenRegion,
    SegmentStatus,
    SessionStartCommand,
    SessionState,
    SessionStateEvent,
    SettingsPatch,
    SummaryCheckpointEvent,
    SummaryFinalEvent,
    TranscriptSegmentEvent,
    TranslationStatus,
)
from tpdk_overlay.mt.languages import to_nllb
from tpdk_overlay.pipeline.events import EventBus
from tpdk_overlay.pipeline.metrics import MetricsRecorder
from tpdk_overlay.pipeline.session import SessionClock, SessionStore
from tpdk_overlay.ports import (
    AsrClient,
    AsrUpdate,
    AudioSource,
    CheckpointSummaryRequest,
    FinalSummaryRequest,
    OcrEngine,
    OcrLine,
    ScreenSource,
    Summarizer,
    Translator,
)
from tpdk_overlay.utils.clock import Stopwatch, now_ms

logger = logging.getLogger(__name__)

_ASR_QUEUE_SIZE = 8
# MT jobs are small; a deeper queue absorbs the one-time NLLB load without
# dropping translations while source captions are already flowing.
_MT_QUEUE_SIZE = 64
_STOP_BUDGET_S = 3.0
_PUT_BUDGET_S = 5.0
_OCR_IDLE_WAIT_S = 0.2
# Send audio to the ASR worker in 0.25 s chunks instead of 32 ms frames: a
# bounded queue then holds ~2 s of audio before it applies backpressure.
_ASR_CHUNK_SAMPLES = 4_000

SegmenterFactory = Callable[[], Any]
PortsFactory = Callable[[AppConfig], "PipelinePorts"]
BlockGrouper = Callable[[Sequence[OcrLine]], Sequence[OcrBlock]]

_SENTINEL = object()

_LOG_LEVEL = {
    NoticeLevel.INFO: logging.INFO,
    NoticeLevel.WARNING: logging.WARNING,
    NoticeLevel.ERROR: logging.ERROR,
}


@dataclass(frozen=True, slots=True)
class PipelinePorts:
    """Adapters behind the orchestrator seams.

    ``screen_source``/``ocr_engine``/``summarizer`` are optional so phase 3 runs
    transcript-only; later phases supply the real adapters. A supplied
    ``segmenter_factory`` replaces Silero (integration tests use it).
    """

    audio_source: AudioSource
    asr_client: AsrClient
    translator: Translator
    segmenter_factory: SegmenterFactory | None = None
    screen_source: ScreenSource | None = None
    ocr_engine: OcrEngine | None = None
    change_detector: Any | None = None
    summarizer: Summarizer | None = None
    block_grouper: BlockGrouper | None = None


@dataclass(slots=True)
class SessionSettings:
    source_lang: str
    target_lang: str
    checkpoint_interval_s: int
    region: ScreenRegion | None = None
    ocr_enabled: bool = True
    screen_enabled: bool = True


@dataclass(frozen=True, slots=True)
class _OpenStream:
    segment_id: int
    audio: np.ndarray
    t_start_ms: int


@dataclass(frozen=True, slots=True)
class _AudioFrame:
    frame: np.ndarray
    t_end_ms: int


@dataclass(frozen=True, slots=True)
class _CloseStream:
    t_end_ms: int
    audio_end_ts_ms: int


@dataclass(frozen=True, slots=True)
class _TranslationJob:
    segment_id: int
    revision: int
    text: str
    t_start_ms: int
    t_end_ms: int
    audio_end_ts_ms: int
    source_lang: str
    target_lang: str
    asr_latency_ms: int
    audio_end_to_text_ms: int


def build_default_ports(
    config: AppConfig, *, translator: Translator | None = None
) -> PipelinePorts:
    """Build the production adapters; screen/OCR/summarizer appear in later phases.

    The optional imports are guarded so the phase 3 backend runs transcript-only
    until the phase 5/6 modules land, without a speculative import error.
    """

    from tpdk_overlay.asr import create_asr_client
    from tpdk_overlay.audio.capture import create_audio_source
    from tpdk_overlay.mt import create_translator

    screen_source: ScreenSource | None = None
    change_detector: Any | None = None
    ocr_engine: OcrEngine | None = None
    if config.screen.enabled and config.ocr.enabled:
        try:
            from tpdk_overlay.ocr.paddle import PaddleOcrEngine
            from tpdk_overlay.screen.capture import MssScreenSource
            from tpdk_overlay.screen.change import SlideChangeDetector
        except ImportError:
            logger.debug("screen/OCR adapters are not available yet; OCR stays off")
        else:
            screen_source = MssScreenSource(config.screen)
            change_detector = SlideChangeDetector(config.screen)
            ocr_engine = PaddleOcrEngine(config.ocr)

    summarizer: Summarizer | None = None
    if config.summary.enabled and config.summary.api_key():
        try:
            from tpdk_overlay.summarize.gemini import GeminiSummarizer
        except ImportError:
            logger.debug("summarizer adapter is not available yet; summaries stay off")
        else:
            summarizer = GeminiSummarizer(config.summary)

    return PipelinePorts(
        audio_source=create_audio_source(config.audio),
        asr_client=create_asr_client(config.asr),
        translator=translator or create_translator(config.mt),
        screen_source=screen_source,
        ocr_engine=ocr_engine,
        change_detector=change_detector,
        summarizer=summarizer,
    )


def _block_id(bbox: BBox, text: str) -> str:
    payload = f"{bbox.x}:{bbox.y}:{bbox.w}:{bbox.h}:{text}".encode()
    return hashlib.blake2s(payload, digest_size=8).hexdigest()


class Orchestrator:
    """Own one session's threads, lifecycle, and event stream."""

    def __init__(
        self,
        config: AppConfig,
        *,
        bus: EventBus | None = None,
        ports_factory: PortsFactory | None = None,
    ) -> None:
        self._config = config
        # ``session.start`` may override the capture device; ``start`` replaces
        # this with a config copy carrying the command overrides.
        self._ports_config = config
        # ``EventBus`` defines ``__len__``, so an empty bus is falsy; do not
        # use ``or`` here or the app and the orchestrator get separate buses.
        self._bus = bus if bus is not None else EventBus()
        self._ports_factory = ports_factory or build_default_ports

        self._state = SessionState.IDLE
        self._settings: SessionSettings | None = None
        self._session: SessionStore | None = None
        self._clock: SessionClock | None = None
        self._start_wall_ms = 0
        self._ports: PipelinePorts | None = None
        self._metrics: MetricsRecorder | None = None
        self._segmenter: Any | None = None

        self._region_override: ScreenRegion | None = None
        self._overlay_visibility: dict[str, bool] = {"ocr": True, "caption": True}

        self._loop: asyncio.AbstractEventLoop | None = None
        self._startup_task: asyncio.Task[None] | None = None
        self._partial_task: asyncio.Task[None] | None = None
        self._checkpoint_task: asyncio.Task[None] | None = None
        self._summary_task: asyncio.Task[None] | None = None
        self._partial_event: asyncio.Event | None = None

        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._asr_queue: queue.Queue[Any] = queue.Queue(maxsize=_ASR_QUEUE_SIZE)
        self._mt_queue: queue.Queue[Any] = queue.Queue(maxsize=_MT_QUEUE_SIZE)
        self._capture_thread: threading.Thread | None = None
        self._asr_thread: threading.Thread | None = None
        self._mt_thread: threading.Thread | None = None
        self._ocr_thread: threading.Thread | None = None

        self._segment_ids = count(1)
        self._frame_ids = count(1)
        self._last_revision: dict[int, int] = {}
        self._revision_lock = threading.Lock()
        self._partial_lock = threading.Lock()
        self._pending_partials: dict[int, TranscriptSegmentEvent] = {}

        self._records_lock = threading.Lock()
        self._final_records: list[dict[str, Any]] = []
        self._final_record_index: dict[int, dict[str, Any]] = {}
        self._slide_records: list[dict[str, Any]] = []
        self._checkpoint_markdowns: list[str] = []
        self._checkpoint_id = 0
        self._last_checkpoint_to_ms = 0

    # -- public surface ----------------------------------------------------

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session.session_id if self._session is not None else None

    def session_state_event(self) -> SessionStateEvent:
        """Current state for a newly connected UI."""

        return SessionStateEvent(
            session_id=self.session_id,
            state=self._state,
            started_at_ms=self._start_wall_ms or None,
            elapsed_ms=self._clock.elapsed_ms() if self._clock is not None else 0,
        )

    async def start(self, command: SessionStartCommand) -> None:
        """Validate the command, create artifacts, and begin startup off-loop."""

        if self._state not in (SessionState.IDLE, SessionState.STOPPED, SessionState.ERROR):
            self._notice(
                NoticeLevel.WARNING,
                "session_active",
                f"a session is already {self._state.value}",
            )
            return

        try:
            to_nllb(command.source_lang)
            to_nllb(command.target_lang)
        except ValueError as exc:
            self._notice(NoticeLevel.ERROR, "unsupported_language", str(exc))
            return

        # A slow NLLB call can outlive stop()'s 3 s budget. Starting a new
        # session would reset the queues/stop event/session id while that
        # worker is still alive, letting it consume the new session's work.
        # Wait for the retired workers first; refuse if they are truly stuck.
        retired = await self._wait_for_retired_workers()
        if retired:
            self._notice(
                NoticeLevel.WARNING,
                "session_stopping",
                "the previous session is still shutting down; try again in a moment",
            )
            return

        checkpoint_interval_s = (
            command.checkpoint_interval_s or self._config.summary.checkpoint_interval_s
        )
        region = command.region or self._region_override or self._config.screen.region
        self._settings = SessionSettings(
            source_lang=command.source_lang,
            target_lang=command.target_lang,
            checkpoint_interval_s=checkpoint_interval_s,
            region=region,
            ocr_enabled=self._config.ocr.enabled,
            screen_enabled=self._config.screen.enabled,
        )
        self._reset_session_state()
        self._ports_config = self._config
        if command.audio_device:
            self._ports_config = self._config.model_copy(
                update={
                    "audio": self._config.audio.model_copy(
                        update={"input_device": command.audio_device}
                    )
                }
            )
        self._loop = asyncio.get_running_loop()
        self._clock = SessionClock()
        self._start_wall_ms = now_ms()
        self._session = SessionStore(
            self._config.data_dir / "sessions",
            source_lang=command.source_lang,
            target_lang=command.target_lang,
            checkpoint_interval_s=checkpoint_interval_s,
            config_snapshot=self._config.model_dump(mode="json"),
            started_at_ms=self._start_wall_ms,
        )
        self._metrics = MetricsRecorder(self._session.dir / "metrics.csv")
        self._transition(SessionState.STARTING)
        self._startup_task = asyncio.create_task(self._startup(), name="tpdk-startup")

    async def stop(self) -> None:
        """Signal every thread and finish within three seconds; idempotent."""

        if self._state in (SessionState.IDLE, SessionState.STOPPED):
            self._notice(NoticeLevel.WARNING, "session_not_running", "no session is running")
            return
        if self._state is SessionState.STOPPING:
            return

        self._transition(SessionState.STOPPING)
        self._stop_event.set()
        self._paused.clear()

        startup = self._startup_task
        self._startup_task = None
        if startup is not None and not startup.done():
            startup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await startup

        await self._cancel_task(self._partial_task)
        self._partial_task = None
        await self._cancel_task(self._checkpoint_task)
        self._checkpoint_task = None

        deadline = time.monotonic() + _STOP_BUDGET_S
        ports = self._ports
        if ports is not None:
            alive = await asyncio.to_thread(self._join_workers, ports, deadline)
            ports_to_release = self._ports
            self._ports = None
            if alive:
                # A worker still holds a local ports reference (one slow model
                # call). Closing its adapter here would be a use-after-close;
                # leave the adapters alive and let the worker finish.
                logger.warning(
                    "workers still running after the stop budget (%s); "
                    "adapters kept alive",
                    ", ".join(alive),
                )
            else:
                remaining = max(0.05, deadline - time.monotonic())
                with contextlib.suppress(asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(
                        asyncio.to_thread(self._release_ports, ports_to_release),
                        timeout=remaining,
                    )

        if self._session is not None:
            self._transition(SessionState.STOPPED)
            self._session.finalize(SessionState.STOPPED.value, ended_at_ms=now_ms())
        else:
            self._transition(SessionState.STOPPED)
        self._segmenter = None

    async def shutdown(self) -> None:
        """Stop an active session and close the event bus at app shutdown."""

        if self._state not in (SessionState.IDLE, SessionState.STOPPED):
            with contextlib.suppress(Exception):
                await self.stop()
        self._bus.close()

    def pause(self) -> None:
        if self._state is not SessionState.RUNNING:
            self._notice(
                NoticeLevel.WARNING,
                "session_not_running",
                "pause requires a running session",
            )
            return
        # The capture loop feeds zeros while paused: VAD timestamps keep
        # tracking wall time and an open utterance closes on silence.
        self._paused.set()
        self._transition(SessionState.PAUSED)

    def resume(self) -> None:
        if self._state is not SessionState.PAUSED:
            self._notice(
                NoticeLevel.WARNING,
                "session_not_running",
                "resume requires a paused session",
            )
            return
        self._paused.clear()
        self._transition(SessionState.RUNNING)

    def update_settings(self, patch: SettingsPatch) -> None:
        if self._settings is None:
            self._notice(
                NoticeLevel.WARNING,
                "session_not_running",
                "settings can only change during a session",
            )
            return
        try:
            if patch.source_lang is not None:
                to_nllb(patch.source_lang)
            if patch.target_lang is not None:
                to_nllb(patch.target_lang)
        except ValueError as exc:
            self._notice(NoticeLevel.WARNING, "unsupported_language", str(exc))
            return

        if patch.source_lang is not None:
            self._settings.source_lang = patch.source_lang
        if patch.target_lang is not None:
            self._settings.target_lang = patch.target_lang
        if patch.checkpoint_interval_s is not None:
            self._settings.checkpoint_interval_s = patch.checkpoint_interval_s
        if patch.ocr_enabled is not None:
            self._settings.ocr_enabled = patch.ocr_enabled
        if patch.screen_enabled is not None:
            self._settings.screen_enabled = patch.screen_enabled
        if self._session is not None:
            self._session.set_languages(
                self._settings.source_lang, self._settings.target_lang
            )

    def update_region(self, region: ScreenRegion) -> None:
        self._region_override = region
        if self._settings is not None:
            self._settings.region = region

    def set_overlay_visibility(self, target: str, visible: bool) -> None:
        """Remember the UI's local caption/OCR visibility toggle."""

        self._overlay_visibility[target] = visible
        logger.debug("overlay %s visibility -> %s", target, visible)

    def record_render_ack(self, event_id: str, painted_ts_ms: int) -> None:
        if self._metrics is not None:
            self._metrics.ack(event_id, painted_ts_ms)

    def request_final_summary(self) -> None:
        """Synthesize the consolidated summary in the background."""

        if self._ports is None or self._ports.summarizer is None or self._session is None:
            self._notice(
                NoticeLevel.WARNING,
                "summary_unavailable",
                "no summarizer is configured; final summary skipped",
            )
            return
        if self._summary_task is not None and not self._summary_task.done():
            self._notice(
                NoticeLevel.INFO,
                "summary_pending",
                "the final summary is already being generated",
            )
            return
        ports = self._ports
        session = self._session
        settings = self._settings
        with self._records_lock:
            records = list(self._final_records)
            checkpoints = list(self._checkpoint_markdowns)
        self._summary_task = asyncio.create_task(
            self._run_final_summary(
                ports.summarizer,
                session,
                settings,
                records,
                checkpoints,
                self._elapsed_ms(),
            ),
            name="tpdk-final-summary",
        )

    # -- startup -----------------------------------------------------------

    async def _startup(self) -> None:
        try:
            ports = self._ports_factory(self._ports_config)
            self._ports = ports
            await asyncio.to_thread(ports.asr_client.warmup)
            # Warm the translator in the background: the PLaMo server is already
            # loaded (this is just a health check), and NLLB loads in 12-23 s.
            # Do not hold capture hostage to it: source captions should start
            # now, and MT jobs queue behind the one-time load.
            translator_warmup = getattr(ports.translator, "warmup", None)
            if callable(translator_warmup):
                threading.Thread(
                    target=self._warm_translator,
                    args=(translator_warmup,),
                    name="tpdk-mt-warm",
                    daemon=True,
                ).start()
            ocr_warmup = getattr(ports.ocr_engine, "warmup", None)
            if callable(ocr_warmup):
                await asyncio.to_thread(ocr_warmup)
            if self._stop_event.is_set():
                return

            # Build the VAD before capture starts. Starting the capture thread
            # first lets scipy's resampler probe torch while Silero is still
            # importing it, which fails with a partially initialized torch.
            factory = ports.segmenter_factory or self._default_segmenter_factory()
            self._segmenter = await asyncio.to_thread(factory)

            await asyncio.to_thread(ports.audio_source.start)
            if self._stop_event.is_set():
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(ports.audio_source.stop)
                return

            reset = getattr(ports.change_detector, "reset", None)
            if callable(reset):
                reset()

            # Create the coalescing pump before any worker can submit a
            # partial; otherwise the first preview can be stored with no wakeup.
            self._start_event_tasks()
            self._start_threads()
            self._transition(SessionState.RUNNING)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - startup failures are user-visible
            logger.exception("session startup failed")
            ports = self._ports
            self._ports = None
            if ports is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(ports.audio_source.stop)
                self._release_ports(ports)
            self._transition(SessionState.ERROR, detail=str(exc))
            self._notice(NoticeLevel.ERROR, "session_start_failed", str(exc))

    @staticmethod
    def _warm_translator(warmup: Any) -> None:
        try:
            warmup()
        except Exception as exc:  # noqa: BLE001 - the MT worker reports real errors
            logger.warning("translator warmup failed: %s", exc)

    def _default_segmenter_factory(self) -> SegmenterFactory:
        def build() -> Any:
            from tpdk_overlay.audio.vad import SileroVadSegmenter

            return SileroVadSegmenter(self._config.audio.vad)

        return build

    def _start_threads(self) -> None:
        assert self._ports is not None
        ports = self._ports
        self._capture_thread = threading.Thread(
            target=self._capture_loop, args=(ports,), name="tpdk-capture", daemon=True
        )
        self._asr_thread = threading.Thread(
            target=self._asr_loop, args=(ports,), name="tpdk-asr", daemon=True
        )
        self._mt_thread = threading.Thread(
            target=self._mt_loop, args=(ports,), name="tpdk-mt", daemon=True
        )
        self._capture_thread.start()
        self._asr_thread.start()
        self._mt_thread.start()

        if ports.screen_source is not None and ports.ocr_engine is not None:
            self._ocr_thread = threading.Thread(
                target=self._ocr_loop, args=(ports,), name="tpdk-ocr", daemon=True
            )
            self._ocr_thread.start()
        elif self._config.screen.enabled and self._config.ocr.enabled:
            self._notice(
                NoticeLevel.INFO,
                "ocr_disabled",
                "screen OCR adapter is not available; running transcript only",
            )
        if self._config.summary.enabled and self._ports.summarizer is None:
            self._notice(
                NoticeLevel.INFO,
                "summary_disabled",
                "summarizer adapter is not available; checkpoints are off",
            )

    def _start_event_tasks(self) -> None:
        self._partial_event = asyncio.Event()
        self._partial_task = asyncio.create_task(self._partial_pump(), name="tpdk-partials")
        assert self._ports is not None
        if self._ports.summarizer is not None and self._config.summary.enabled:
            self._checkpoint_task = asyncio.create_task(
                self._checkpoint_loop(), name="tpdk-checkpoints"
            )

    # -- capture / VAD -----------------------------------------------------

    def _capture_loop(self, ports: PipelinePorts) -> None:
        assert self._segmenter is not None
        source = ports.audio_source
        segmenter = self._segmenter
        stream_open = False
        stream_t_end_ms = 0
        pending: list[np.ndarray] = []
        pending_samples = 0
        pending_t_end_ms = 0

        def flush_pending() -> None:
            nonlocal pending, pending_samples
            if pending:
                self._put_asr(_AudioFrame(np.concatenate(pending), pending_t_end_ms))
                pending = []
                pending_samples = 0

        try:
            while not self._stop_event.is_set():
                frame = source.read(VAD_FRAME_SAMPLES)
                if frame.size < VAD_FRAME_SAMPLES:
                    if self._stop_event.is_set():
                        break
                    frame = np.pad(frame, (0, VAD_FRAME_SAMPLES - frame.size))
                if self._paused.is_set():
                    frame = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)

                event = segmenter.feed(frame)
                if isinstance(event, SpeechStart):
                    stream_open = True
                    pending = []
                    pending_samples = 0
                    stream_t_end_ms = event.t_start_ms + (
                        event.audio.size // VAD_FRAME_SAMPLES
                    ) * VAD_FRAME_MS
                    self._put_asr(
                        _OpenStream(next(self._segment_ids), event.audio, event.t_start_ms)
                    )
                elif isinstance(event, Utterance):
                    if not stream_open:
                        self._put_asr(
                            _OpenStream(next(self._segment_ids), event.audio, event.t_start_ms)
                        )
                    else:
                        flush_pending()
                    stream_open = False
                    self._put_asr(_CloseStream(event.t_end_ms, now_ms()))
                elif stream_open:
                    stream_t_end_ms += VAD_FRAME_MS
                    pending.append(frame)
                    pending_samples += int(frame.size)
                    pending_t_end_ms = stream_t_end_ms
                    if pending_samples >= _ASR_CHUNK_SAMPLES:
                        flush_pending()
        except Exception as exc:  # noqa: BLE001 - component failures become notices
            if not self._stop_event.is_set():
                logger.exception("audio capture loop failed")
                self._notice(NoticeLevel.ERROR, "audio_error", f"audio capture failed: {exc}")
        finally:
            with contextlib.suppress(Exception):
                source.stop()

    def _put_asr(self, item: Any) -> bool:
        deadline = time.monotonic() + _PUT_BUDGET_S
        control = isinstance(item, (_OpenStream, _CloseStream))
        while not self._stop_event.is_set():
            try:
                self._asr_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                if control:
                    # Open/close must not queue behind a backlog of audio, but
                    # never evict another control item: only drop a chunk.
                    try:
                        head = self._asr_queue.queue[0]
                    except IndexError:
                        continue
                    if isinstance(head, _AudioFrame):
                        with contextlib.suppress(queue.Empty):
                            self._asr_queue.get_nowait()
                    continue
                if time.monotonic() > deadline:
                    logger.warning(
                        "dropping ASR input after %.1fs of backpressure", _PUT_BUDGET_S
                    )
                    return False
        return False

    # -- ASR ---------------------------------------------------------------

    def _asr_loop(self, ports: PipelinePorts) -> None:
        stream = None
        segment_id = 0
        t_start_ms = 0
        t_end_ms = 0
        revision = 0
        last_text = ""
        try:
            while True:
                item = self._asr_queue.get()
                if item is _SENTINEL:
                    self._finish_on_stop(
                        stream, segment_id, t_start_ms, t_end_ms, revision, last_text
                    )
                    break
                try:
                    if isinstance(item, _OpenStream):
                        if stream is not None:
                            with contextlib.suppress(Exception):
                                stream.close()
                        stream = ports.asr_client.open(
                            language=self._settings.source_lang
                        )
                        segment_id = item.segment_id
                        t_start_ms = item.t_start_ms
                        t_end_ms = item.t_start_ms + (
                            item.audio.size // VAD_FRAME_SAMPLES
                        ) * VAD_FRAME_MS
                        revision = 0
                        last_text = ""
                        update = stream.push(item.audio)
                        revision, last_text = self._emit_partial(
                            update, segment_id, revision, last_text, t_start_ms, t_end_ms
                        )
                    elif isinstance(item, _AudioFrame):
                        if stream is None:
                            continue
                        t_end_ms = item.t_end_ms
                        update = stream.push(item.frame)
                        revision, last_text = self._emit_partial(
                            update, segment_id, revision, last_text, t_start_ms, t_end_ms
                        )
                    elif isinstance(item, _CloseStream):
                        if stream is None:
                            continue
                        t_end_ms = item.t_end_ms
                        stopwatch = Stopwatch()
                        update = stream.finish()
                        asr_ms = stopwatch.elapsed_ms()
                        stream.close()
                        stream = None
                        text = update.text.strip()
                        if not text:
                            logger.debug("empty final for segment %d", segment_id)
                            continue
                        self._finalize_source(
                            segment_id=segment_id,
                            revision=revision,
                            text=text,
                            t_start_ms=t_start_ms,
                            t_end_ms=t_end_ms,
                            audio_end_ts_ms=item.audio_end_ts_ms,
                            asr_ms=asr_ms,
                        )
                except (AsrUnavailable, AsrTimeout) as exc:
                    logger.warning("ASR unavailable: %s", exc)
                    self._notice(NoticeLevel.WARNING, "asr_unavailable", str(exc))
                    if stream is not None:
                        with contextlib.suppress(Exception):
                            stream.close()
                    stream = None
                except Exception as exc:  # noqa: BLE001 - ASR failures keep the session alive
                    logger.warning("ASR failure: %s", exc, exc_info=True)
                    self._notice(NoticeLevel.ERROR, "asr_error", str(exc))
                    if stream is not None:
                        with contextlib.suppress(Exception):
                            stream.close()
                    stream = None
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()

    def _emit_partial(
        self,
        update: AsrUpdate,
        segment_id: int,
        revision: int,
        last_text: str,
        t_start_ms: int,
        t_end_ms: int,
    ) -> tuple[int, str]:
        text = update.text.strip()
        if not text or text == last_text:
            return revision, last_text
        revision += 1
        event = TranscriptSegmentEvent(
            session_id=self.session_id,
            segment_id=segment_id,
            revision=revision,
            status=SegmentStatus.PARTIAL,
            t_start_ms=t_start_ms,
            t_end_ms=max(t_start_ms, t_end_ms),
            source_lang=self._settings.source_lang,
            target_lang=self._settings.target_lang,
            text=text,
            translation_status=TranslationStatus.SKIPPED,
        )
        self._submit_partial(event)
        return revision, text

    def _finalize_source(
        self,
        *,
        segment_id: int,
        revision: int,
        text: str,
        t_start_ms: int,
        t_end_ms: int,
        audio_end_ts_ms: int,
        asr_ms: int,
    ) -> None:
        self._drop_pending_partial(segment_id)
        revision += 1
        audio_end_to_text_ms = max(0, now_ms() - audio_end_ts_ms)
        event = TranscriptSegmentEvent(
            session_id=self.session_id,
            segment_id=segment_id,
            revision=revision,
            status=SegmentStatus.FINAL,
            t_start_ms=t_start_ms,
            t_end_ms=max(t_start_ms, t_end_ms),
            source_lang=self._settings.source_lang,
            target_lang=self._settings.target_lang,
            text=text,
            translation_status=TranslationStatus.PENDING,
            asr_latency_ms=asr_ms,
            audio_end_to_text_ms=audio_end_to_text_ms,
            audio_end_ts_ms=audio_end_ts_ms,
        )
        self._emit_transcript(event)
        record = {
            "segment_id": segment_id,
            "t_start_ms": t_start_ms,
            "t_end_ms": max(t_start_ms, t_end_ms),
            "source_lang": self._settings.source_lang,
            "target_lang": self._settings.target_lang,
            "text": text,
            "translation": "",
            "checkpointed": False,
        }
        with self._records_lock:
            self._final_records.append(record)
            self._final_record_index[segment_id] = record
        job = _TranslationJob(
            segment_id=segment_id,
            revision=revision,
            text=text,
            t_start_ms=t_start_ms,
            t_end_ms=max(t_start_ms, t_end_ms),
            audio_end_ts_ms=audio_end_ts_ms,
            source_lang=self._settings.source_lang,
            target_lang=self._settings.target_lang,
            asr_latency_ms=asr_ms,
            audio_end_to_text_ms=audio_end_to_text_ms,
        )
        if not self._put_mt(job):
            fallback = event.model_copy(
                update={
                    "revision": revision + 1,
                    "translation": None,
                    "translation_status": TranslationStatus.FAILED,
                }
            )
            self._emit_transcript(fallback)
            self._persist_transcript(fallback, None)

    def _finish_on_stop(
        self,
        stream: Any,
        segment_id: int,
        t_start_ms: int,
        t_end_ms: int,
        revision: int,
        last_text: str,
    ) -> None:
        if stream is None:
            return
        try:
            stopwatch = Stopwatch()
            update = stream.finish()
            asr_ms = stopwatch.elapsed_ms()
            text = update.text.strip()
            if text:
                self._finalize_source(
                    segment_id=segment_id,
                    revision=revision,
                    text=text,
                    t_start_ms=t_start_ms,
                    t_end_ms=max(t_start_ms, t_end_ms),
                    audio_end_ts_ms=now_ms(),
                    asr_ms=asr_ms,
                )
        except Exception as exc:  # noqa: BLE001 - stop is best effort
            logger.warning("failed to finalize the open ASR stream on stop: %s", exc)
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    # -- MT ----------------------------------------------------------------

    def _mt_loop(self, ports: PipelinePorts) -> None:
        while True:
            job = self._mt_queue.get()
            if job is _SENTINEL:
                break
            assert isinstance(job, _TranslationJob)
            if self._stop_event.is_set():
                # Shutdown: do not start another model call. Resolve the
                # segment as skipped so no client is left with a permanently
                # pending translation, and drain the queue quickly.
                self._emit_shutdown_skip(job)
                continue
            translation: str | None = None
            status = TranslationStatus.FAILED
            stopwatch = Stopwatch()
            try:
                [translation] = ports.translator.translate(
                    [job.text],
                    source_lang=job.source_lang,
                    target_lang=job.target_lang,
                )
                status = TranslationStatus.OK
            except Exception as exc:  # noqa: BLE001 - MT failure degrades one segment
                logger.warning(
                    "translation failed for segment %d: %s", job.segment_id, exc, exc_info=True
                )
                self._notice(
                    NoticeLevel.WARNING,
                    "mt_unavailable",
                    f"translation failed for segment {job.segment_id}: {exc}",
                )
            mt_ms = stopwatch.elapsed_ms()
            event = TranscriptSegmentEvent(
                session_id=self.session_id,
                segment_id=job.segment_id,
                revision=job.revision + 1,
                status=SegmentStatus.FINAL,
                t_start_ms=job.t_start_ms,
                t_end_ms=job.t_end_ms,
                source_lang=job.source_lang,
                target_lang=job.target_lang,
                text=job.text,
                translation=translation,
                translation_status=status,
                asr_latency_ms=job.asr_latency_ms,
                mt_latency_ms=mt_ms,
                audio_end_to_text_ms=job.audio_end_to_text_ms,
                audio_end_ts_ms=job.audio_end_ts_ms,
            )
            try:
                self._emit_transcript(event)
                self._persist_transcript(event, translation)
            except Exception as exc:  # noqa: BLE001 - one bad write must not kill MT
                logger.error("failed to publish/persist segment %d: %s", event.segment_id, exc)
                self._notice(
                    NoticeLevel.ERROR,
                    "persist_error",
                    f"failed to persist segment {event.segment_id}: {exc}",
                )

    def _emit_shutdown_skip(self, job: _TranslationJob) -> None:
        """Resolve a queued translation as skipped instead of starting inference."""

        event = TranscriptSegmentEvent(
            session_id=self.session_id,
            segment_id=job.segment_id,
            revision=job.revision + 1,
            status=SegmentStatus.FINAL,
            t_start_ms=job.t_start_ms,
            t_end_ms=job.t_end_ms,
            source_lang=job.source_lang,
            target_lang=job.target_lang,
            text=job.text,
            translation=None,
            translation_status=TranslationStatus.SKIPPED,
            asr_latency_ms=job.asr_latency_ms,
            mt_latency_ms=0,
            audio_end_to_text_ms=job.audio_end_to_text_ms,
            audio_end_ts_ms=job.audio_end_ts_ms,
        )
        try:
            self._emit_transcript(event)
            self._persist_transcript(event, None)
        except Exception as exc:  # noqa: BLE001 - stop must not be blocked by a write
            logger.warning("failed to resolve skipped segment %d: %s", job.segment_id, exc)

    def _put_mt(self, job: _TranslationJob) -> bool:
        deadline = time.monotonic() + _PUT_BUDGET_S
        while True:
            try:
                self._mt_queue.put(job, timeout=0.1)
                return True
            except queue.Full:
                if time.monotonic() > deadline:
                    logger.error("MT queue stayed full; dropping segment %d", job.segment_id)
                    self._notice(
                        NoticeLevel.ERROR,
                        "mt_backlog",
                        f"translation backlog: segment {job.segment_id} was dropped",
                    )
                    return False

    def _persist_transcript(
        self, event: TranscriptSegmentEvent, translation: str | None
    ) -> None:
        if self._session is not None:
            self._session.append_transcript(event)
        with self._records_lock:
            record = self._final_record_index.get(event.segment_id)
            if record is None:
                record = {
                    "segment_id": event.segment_id,
                    "t_start_ms": event.t_start_ms,
                    "t_end_ms": event.t_end_ms,
                    "source_lang": event.source_lang,
                    "target_lang": event.target_lang,
                    "text": event.text,
                    "translation": translation or "",
                    "checkpointed": False,
                }
                self._final_records.append(record)
                self._final_record_index[event.segment_id] = record
            else:
                record["translation"] = translation or ""

    # -- transcript event emission ----------------------------------------

    def _submit_partial(self, event: TranscriptSegmentEvent) -> None:
        with self._partial_lock:
            self._pending_partials[event.segment_id] = event
        if self._loop is not None and self._partial_event is not None:
            self._loop.call_soon_threadsafe(self._partial_event.set)

    def _drop_pending_partial(self, segment_id: int) -> None:
        with self._partial_lock:
            self._pending_partials.pop(segment_id, None)

    async def _partial_pump(self) -> None:
        assert self._partial_event is not None
        while True:
            await self._partial_event.wait()
            self._partial_event.clear()
            with self._partial_lock:
                pending = sorted(
                    self._pending_partials.values(), key=lambda item: item.segment_id
                )
                self._pending_partials.clear()
            for event in pending:
                self._emit_transcript(event)

    def _emit_transcript(self, event: TranscriptSegmentEvent) -> None:
        with self._revision_lock:
            previous = self._last_revision.get(event.segment_id, -1)
            if event.revision <= previous:
                return
            self._last_revision[event.segment_id] = event.revision
        self._publish(event)
        if self._metrics is not None:
            self._metrics.note(event)

    # -- OCR ---------------------------------------------------------------

    def _ocr_loop(self, ports: PipelinePorts) -> None:
        assert ports.screen_source is not None
        assert ports.ocr_engine is not None
        interval_s = max(0.1, self._config.screen.capture_interval_ms / 1000.0)
        while not self._stop_event.is_set():
            settings = self._settings
            if (
                settings is None
                or not settings.screen_enabled
                or not settings.ocr_enabled
                or settings.region is None
            ):
                self._stop_event.wait(_OCR_IDLE_WAIT_S)
                continue
            started = time.monotonic()
            try:
                self._run_ocr_pass(settings.region, ports)
            except Exception as exc:  # noqa: BLE001 - OCR failure degrades frames
                logger.warning("screen OCR pass failed: %s", exc, exc_info=True)
                self._notice(NoticeLevel.WARNING, "ocr_error", f"screen OCR failed: {exc}")
            elapsed_s = time.monotonic() - started
            self._stop_event.wait(max(0.0, interval_s - elapsed_s))

    def _run_ocr_pass(self, region: ScreenRegion, ports: PipelinePorts) -> None:
        assert ports.screen_source is not None and ports.ocr_engine is not None
        assert self._settings is not None
        image = ports.screen_source.capture(region)
        change_ts_ms = now_ms()
        score = 1.0
        if ports.change_detector is not None:
            score = float(ports.change_detector.score(image))
        if score < self._config.screen.change_threshold:
            return

        stopwatch = Stopwatch()
        lines = ports.ocr_engine.recognize(image)
        ocr_ms = stopwatch.elapsed_ms()
        lines = [
            line for line in lines if line.confidence >= self._config.ocr.min_confidence
        ][: self._config.ocr.max_blocks]
        if not lines:
            return

        blocks = self._group_blocks(lines, ports)
        mt_ms: int | None = None
        stopwatch = Stopwatch()
        try:
            translations = ports.translator.translate(
                [block.text for block in blocks],
                source_lang=self._settings.source_lang,
                target_lang=self._settings.target_lang,
            )
            mt_ms = stopwatch.elapsed_ms()
            blocks = tuple(
                block.model_copy(
                    update={
                        "translation": translation,
                        "translation_status": TranslationStatus.OK,
                    }
                )
                for block, translation in zip(blocks, translations, strict=True)
            )
        except Exception as exc:  # noqa: BLE001 - keep the source blocks visible
            mt_ms = stopwatch.elapsed_ms()
            logger.warning("OCR block translation failed: %s", exc)
            self._notice(NoticeLevel.WARNING, "mt_unavailable", f"OCR translation failed: {exc}")
            blocks = tuple(
                block.model_copy(update={"translation_status": TranslationStatus.FAILED})
                for block in blocks
            )

        event = OcrFrameEvent(
            session_id=self.session_id,
            frame_id=next(self._frame_ids),
            region=region,
            t_capture_ms=self._elapsed_ms(),
            blocks=blocks,
            ocr_latency_ms=ocr_ms,
            mt_latency_ms=mt_ms,
            change_ts_ms=change_ts_ms,
        )
        self._publish(event)
        if self._metrics is not None:
            self._metrics.note(event)
        if self._session is not None:
            self._session.append_slide(event)
        with self._records_lock:
            self._slide_records.append(
                {
                    "t_capture_ms": event.t_capture_ms,
                    "blocks": [block.model_dump() for block in event.blocks],
                    "checkpointed": False,
                }
            )

    def _group_blocks(
        self, lines: Sequence[OcrLine], ports: PipelinePorts
    ) -> tuple[OcrBlock, ...]:
        if ports.block_grouper is not None:
            blocks = tuple(ports.block_grouper(lines))
        else:
            blocks = tuple(
                OcrBlock(
                    block_id=_block_id(line.bbox, line.text),
                    bbox=line.bbox,
                    text=line.text,
                    confidence=line.confidence,
                )
                for line in lines
            )
        return blocks[: self._config.ocr.max_blocks]

    # -- checkpoints -------------------------------------------------------

    async def _checkpoint_loop(self) -> None:
        assert self._settings is not None and self._ports is not None
        assert self._ports.summarizer is not None
        while not self._stop_event.is_set():
            interval_ms = max(1, int(self._settings.checkpoint_interval_s * 1000))
            next_at_ms = self._last_checkpoint_to_ms + interval_ms
            now = self._elapsed_ms()
            if now < next_at_ms:
                await asyncio.sleep(min(1.0, max(0.05, (next_at_ms - now) / 1000.0)))
                continue
            t_from_ms = self._last_checkpoint_to_ms
            self._last_checkpoint_to_ms = now
            await self._run_checkpoint(t_from_ms, now)

    async def _run_checkpoint(self, t_from_ms: int, t_to_ms: int) -> None:
        ports = self._ports
        settings = self._settings
        session = self._session
        assert ports is not None and ports.summarizer is not None
        assert settings is not None and session is not None
        with self._records_lock:
            # A record is emitted when the source final exists and updated when
            # its translation lands. Selecting by the "checkpointed" flag (not
            # just by the window) guarantees a segment that finished while the
            # previous checkpoint was being written still enters a summary.
            segments = [
                record
                for record in self._final_records
                if not record["checkpointed"] and record["t_end_ms"] < t_to_ms
            ]
            slides = [
                record
                for record in self._slide_records
                if not record["checkpointed"] and record["t_capture_ms"] < t_to_ms
            ]
            for record in segments:
                record["checkpointed"] = True
            for record in slides:
                record["checkpointed"] = True
        if not segments and not slides:
            return

        transcript = "\n".join(
            f"[{record['t_start_ms']}-{record['t_end_ms']} ms] {record['text']}"
            f" | {record['translation']}"
            for record in segments
        )
        slide_text = "\n\n".join(
            f"[{record['t_capture_ms']} ms]\n"
            + "\n".join(
                block.get("translation") or block.get("text", "")
                for block in record["blocks"]
            )
            for record in slides
        )
        self._checkpoint_id += 1
        checkpoint_id = self._checkpoint_id
        request = CheckpointSummaryRequest(
            checkpoint_id=checkpoint_id,
            t_from_ms=t_from_ms,
            t_to_ms=t_to_ms,
            transcript=transcript,
            slides=slide_text,
            source_lang=settings.source_lang,
            target_lang=settings.target_lang,
        )
        stopwatch = Stopwatch()
        status: str = "ok"
        try:
            markdown = await ports.summarizer.checkpoint(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - summaries degrade, they never stop capture
            logger.warning("checkpoint %d failed: %s", checkpoint_id, exc)
            markdown = f"_Checkpoint {checkpoint_id} unavailable: {exc}_"
            status = "degraded"
        latency_ms = stopwatch.elapsed_ms()
        session.write_checkpoint(checkpoint_id, markdown)
        self._checkpoint_markdowns.append(markdown)
        self._publish(
            SummaryCheckpointEvent(
                session_id=self.session_id,
                checkpoint_id=checkpoint_id,
                t_from_ms=t_from_ms,
                t_to_ms=t_to_ms,
                markdown=markdown,
                status=status,  # type: ignore[arg-type]
                latency_ms=latency_ms,
            )
        )

    async def _run_final_summary(
        self,
        summarizer: Summarizer,
        session: SessionStore,
        settings: SessionSettings,
        records: list[dict[str, Any]],
        checkpoints: list[str],
        t_to_ms: int,
    ) -> None:
        transcript = "\n".join(
            f"[{record['t_start_ms']}-{record['t_end_ms']} ms] {record['text']}"
            f" | {record['translation']}"
            for record in records
            if not record["checkpointed"]
        )
        request = FinalSummaryRequest(
            checkpoints=checkpoints,
            transcript=transcript,
            source_lang=settings.source_lang,
            target_lang=settings.target_lang,
        )
        try:
            markdown = await summarizer.final(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed final summary is a notice
            logger.warning("final summary failed: %s", exc)
            self._publish(
                NoticeEvent(
                    session_id=session.session_id,
                    level=NoticeLevel.WARNING,
                    code="summary_failed",
                    message=f"final summary failed: {exc}",
                )
            )
            return
        path = session.write_final(markdown)
        self._publish(
            SummaryFinalEvent(
                session_id=session.session_id,
                path=str(path),
                markdown=markdown,
                t_from_ms=0,
                t_to_ms=t_to_ms,
                checkpoint_count=len(checkpoints),
            )
        )

    # -- lifecycle helpers -------------------------------------------------

    def _live_workers(self) -> list[threading.Thread]:
        threads = (
            self._capture_thread,
            self._asr_thread,
            self._mt_thread,
            self._ocr_thread,
        )
        return [thread for thread in threads if thread is not None and thread.is_alive()]

    @staticmethod
    def _join_threads(threads: list[threading.Thread], deadline: float) -> None:
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(timeout=remaining)

    async def _wait_for_retired_workers(self, timeout_s: float = 6.0) -> list[str]:
        """Block a new session until workers from the previous one have exited."""

        live = self._live_workers()
        if not live:
            return []
        logger.info(
            "waiting for previous workers before starting a new session: %s",
            ", ".join(thread.name for thread in live),
        )
        deadline = time.monotonic() + timeout_s
        await asyncio.to_thread(self._join_threads, live, deadline)
        return [thread.name for thread in live if thread.is_alive()]

    def _reset_session_state(self) -> None:
        self._session = None
        self._metrics = None
        self._ports = None
        self._segmenter = None
        self._startup_task = None
        self._partial_task = None
        self._checkpoint_task = None
        self._summary_task = None
        self._partial_event = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._asr_queue = queue.Queue(maxsize=_ASR_QUEUE_SIZE)
        self._mt_queue = queue.Queue(maxsize=_MT_QUEUE_SIZE)
        self._capture_thread = None
        self._asr_thread = None
        self._mt_thread = None
        self._ocr_thread = None
        self._segment_ids = count(1)
        self._frame_ids = count(1)
        with self._revision_lock:
            self._last_revision.clear()
        with self._partial_lock:
            self._pending_partials.clear()
        with self._records_lock:
            self._final_records.clear()
            self._final_record_index.clear()
            self._slide_records.clear()
        self._checkpoint_markdowns.clear()
        self._checkpoint_id = 0
        self._last_checkpoint_to_ms = 0

    def _transition(self, state: SessionState, detail: str | None = None) -> None:
        self._state = state
        if self._session is not None:
            self._session.record_state(state.value, detail)
        self._publish(
            SessionStateEvent(
                session_id=self.session_id,
                state=state,
                started_at_ms=self._start_wall_ms or None,
                elapsed_ms=self._elapsed_ms(),
                detail=detail,
            )
        )

    def _publish(self, event: Any) -> None:
        self._bus.publish(event)

    def _notice(self, level: NoticeLevel, code: str, message: str) -> None:
        logger.log(_LOG_LEVEL[level], "%s: %s", code, message)
        self._publish(
            NoticeEvent(
                session_id=self.session_id,
                level=level,
                code=code,
                message=message,
            )
        )

    def _elapsed_ms(self) -> int:
        return self._clock.elapsed_ms() if self._clock is not None else 0

    def _release_ports(self, ports: PipelinePorts | None) -> None:
        if ports is None:
            return
        for adapter in (ports.asr_client, ports.translator, ports.ocr_engine):
            closer = getattr(adapter, "close", None)
            if callable(closer):
                with contextlib.suppress(Exception):
                    closer()

    @staticmethod
    async def _cancel_task(task: asyncio.Task[None] | None) -> None:
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _join_workers(self, ports: PipelinePorts, deadline: float) -> list[str]:
        """Stop capture and join workers without blocking the event loop."""

        with contextlib.suppress(Exception):
            ports.audio_source.stop()
        self._join(self._capture_thread, deadline)
        self._send_sentinel(self._asr_queue, deadline)
        self._join(self._asr_thread, deadline)
        self._send_sentinel(self._mt_queue, deadline)
        self._join(self._mt_thread, deadline)
        self._join(self._ocr_thread, deadline)
        threads = (
            self._capture_thread,
            self._asr_thread,
            self._mt_thread,
            self._ocr_thread,
        )
        return [thread.name for thread in threads if thread is not None and thread.is_alive()]

    @staticmethod
    def _join(thread: threading.Thread | None, deadline: float) -> None:
        if thread is None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("thread %s did not stop within the budget", thread.name)
            return
        thread.join(timeout=remaining)
        if thread.is_alive():
            logger.warning("thread %s did not stop within the budget", thread.name)

    @staticmethod
    def _send_sentinel(target: queue.Queue[Any], deadline: float) -> None:
        while True:
            try:
                target.put(_SENTINEL, timeout=0.05)
                return
            except queue.Full:
                if time.monotonic() > deadline:
                    logger.warning("queue stayed full while stopping")
                    return


__all__ = ["Orchestrator", "PipelinePorts", "SessionSettings", "build_default_ports"]
