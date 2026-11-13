"""Wire contracts for the backend <-> overlay protocol (version 1.0).

This module is the single source of truth for every message that crosses the
WebSocket. Do not restate these shapes elsewhere; import them.

Conventions
-----------
* ``Event.ts_ms`` is a wall-clock Unix epoch in milliseconds on the machine
  running the backend and the UI (same host), so it is comparable across both.
* ``t_*_ms`` fields on payloads are **session-relative** monotonic milliseconds,
  starting at 0 when the session starts. They never jump when the OS clock
  changes.
* Language codes on the wire are ISO 639-1 (``"en"``, ``"id"``). Model dialect
  codes (NLLB FLORES, PLaMo codes) and ASR language tags stay inside their
  adapters.
* Events are immutable. Build a new event instead of mutating one.

Requirement anchors (Bab III, Tabel 3.7/3.8): FR-02 (partial/final transcript
with segment id and timestamps), FR-03 (paired transcript/translation), FR-05
(OCR blocks with coordinates), FR-06 (per-block visibility), FR-07/FR-08
(checkpoint and final summaries), NFR-01/NFR-02 (latency fields).
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

PROTOCOL_VERSION = "1.0"


def _event_id() -> str:
    return uuid.uuid4().hex


def _wall_ms() -> int:
    return int(time.time() * 1000)


class WireModel(BaseModel):
    """Base for all protocol models: reject unknown fields, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class TranslationStatus(StrEnum):
    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class SegmentStatus(StrEnum):
    PARTIAL = "partial"
    FINAL = "final"


class NoticeLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class BBox(WireModel):
    """Axis-aligned box in physical pixels, relative to the captured region."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class ScreenRegion(WireModel):
    """Rectangle on the virtual desktop in physical pixels.

    ``left``/``top`` may be negative on multi-monitor setups. ``monitor`` is the
    logical monitor index used by the screen-capture adapter.
    """

    left: int
    top: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    monitor: int = Field(default=1, ge=1)


# --------------------------------------------------------------------------- #
# Server -> client events
# --------------------------------------------------------------------------- #


class Event(WireModel):
    type: str
    event_id: str = Field(default_factory=_event_id)
    session_id: str | None = None
    ts_ms: int = Field(default_factory=_wall_ms)


class SessionStateEvent(Event):
    """Lifecycle transition. Emitted on connect and on every state change."""

    type: Literal["session.state"] = "session.state"
    state: SessionState
    started_at_ms: int | None = None
    elapsed_ms: int = Field(default=0, ge=0)
    detail: str | None = None


class TranscriptSegmentEvent(Event):
    """One ASR segment, revised in place until ``status == "final"``.

    ``segment_id`` identifies the utterance; ``revision`` increases for every
    source replacement. A translated partial reuses its source revision (the
    same revision is enriched in place), so no preview update is lost when the
    translation lags behind. ``translation`` is filled once the MT adapter
    returns.
    """

    type: Literal["transcript.segment"] = "transcript.segment"
    segment_id: int = Field(ge=0)
    revision: int = Field(default=0, ge=0)
    status: SegmentStatus
    t_start_ms: int = Field(ge=0)
    t_end_ms: int = Field(ge=0)
    source_lang: str
    target_lang: str
    text: str
    translation: str | None = None
    translation_status: TranslationStatus = TranslationStatus.PENDING
    asr_latency_ms: int | None = Field(default=None, ge=0)
    mt_latency_ms: int | None = Field(default=None, ge=0)
    audio_end_to_text_ms: int | None = Field(default=None, ge=0)
    # Wall-clock anchor for NFR-01: painted_ts_ms - audio_end_ts_ms is the
    # end-to-end latency, including UI paint, measured via RenderAckCommand.
    audio_end_ts_ms: int | None = None

    @model_validator(mode="after")
    def _ordered(self) -> TranscriptSegmentEvent:
        if self.t_end_ms < self.t_start_ms:
            raise ValueError("t_end_ms must be greater than or equal to t_start_ms")
        return self


class OcrBlock(WireModel):
    """A grouped text block from a slide frame.

    ``block_id`` is stable across frames while the block keeps its position and
    text, so the UI can animate/hide individual blocks.
    """

    block_id: str
    bbox: BBox
    text: str
    translation: str | None = None
    translation_status: TranslationStatus = TranslationStatus.PENDING
    confidence: float = Field(ge=0.0, le=1.0)
    line_count: int = Field(default=1, ge=1)


class OcrFrameEvent(Event):
    """One OCR pass over the captured slide region."""

    type: Literal["ocr.frame"] = "ocr.frame"
    frame_id: int = Field(ge=0)
    region: ScreenRegion
    t_capture_ms: int = Field(ge=0)
    blocks: tuple[OcrBlock, ...] = ()
    ocr_latency_ms: int | None = Field(default=None, ge=0)
    mt_latency_ms: int | None = Field(default=None, ge=0)
    # Wall-clock anchor for NFR-03: painted_ts_ms - change_ts_ms is the
    # end-to-end slide-change latency.
    change_ts_ms: int | None = None


class OverlayVisibilityEvent(Event):
    """Transient overlay hide/show around a screen grab (recording mode).

    ``reason == "capture"`` distinguishes this from the user's tray toggle: the
    UI hides the addressed window (the OCR overlay, or the caption/history
    window when it overlaps the captured region), acknowledges with
    ``RenderAckCommand`` once the window is off the compositor, and restores its
    own visibility afterwards. The backend only sends it when
    ``screen.recording_mode`` is on, so the overlays stay visible to
    OBS/screen shares and the OCR loop still grabs clean pixels.
    """

    type: Literal["overlay.visibility"] = "overlay.visibility"
    target: Literal["ocr", "caption", "history"] = "ocr"
    visible: bool
    reason: Literal["capture"] = "capture"



class SummaryCheckpointEvent(Event):
    """Partial summary for one checkpoint window.

    ``status == "degraded"`` means the cloud call failed or was rate-limited;
    transcription continues (NFR-06).
    """

    type: Literal["summary.checkpoint"] = "summary.checkpoint"
    checkpoint_id: int = Field(ge=0)
    t_from_ms: int = Field(ge=0)
    t_to_ms: int = Field(ge=0)
    markdown: str
    status: Literal["ok", "degraded"] = "ok"
    latency_ms: int | None = Field(default=None, ge=0)


class SummaryFinalEvent(Event):
    """Consolidated end-of-session summary persisted on disk."""

    type: Literal["summary.final"] = "summary.final"
    path: str
    markdown: str
    t_from_ms: int = Field(ge=0)
    t_to_ms: int = Field(ge=0)
    checkpoint_count: int = Field(default=0, ge=0)


class NoticeEvent(Event):
    """Human-readable status or error message for the overlay."""

    type: Literal["notice"] = "notice"
    level: NoticeLevel = NoticeLevel.INFO
    code: str
    message: str


class PongEvent(Event):
    type: Literal["pong"] = "pong"
    echo_ts_ms: int


# --------------------------------------------------------------------------- #
# Client -> server commands
# --------------------------------------------------------------------------- #


class Command(WireModel):
    type: str
    client_ts_ms: int = Field(default_factory=_wall_ms)


class SessionStartCommand(Command):
    type: Literal["session.start"] = "session.start"
    source_lang: str
    target_lang: str
    checkpoint_interval_s: int | None = Field(default=None, gt=0)
    region: ScreenRegion | None = None
    audio_device: str | None = None


class SessionStopCommand(Command):
    type: Literal["session.stop"] = "session.stop"


class SessionPauseCommand(Command):
    type: Literal["session.pause"] = "session.pause"


class SessionResumeCommand(Command):
    type: Literal["session.resume"] = "session.resume"


class SettingsPatch(WireModel):
    source_lang: str | None = None
    target_lang: str | None = None
    checkpoint_interval_s: int | None = Field(default=None, gt=0)
    ocr_enabled: bool | None = None
    screen_enabled: bool | None = None


class SettingsUpdateCommand(Command):
    type: Literal["settings.update"] = "settings.update"
    patch: SettingsPatch


class RegionUpdateCommand(Command):
    type: Literal["region.update"] = "region.update"
    region: ScreenRegion


class OverlayToggleCommand(Command):
    type: Literal["overlay.toggle"] = "overlay.toggle"
    target: Literal["ocr", "caption"]
    visible: bool


class SummaryFinalRequestCommand(Command):
    type: Literal["summary.final_request"] = "summary.final_request"


class RenderAckCommand(Command):
    """UI reports when it painted an event; closes the NFR-01 end-to-end loop.

    ``rect`` rides along with a capture-hide acknowledgement: the physical rect
    of the window addressed by that ``overlay.visibility`` event, which the
    backend masks out of the slide-change score so a caption or history repaint
    never looks like a new slide. Other acks leave it unset.
    """

    type: Literal["render.ack"] = "render.ack"
    event_id: str
    painted_ts_ms: int
    rect: ScreenRegion | None = None


class PingCommand(Command):
    type: Literal["ping"] = "ping"


ServerEvent = Annotated[
    SessionStateEvent
    | TranscriptSegmentEvent
    | OcrFrameEvent
    | OverlayVisibilityEvent
    | SummaryCheckpointEvent
    | SummaryFinalEvent
    | NoticeEvent
    | PongEvent,
    Field(discriminator="type"),
]

ClientCommand = Annotated[
    SessionStartCommand
    | SessionStopCommand
    | SessionPauseCommand
    | SessionResumeCommand
    | SettingsUpdateCommand
    | RegionUpdateCommand
    | OverlayToggleCommand
    | SummaryFinalRequestCommand
    | RenderAckCommand
    | PingCommand,
    Field(discriminator="type"),
]

_server_event_adapter: TypeAdapter[ServerEvent] = TypeAdapter(ServerEvent)
_client_command_adapter: TypeAdapter[ClientCommand] = TypeAdapter(ClientCommand)


def parse_server_event(payload: str | bytes) -> ServerEvent:
    """Validate a JSON payload received by the UI; raises ``ValidationError``."""

    return _server_event_adapter.validate_json(payload)


def parse_client_command(payload: str | bytes) -> ClientCommand:
    """Validate a JSON payload received by the backend; raises ``ValidationError``."""

    return _client_command_adapter.validate_json(payload)


__all__ = [
    "PROTOCOL_VERSION",
    "BBox",
    "ClientCommand",
    "Command",
    "NoticeEvent",
    "NoticeLevel",
    "OcrBlock",
    "OcrFrameEvent",
    "OverlayToggleCommand",
    "OverlayVisibilityEvent",
    "PingCommand",
    "PongEvent",
    "RegionUpdateCommand",
    "RenderAckCommand",
    "ScreenRegion",
    "SegmentStatus",
    "ServerEvent",
    "SessionPauseCommand",
    "SessionResumeCommand",
    "SessionStartCommand",
    "SessionState",
    "SessionStateEvent",
    "SessionStopCommand",
    "SettingsPatch",
    "SettingsUpdateCommand",
    "SummaryCheckpointEvent",
    "SummaryFinalEvent",
    "SummaryFinalRequestCommand",
    "TranscriptSegmentEvent",
    "TranslationStatus",
    "parse_client_command",
    "parse_server_event",
]
