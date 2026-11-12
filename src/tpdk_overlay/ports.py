"""Seams between the orchestrator and each replaceable adapter.

These are ``Protocol`` classes, so an adapter needs no inheritance: implement the
methods and satisfy the type. Tests replace a port with a fake; production wires
the real adapter. Keep the interfaces small: behavior belongs behind the seam.

Invariants every adapter must honor
-----------------------------------
* ``AsrStream.push`` accepts 16 kHz mono ``float32`` in ``[-1, 1]`` and returns
  the **cumulative** text for that stream. ``finish`` flushes and is idempotent.
* ``Translator.translate`` preserves input order and returns one output per
  input, even when it batches internally.
* ``OcrEngine.recognize`` returns reading order, boxes in the coordinate space of
  the supplied image (physical pixels), and never invents text.
* ``Summarizer`` methods raise ``SummarizerUnavailable`` on failure; the
  orchestrator degrades instead of stopping transcription (NFR-06).
* ``AudioSource.read`` returns one mono ``float32`` block at the configured
  capture rate. ``ScreenSource.capture`` returns BGR ``uint8`` (OpenCV order).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from tpdk_overlay.contracts import BBox, ScreenRegion

FloatArray = npt.NDArray[np.float32]
ImageArray = npt.NDArray[np.uint8]


class SummarizerUnavailable(RuntimeError):
    """The cloud summarizer cannot serve this request; callers degrade."""


@dataclass(frozen=True, slots=True)
class AsrUpdate:
    language: str
    text: str


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True, slots=True)
class CheckpointSummaryRequest:
    checkpoint_id: int
    t_from_ms: int
    t_to_ms: int
    transcript: str
    slides: str
    source_lang: str
    target_lang: str


@dataclass(frozen=True, slots=True)
class FinalSummaryRequest:
    checkpoints: Sequence[str]
    transcript: str
    source_lang: str
    target_lang: str


class AsrStream(Protocol):
    def push(self, pcm16k: FloatArray) -> AsrUpdate:
        """Feed the next chunk; return the cumulative transcript so far."""
        ...

    def finish(self) -> AsrUpdate:
        """Flush buffered audio and return the final cumulative transcript."""
        ...

    def close(self) -> None:
        """Release the server-side stream. Safe to call after ``finish``."""
        ...


class AsrClient(Protocol):
    def warmup(self) -> None:
        """Load the model or verify the service. May block; call off the event loop."""
        ...

    def open(self, *, language: str | None = None) -> AsrStream:
        """Start a new ASR stream for one utterance."""
        ...

    def close(self) -> None:
        """Release the client and any loaded model."""
        ...


class Translator(Protocol):
    def translate(
        self,
        texts: Sequence[str],
        *,
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a batch, preserving order and length."""
        ...


class OcrEngine(Protocol):
    def recognize(self, image: ImageArray) -> list[OcrLine]:
        """Detect and recognize text in a BGR frame."""
        ...


class Summarizer(Protocol):
    async def checkpoint(self, request: CheckpointSummaryRequest) -> str:
        """Return markdown for one checkpoint window."""
        ...

    async def final(self, request: FinalSummaryRequest) -> str:
        """Return markdown for the consolidated end-of-session summary."""
        ...


class AudioSource(Protocol):
    def start(self) -> None: ...

    def read(self, frames: int) -> FloatArray:
        """Block until ``frames`` samples are available; return mono float32."""
        ...

    def stop(self) -> None: ...


class ScreenSource(Protocol):
    def capture(self, region: ScreenRegion) -> ImageArray:
        """Grab the region as a BGR image in physical pixels."""
        ...


__all__ = [
    "AsrClient",
    "AsrStream",
    "AsrUpdate",
    "AudioSource",
    "CheckpointSummaryRequest",
    "FinalSummaryRequest",
    "FloatArray",
    "ImageArray",
    "OcrEngine",
    "OcrLine",
    "ScreenSource",
    "Summarizer",
    "SummarizerUnavailable",
    "Translator",
]
