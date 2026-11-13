"""Checkpoint windows, prompt assembly, retry, and rate limiting (FR-07/FR-08).

The orchestrator owns the session clock, the record lists, and persistence;
this module owns the policy between them:

* :func:`take_window` selects the final transcript segments and OCR rows that
  have not entered a summary yet and marks them ``checkpointed``. The caller
  runs it under its records lock and restores the flags with
  :func:`release_window` when the cloud call is cancelled.
* :func:`format_transcript` / :func:`format_slides` render the window, and
  :func:`checkpoint_prompt` / :func:`final_prompt` turn it into the text-only
  Gemini prompts (audio and screenshot pixels never leave the machine).
* :func:`is_retryable` / :func:`retry_delay` / :class:`RequestPacer` implement
  the free-tier safety policy: retry only 429/5xx and transport failures with
  jittered backoff, and keep request starts at least
  ``summary.min_request_interval_s`` apart.

Prompts ask for the four sections the report requires (topic, key points,
technical terms, questions) in the target language, with technical terms kept
in the source language.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping
from typing import Any

from tpdk_overlay.ports import CheckpointSummaryRequest, FinalSummaryRequest

logger = logging.getLogger(__name__)

# 429 is the free-tier rate limit; the rest are transient server failures.
# Anything else (bad key, unknown model, bad request) fails fast: retrying it
# only burns quota and delays the degraded notice.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

try:  # google-genai talks through httpx; its transport errors are retryable.
    import httpx
except ImportError:  # pragma: no cover - httpx is a hard project dependency
    _TRANSPORT_ERRORS: tuple[type[BaseException], ...] = ()
else:
    _TRANSPORT_ERRORS = (httpx.TransportError,)

_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    *_TRANSPORT_ERRORS,
)

# (topic, key points, technical terms, questions) per target language. Unknown
# languages get the English headings; the model still writes the body in the
# requested language.
_SECTION_HEADINGS: dict[str, tuple[str, str, str, str]] = {
    "id": ("Topik", "Poin Penting", "Istilah Teknis", "Pertanyaan"),
    "en": ("Topic", "Key Points", "Technical Terms", "Questions"),
}
_FINAL_HEADINGS: dict[str, tuple[str, str, str, str]] = {
    "id": ("Ringkasan", "Poin Penting", "Istilah Teknis", "Pertanyaan"),
    "en": ("Overview", "Key Points", "Technical Terms", "Questions"),
}


def _headings(target_lang: str, final: bool) -> tuple[str, str, str, str]:
    table = _FINAL_HEADINGS if final else _SECTION_HEADINGS
    return table.get(target_lang.strip().lower(), table["en"])


def format_transcript(records: Iterable[Mapping[str, Any]]) -> str:
    """Render final segments as ``[start-end ms] source | translation`` lines."""

    lines: list[str] = []
    for record in records:
        line = (
            f"[{record['t_start_ms']}-{record['t_end_ms']} ms] "
            f"{record['text']}"
        )
        translation = str(record.get("translation") or "").strip()
        if translation:
            line += f" | {translation}"
        lines.append(line)
    return "\n".join(lines)


def format_slides(records: Iterable[Mapping[str, Any]]) -> str:
    """Render OCR frames as timestamped block text, preferring translations."""

    frames: list[str] = []
    for record in records:
        texts = []
        for block in record.get("blocks", ()):
            text = str(block.get("translation") or block.get("text") or "").strip()
            if text:
                texts.append(text)
        if texts:
            frames.append(f"[{record['t_capture_ms']} ms]\n" + "\n".join(texts))
    return "\n\n".join(frames)


def take_window(
    transcripts: Iterable[MutableMapping[str, Any]],
    slides: Iterable[MutableMapping[str, Any]],
    *,
    t_to_ms: int,
) -> tuple[list[MutableMapping[str, Any]], list[MutableMapping[str, Any]]]:
    """Return and mark the records that belong to the next checkpoint.

    A record joins the window when it has not been summarized yet and its end
    (transcript) or capture time (slide) is strictly before ``t_to_ms``.
    Marking them here means a retried checkpoint never summarizes a segment
    twice; a cancelled cloud call must restore the flags with
    :func:`release_window` so the final summary still sees those records.
    """

    segments = [
        record
        for record in transcripts
        if not record["checkpointed"] and record["t_end_ms"] < t_to_ms
    ]
    slide_rows = [
        record
        for record in slides
        if not record["checkpointed"] and record["t_capture_ms"] < t_to_ms
    ]
    for record in (*segments, *slide_rows):
        record["checkpointed"] = True
    return segments, slide_rows


def release_window(
    segments: Iterable[MutableMapping[str, Any]],
    slides: Iterable[MutableMapping[str, Any]],
) -> None:
    """Undo :func:`take_window` after a cancelled checkpoint call."""

    for record in (*segments, *slides):
        record["checkpointed"] = False


def remaining_transcript(records: Iterable[Mapping[str, Any]]) -> str:
    """Render only the segments no checkpoint has covered yet."""

    return format_transcript(record for record in records if not record["checkpointed"])


def checkpoint_prompt(request: CheckpointSummaryRequest) -> tuple[str, str]:
    """Return ``(system, user)`` prompts for one checkpoint window."""

    topic, points, terms, questions = _headings(request.target_lang, final=False)
    system = f"""\
You summarize live lecture captions for exchange students. Turn each window into \
a compact study note. Reply with markdown only: no preamble, no closing remarks.

Language rules:
- Write all prose in the target language (ISO 639-1: {request.target_lang}).
- Keep technical terms, formulas, acronyms, and product names in the source \
language (ISO 639-1: {request.source_lang}) when translating them would be \
ambiguous; add a short translation in parentheses the first time.

Use exactly these sections in this order:
## {topic}
One to three sentences naming what this window covered.
## {points}
Five to twelve bullets with the key claims, definitions, and results, in the \
order they were taught.
## {terms}
Bullets in the form "- **term** — short explanation" for terms a student would \
need to look up; write "- (none)" when there are none.
## {questions}
Bullets with questions the lecturer asked, exam hints, and open questions; \
write "- (none)" when there are none."""
    transcript = request.transcript.strip() or "(no speech in this window)"
    slides = request.slides.strip() or "(no slide text captured)"
    user = f"""\
Lecture window {request.checkpoint_id}: {request.t_from_ms / 1000:.1f} s to \
{request.t_to_ms / 1000:.1f} s.

Transcript (source | translation):
{transcript}

Slide text (newest OCR frame per slide change):
{slides}"""
    return system, user


def final_prompt(request: FinalSummaryRequest) -> tuple[str, str]:
    """Return ``(system, user)`` prompts for the consolidated summary."""

    overview, points, terms, questions = _headings(request.target_lang, final=True)
    system = f"""\
You write the consolidated study summary of one recorded lecture for exchange \
students, from its checkpoint summaries and the remaining transcript. Merge \
overlapping points, remove duplicates, and keep the lecture's order. Reply with \
markdown only: no preamble, no closing remarks.

Language rules:
- Write all prose in the target language (ISO 639-1: {request.target_lang}).
- Keep technical terms, formulas, acronyms, and product names in the source \
language (ISO 639-1: {request.source_lang}) when translating them would be \
ambiguous; add a short translation in parentheses the first time.

Use exactly these sections in this order:
## {overview}
Three to six sentences summarizing the whole lecture.
## {points}
A comprehensive bullet list of the key claims, definitions, and results, \
ordered by the lecture.
## {terms}
Bullets in the form "- **term** — short explanation" for every important \
technical term; write "- (none)" when there are none.
## {questions}
Questions raised, exam hints, and open questions; write "- (none)" when there \
are none."""
    checkpoints = [
        markdown.strip() for markdown in request.checkpoints if markdown.strip()
    ]
    body = "\n\n---\n\n".join(checkpoints) if checkpoints else "(no checkpoints)"
    remaining = request.transcript.strip() or "(none)"
    user = f"""\
Checkpoint summaries ({len(checkpoints)}):
{body}

Remaining transcript not covered by a checkpoint:
{remaining}"""
    return system, user


def is_retryable(exc: BaseException) -> bool:
    """True when retrying the same cloud request can plausibly succeed."""

    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and value in RETRYABLE_STATUS:
            return True
    return isinstance(exc, _RETRYABLE_TYPES)


def retry_delay(
    attempt: int,
    *,
    base_s: float,
    max_s: float,
    rng: Callable[[], float] = random.random,
) -> float:
    """Jittered exponential backoff for retry number ``attempt`` (1-based).

    The delay stays in ``[cap / 2, cap]`` with ``cap = min(max_s, base_s *
    2 ** (attempt - 1))``; full jitter keeps concurrent clients from retrying
    in lockstep against the same quota.
    """

    cap = min(max_s, base_s * (2 ** max(0, attempt - 1)))
    return cap * (0.5 + 0.5 * rng())


class RequestPacer:
    """Keep cloud request starts at least ``min_interval_s`` apart.

    One pacer per summarizer instance, shared by checkpoint and final calls, so
    a user-triggered final summary cannot slip past the requests-per-minute
    budget while checkpoints are running. The lock only serializes the spacing
    decision, not the request itself.
    """

    def __init__(
        self,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._min_interval_s = max(0.0, min_interval_s)
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last_at: float | None = None

    async def wait(self) -> None:
        async with self._lock:
            now = self._clock()
            if self._last_at is not None:
                delay = self._min_interval_s - (now - self._last_at)
                if delay > 0:
                    await self._sleep(delay)
                    now = self._clock()
            self._last_at = now


__all__ = [
    "RETRYABLE_STATUS",
    "RequestPacer",
    "checkpoint_prompt",
    "final_prompt",
    "format_slides",
    "format_transcript",
    "is_retryable",
    "release_window",
    "remaining_transcript",
    "retry_delay",
    "take_window",
]
