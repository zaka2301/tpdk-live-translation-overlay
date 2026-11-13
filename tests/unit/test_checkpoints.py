"""Checkpoint windows, prompt assembly, retry policy, and the Gemini adapter.

These tests pin the phase 6 contracts: a window never summarizes a segment
twice, prompts carry only transcript/slide text in the requested languages, and
cloud failures either retry (429/5xx) or degrade fast with
``SummarizerUnavailable`` so capture keeps running (NFR-06).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tpdk_overlay.config import SummaryConfig
from tpdk_overlay.ports import (
    CheckpointSummaryRequest,
    FinalSummaryRequest,
    SummarizerUnavailable,
)
from tpdk_overlay.summarize import create_summarizer
from tpdk_overlay.summarize.checkpoints import (
    RequestPacer,
    checkpoint_prompt,
    final_prompt,
    format_slides,
    format_transcript,
    is_retryable,
    release_window,
    remaining_transcript,
    retry_delay,
    take_window,
)
from tpdk_overlay.summarize.gemini import GeminiSummarizer


def _segment(
    segment_id: int,
    t_start_ms: int,
    t_end_ms: int,
    text: str,
    translation: str = "",
    *,
    checkpointed: bool = False,
) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "t_start_ms": t_start_ms,
        "t_end_ms": t_end_ms,
        "text": text,
        "translation": translation,
        "checkpointed": checkpointed,
    }


def _slide(
    t_capture_ms: int,
    blocks: list[dict[str, Any]],
    *,
    checkpointed: bool = False,
) -> dict[str, Any]:
    return {
        "t_capture_ms": t_capture_ms,
        "blocks": blocks,
        "checkpointed": checkpointed,
    }


def _checkpoint_request(
    *, target_lang: str = "id", source_lang: str = "en"
) -> CheckpointSummaryRequest:
    return CheckpointSummaryRequest(
        checkpoint_id=2,
        t_from_ms=0,
        t_to_ms=60_000,
        transcript="[0-1200 ms] hello world | halo dunia",
        slides="[500 ms]\nGradient descent",
        source_lang=source_lang,
        target_lang=target_lang,
    )


def _final_request(*, target_lang: str = "id") -> FinalSummaryRequest:
    return FinalSummaryRequest(
        checkpoints=["## Topik\ncheckpoint one", "## Topik\ncheckpoint two"],
        transcript="[90000-91000 ms] tail segment | segmen akhir",
        source_lang="en",
        target_lang=target_lang,
    )


def _summary_config(**overrides: Any) -> SummaryConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "provider": "gemini",
        "model": "gemini-test",
        "api_key_env": "TPDK_TEST_GEMINI_KEY",
        "min_request_interval_s": 0.0,
        "max_attempts": 2,
        "retry_base_s": 0.001,
        "retry_max_s": 0.002,
    }
    defaults.update(overrides)
    return SummaryConfig(**defaults)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _ApiError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class _FakeModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeResponse(outcome)


class _FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.models = _FakeModels(outcomes)
        self.aio = SimpleNamespace(models=self.models)


# -- windows ---------------------------------------------------------------


def test_take_window_selects_uncheckpointed_records_before_the_horizon() -> None:
    segments = [
        _segment(1, 0, 1_000, "one", "satu"),
        _segment(2, 5_000, 6_000, "two", "dua"),
        _segment(3, 10_000, 11_000, "after horizon"),
    ]
    slides = [
        _slide(500, [{"text": "Slide one", "translation": "Slid satu"}]),
        _slide(20_000, [{"text": "after horizon"}]),
    ]

    picked_segments, picked_slides = take_window(segments, slides, t_to_ms=7_000)

    assert [record["segment_id"] for record in picked_segments] == [1, 2]
    assert picked_slides == [slides[0]]
    assert [record["checkpointed"] for record in segments] == [True, True, False]
    assert [record["checkpointed"] for record in slides] == [True, False]


def test_take_window_uses_a_strict_horizon_and_never_repeats() -> None:
    segments = [_segment(1, 0, 5_000, "boundary")]
    assert take_window(segments, [], t_to_ms=5_000) == ([], [])

    picked, _ = take_window(segments, [], t_to_ms=5_001)
    assert [record["segment_id"] for record in picked] == [1]
    assert take_window(segments, [], t_to_ms=9_000) == ([], [])


def test_release_window_returns_records_to_the_final_summary() -> None:
    segments = [_segment(1, 0, 1_000, "one")]
    slides = [_slide(500, [{"text": "Slide"}])]

    picked_segments, picked_slides = take_window(segments, slides, t_to_ms=2_000)
    release_window(picked_segments, picked_slides)

    assert segments[0]["checkpointed"] is False
    assert slides[0]["checkpointed"] is False
    assert remaining_transcript(segments) == "[0-1000 ms] one"


# -- formatting ------------------------------------------------------------


def test_format_transcript_keeps_timestamps_and_translation() -> None:
    text = format_transcript(
        [
            _segment(1, 0, 1_500, "hello world", "halo dunia"),
            _segment(2, 1_500, 2_000, "no translation"),
        ]
    )
    assert text == (
        "[0-1500 ms] hello world | halo dunia\n"
        "[1500-2000 ms] no translation"
    )


def test_format_slides_prefers_translation_and_skips_empty_frames() -> None:
    text = format_slides(
        [
            _slide(500, []),
            _slide(
                900,
                [
                    {"text": "Gradient descent", "translation": "Penurunan gradien"},
                    {"text": "Learning rate", "translation": ""},
                ],
            ),
        ]
    )
    assert text == "[900 ms]\nPenurunan gradien\nLearning rate"


def test_remaining_transcript_excludes_checkpointed_segments() -> None:
    records = [
        _segment(1, 0, 1_000, "done", checkpointed=True),
        _segment(2, 1_000, 2_000, "left", "sisa"),
    ]
    assert remaining_transcript(records) == "[1000-2000 ms] left | sisa"


def test_checkpoint_prompt_carries_text_languages_and_sections() -> None:
    system, user = checkpoint_prompt(_checkpoint_request())

    for heading in ("Topik", "Poin Penting", "Istilah Teknis", "Pertanyaan"):
        assert f"## {heading}" in system
    assert "target language (ISO 639-1: id)" in system
    assert "source language (ISO 639-1: en)" in system
    assert "0.0 s to 60.0 s" in user
    assert "hello world | halo dunia" in user
    assert "Gradient descent" in user


def test_checkpoint_prompt_uses_english_headings_for_english_target() -> None:
    system, _ = checkpoint_prompt(_checkpoint_request(target_lang="en"))
    assert "## Topic" in system
    assert "## Key Points" in system
    assert "## Topik" not in system


def test_checkpoint_prompt_labels_empty_windows() -> None:
    request = CheckpointSummaryRequest(
        checkpoint_id=1,
        t_from_ms=0,
        t_to_ms=1_000,
        transcript="",
        slides="",
        source_lang="en",
        target_lang="id",
    )
    _, user = checkpoint_prompt(request)
    assert "(no speech in this window)" in user
    assert "(no slide text captured)" in user


def test_final_prompt_merges_checkpoints_and_remaining_transcript() -> None:
    system, user = final_prompt(_final_request())

    assert "## Ringkasan" in system
    assert "## Poin Penting" in system
    assert "Checkpoint summaries (2):" in user
    assert "checkpoint one" in user
    assert "checkpoint two" in user
    assert "\n\n---\n\n" in user
    assert "tail segment | segmen akhir" in user


# -- retry policy ----------------------------------------------------------


def test_retry_delay_is_bounded_and_grows() -> None:
    def no_jitter() -> float:
        return 0.0

    delays = [
        retry_delay(attempt, base_s=1.0, max_s=20.0, rng=no_jitter)
        for attempt in range(1, 8)
    ]
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 10.0]

    def full_jitter() -> float:
        return 1.0

    assert retry_delay(3, base_s=1.0, max_s=20.0, rng=full_jitter) == 4.0


def test_is_retryable_accepts_rate_limits_and_transient_failures() -> None:
    assert is_retryable(_ApiError(429))
    assert is_retryable(_ApiError(503))
    assert not is_retryable(_ApiError(400))
    assert not is_retryable(_ApiError(404))
    assert is_retryable(TimeoutError("deadline"))
    assert is_retryable(ConnectionError("reset"))
    assert not is_retryable(ValueError("bad prompt"))

    class StatusError(Exception):
        status_code = 500

    assert is_retryable(StatusError("server exploded"))


@pytest.mark.asyncio
async def test_request_pacer_spaces_request_starts() -> None:
    now = [100.0]
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    pacer = RequestPacer(4.5, clock=lambda: now[0], sleep=fake_sleep)

    await pacer.wait()
    now[0] += 1.0  # the first request took 1 s
    await pacer.wait()
    assert sleeps == [pytest.approx(3.5)]

    now[0] += 10.0  # plenty of time passed; no extra sleep
    await pacer.wait()
    assert len(sleeps) == 1


# -- Gemini adapter --------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_checkpoint_sends_prompt_and_returns_text() -> None:
    client = _FakeClient(["## Topik\nok"])
    summarizer = GeminiSummarizer(_summary_config(), client=client)

    markdown = await summarizer.checkpoint(_checkpoint_request())

    assert markdown == "## Topik\nok"
    call = client.models.calls[0]
    assert call["model"] == "gemini-test"
    assert "hello world" in call["contents"]
    config = call["config"]
    assert "## Topik" in config.system_instruction
    assert config.temperature == pytest.approx(0.2)
    assert config.max_output_tokens == 2048


@pytest.mark.asyncio
async def test_gemini_retries_a_transient_failure() -> None:
    client = _FakeClient([_ApiError(503), "recovered"])
    summarizer = GeminiSummarizer(_summary_config(), client=client)

    assert await summarizer.final(_final_request()) == "recovered"
    assert len(client.models.calls) == 2


@pytest.mark.asyncio
async def test_gemini_fails_fast_on_a_permanent_failure() -> None:
    client = _FakeClient([_ApiError(400)])
    summarizer = GeminiSummarizer(_summary_config(), client=client)

    with pytest.raises(SummarizerUnavailable, match="400"):
        await summarizer.checkpoint(_checkpoint_request())
    assert len(client.models.calls) == 1


@pytest.mark.asyncio
async def test_gemini_gives_up_after_max_attempts() -> None:
    client = _FakeClient([_ApiError(429), _ApiError(429)])
    summarizer = GeminiSummarizer(_summary_config(), client=client)

    with pytest.raises(SummarizerUnavailable):
        await summarizer.checkpoint(_checkpoint_request())
    assert len(client.models.calls) == 2


@pytest.mark.asyncio
async def test_gemini_empty_response_degrades_without_retrying() -> None:
    client = _FakeClient([""])
    summarizer = GeminiSummarizer(_summary_config(), client=client)

    with pytest.raises(SummarizerUnavailable, match="empty"):
        await summarizer.final(_final_request())
    assert len(client.models.calls) == 1


@pytest.mark.asyncio
async def test_gemini_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TPDK_TEST_GEMINI_KEY", raising=False)
    summarizer = GeminiSummarizer(_summary_config())

    with pytest.raises(SummarizerUnavailable, match="not set"):
        await summarizer.checkpoint(_checkpoint_request())


def test_create_summarizer_respects_enabled_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert create_summarizer(_summary_config(enabled=False)) is None

    monkeypatch.delenv("TPDK_TEST_GEMINI_KEY", raising=False)
    assert create_summarizer(_summary_config()) is None

    monkeypatch.setenv("TPDK_TEST_GEMINI_KEY", "test-key")
    summarizer = create_summarizer(_summary_config())
    assert isinstance(summarizer, GeminiSummarizer)
