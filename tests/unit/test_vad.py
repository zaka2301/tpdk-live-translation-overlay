from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from tpdk_overlay.audio.vad import (
    VAD_FRAME_SAMPLES,
    SileroVadSegmenter,
    SpeechStart,
    Utterance,
    VadEvent,
)
from tpdk_overlay.config import VadConfig


def make_frame(index: int) -> np.ndarray:
    return np.full(VAD_FRAME_SAMPLES, index / 1000, dtype=np.float32)


def _config(**overrides: int | float) -> VadConfig:
    defaults: dict[str, int | float] = {
        "threshold": 0.5,
        "min_speech_ms": 96,
        "min_silence_ms": 64,
        "max_utterance_ms": 1000,
        "pre_roll_ms": 64,
    }
    defaults.update(overrides)
    return VadConfig(**defaults)


def _frames(start: int, stop: int) -> np.ndarray:
    return np.concatenate([make_frame(index) for index in range(start, stop)])


def _run_events(
    probs: Sequence[float],
    *,
    t0_ms: int = 0,
    **overrides: int | float,
) -> list[VadEvent]:
    iterator = iter(probs)
    segmenter = SileroVadSegmenter(
        _config(**overrides),
        prob_fn=lambda _frame: next(iterator),
        t0_ms=t0_ms,
    )
    events: list[VadEvent] = []
    for index in range(len(probs)):
        event = segmenter.feed(make_frame(index))
        if event is not None:
            events.append(event)
    return events


def _run(
    probs: Sequence[float],
    *,
    t0_ms: int = 0,
    **overrides: int | float,
) -> list[Utterance]:
    return [
        event
        for event in _run_events(probs, t0_ms=t0_ms, **overrides)
        if isinstance(event, Utterance)
    ]


def test_all_silence_emits_nothing() -> None:
    assert _run([0.0] * 20) == []


def test_speech_burst_emits_trimmed_utterance() -> None:
    utterances = _run([0, 0, 1, 1, 1, 1, 0, 0], t0_ms=10_000)

    assert len(utterances) == 1
    utterance = utterances[0]
    assert utterance.reason == "silence"
    assert (utterance.t_start_ms, utterance.t_end_ms) == (10_000, 10_192)
    np.testing.assert_array_equal(utterance.audio, _frames(0, 6))


def test_speech_start_carries_pre_roll_before_utterance() -> None:
    events = _run_events([0, 0, 1, 1, 1, 1, 0, 0], t0_ms=10_000)

    assert len(events) == 2
    start, utterance = events
    assert isinstance(start, SpeechStart)
    assert isinstance(utterance, Utterance)
    assert start.t_start_ms == utterance.t_start_ms == 10_000
    np.testing.assert_array_equal(start.audio, _frames(0, 5))
    np.testing.assert_array_equal(utterance.audio[: start.audio.size], start.audio)


def test_short_speech_is_discarded() -> None:
    assert _run([1, 0, 0, 0]) == []
    assert _run_events([1, 0, 0, 0]) == []


def test_pre_roll_is_bounded_by_config() -> None:
    utterances = _run([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])

    assert len(utterances) == 1
    utterance = utterances[0]
    assert utterance.reason == "silence"
    assert (utterance.t_start_ms, utterance.t_end_ms) == (96, 256)
    np.testing.assert_array_equal(utterance.audio, _frames(3, 8))


def test_brief_dip_does_not_split_utterance() -> None:
    utterances = _run([1, 1, 1, 0, 1, 0, 0])

    assert len(utterances) == 1
    utterance = utterances[0]
    assert utterance.reason == "silence"
    assert (utterance.t_start_ms, utterance.t_end_ms) == (0, 160)
    np.testing.assert_array_equal(utterance.audio, _frames(0, 5))


def test_max_length_cuts_and_rearms() -> None:
    events = _run_events(
        [1.0] * 30,
        min_speech_ms=32,
        min_silence_ms=64,
        max_utterance_ms=320,
        pre_roll_ms=0,
    )

    starts = [event for event in events if isinstance(event, SpeechStart)]
    utterances = [event for event in events if isinstance(event, Utterance)]
    assert [start.t_start_ms for start in starts] == [0, 320, 640]
    assert len(utterances) == 3
    boundaries = [(0, 320), (320, 640), (640, 960)]
    frame_starts = (0, 10, 20)
    for utterance, (t_start, t_end), frame_start in zip(
        utterances, boundaries, frame_starts, strict=True
    ):
        assert utterance.reason == "max_len"
        assert (utterance.t_start_ms, utterance.t_end_ms) == (t_start, t_end)
        np.testing.assert_array_equal(
            utterance.audio, _frames(frame_start, frame_start + 10)
        )


def test_max_len_on_promoting_frame_emits_only_utterance() -> None:
    events = _run_events(
        [1.0] * 3,
        min_speech_ms=96,
        max_utterance_ms=32,
        pre_roll_ms=0,
    )

    assert len(events) == 1
    utterance = events[0]
    assert isinstance(utterance, Utterance)
    assert utterance.reason == "max_len"
    assert (utterance.t_start_ms, utterance.t_end_ms) == (0, 96)


def test_invalid_frame_shapes_raise() -> None:
    segmenter = SileroVadSegmenter(_config(), prob_fn=lambda _frame: 0.0)
    invalid_frames = (
        np.zeros(VAD_FRAME_SAMPLES - 1, dtype=np.float32),
        np.zeros(VAD_FRAME_SAMPLES + 1, dtype=np.float32),
        np.zeros((1, VAD_FRAME_SAMPLES), dtype=np.float32),
    )
    for frame in invalid_frames:
        with pytest.raises(ValueError):
            segmenter.feed(frame)


def test_reset_restarts_indices_and_discards_partial_state() -> None:
    prob = [1.0]
    segmenter = SileroVadSegmenter(
        _config(min_speech_ms=96), prob_fn=lambda _frame: prob[0]
    )
    assert segmenter.feed(make_frame(0)) is None

    segmenter.reset()

    utterances: list[Utterance] = []
    for index, value in enumerate([0, 0, 1, 1, 1, 0, 0]):
        prob[0] = value
        event = segmenter.feed(make_frame(index))
        if isinstance(event, Utterance):
            utterances.append(event)

    assert len(utterances) == 1
    utterance = utterances[0]
    assert utterance.reason == "silence"
    assert (utterance.t_start_ms, utterance.t_end_ms) == (0, 160)
    np.testing.assert_array_equal(utterance.audio, _frames(0, 5))


def test_non_silero_engine_requires_prob_fn() -> None:
    with pytest.raises(ValueError):
        SileroVadSegmenter(VadConfig(engine="webrtc"))


def test_real_silero_model_silence_is_ignored() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("silero_vad")

    segmenter = SileroVadSegmenter(VadConfig())
    assert segmenter.feed(np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)) is None
