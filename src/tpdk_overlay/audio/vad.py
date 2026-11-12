"""Voice-activity detection and utterance segmentation (FR-01).

:class:`SileroVadSegmenter` turns a stream of 16 kHz mono frames into speech
events. Frames are 512 samples (32 ms) because the bundled Silero VAD model is
trained on exactly that window. Construction loads that model once unless a
``prob_fn`` is injected; after that the segmenter does no I/O of its own:
callers push frames through :meth:`SileroVadSegmenter.feed`, receive a
:class:`SpeechStart` when speech is confirmed (carrying the pre-roll), stream
the following frames to the ASR client, and receive the :class:`Utterance`
when the utterance closes. Opening the one ASR stream per utterance at
``SpeechStart`` is what lets the first partial track the lecturer.

Instances are single-threaded: every ``feed``/``reset`` call must happen on the
same thread (in the application, the capture reader thread owns it).
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np
import numpy.typing as npt

from tpdk_overlay.config import VadConfig
from tpdk_overlay.utils.audio import SAMPLE_RATE_16K

logger = logging.getLogger(__name__)

VAD_FRAME_SAMPLES = 512
VAD_FRAME_MS = 32  # 512 samples / 16 kHz

UtteranceReason = Literal["silence", "max_len"]
SegmenterState = Literal["idle", "candidate", "speech"]

_Frame = tuple[int, npt.NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class SpeechStart:
    """Speech onset: pre-roll plus the frames that confirmed the detection.

    ``audio`` is exactly what the ASR stream must receive first; subsequent
    ``feed`` calls return no event while the utterance is active, so the caller
    pushes each following frame to the stream itself.
    """

    audio: npt.NDArray[np.float32]
    t_start_ms: int


@dataclass(frozen=True, slots=True)
class Utterance:
    """A completed speech segment with 16 kHz mono audio and frame timestamps.

    ``audio`` is the full segment and repeats the frames already delivered in
    the matching :class:`SpeechStart`; the caller streams each frame once.
    """

    audio: npt.NDArray[np.float32]
    t_start_ms: int
    t_end_ms: int
    reason: UtteranceReason


VadEvent = SpeechStart | Utterance


def _load_silero_model() -> tuple[
    Callable[[npt.NDArray[np.float32]], float],
    Callable[[], None] | None,
]:
    """Load the bundled Silero VAD model and wrap it as a frame probability function."""

    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad()

    def probability(frame: npt.NDArray[np.float32]) -> float:
        tensor = torch.from_numpy(frame if frame.flags.writeable else frame.copy())
        with torch.inference_mode():
            return float(model(tensor, SAMPLE_RATE_16K))

    return probability, getattr(model, "reset_states", None)


class SileroVadSegmenter:
    """Frame-by-frame VAD state machine producing speech-start and end events.

    States: ``idle`` accumulates silence into the pre-roll buffer, ``candidate``
    waits until enough consecutive speech frames accumulate, and ``speech``
    grows the active utterance until silence, a brief dip, or the maximum length
    closes it. The promoting frame returns :class:`SpeechStart` with the
    pre-roll already included; the closing frame returns the :class:`Utterance`
    with trailing silence trimmed. When the maximum length is reached on the
    promoting frame, only the :class:`Utterance` is returned, since no partial
    could have been streamed.
    """

    def __init__(
        self,
        config: VadConfig,
        *,
        prob_fn: Callable[[npt.NDArray[np.float32]], float] | None = None,
        t0_ms: int = 0,
    ) -> None:
        if prob_fn is None:
            if config.engine != "silero":
                raise ValueError(
                    f"prob_fn is required for the {config.engine!r} VAD engine"
                )
            prob_fn, reset_model = _load_silero_model()
        else:
            reset_model = None

        self._prob_fn = prob_fn
        self._reset_model = reset_model
        self._t0_ms = t0_ms
        self._threshold = config.threshold
        self._max_utterance_ms = config.max_utterance_ms
        self._min_speech_frames = max(1, ceil(config.min_speech_ms / VAD_FRAME_MS))
        self._min_silence_frames = max(1, ceil(config.min_silence_ms / VAD_FRAME_MS))
        self._pre_roll_frames = ceil(config.pre_roll_ms / VAD_FRAME_MS)

        self._recent: deque[_Frame] = deque(maxlen=self._pre_roll_frames)
        self._candidate: list[_Frame] = []
        self._utterance: list[_Frame] = []
        self._state: SegmenterState = "idle"
        self._frame_index = 0
        self._speech_run = 0
        self._silence_run = 0
        self._last_speech_end = 0

    def feed(self, frame: npt.NDArray[np.float32]) -> VadEvent | None:
        """Consume one 512-sample frame and return the event it triggered.

        A :class:`SpeechStart` is returned on the frame that confirms speech
        onset, carrying the pre-roll plus the confirming frames; this is when
        the caller opens the utterance's ASR stream. The matching
        :class:`Utterance` is returned when the utterance closes. If the
        maximum length is reached on the promoting frame, only the
        :class:`Utterance` is returned.

        The call is single-threaded and non-reentrant: callers must serialize
        ``feed``/``reset``. Invalid frame shapes raise :class:`ValueError`.
        """

        samples = self._validate_frame(frame)
        index = self._frame_index
        self._frame_index += 1
        speech = self._prob_fn(samples) >= self._threshold

        if self._state == "idle":
            if not speech:
                self._recent.append((index, samples))
                return None
            self._candidate = [*self._recent, (index, samples)]
            self._speech_run = 1
            self._state = "candidate"
            return self._promote_if_ready(index)

        if self._state == "candidate":
            self._candidate.append((index, samples))
            if speech:
                self._speech_run += 1
                return self._promote_if_ready(index)
            self._recent.extend(self._candidate)
            self._candidate.clear()
            self._speech_run = 0
            self._state = "idle"
            return None

        self._utterance.append((index, samples))
        if speech:
            self._silence_run = 0
            self._last_speech_end = index + 1
        else:
            self._silence_run += 1

        if self._exceeds_max_len(index):
            return self._close_max_len(index)
        if not speech and self._silence_run >= self._min_silence_frames:
            return self._close_silence()
        return None

    def reset(self) -> None:
        """Return to the idle state, restart frame numbering, and reset the model."""

        self._state = "idle"
        self._frame_index = 0
        self._recent.clear()
        self._candidate.clear()
        self._utterance.clear()
        self._speech_run = 0
        self._silence_run = 0
        self._last_speech_end = 0
        if self._reset_model is not None:
            self._reset_model()

    @staticmethod
    def _validate_frame(frame: npt.ArrayLike) -> npt.NDArray[np.float32]:
        samples = np.asarray(frame)
        if samples.ndim != 1 or samples.shape[0] != VAD_FRAME_SAMPLES:
            raise ValueError(
                "VAD frames must be 1-D with "
                f"{VAD_FRAME_SAMPLES} samples, got shape {samples.shape}"
            )
        return np.ascontiguousarray(samples, dtype=np.float32)

    def _promote_if_ready(self, index: int) -> VadEvent | None:
        if self._speech_run < self._min_speech_frames:
            return None
        self._utterance.extend(self._candidate)
        self._candidate.clear()
        self._state = "speech"
        self._silence_run = 0
        self._last_speech_end = index + 1
        if self._exceeds_max_len(index):
            return self._close_max_len(index)
        return self._speech_start()

    def _speech_start(self) -> SpeechStart:
        first_index = self._utterance[0][0]
        audio = np.concatenate([samples for _, samples in self._utterance])
        logger.debug(
            "speech start at frame %d with %d pre-roll/confirm frames",
            first_index,
            len(self._utterance),
        )
        return SpeechStart(
            audio=audio,
            t_start_ms=self._t0_ms + first_index * VAD_FRAME_MS,
        )

    def _exceeds_max_len(self, index: int) -> bool:
        first_index = self._utterance[0][0]
        return (index + 1 - first_index) * VAD_FRAME_MS >= self._max_utterance_ms

    def _close_max_len(self, index: int) -> Utterance:
        utterance = self._emit(index + 1, "max_len")
        self._state = "idle"
        self._utterance.clear()
        self._recent.clear()
        self._speech_run = 0
        self._silence_run = 0
        self._last_speech_end = 0
        return utterance

    def _close_silence(self) -> Utterance:
        utterance = self._emit(self._last_speech_end, "silence")
        tail = list(self._utterance[-self._pre_roll_frames :]) if self._pre_roll_frames else []
        self._utterance.clear()
        self._recent.clear()
        self._recent.extend(tail)
        self._state = "idle"
        self._speech_run = 0
        self._silence_run = 0
        self._last_speech_end = 0
        return utterance

    def _emit(self, end_index: int, reason: UtteranceReason) -> Utterance:
        included = [entry for entry in self._utterance if entry[0] < end_index]
        first_index = included[0][0]
        audio = np.concatenate([samples for _, samples in included])
        logger.debug(
            "emitting utterance frames %d..%d (%s)",
            first_index,
            end_index - 1,
            reason,
        )
        return Utterance(
            audio=audio,
            t_start_ms=self._t0_ms + first_index * VAD_FRAME_MS,
            t_end_ms=self._t0_ms + end_index * VAD_FRAME_MS,
            reason=reason,
        )


__all__ = [
    "VAD_FRAME_MS",
    "VAD_FRAME_SAMPLES",
    "SileroVadSegmenter",
    "SpeechStart",
    "Utterance",
    "UtteranceReason",
    "VadEvent",
]
