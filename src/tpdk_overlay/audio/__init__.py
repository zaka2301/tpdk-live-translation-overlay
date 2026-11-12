"""Capture, VAD, and utterance segmentation (FR-01).

``capture`` provides WASAPI loopback and microphone sources that deliver
16 kHz mono float32 blocks; ``vad`` turns that stream into ``SpeechStart`` and
``Utterance`` events with session-relative timestamps and pre-roll, so the
orchestrator opens one ASR stream per utterance at speech onset.
"""

from __future__ import annotations

from tpdk_overlay.audio.capture import (
    AudioCaptureError,
    LoopbackSource,
    MicrophoneSource,
    create_audio_source,
)
from tpdk_overlay.audio.vad import (
    VAD_FRAME_MS,
    VAD_FRAME_SAMPLES,
    SileroVadSegmenter,
    SpeechStart,
    Utterance,
    UtteranceReason,
    VadEvent,
)

__all__ = [
    "VAD_FRAME_MS",
    "VAD_FRAME_SAMPLES",
    "AudioCaptureError",
    "LoopbackSource",
    "MicrophoneSource",
    "SileroVadSegmenter",
    "SpeechStart",
    "Utterance",
    "UtteranceReason",
    "VadEvent",
    "create_audio_source",
]
