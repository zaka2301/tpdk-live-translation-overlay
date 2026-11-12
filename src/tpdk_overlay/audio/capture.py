"""WASAPI loopback and microphone capture, delivered as 16 kHz mono float32.

Both sources implement :class:`tpdk_overlay.ports.AudioSource`. ``soundcard`` is
imported lazily and the device is resolved in :meth:`start`, never in
``__init__``, so this module imports cleanly on machines without audio devices.

Capture runs on a daemon thread that records fixed-size blocks at the configured
rate, converts them to mono float32, resamples to 16 kHz, and appends them to an
internal buffer. ``read()`` blocks on that buffer; ``stop()`` drains whatever is
left so no samples are lost on shutdown.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from tpdk_overlay.config import AudioConfig
from tpdk_overlay.ports import AudioSource, FloatArray
from tpdk_overlay.utils.audio import SAMPLE_RATE_16K, as_float32_mono, resample

logger = logging.getLogger(__name__)

_CAPTURE_THREAD_NAME = "tpdk-audio-capture"
_STOP_JOIN_TIMEOUT_S = 2.0
_FORCED_STOP_JOIN_TIMEOUT_S = 1.0


class AudioCaptureError(RuntimeError):
    """The capture device could not be resolved, opened, or read."""


def _import_soundcard() -> Any:
    """Import ``soundcard`` on demand and normalize backend import failures."""

    try:
        import soundcard
    except Exception as exc:  # pragma: no cover - depends on the host audio stack
        raise AudioCaptureError(f"soundcard backend is unavailable: {exc}") from exc
    return soundcard


class _CaptureSource:
    """Shared recorder plumbing for loopback and microphone capture.

    ``read(frames)`` returns exactly ``frames`` samples of mono ``float32`` at
    :data:`~tpdk_overlay.utils.audio.SAMPLE_RATE_16K` (16 kHz), blocking until
    that many are buffered. ``stop()`` halts the capture thread and drains the
    remainder: a read blocked at that moment wakes and returns what was already
    buffered, possibly fewer samples than requested or none at all. The
    recorder is closed exactly once, either by the capture thread or by a
    ``stop()`` that had to force it.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._device_block_frames = max(
            1, round(config.sample_rate * config.block_ms / 1000)
        )

        self._condition = threading.Condition()
        self._chunks: list[FloatArray] = []
        self._buffered = 0
        self._error: Exception | None = None

        self._thread: threading.Thread | None = None
        self._recorder_context: Any | None = None
        self._close_lock = threading.Lock()
        self._closed = False

        self._stop_event = threading.Event()
        self._started = False
        self._stopped = False
        self._finished = False

    # -- subclass hooks ---------------------------------------------------

    def _device_label(self) -> str:
        """Human-readable device name used in error messages."""

        raise NotImplementedError

    def _resolve_device(self) -> Any:
        """Return the ``soundcard`` microphone/loopback object to record from."""

        raise NotImplementedError

    # -- AudioSource ------------------------------------------------------

    def start(self) -> None:
        """Resolve and open the device, then begin capturing on a daemon thread."""

        with self._condition:
            if self._started:
                raise RuntimeError("audio capture has already been started")
            if self._stopped:
                raise RuntimeError("audio capture was stopped and cannot be restarted")
            self._started = True

        label = self._device_label()
        try:
            microphone = self._resolve_device()
            context = microphone.recorder(
                samplerate=self._config.sample_rate,
                channels=None,
                blocksize=self._device_block_frames,
            )
            # Record the context before entering it so a failed open is still
            # released by _close_recorder instead of leaking the audio client.
            self._recorder_context = context
            recorder = context.__enter__()
        except AudioCaptureError as exc:
            self._close_recorder()
            self._fail_start(exc)
            raise
        except Exception as exc:
            self._close_recorder()
            error = AudioCaptureError(f"cannot open audio device {label!r}: {exc}")
            self._fail_start(error)
            raise error from exc

        thread = threading.Thread(
            target=self._capture_loop,
            args=(recorder,),
            name=_CAPTURE_THREAD_NAME,
            daemon=True,
        )
        try:
            thread.start()
        except Exception as exc:
            error = AudioCaptureError(
                f"cannot start the capture thread for {label!r}: {exc}"
            )
            self._close_recorder()
            self._fail_start(error)
            raise error from exc
        self._thread = thread

        logger.info(
            "audio capture started: %r, %d Hz, %d-frame blocks -> %d Hz mono",
            label,
            self._config.sample_rate,
            self._device_block_frames,
            SAMPLE_RATE_16K,
        )

    def read(self, frames: int) -> FloatArray:
        """Return up to ``frames`` samples of 16 kHz mono ``float32`` audio.

        Blocks until the request is satisfied. If the source stops or fails
        first, returns the buffered remainder (possibly empty) or raises
        :class:`AudioCaptureError` for a capture failure.
        """

        if frames <= 0:
            raise ValueError("frames must be positive")
        with self._condition:
            if not self._started:
                raise RuntimeError("audio capture has not been started")
            while (
                self._buffered < frames
                and not self._finished
                and self._error is None
                and not self._stop_event.is_set()
            ):
                self._condition.wait()
            if self._error is not None:
                raise AudioCaptureError(f"audio capture failed: {self._error}") from self._error
            return self._drain(min(frames, self._buffered))

    def stop(self) -> None:
        """Stop capturing and release the device; safe to call repeatedly."""

        self._stop_event.set()
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
        if thread.is_alive():
            logger.warning("audio capture thread is still running; closing the recorder")
            self._close_recorder()
            thread.join(timeout=_FORCED_STOP_JOIN_TIMEOUT_S)

    # -- internals --------------------------------------------------------

    def _capture_loop(self, recorder: Any) -> None:
        try:
            while not self._stop_event.is_set():
                block = recorder.record(numframes=self._device_block_frames)
                pcm = resample(
                    as_float32_mono(block),
                    self._config.sample_rate,
                    SAMPLE_RATE_16K,
                )
                if pcm.size == 0:
                    continue
                with self._condition:
                    self._chunks.append(pcm)
                    self._buffered += int(pcm.size)
                    self._condition.notify_all()
        except Exception as exc:
            logger.exception("audio capture loop failed")
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            self._close_recorder()
            with self._condition:
                self._finished = True
                self._condition.notify_all()

    def _fail_start(self, error: Exception) -> None:
        with self._condition:
            self._error = error
            self._finished = True
            self._condition.notify_all()

    def _close_recorder(self) -> None:
        """Exit the recorder context at most once, whatever the caller."""

        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            context = self._recorder_context
        if context is None:
            return
        try:
            context.__exit__(None, None, None)
        except Exception:
            logger.debug("failed to close the audio recorder", exc_info=True)

    def _drain(self, count: int) -> FloatArray:
        """Remove ``count`` samples from the head of the buffer; caller holds the lock."""

        if count <= 0:
            return np.empty(0, dtype=np.float32)
        if count >= self._buffered:
            chunks = self._chunks
            self._chunks = []
            self._buffered = 0
            return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)

        pieces: list[FloatArray] = []
        remaining = count
        while remaining > 0:
            head = self._chunks[0]
            if head.size <= remaining:
                pieces.append(head)
                self._chunks.pop(0)
                remaining -= int(head.size)
            else:
                pieces.append(head[:remaining])
                self._chunks[0] = head[remaining:]
                remaining = 0
        self._buffered -= count
        return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)


class LoopbackSource(_CaptureSource):
    """System-audio capture through WASAPI loopback at 16 kHz mono float32.

    ``read`` returns 16 kHz mono ``float32`` samples; ``stop`` halts the capture
    thread and drains the buffered remainder.
    """

    def _device_label(self) -> str:
        return self._config.input_device or "default speaker"

    def _resolve_device(self) -> Any:
        soundcard = _import_soundcard()
        label = self._device_label()
        try:
            if self._config.input_device is not None:
                return soundcard.get_microphone(
                    self._config.input_device, include_loopback=True
                )
            return soundcard.get_microphone(
                soundcard.default_speaker().name, include_loopback=True
            )
        except Exception as exc:
            raise AudioCaptureError(
                f"cannot find loopback device {label!r}: {exc}"
            ) from exc


class MicrophoneSource(_CaptureSource):
    """Microphone capture at 16 kHz mono float32.

    ``read`` returns 16 kHz mono ``float32`` samples; ``stop`` halts the capture
    thread and drains the buffered remainder.
    """

    def _device_label(self) -> str:
        return self._config.input_device or "default microphone"

    def _resolve_device(self) -> Any:
        soundcard = _import_soundcard()
        label = self._device_label()
        try:
            if self._config.input_device is not None:
                return soundcard.get_microphone(self._config.input_device)
            return soundcard.default_microphone()
        except Exception as exc:
            raise AudioCaptureError(
                f"cannot find microphone {label!r}: {exc}"
            ) from exc


def create_audio_source(config: AudioConfig) -> AudioSource:
    """Build the source named by ``config.source``."""

    if config.source == "loopback":
        return LoopbackSource(config)
    if config.source == "microphone":
        return MicrophoneSource(config)
    raise ValueError(f"unknown audio source: {config.source!r}")


__all__ = [
    "AudioCaptureError",
    "LoopbackSource",
    "MicrophoneSource",
    "create_audio_source",
]
