"""Audio conversion helpers shared by capture, VAD, and the ASR client.

All helpers are pure functions over numpy arrays; no device access lives here.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import gcd

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

SAMPLE_RATE_16K = 16_000


def as_float32_mono(samples: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Coerce arbitrary channel/int data to contiguous mono float32 in [-1, 1]."""

    data = np.asarray(samples)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype.kind == "f":
        return np.ascontiguousarray(data, dtype=np.float32)
    if data.dtype.kind == "i":
        scale = float(2 ** (data.dtype.itemsize * 8 - 1))
        return np.ascontiguousarray(data.astype(np.float32) / scale, dtype=np.float32)
    if data.dtype.kind == "u":
        info = np.iinfo(data.dtype)
        midpoint = float(info.max) / 2.0
        return np.ascontiguousarray(
            (data.astype(np.float32) - midpoint) / midpoint, dtype=np.float32
        )
    raise TypeError(f"unsupported audio dtype: {data.dtype}")


def resample(
    samples: npt.ArrayLike,
    src_rate: int,
    dst_rate: int,
) -> npt.NDArray[np.float32]:
    """Polyphase resample to ``dst_rate``; returns float32 mono."""

    data = as_float32_mono(samples)
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if src_rate == dst_rate:
        return data
    divisor = gcd(src_rate, dst_rate)
    return resample_poly(data, dst_rate // divisor, src_rate // divisor).astype(
        np.float32, copy=False
    )


def to_pcm16(samples: npt.ArrayLike) -> bytes:
    """Encode float32 mono in [-1, 1] as little-endian int16 PCM bytes."""

    data = np.clip(as_float32_mono(samples), -1.0, 1.0)
    return (data * 32767.0).astype("<i2").tobytes()


def from_pcm16(data: bytes) -> npt.NDArray[np.float32]:
    """Decode little-endian int16 PCM bytes to float32 mono in [-1, 1]."""

    if len(data) % 2:
        raise ValueError("PCM16 byte length must be a multiple of 2")
    return (np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0).copy()


def rms(samples: npt.ArrayLike) -> float:
    """Root-mean-square level, useful for silence gates and debugging."""

    data = as_float32_mono(samples)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))


def iter_frames(samples: npt.ArrayLike, frame_size: int) -> Iterator[npt.NDArray[np.float32]]:
    """Yield exact-size frames; the caller keeps any remainder."""

    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    data = as_float32_mono(samples)
    for start in range(0, data.size - frame_size + 1, frame_size):
        yield data[start : start + frame_size]


__all__ = [
    "SAMPLE_RATE_16K",
    "as_float32_mono",
    "from_pcm16",
    "iter_frames",
    "resample",
    "rms",
    "to_pcm16",
]
