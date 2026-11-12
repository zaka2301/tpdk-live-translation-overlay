from __future__ import annotations

import numpy as np

from tpdk_overlay.utils.audio import (
    as_float32_mono,
    from_pcm16,
    iter_frames,
    resample,
    rms,
    to_pcm16,
)


def test_mono_conversion_averages_channels() -> None:
    stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    mono = as_float32_mono(stereo)
    assert mono.shape == (2,)
    assert mono[0] == 0.5


def test_int16_conversion_scales() -> None:
    pcm = np.array([0, 16384, -32768], dtype=np.int16)
    converted = as_float32_mono(pcm)
    assert converted.dtype == np.float32
    assert converted[1] == 0.5


def test_resample_48k_to_16k_length() -> None:
    source = np.ones(48_000, dtype=np.float32)
    out = resample(source, 48_000, 16_000)
    assert out.dtype == np.float32
    assert out.shape == (16_000,)


def test_resample_identity_is_noop() -> None:
    source = np.linspace(-1.0, 1.0, 1600, dtype=np.float32)
    out = resample(source, 16_000, 16_000)
    np.testing.assert_array_equal(out, source)


def test_pcm16_roundtrip() -> None:
    source = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    decoded = from_pcm16(to_pcm16(source))
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, source, atol=1 / 32767)


def test_rms_silence_and_tone() -> None:
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0
    assert rms(np.ones(100, dtype=np.float32)) == 1.0


def test_iter_frames_drops_remainder() -> None:
    frames = list(iter_frames(np.arange(10, dtype=np.float32), 4))
    assert len(frames) == 2
    assert len(frames[0]) == 4
