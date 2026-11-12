from __future__ import annotations

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tpdk_overlay.asr import create_asr_client  # noqa: E402
from tpdk_overlay.asr.http_backend import HttpAsrClient  # noqa: E402
from tpdk_overlay.asr.transformers_backend import TransformersAsrClient  # noqa: E402
from tpdk_overlay.config import AsrConfig, NativeAsrConfig  # noqa: E402


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int, str | None]] = []

    def transcribe(self, audio, language=None):
        wav, sample_rate = audio
        self.calls.append((wav.copy(), sample_rate, language))
        return [types.SimpleNamespace(language=language or "English", text="native text")]


class _FakeQwen3ASRModel:
    load_calls = 0
    last_kwargs: dict[str, object] = {}
    instance: _FakeModel | None = None

    @classmethod
    def from_pretrained(cls, model: str, **kwargs):
        cls.load_calls += 1
        cls.last_kwargs = kwargs
        cls.instance = _FakeModel()
        return cls.instance


@pytest.fixture()
def fake_qwen(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("qwen_asr")
    module.Qwen3ASRModel = _FakeQwen3ASRModel
    _FakeQwen3ASRModel.load_calls = 0
    _FakeQwen3ASRModel.last_kwargs = {}
    _FakeQwen3ASRModel.instance = None
    monkeypatch.setitem(sys.modules, "qwen_asr", module)
    return _FakeQwen3ASRModel


def _config(**native_kwargs) -> AsrConfig:
    return AsrConfig(backend="native", native=NativeAsrConfig(**native_kwargs))


def test_native_buffers_utterance_and_transcribes_on_finish(fake_qwen) -> None:
    client = TransformersAsrClient(_config(device="cpu", dtype="float16", max_new_tokens=64))
    stream = client.open(language="Indonesian")

    assert stream.push(np.ones(8_000, dtype=np.float32)).text == ""
    assert stream.push(np.ones(4_000, dtype=np.float32)).text == ""

    final = stream.finish()
    assert final.text == "native text"
    assert final.language == "Indonesian"

    assert fake_qwen.instance is not None
    wav, sample_rate, language = fake_qwen.instance.calls[0]
    assert sample_rate == 16_000
    assert wav.size == 12_000
    assert language == "Indonesian"
    assert fake_qwen.load_calls == 1

    kwargs = fake_qwen.last_kwargs
    assert kwargs["device_map"] == "cpu"
    assert kwargs["dtype"] == torch.float32  # float16 is upgraded on CPU
    assert kwargs["max_new_tokens"] == 64


def test_native_finish_is_idempotent(fake_qwen) -> None:
    client = TransformersAsrClient(_config(device="cpu"))
    stream = client.open()
    stream.push(np.ones(100, dtype=np.float32))

    assert stream.finish().text == "native text"
    assert stream.finish().text == "native text"
    assert fake_qwen.instance is not None
    assert len(fake_qwen.instance.calls) == 1


def test_native_empty_stream_skips_model_load(fake_qwen) -> None:
    client = TransformersAsrClient(_config(device="cpu"))
    assert client.open().finish().text == ""
    assert fake_qwen.load_calls == 0


def test_native_warmup_runs_inference_and_is_idempotent(fake_qwen) -> None:
    client = TransformersAsrClient(_config(device="cpu"))
    client.warmup()

    assert fake_qwen.load_calls == 1
    assert fake_qwen.instance is not None
    calls = fake_qwen.instance.calls
    assert calls
    assert all(sample_rate == 16_000 for _, sample_rate, _ in calls)

    count = len(calls)
    client.warmup()
    assert fake_qwen.load_calls == 1
    assert len(fake_qwen.instance.calls) == count


def test_factory_selects_native(fake_qwen) -> None:
    client = create_asr_client(_config(device="cpu"))
    assert isinstance(client, TransformersAsrClient)


def test_factory_selects_http() -> None:
    client = create_asr_client(AsrConfig())
    assert isinstance(client, HttpAsrClient)
    client.close()
