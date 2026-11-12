from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tpdk_overlay.config import MtConfig, MtCt2Config
from tpdk_overlay.mt.base import TranslationError
from tpdk_overlay.mt.nllb_ct2 import NllbCt2Translator, _resolve_ct2_device


class _FakeTokenizer:
    """Deterministic stand-in for the NLLB tokenizer."""

    def __init__(self) -> None:
        self.src_lang = ""
        self.encoded: list[tuple[str, int, bool]] = []
        self.decoded_tokens: list[list[str]] = []

    def encode(self, text: str, max_length: int, truncation: bool) -> list[int]:
        self.encoded.append((text, max_length, truncation))
        return list(range(len(text)))

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"src{index}" for index in ids]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        self.decoded_tokens.append(list(tokens))
        return [int(token.removeprefix("out")) for token in tokens]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"out{index}" for index in ids)


class _FakeCt2Translator:
    """Records ``translate_batch`` calls and replays the target prefix."""

    def __init__(
        self,
        model_path: str,
        device: str,
        compute_type: str,
        *,
        echo_prefix: bool = True,
        empty_hypotheses: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.calls: list[dict[str, Any]] = []
        self._echo_prefix = echo_prefix
        self._empty_hypotheses = empty_hypotheses
        self._produced = 0

    def translate_batch(
        self,
        tokens: list[list[str]],
        *,
        target_prefix: list[list[str]],
        max_decoding_length: int,
        beam_size: int,
    ) -> list[SimpleNamespace]:
        self.calls.append(
            {
                "tokens": tokens,
                "target_prefix": target_prefix,
                "max_decoding_length": max_decoding_length,
                "beam_size": beam_size,
            }
        )
        results: list[SimpleNamespace] = []
        for prefix in target_prefix:
            if self._empty_hypotheses:
                results.append(SimpleNamespace(hypotheses=[]))
                continue
            hypothesis = [f"out{self._produced}"]
            if self._echo_prefix:
                hypothesis.insert(0, prefix[0])
            results.append(SimpleNamespace(hypotheses=[hypothesis]))
            self._produced += 1
        return results


def _install_fake_ct2(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_count: int = 0,
    load_error: Exception | None = None,
    echo_prefix: bool = True,
    empty_hypotheses: bool = False,
) -> list[_FakeCt2Translator]:
    created: list[_FakeCt2Translator] = []

    class _RecordingTranslator(_FakeCt2Translator):
        def __init__(self, model_path: str, device: str, compute_type: str) -> None:
            super().__init__(
                model_path,
                device,
                compute_type,
                echo_prefix=echo_prefix,
                empty_hypotheses=empty_hypotheses,
            )
            if load_error is not None:
                raise load_error
            created.append(self)

    def _get_cuda_device_count() -> int:
        return cuda_count

    module = ModuleType("ctranslate2")
    module.get_cuda_device_count = _get_cuda_device_count
    module.Translator = _RecordingTranslator
    monkeypatch.setitem(sys.modules, "ctranslate2", module)
    return created


def _install_fake_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tokenizer: _FakeTokenizer
) -> list[str]:
    calls: list[str] = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str) -> _FakeTokenizer:
            calls.append(model_id)
            return tokenizer

    module = ModuleType("transformers")
    module.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    return calls


def _model_dir(tmp_path: Path) -> Path:
    (tmp_path / "model.bin").touch()
    return tmp_path


def _config(
    model_dir: Path,
    *,
    device: str = "cpu",
    batch_size: int = 2,
    compute_type: str = "int8",
    ct2_max_new_tokens: int = 64,
) -> MtConfig:
    return MtConfig(
        provider="nllb_ct2",
        model="fake-tokenizer",
        device=device,
        max_input_tokens=48,
        max_new_tokens=96,
        batch_size=batch_size,
        ct2=MtCt2Config(
            model=str(model_dir),
            compute_type=compute_type,
            max_new_tokens=ct2_max_new_tokens,
        ),
    )


@contextmanager
def _managed(config: MtConfig) -> Iterator[NllbCt2Translator]:
    translator = NllbCt2Translator(config)
    try:
        yield translator
    finally:
        translator.close()


def test_translate_preserves_order_across_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = _FakeTokenizer()
    _install_fake_tokenizer(monkeypatch, tokenizer)
    created = _install_fake_ct2(monkeypatch)
    config = _config(_model_dir(tmp_path), batch_size=2)

    with _managed(config) as translator:
        result = translator.translate(
            ["aa", "bbb", "c"], source_lang="en", target_lang="id"
        )

    assert result == ["out0", "out1", "out2"]
    assert [len(call["tokens"]) for call in created[0].calls] == [2, 1]
    assert tokenizer.src_lang == "eng_Latn"


def test_batch_parameters_propagate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer()
    tokenizer_calls = _install_fake_tokenizer(monkeypatch, tokenizer)
    created = _install_fake_ct2(monkeypatch)
    config = _config(
        _model_dir(tmp_path),
        batch_size=2,
        compute_type="int8_float16",
        ct2_max_new_tokens=64,
    )

    with _managed(config) as translator:
        translator.translate(["aa", "bb"], source_lang="en", target_lang="id")

    assert tokenizer_calls == ["fake-tokenizer"]
    assert created[0].model_path == str(tmp_path)
    assert created[0].device == "cpu"
    assert created[0].compute_type == "int8_float16"
    call = created[0].calls[0]
    assert call["target_prefix"] == [["ind_Latn"], ["ind_Latn"]]
    assert call["beam_size"] == 1
    assert call["max_decoding_length"] == 64
    assert tokenizer.encoded[0] == ("aa", 48, True)


def test_echoed_prefix_is_stripped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer()
    _install_fake_tokenizer(monkeypatch, tokenizer)
    _install_fake_ct2(monkeypatch)

    with _managed(_config(_model_dir(tmp_path))) as translator:
        result = translator.translate(["aa"], source_lang="en", target_lang="id")

    assert result == ["out0"]
    assert tokenizer.decoded_tokens == [["out0"]]


def test_prefix_stripped_only_when_echoed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = _FakeTokenizer()
    _install_fake_tokenizer(monkeypatch, tokenizer)
    _install_fake_ct2(monkeypatch, echo_prefix=False)

    with _managed(_config(_model_dir(tmp_path))) as translator:
        result = translator.translate(["aa"], source_lang="en", target_lang="id")

    assert result == ["out0"]
    assert tokenizer.decoded_tokens == [["out0"]]


def test_empty_hypothesis_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_tokenizer(monkeypatch, _FakeTokenizer())
    _install_fake_ct2(monkeypatch, empty_hypotheses=True)

    config = _config(_model_dir(tmp_path))

    with (
        pytest.raises(TranslationError, match="empty hypothesis"),
        _managed(config) as translator,
    ):
        translator.translate(["aa"], source_lang="en", target_lang="id")


def test_blank_inputs_never_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created = _install_fake_ct2(monkeypatch)

    with _managed(_config(_model_dir(tmp_path))) as translator:
        result = translator.translate(["", "   "], source_lang="en", target_lang="id")

    assert result == ["", ""]
    assert created == []


def test_missing_model_bin_raises_with_conversion_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_tokenizer(monkeypatch, _FakeTokenizer())
    _install_fake_ct2(monkeypatch)
    model_dir = tmp_path / "converted"
    model_dir.mkdir()
    config = _config(model_dir)

    with _managed(config) as translator, pytest.raises(TranslationError) as excinfo:
        translator.warmup()

    message = str(excinfo.value)
    assert "model.bin" in message
    assert "ct2-transformers-converter" in message
    assert "--quantization int8" in message


def test_import_error_points_to_optional_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    config = _config(_model_dir(tmp_path))

    with _managed(config) as translator, pytest.raises(TranslationError) as excinfo:
        translator.warmup()

    assert "[ct2]" in str(excinfo.value)


def test_load_failure_is_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tokenizer(monkeypatch, _FakeTokenizer())
    _install_fake_ct2(monkeypatch, load_error=RuntimeError("bad compute type"))
    model_dir = _model_dir(tmp_path)
    config = _config(model_dir)

    with _managed(config) as translator, pytest.raises(TranslationError) as excinfo:
        translator.warmup()

    message = str(excinfo.value)
    assert str(model_dir) in message
    assert "bad compute type" in message


def test_cuda_request_without_gpus_falls_back_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_tokenizer(monkeypatch, _FakeTokenizer())
    created = _install_fake_ct2(monkeypatch, cuda_count=0)
    config = _config(_model_dir(tmp_path), device="cuda")

    with caplog.at_level(logging.WARNING), _managed(config) as translator:
        translator.warmup()

    assert created[0].device == "cpu"
    assert "CUDA" in caplog.text


def test_close_drops_model_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_tokenizer(monkeypatch, _FakeTokenizer())
    _install_fake_ct2(monkeypatch)

    translator = NllbCt2Translator(_config(_model_dir(tmp_path)))
    try:
        translator.warmup()
        translator.close()
        assert translator._translator is None
        assert translator._tokenizer is None
    finally:
        translator.close()


@pytest.mark.parametrize(
    ("device", "cuda_count", "expected"),
    [
        ("cpu", 0, ("cpu", None)),
        ("CPU", 4, ("cpu", None)),
        ("", 0, ("cpu", None)),
        ("auto", 0, ("cpu", None)),
        ("auto", 2, ("cuda", None)),
        (" cuda ", 1, ("cuda", None)),
        ("cuda:1", 2, ("cuda", None)),
    ],
)
def test_resolve_ct2_device(
    device: str, cuda_count: int, expected: tuple[str, str | None]
) -> None:
    assert _resolve_ct2_device(device, cuda_count) == expected


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "CUDA"])
def test_resolve_ct2_device_without_cuda_warns_and_falls_back(device: str) -> None:
    resolved, warning = _resolve_ct2_device(device, 0)
    assert resolved == "cpu"
    assert warning is not None
    assert "CUDA" in warning


def test_resolve_ct2_device_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        _resolve_ct2_device("tpu", 1)
