from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
import transformers

from tpdk_overlay.config import MtConfig
from tpdk_overlay.mt.base import TranslationError
from tpdk_overlay.mt.nllb import NllbTranslator, _resolve_runtime

_MODEL = "facebook/nllb-200-distilled-600M"
_LANG_IDS = {"eng_Latn": 256047, "ind_Latn": 256057}


class _FakeTokenizer:
    """Deterministic offline stand-in for the NLLB tokenizer."""

    unk_token_id = 0

    def __init__(self, texts: list[str]) -> None:
        self.src_lang: str | None = None
        self.src_lang_at_call: str | None = None
        self.calls: list[list[str]] = []
        self.call_options: list[dict[str, Any]] = []
        self._token_ids = {text: 3000 + index for index, text in enumerate(texts)}
        self._decoded = {token: f" tr:{text} " for text, token in self._token_ids.items()}

    def token_id(self, text: str) -> int:
        """Return the fake input token id for ``text``."""

        return self._token_ids[text]

    def convert_tokens_to_ids(self, token: str) -> int:
        return _LANG_IDS.get(token, self.unk_token_id)

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        self.src_lang_at_call = self.src_lang
        self.calls.append(list(texts))
        self.call_options.append(
            {
                "return_tensors": return_tensors,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        token_ids = [self._token_ids[text] for text in texts]
        return {
            "input_ids": torch.tensor([[1, token] for token in token_ids], dtype=torch.long),
            "attention_mask": torch.ones((len(token_ids), 2), dtype=torch.long),
        }

    def batch_decode(
        self,
        sequences: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        assert skip_special_tokens is True
        return [
            "".join(self._decoded.get(token, "") for token in row.tolist()) for row in sequences
        ]


class _FakeModel:
    """Record generation kwargs and echo each input token id back out."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.warmed_up = False
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> _FakeModel:
        self.warmed_up = True
        return self

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        rows = input_ids.shape[0]
        bos = kwargs["forced_bos_token_id"]
        return torch.cat(
            [
                torch.full((rows, 1), bos, dtype=torch.long),
                input_ids[:, 1:2],
                torch.full((rows, 1), 2, dtype=torch.long),
            ],
            dim=1,
        )


class _FakeStack:
    """Swap the transformers factories for fakes and record what was loaded."""

    def __init__(self, texts: list[str]) -> None:
        self.tokenizer = _FakeTokenizer(texts)
        self.model = _FakeModel()
        self.loads: list[tuple[str, dict[str, Any]]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def load_tokenizer(model: str, **kwargs: Any) -> _FakeTokenizer:
            self.loads.append((model, kwargs))
            return self.tokenizer

        def load_model(model: str, **kwargs: Any) -> _FakeModel:
            self.loads.append((model, kwargs))
            return self.model

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
        monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", load_model)


def _config(**overrides: Any) -> MtConfig:
    values: dict[str, Any] = {
        "provider": "nllb",
        "model": _MODEL,
        "device": "cpu",
        "dtype": "float32",
        "max_input_tokens": 64,
        "max_new_tokens": 17,
        "batch_size": 8,
    }
    values.update(overrides)
    return MtConfig(**values)


def test_batch_order_is_preserved_and_blanks_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _FakeStack(["one", "two"])
    stack.install(monkeypatch)
    translator = NllbTranslator(_config())
    try:
        result = translator.translate(
            ["one", "", "   ", "two"],
            source_lang="en",
            target_lang="id",
        )
        assert result == ["tr:one", "", "", "tr:two"]
        assert stack.tokenizer.calls == [["one", "two"]]
    finally:
        translator.close()


def test_chunking_honors_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _FakeStack(["first", "second", "third", "fourth"])
    stack.install(monkeypatch)
    translator = NllbTranslator(_config(batch_size=2))
    try:
        result = translator.translate(
            ["first", "", "second", "third", "fourth"],
            source_lang="en",
            target_lang="id",
        )
    finally:
        translator.close()

    assert result == ["tr:first", "", "tr:second", "tr:third", "tr:fourth"]
    assert stack.tokenizer.calls == [["first", "second"], ["third", "fourth"]]
    assert len(stack.model.generate_calls) == 2


def test_generation_parameters_and_load_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _FakeStack(["hello"])
    stack.install(monkeypatch)
    translator = NllbTranslator(_config())
    try:
        translator.translate(["hello"], source_lang="en", target_lang="id")
    finally:
        translator.close()

    assert stack.loads == [
        (_MODEL, {"src_lang": "eng_Latn"}),
        (_MODEL, {"torch_dtype": torch.float32}),
    ]
    assert stack.model.warmed_up is True
    assert stack.model.device == "cpu"
    assert stack.tokenizer.src_lang_at_call == "eng_Latn"
    assert stack.tokenizer.call_options == [
        {"return_tensors": "pt", "padding": True, "truncation": True, "max_length": 64}
    ]
    (call,) = stack.model.generate_calls
    assert call["forced_bos_token_id"] == _LANG_IDS["ind_Latn"]
    assert call["max_new_tokens"] == 17
    assert call["num_beams"] == 1
    assert call["input_ids"].tolist() == [[1, stack.tokenizer.token_id("hello")]]


def test_blank_only_input_never_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _FakeStack([])
    stack.install(monkeypatch)
    translator = NllbTranslator(_config())
    try:
        assert translator.translate(["", "   "], source_lang="en", target_lang="id") == ["", ""]
    finally:
        translator.close()

    assert stack.loads == []
    assert stack.model.generate_calls == []


@pytest.mark.parametrize(
    ("device", "dtype", "cuda_available", "expected_device", "expected_dtype", "warns"),
    [
        ("cuda", "float16", True, "cuda", torch.float16, False),
        ("cuda", "float16", False, "cpu", torch.float32, True),
        ("cuda:1", "bfloat16", True, "cuda:1", torch.bfloat16, False),
        ("cuda:0", "float32", False, "cpu", torch.float32, True),
        ("cpu", "float16", False, "cpu", torch.float32, True),
        ("cpu", "bfloat16", False, "cpu", torch.bfloat16, False),
        ("auto", "float16", True, "cuda", torch.float16, False),
        ("auto", "float16", False, "cpu", torch.float32, True),
        ("", "float32", True, "cuda", torch.float32, False),
        ("mps", "float32", False, "mps", torch.float32, False),
    ],
)
def test_resolve_runtime_matrix(
    device: str,
    dtype: str,
    cuda_available: bool,
    expected_device: str,
    expected_dtype: torch.dtype,
    warns: bool,
) -> None:
    resolved_device, torch_dtype, warning = _resolve_runtime(
        device,
        dtype,
        cuda_available=cuda_available,
    )

    assert resolved_device == expected_device
    assert torch_dtype is expected_dtype
    assert (warning is not None) is warns


def test_resolve_runtime_queries_torch_when_availability_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert _resolve_runtime("auto", "float16") == ("cuda", torch.float16, None)


def test_resolve_runtime_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        _resolve_runtime("cpu", "int8", cuda_available=False)


def test_load_logs_device_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stack = _FakeStack(["hello"])
    stack.install(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    translator = NllbTranslator(_config(device="cuda", dtype="float16"))
    try:
        with caplog.at_level(logging.WARNING, logger="tpdk_overlay.mt.nllb"):
            translator.warmup()
    finally:
        translator.close()

    assert "falling back to CPU" in caplog.text
    assert "float16 is not supported on CPU" in caplog.text
    assert stack.model.device == "cpu"


def test_load_failure_surfaces_as_translation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(model: str, **kwargs: Any) -> Any:
        raise OSError("checkpoint unavailable")

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", explode)
    translator = NllbTranslator(_config())
    try:
        with pytest.raises(TranslationError, match=_MODEL):
            translator.warmup()
    finally:
        translator.close()
