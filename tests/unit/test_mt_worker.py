from __future__ import annotations

import threading

import pytest

from tpdk_overlay.mt.base import TranslationError, WorkerTranslator


class _FakeWorker(WorkerTranslator):
    def __init__(self) -> None:
        super().__init__()
        self.load_count = 0
        self.load_thread: int | None = None
        self.calls: list[tuple[tuple[str, ...], str, str, int]] = []
        self.fail_load = False
        self.fail_translate = False
        self.drop_one = False

    def _load(self) -> None:
        if self.fail_load:
            raise TranslationError("load failed")
        self.load_count += 1
        self.load_thread = threading.get_ident()

    def _translate_batch(
        self,
        texts: list[str],
        source_nllb: str,
        target_nllb: str,
    ) -> list[str]:
        if self.fail_translate:
            raise ValueError("boom")
        self.calls.append((tuple(texts), source_nllb, target_nllb, threading.get_ident()))
        outputs = [f"{source_nllb}->{target_nllb}:{text}" for text in texts]
        return outputs[:-1] if self.drop_one else outputs


def test_empty_batch_does_not_load() -> None:
    worker = _FakeWorker()
    try:
        assert worker.translate([], source_lang="en", target_lang="id") == []
        assert worker.load_count == 0
    finally:
        worker.close()


def test_blank_inputs_return_empty_without_loading() -> None:
    worker = _FakeWorker()
    try:
        assert worker.translate(["", "   "], source_lang="en", target_lang="id") == ["", ""]
        assert worker.load_count == 0
    finally:
        worker.close()


def test_batch_order_and_thread_affinity() -> None:
    worker = _FakeWorker()
    try:
        result = worker.translate(
            ["Hello", "", "World"],
            source_lang="EN",
            target_lang="id",
        )
        assert result == ["eng_Latn->ind_Latn:Hello", "", "eng_Latn->ind_Latn:World"]
        assert worker.load_count == 1
        assert worker.load_thread is not None
        assert worker.load_thread != threading.get_ident()
        texts, source, target, call_thread = worker.calls[0]
        assert texts == ("Hello", "World")
        assert source == "eng_Latn"
        assert target == "ind_Latn"
        assert call_thread == worker.load_thread
    finally:
        worker.close()


def test_model_loads_once_across_calls() -> None:
    worker = _FakeWorker()
    try:
        worker.translate(["one"], source_lang="en", target_lang="id")
        worker.translate(["two"], source_lang="en", target_lang="id")
        assert worker.load_count == 1
        assert len(worker.calls) == 2
    finally:
        worker.close()


def test_same_language_returns_inputs_without_loading() -> None:
    worker = _FakeWorker()
    try:
        assert worker.translate(["Halo"], source_lang="id", target_lang="id") == ["Halo"]
        assert worker.load_count == 0
    finally:
        worker.close()


def test_unknown_language_raises_before_loading() -> None:
    worker = _FakeWorker()
    try:
        with pytest.raises(ValueError):
            worker.translate(["text"], source_lang="xx", target_lang="id")
        assert worker.load_count == 0
    finally:
        worker.close()


def test_warmup_loads_without_translating() -> None:
    worker = _FakeWorker()
    try:
        worker.warmup()
        assert worker.load_count == 1
        assert worker.load_thread != threading.get_ident()
        assert worker.calls == []
    finally:
        worker.close()


def test_adapter_exception_becomes_translation_error() -> None:
    worker = _FakeWorker()
    worker.fail_translate = True
    try:
        with pytest.raises(TranslationError, match="translation failed: boom"):
            worker.translate(["text"], source_lang="en", target_lang="id")
    finally:
        worker.close()


def test_length_mismatch_is_rejected() -> None:
    worker = _FakeWorker()
    worker.drop_one = True
    try:
        with pytest.raises(TranslationError, match="returned 1 translations for 2 inputs"):
            worker.translate(["one", "two"], source_lang="en", target_lang="id")
    finally:
        worker.close()


def test_load_failure_can_be_retried() -> None:
    worker = _FakeWorker()
    worker.fail_load = True
    try:
        with pytest.raises(TranslationError, match="load failed"):
            worker.translate(["text"], source_lang="en", target_lang="id")
        worker.fail_load = False
        result = worker.translate(["text"], source_lang="en", target_lang="id")
        assert result == ["eng_Latn->ind_Latn:text"]
        assert worker.load_count == 1
    finally:
        worker.close()


def test_close_is_idempotent_and_rejects_new_work() -> None:
    worker = _FakeWorker()
    worker.close()
    worker.close()
    with pytest.raises(TranslationError, match="closed"):
        worker.translate(["text"], source_lang="en", target_lang="id")
