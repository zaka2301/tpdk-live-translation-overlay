"""Shared single-worker machinery for the NLLB adapters.

Both adapters load their model lazily on a dedicated thread and serve one batch
at a time. That gives the model a stable thread affinity (CUDA contexts and
CTranslate2 translators are not cheap to move between threads) and enforces the
"single GPU writer" rule: callers may submit from the asyncio loop or worker
threads, but inference is serialized.

Subclasses implement :meth:`WorkerTranslator._load`,
:meth:`WorkerTranslator._translate_batch`, and optionally
:meth:`WorkerTranslator._release`. They receive FLORES-200 codes, non-empty
texts, and must return exactly one translation per input.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tpdk_overlay.mt.languages import to_nllb

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """The translator cannot load or serve this request."""


class WorkerTranslator:
    """Run model load and inference on one dedicated thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt")
        self._loaded = False
        self._closed = False

    def warmup(self) -> None:
        """Load the model now instead of on the first :meth:`translate` call."""

        with self._lock:
            self._submit(self._ensure_loaded)

    def translate(
        self,
        texts: Sequence[str],
        *,
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a batch, preserving order and length.

        Empty and whitespace-only inputs return ``""`` without touching the
        model. A same-language request returns the inputs unchanged.
        """

        items = list(texts)
        if not items:
            return []
        source = to_nllb(source_lang)
        target = to_nllb(target_lang)
        if source == target:
            return items

        results = [""] * len(items)
        pending = [(index, text) for index, text in enumerate(items) if text.strip()]
        if not pending:
            return results

        payload = [text for _, text in pending]
        with self._lock:
            translated = self._submit(self._run_batch, payload, source, target)
        for (index, _), text in zip(pending, translated, strict=True):
            results[index] = text
        return results

    def close(self) -> None:
        """Stop the worker and release model memory. Safe to call twice."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._loaded = False
            self._release()

    # -- worker-thread hooks -------------------------------------------------

    def _load(self) -> None:
        """Load model resources on the worker thread; raise TranslationError."""

        raise NotImplementedError

    def _translate_batch(
        self,
        texts: list[str],
        source_nllb: str,
        target_nllb: str,
    ) -> list[str]:
        """Translate non-empty texts on the worker thread."""

        raise NotImplementedError

    def _release(self) -> None:
        """Drop model references after the worker has stopped."""

    # -- internals -----------------------------------------------------------

    def _run_batch(
        self,
        texts: list[str],
        source_nllb: str,
        target_nllb: str,
    ) -> list[str]:
        self._ensure_loaded()
        try:
            outputs = self._translate_batch(texts, source_nllb, target_nllb)
        except TranslationError:
            raise
        except Exception as exc:  # noqa: BLE001 - adapter failures become port errors
            raise TranslationError(f"translation failed: {exc}") from exc

        if len(outputs) != len(texts):
            raise TranslationError(
                f"adapter returned {len(outputs)} translations for {len(texts)} inputs"
            )
        return [text.strip() for text in outputs]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _submit(self, fn: Any, *args: Any) -> Any:
        if self._closed:
            raise TranslationError("translator is closed")
        try:
            return self._executor.submit(fn, *args).result()
        except TranslationError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the worker boundary opaque
            raise TranslationError(f"translator worker failed: {exc}") from exc


__all__ = ["TranslationError", "WorkerTranslator"]
