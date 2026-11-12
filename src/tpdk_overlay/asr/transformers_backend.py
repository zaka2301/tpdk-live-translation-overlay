"""Native Windows ASR backend using the ``qwen_asr`` transformers engine.

Backup path when WSL2/vLLM is unavailable. The transformers backend has no
streaming decode, so each utterance is buffered and transcribed once at
``finish``; callers get no partial transcripts from this backend.

Model loading is lazy and serialized. ``warmup()`` loads the weights at session
start so the first utterance does not pay the load cost.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from tpdk_overlay.asr.errors import AsrUnavailable
from tpdk_overlay.config import AsrConfig
from tpdk_overlay.ports import AsrStream, AsrUpdate
from tpdk_overlay.utils.audio import SAMPLE_RATE_16K

logger = logging.getLogger(__name__)


class TransformersAsrClient:
    def __init__(self, config: AsrConfig) -> None:
        self._native = config.native
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def warmup(self) -> None:
        self._model_or_load()

    def open(self, *, language: str | None = None) -> AsrStream:
        return _TransformersAsrStream(self, language)

    def close(self) -> None:
        self._model = None

    def transcribe(self, audio: np.ndarray, language: str | None) -> AsrUpdate:
        model = self._model_or_load()
        try:
            with self._infer_lock:
                results = model.transcribe(
                    audio=(np.asarray(audio, dtype=np.float32), SAMPLE_RATE_16K),
                    language=language,
                )
            result = results[0]
        except Exception as exc:
            raise AsrUnavailable(f"native ASR inference failed: {exc}") from exc
        return AsrUpdate(
            language=str(getattr(result, "language", "") or language or ""),
            text=str(getattr(result, "text", "") or ""),
        )

    def _model_or_load(self) -> Any:
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from qwen_asr import Qwen3ASRModel

                device = self._resolve_device(torch)
                dtype = self._resolve_dtype(torch, device)
                logger.info(
                    "loading native ASR model %s on %s (%s)",
                    self._native.model,
                    device,
                    dtype,
                )
                self._model = Qwen3ASRModel.from_pretrained(
                    self._native.model,
                    dtype=dtype,
                    device_map=device,
                    max_inference_batch_size=self._native.max_inference_batch_size,
                    max_new_tokens=self._native.max_new_tokens,
                )
            except Exception as exc:
                raise AsrUnavailable(f"failed to load native ASR model: {exc}") from exc
            return self._model

    def _resolve_device(self, torch: Any) -> str:
        if self._native.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if self._native.device == "cuda":
            return "cuda:0"
        return "cpu"

    def _resolve_dtype(self, torch: Any, device: str) -> Any:
        if device == "cpu" and self._native.dtype == "float16":
            return torch.float32
        return getattr(torch, self._native.dtype)


class _TransformersAsrStream:
    def __init__(self, client: TransformersAsrClient, language: str | None) -> None:
        self._client = client
        self._language = language
        self._chunks: list[np.ndarray] = []
        self._final: AsrUpdate | None = None
        self._closed = False

    def push(self, pcm16k: np.ndarray) -> AsrUpdate:
        if self._closed:
            raise AsrUnavailable("ASR stream is closed")
        pcm = np.array(pcm16k, dtype=np.float32, copy=True).reshape(-1)
        if pcm.size:
            self._chunks.append(pcm)
        # Offline backend: partial text stays empty until finish().
        return AsrUpdate(language=self._language or "", text="")

    def finish(self) -> AsrUpdate:
        if self._final is not None:
            return self._final
        if self._closed:
            raise AsrUnavailable("ASR stream is closed")
        if self._chunks:
            audio = np.concatenate(self._chunks)
            self._final = self._client.transcribe(audio, self._language)
        else:
            self._final = AsrUpdate(language=self._language or "", text="")
        self._closed = True
        self._chunks.clear()
        return self._final

    def close(self) -> None:
        self._closed = True
        self._chunks.clear()
