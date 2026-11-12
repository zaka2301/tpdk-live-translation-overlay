"""Native Windows ASR backend using the ``qwen_asr`` transformers engine.

Backup path when WSL2/vLLM is unavailable. The transformers backend has no
streaming decode, so each utterance is buffered and transcribed once at
``finish``; callers get no partial transcripts from this backend.

Model loading is lazy and serialized. ``warmup()`` loads the weights and warms
the 1 s and 20 s encoder shapes on CUDA (1 s on CPU) at session start, so the
first real utterance does not pay the load or encoder-shape cold start.
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import Any

import numpy as np

from tpdk_overlay.asr.errors import AsrUnavailable
from tpdk_overlay.config import AsrConfig
from tpdk_overlay.ports import AsrStream, AsrUpdate
from tpdk_overlay.utils.audio import SAMPLE_RATE_16K

logger = logging.getLogger(__name__)

# Longest utterance shape warmed up front; matches the VAD default max_utterance_ms.
_WARMUP_LONG_SEC = 20.0


class TransformersAsrClient:
    def __init__(self, config: AsrConfig) -> None:
        self._native = config.native
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._warmed = False

    def warmup(self) -> None:
        if self._warmed:
            return
        model = self._model_or_load()
        self._warm_inference(model)
        self._warmed = True

    def open(self, *, language: str | None = None) -> AsrStream:
        return _TransformersAsrStream(self, language)

    def close(self) -> None:
        with self._load_lock:
            model = self._model
            self._model = None
            self._warmed = False
        if model is None:
            return
        try:
            del model
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("native ASR cleanup failed", exc_info=True)

    def transcribe(self, audio: np.ndarray, language: str | None) -> AsrUpdate:
        model = self._model_or_load()
        try:
            import torch

            with self._infer_lock, torch.inference_mode():
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

    def _warm_inference(self, model: Any) -> None:
        try:
            import torch

            inner = getattr(model, "model", None)
            if inner is not None:
                inner.eval()
                inner.requires_grad_(False)
            device_type = getattr(getattr(model, "device", None), "type", "")
            shapes = (1.0,) if device_type == "cpu" else (1.0, _WARMUP_LONG_SEC)
            saved = getattr(model, "max_new_tokens", None)
            if saved is not None:
                model.max_new_tokens = 1
            try:
                with self._infer_lock, torch.inference_mode():
                    for sec in shapes:
                        model.transcribe(
                            audio=(
                                np.zeros(int(SAMPLE_RATE_16K * sec), dtype=np.float32),
                                SAMPLE_RATE_16K,
                            ),
                            language=None,
                        )
            finally:
                if saved is not None:
                    model.max_new_tokens = saved
        except Exception as exc:
            raise AsrUnavailable(f"native ASR warmup failed: {exc}") from exc

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
                model = Qwen3ASRModel.from_pretrained(
                    self._native.model,
                    dtype=dtype,
                    device_map=device,
                    max_inference_batch_size=self._native.max_inference_batch_size,
                    max_new_tokens=self._native.max_new_tokens,
                )
                inner = getattr(model, "model", None)
                if inner is not None:
                    inner.eval()
                    inner.requires_grad_(False)
                self._model = model
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
