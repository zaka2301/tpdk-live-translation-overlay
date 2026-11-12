"""NLLB-200 through a CTranslate2 int8 conversion.

This is the optional quantized path (``mt.provider: nllb_ct2``) used when the
GPU must stay free for ASR: CT2 int8 runs comfortably on CPU and needs no
torch. The converted directory only contains the weights, so the tokenizer is
still loaded from the Hugging Face checkpoint named by ``mt.model``.

CTranslate2 and transformers are imported inside :meth:`_load`, keeping
``import tpdk_overlay.mt.nllb_ct2`` usable without the optional ``ct2`` extra.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from tpdk_overlay.config import MtConfig, project_root
from tpdk_overlay.mt.base import TranslationError, WorkerTranslator

logger = logging.getLogger(__name__)

_CONVERT_HINT = (
    "ct2-transformers-converter --model facebook/nllb-200-distilled-600M "
    "--output_dir data/models/nllb-ct2 --quantization int8"
)


def _resolve_ct2_device(device: str, cuda_count: int) -> tuple[str, str | None]:
    """Map a config device name onto a CTranslate2 device and optional warning.

    ``"auto"`` and an empty value pick CUDA when at least one GPU is visible and
    CPU otherwise. A ``cuda*`` request falls back to CPU with a warning instead
    of failing at load time. Anything else is a configuration error.
    """

    normalized = device.strip().lower()
    if not normalized or normalized == "auto":
        return ("cuda", None) if cuda_count > 0 else ("cpu", None)
    if normalized == "cpu":
        return "cpu", None
    if normalized.startswith("cuda"):
        if cuda_count > 0:
            return "cuda", None
        return (
            "cpu",
            f"device {device.strip()!r} requested but no CUDA device is available; "
            "using CPU for NLLB CTranslate2",
        )
    raise ValueError(f"unsupported device for nllb_ct2: {device!r}")


class NllbCt2Translator(WorkerTranslator):
    """Serve NLLB-200 translations from a CTranslate2 conversion.

    The model is loaded once, on the worker thread, and every batch is encoded,
    translated, and decoded there. ``target_prefix`` forces the target FLORES
    code; CT2 echoes it back as the first hypothesis token, which is dropped
    before decoding.
    """

    def __init__(self, config: MtConfig) -> None:
        super().__init__()
        self._config = config
        self._translator: Any = None
        self._tokenizer: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        try:
            import ctranslate2
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TranslationError(
                "mt.provider 'nllb_ct2' needs ctranslate2 and transformers: "
                'pip install -e ".[ct2]"'
            ) from exc

        model_dir = Path(self._config.ct2.model)
        if not model_dir.is_absolute():
            model_dir = project_root() / model_dir
        model_bin = model_dir / "model.bin"
        if not model_bin.is_file():
            raise TranslationError(
                f"CTranslate2 model not found at {model_bin}. Convert it first: {_CONVERT_HINT}"
            )

        device, warning = _resolve_ct2_device(
            self._config.device,
            ctranslate2.get_cuda_device_count(),
        )
        if warning is not None:
            logger.warning(warning)

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._config.model)
            translator = ctranslate2.Translator(
                str(model_dir),
                device=device,
                compute_type=self._config.ct2.compute_type,
            )
        except Exception as exc:
            raise TranslationError(
                f"failed to load the CTranslate2 NLLB model from {model_dir}: {exc}"
            ) from exc

        self._tokenizer = tokenizer
        self._translator = translator
        self._device = device
        logger.info(
            "CT2 NLLB model %s loaded (device %s, %s)",
            model_dir,
            device,
            self._config.ct2.compute_type,
        )

    def _translate_batch(
        self,
        texts: list[str],
        source_nllb: str,
        target_nllb: str,
    ) -> list[str]:
        if self._tokenizer is None or self._translator is None:
            raise TranslationError("CTranslate2 NLLB model is not loaded")

        tokenizer = self._tokenizer
        tokenizer.src_lang = source_nllb
        tokenized = [
            tokenizer.convert_ids_to_tokens(
                tokenizer.encode(
                    text,
                    max_length=self._config.max_input_tokens,
                    truncation=True,
                )
            )
            for text in texts
        ]

        outputs: list[str] = []
        batch_size = self._config.batch_size
        for start in range(0, len(tokenized), batch_size):
            chunk = tokenized[start : start + batch_size]
            results = self._translator.translate_batch(
                chunk,
                target_prefix=[[target_nllb]] * len(chunk),
                max_decoding_length=self._config.ct2.max_new_tokens,
                beam_size=1,
            )
            for result in results:
                if not result.hypotheses or not result.hypotheses[0]:
                    raise TranslationError("CTranslate2 NLLB returned an empty hypothesis")
                hypothesis = list(result.hypotheses[0])
                if hypothesis[0] == target_nllb:
                    hypothesis = hypothesis[1:]
                target_ids = tokenizer.convert_tokens_to_ids(hypothesis)
                outputs.append(
                    tokenizer.decode(target_ids, skip_special_tokens=True).strip()
                )
        return outputs

    def _release(self) -> None:
        self._translator = None
        self._tokenizer = None
        gc.collect()


__all__ = ["NllbCt2Translator"]
