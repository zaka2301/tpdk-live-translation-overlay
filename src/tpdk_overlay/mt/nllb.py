"""NLLB-200 translation through transformers.

The model loads lazily on the dedicated worker thread created by
:class:`~tpdk_overlay.mt.base.WorkerTranslator`. CUDA with float16 is the fast
path; CPU falls back to float32 because CPU kernels do not implement half
precision. Torch and transformers are imported inside the helpers so importing
this module stays cheap and works when the ML extras are absent.
"""

from __future__ import annotations

import contextlib
import gc
import logging
from typing import Any

from tpdk_overlay.config import MtConfig
from tpdk_overlay.mt.base import TranslationError, WorkerTranslator
from tpdk_overlay.mt.languages import to_nllb

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "NLLB translation requires 'torch' and 'transformers'. "
    "Run: pip install torch transformers sentencepiece"
)


def _resolve_runtime(
    device: str,
    dtype: str,
    *,
    cuda_available: bool | None = None,
) -> tuple[str, object, str | None]:
    """Resolve the configured device and dtype into their torch counterparts.

    ``"auto"`` (or an empty device) picks CUDA when it is available. An explicit
    ``cuda*`` request degrades to CPU when CUDA is missing. float16 is promoted
    to float32 on CPU. Returns ``(device, torch_dtype, warning)`` where the
    warning is ``None`` when nothing had to be downgraded.
    """

    import torch

    if cuda_available is None:
        cuda_available = torch.cuda.is_available()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        torch_dtype = dtype_map[dtype.strip().lower()]
    except KeyError:
        supported = ", ".join(dtype_map)
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {supported}") from None

    requested = device.strip().lower() or "auto"
    notes: list[str] = []
    if requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
    elif requested.startswith("cuda"):
        if cuda_available:
            resolved = requested
        else:
            resolved = "cpu"
            notes.append(f"CUDA device {device!r} is unavailable; falling back to CPU")
    else:
        resolved = requested

    if resolved == "cpu" and torch_dtype is torch.float16:
        torch_dtype = torch.float32
        notes.append("float16 is not supported on CPU; using float32")

    return resolved, torch_dtype, "; ".join(notes) or None


class NllbTranslator(WorkerTranslator):
    """NLLB-200 translation via ``transformers`` on a single worker thread."""

    def __init__(self, config: MtConfig) -> None:
        super().__init__()
        self._config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._dtype: Any = None

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise TranslationError(_INSTALL_HINT) from exc

        device, torch_dtype, warning = _resolve_runtime(
            self._config.device,
            self._config.dtype,
            cuda_available=torch.cuda.is_available(),
        )
        if warning is not None:
            logger.warning("%s", warning)

        model_name = self._config.model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=to_nllb("en"))
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            ).to(device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - any load failure is a port error
            raise TranslationError(f"failed to load NLLB model {model_name!r}: {exc}") from exc

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._dtype = torch_dtype
        logger.info("NLLB model %r loaded on %s (%s)", model_name, device, torch_dtype)

    def _translate_batch(
        self,
        texts: list[str],
        source_nllb: str,
        target_nllb: str,
    ) -> list[str]:
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise TranslationError("NLLB model is not loaded")

        import torch

        tokenizer.src_lang = source_nllb
        bos_id = tokenizer.convert_tokens_to_ids(target_nllb)
        if bos_id == tokenizer.unk_token_id:
            raise TranslationError(f"tokenizer has no id for target language {target_nllb!r}")

        batch_size = self._config.batch_size
        translations: list[str] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                encoded = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._config.max_input_tokens,
                )
                moved = {name: tensor.to(self._device) for name, tensor in encoded.items()}
                generated = model.generate(
                    **moved,
                    forced_bos_token_id=bos_id,
                    max_new_tokens=self._config.max_new_tokens,
                    num_beams=1,
                )
                translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return [text.strip() for text in translations]

    def _release(self) -> None:
        was_cuda = self._device.startswith("cuda")
        self._model = None
        self._tokenizer = None
        self._dtype = None
        self._device = "cpu"
        gc.collect()
        if was_cuda:
            with contextlib.suppress(ImportError):
                import torch

                torch.cuda.empty_cache()
        logger.info("NLLB model released")


__all__ = ["NllbTranslator"]
