"""Default port factory keeps a caller-injected shared OCR engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from tpdk_overlay.config import load_config
from tpdk_overlay.pipeline.orchestrator import build_default_ports


class _Translator:
    def translate(
        self, texts: Any, *, source_lang: str, target_lang: str
    ) -> list[str]:
        return list(texts)

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        pass


class _OcrEngine:
    def __init__(self) -> None:
        self.calls = 0

    def warmup(self) -> None:
        pass

    def recognize(self, image: np.ndarray) -> list[Any]:
        self.calls += 1
        return []

    def close(self) -> None:
        pass


def test_injected_ocr_engine_is_not_replaced(tmp_path: Any) -> None:
    config = load_config().model_copy(update={"root_dir": tmp_path})
    config = config.model_copy(
        update={
            "screen": config.screen.model_copy(update={"enabled": True}),
            "ocr": config.ocr.model_copy(update={"enabled": True}),
            "summary": config.summary.model_copy(update={"enabled": False}),
        }
    )
    engine = _OcrEngine()

    ports = build_default_ports(
        config, translator=_Translator(), ocr_engine=engine
    )

    assert ports.ocr_engine is engine
    assert ports.screen_source is not None
    assert ports.change_detector is not None
    assert ports.block_grouper is not None
    # The factory must not create a second PaddleOCR behind the shared one.
    assert engine.calls == 0
