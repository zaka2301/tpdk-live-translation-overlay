"""PaddleOCR adapter parsing against the installed 3.x result shape."""

from __future__ import annotations

from typing import Any

import numpy as np

from tpdk_overlay.config import OcrConfig
from tpdk_overlay.ocr.paddle import PaddleOcrEngine


class _FakePaddle:
    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.images: list[np.ndarray] = []

    def predict(self, image: np.ndarray) -> list[Any]:
        self.images.append(image)
        return self.results


def _results() -> list[dict[str, Any]]:
    return [
        {
            "rec_texts": ["Second line", "", "First line", "Out of bounds"],
            "rec_scores": [0.8, 0.9, 0.7, 1.2],
            "rec_polys": [
                np.array([[10, 60], [210, 60], [210, 80], [10, 80]]),
                np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                np.array([[10, 20], [110, 20], [110, 40], [10, 40]]),
                np.array([[-20, -20], [30, -20], [30, 10], [-20, 10]]),
            ],
        }
    ]


def test_recognize_parses_text_boxes_and_confidence() -> None:
    fake = _FakePaddle(_results())
    engine = PaddleOcrEngine(OcrConfig(), factory=lambda **_kwargs: fake)

    lines = engine.recognize(np.zeros((100, 200, 3), dtype=np.uint8))

    # Sorted top-to-bottom: the clipped polygon starts at y=0, then the two
    # in-bounds lines. The empty text is dropped.
    assert [line.text for line in lines] == ["Out of bounds", "First line", "Second line"]
    assert lines[1].bbox.x == 10 and lines[1].bbox.y == 20
    assert lines[1].bbox.w == 100 and lines[1].bbox.h == 20
    assert lines[1].confidence == 0.7
    # Clipped to the image and confidence clamped to 1.0.
    assert lines[0].bbox.x == 0 and lines[0].bbox.y == 0
    assert lines[0].bbox.w == 30 and lines[0].bbox.h == 10
    assert lines[0].confidence == 1.0
    assert len(fake.images) == 1


def test_nested_res_payload_and_dt_polys_fallback() -> None:
    fake = _FakePaddle(
        [
            {
                "res": {
                    "rec_texts": ["Nested"],
                    "rec_scores": [0.55],
                    "dt_polys": [np.array([[5, 5], [105, 5], [105, 25], [5, 25]])],
                }
            }
        ]
    )
    engine = PaddleOcrEngine(OcrConfig(), factory=lambda **_kwargs: fake)

    lines = engine.recognize(np.zeros((50, 120, 3), dtype=np.uint8))

    assert len(lines) == 1
    assert lines[0].text == "Nested"
    assert lines[0].bbox.w == 100 and lines[0].bbox.h == 20
    assert lines[0].confidence == 0.55


def test_engine_is_lazy_and_reused_until_closed() -> None:
    calls: list[dict[str, Any]] = []
    fake = _FakePaddle([])

    def factory(**kwargs: Any) -> _FakePaddle:
        calls.append(kwargs)
        return fake

    config = OcrConfig(
        device="cpu",
        det_model="PP-OCRv6_small_det",
        rec_model="PP-OCRv6_small_rec",
        resize_max_side=960,
    )
    engine = PaddleOcrEngine(config, factory=factory)
    blank = np.zeros((10, 10, 3), dtype=np.uint8)

    assert calls == []
    engine.recognize(blank)
    engine.recognize(blank)
    assert len(calls) == 1
    assert calls[0]["device"] == "cpu"
    assert calls[0]["text_detection_model_name"] == "PP-OCRv6_small_det"
    assert calls[0]["text_recognition_model_name"] == "PP-OCRv6_small_rec"
    assert calls[0]["text_det_limit_side_len"] == 960
    assert calls[0]["text_det_limit_type"] == "max"

    engine.close()
    engine.recognize(blank)
    assert len(calls) == 2


def test_warmup_initializes_once_and_runs_predict() -> None:
    fake = _FakePaddle([])
    created = 0

    def factory(**_kwargs: Any) -> _FakePaddle:
        nonlocal created
        created += 1
        return fake

    engine = PaddleOcrEngine(OcrConfig(), factory=factory)
    engine.warmup()
    engine.warmup()

    assert created == 1
    assert len(fake.images) == 2


def test_recognize_rejects_non_bgr_input() -> None:
    engine = PaddleOcrEngine(OcrConfig(), factory=lambda **_kwargs: _FakePaddle([]))

    try:
        engine.recognize(np.zeros((20, 20), dtype=np.uint8))
    except ValueError as exc:
        assert "BGR" in str(exc)
    else:  # pragma: no cover - the assertion above is the contract
        raise AssertionError("expected ValueError")
