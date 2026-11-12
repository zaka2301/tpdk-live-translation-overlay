"""PaddleOCR 3.x adapter for slide text (FR-05).

The engine is initialized lazily (the backend starts without loading OCR) and
warmed up once at session start. Models are ``PP-OCRv6_small_det`` +
``PP-OCRv6_small_rec`` on CPU by default: ~7.7M parameters and no VRAM cost, so
OCR stays inside the 4 GB GPU budget (NFR-05).

PaddleOCR 3.x returns a list of dict-like result objects whose top level holds
``rec_texts``, ``rec_polys``, ``rec_scores`` (and ``dt_polys``). Older code paths
wrap the same payload under ``res``; both shapes are accepted here. The
``predict`` API is used directly (``ocr()`` is deprecated in 3.x).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from tpdk_overlay.config import OcrConfig
from tpdk_overlay.contracts import BBox
from tpdk_overlay.ports import ImageArray, OcrLine

PaddleFactory = Callable[..., Any]

# Blank white canvas fed through the models once so the first real slide does
# not pay graph construction on the critical path.
_WARMUP_IMAGE = np.full((64, 256, 3), 255, dtype=np.uint8)


def _payload(result: Any) -> Any:
    """Return the dict that holds ``rec_texts`` for one PaddleOCR result."""

    if isinstance(result, dict):
        nested = result.get("res")
        return nested if isinstance(nested, dict) else result
    nested = getattr(result, "res", None)
    if isinstance(nested, dict):
        return nested
    to_dict = getattr(result, "json", None)
    if isinstance(to_dict, dict):
        nested = to_dict.get("res")
        return nested if isinstance(nested, dict) else to_dict
    return {}


def _polygon_bbox(polygon: Any, width: int, height: int) -> BBox | None:
    """Axis-aligned, image-clipped bbox for one detection polygon."""

    try:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if points.size == 0:
        return None
    left = int(np.floor(points[:, 0].min()))
    top = int(np.floor(points[:, 1].min()))
    right = int(np.ceil(points[:, 0].max()))
    bottom = int(np.ceil(points[:, 1].max()))
    left = min(max(0, left), width - 1)
    top = min(max(0, top), height - 1)
    right = min(max(left + 1, right), width)
    bottom = min(max(top + 1, bottom), height)
    if right <= left or bottom <= top:
        return None
    return BBox(x=left, y=top, w=right - left, h=bottom - top)


def _iter_lines(results: Iterable[Any], width: int, height: int) -> list[OcrLine]:
    lines: list[OcrLine] = []
    for result in results:
        payload = _payload(result)
        texts = payload.get("rec_texts")
        scores = payload.get("rec_scores")
        polygons = payload.get("rec_polys")
        if texts is None or scores is None:
            continue
        if polygons is None or len(polygons) == 0:
            polygons = payload.get("dt_polys")
        if polygons is None:
            continue
        for text, score, polygon in zip(texts, scores, polygons, strict=False):
            cleaned = " ".join(str(text).split())
            if not cleaned:
                continue
            bbox = _polygon_bbox(polygon, width, height)
            if bbox is None:
                continue
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                continue
            lines.append(
                OcrLine(
                    text=cleaned,
                    bbox=bbox,
                    confidence=min(1.0, max(0.0, confidence)),
                )
            )
    lines.sort(key=lambda line: (line.bbox.y, line.bbox.x))
    return lines


class PaddleOcrEngine:
    """``OcrEngine`` implementation backed by a lazily created PaddleOCR."""

    def __init__(
        self, config: OcrConfig, *, factory: PaddleFactory | None = None
    ) -> None:
        self._config = config
        self._factory = factory
        self._engine: Any | None = None
        # The backend preloads the shared engine on a daemon thread while the
        # OCR loop may already be running; creation and inference are serialized.
        self._engine_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def _create_engine(self) -> Any:
        factory = self._factory
        if factory is None:
            from paddleocr import PaddleOCR as factory
        return factory(
            device=self._config.device,
            text_detection_model_name=self._config.det_model,
            text_recognition_model_name=self._config.rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # The pipeline default is limit_type "min" with a tiny 64 px floor;
            # feeding it resize_max_side would upscale every frame. "max" caps
            # the long side instead (1920x1080 -> 1280x720) and never upscales.
            text_det_limit_side_len=self._config.resize_max_side,
            text_det_limit_type="max",
        )

    def _ensure_engine(self) -> Any:
        engine = self._engine
        if engine is not None:
            return engine
        with self._engine_lock:
            if self._engine is None:
                self._engine = self._create_engine()
            return self._engine

    def warmup(self) -> None:
        """Load the models and run one tiny inference; call off the event loop."""

        engine = self._ensure_engine()
        with self._predict_lock:
            engine.predict(_WARMUP_IMAGE)

    def recognize(self, image: ImageArray) -> list[OcrLine]:
        """Detect and recognize text in a BGR frame, in reading order."""

        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"expected a BGR image, got shape {image.shape!r}")
        height, width = image.shape[:2]
        engine = self._ensure_engine()
        with self._predict_lock:
            results = engine.predict(image)
        return _iter_lines(results, width, height)

    def close(self) -> None:
        """Drop the loaded models; a later ``recognize`` reinitializes them."""

        with self._engine_lock:
            self._engine = None


__all__ = ["PaddleFactory", "PaddleOcrEngine"]
