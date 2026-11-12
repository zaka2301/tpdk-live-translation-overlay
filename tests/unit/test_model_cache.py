"""The backend keeps the translator and OCR models loaded across sessions."""

from __future__ import annotations

from tpdk_overlay.config import load_config
from tpdk_overlay.server.app import _ModelCache


def test_model_cache_reuses_one_translator() -> None:
    config = load_config()
    cache = _ModelCache(config)
    try:
        first = cache._inner_for(config)
        second = cache._inner_for(config)
        assert first is second
    finally:
        cache.close()


def test_model_cache_reuses_one_ocr_engine() -> None:
    config = load_config()
    cache = _ModelCache(config)
    try:
        first = cache._ocr_for(config)
        second = cache._ocr_for(config)
        assert first is second
    finally:
        cache.close()
