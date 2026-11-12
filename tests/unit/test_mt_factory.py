from __future__ import annotations

from types import SimpleNamespace

import pytest

from tpdk_overlay.config import MtConfig
from tpdk_overlay.mt import create_translator


def test_factory_selects_transformer_adapter() -> None:
    from tpdk_overlay.mt.nllb import NllbTranslator

    translator = create_translator(MtConfig(provider="nllb"))
    try:
        assert isinstance(translator, NllbTranslator)
    finally:
        translator.close()


def test_factory_selects_plamo_adapter() -> None:
    from tpdk_overlay.mt.plamo import PlamoTranslator

    translator = create_translator(MtConfig(provider="plamo_llama"))
    try:
        assert isinstance(translator, PlamoTranslator)
    finally:
        translator.close()


def test_factory_selects_ct2_adapter() -> None:
    from tpdk_overlay.mt.nllb_ct2 import NllbCt2Translator

    translator = create_translator(MtConfig(provider="nllb_ct2"))
    try:
        assert isinstance(translator, NllbCt2Translator)
    finally:
        translator.close()


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unsupported mt.provider"):
        create_translator(SimpleNamespace(provider="other"))
