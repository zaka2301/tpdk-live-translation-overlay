from __future__ import annotations

import pytest

from tpdk_overlay.mt.languages import ISO_TO_NLLB, to_nllb


def test_demo_pair_maps_to_flores() -> None:
    assert to_nllb("en") == "eng_Latn"
    assert to_nllb("id") == "ind_Latn"


def test_iso_lookup_is_case_insensitive_and_trims() -> None:
    assert to_nllb(" EN ") == "eng_Latn"
    assert to_nllb("Id") == "ind_Latn"


def test_flores_codes_pass_through() -> None:
    assert to_nllb("eng_Latn") == "eng_Latn"
    assert to_nllb("ind_Latn") == "ind_Latn"


def test_mapping_values_are_unique() -> None:
    assert len(set(ISO_TO_NLLB.values())) == len(ISO_TO_NLLB)


@pytest.mark.parametrize("code", ["", "   ", "xx", "english", "eng_latn", "en-US"])
def test_unsupported_code_raises(code: str) -> None:
    with pytest.raises(ValueError, match="unsupported language code|empty"):
        to_nllb(code)
