"""ISO 639-1 ↔ NLLB-200 (FLORES-200) language codes.

The caption demo pair is ``en``/``id``. The remaining entries come from the
earlier Live Subtitle prototype so an auto-detected source language can still be
translated instead of failing at the MT stage.
"""

from __future__ import annotations

ISO_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "yue": "yue_Hant",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "tr": "tur_Latn",
    "hi": "hin_Deva",
    "ms": "zsm_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "pl": "pol_Latn",
    "cs": "ces_Latn",
    "fil": "fil_Latn",
    "fa": "pes_Arab",
    "el": "ell_Grek",
    "hu": "hun_Latn",
    "mk": "mkd_Cyrl",
    "ro": "ron_Latn",
}

_NLLB_CODES = frozenset(ISO_TO_NLLB.values())


def to_nllb(code: str) -> str:
    """Return the NLLB-200 code for an ISO 639-1 code or an NLLB code.

    ISO codes are matched case-insensitively; FLORES codes are matched exactly.
    Raises ``ValueError`` for an empty or unsupported code.
    """

    normalized = code.strip()
    if not normalized:
        raise ValueError("language code is empty")
    if normalized in _NLLB_CODES:
        return normalized
    try:
        return ISO_TO_NLLB[normalized.lower()]
    except KeyError:
        supported = ", ".join(sorted(ISO_TO_NLLB))
        raise ValueError(
            f"unsupported language code {code!r}; supported ISO 639-1 codes: {supported}"
        ) from None


__all__ = ["ISO_TO_NLLB", "to_nllb"]
