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

# PLaMo-3-Translate language codes (53 languages, from the model card). Most
# ISO 639-1 codes pass through; the model distinguishes Chinese variants and
# calls Tagalog ``tl``.
PLAMO_LANGUAGE_CODES: frozenset[str] = frozenset(
    {
        "ar", "az", "be", "bg", "bn", "ca", "ceb", "de", "en", "es", "et",
        "eu", "fa", "fr", "gu", "he", "hi", "hr", "hu", "hy", "id", "it",
        "ja", "jv", "ka", "kk", "km", "ko", "lt", "lv", "ml", "mr", "ms",
        "my", "ne", "nl", "pa", "pl", "ru", "si", "sk", "sl", "sr", "sw",
        "ta", "te", "th", "tl", "ur", "uz", "vi", "zh-hans", "zh-hant",
    }
)

_PLAMO_ALIASES: dict[str, str] = {
    "zh": "zh-hans",  # project default for Chinese; zh-hant stays explicit
    "fil": "tl",
}


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


def to_plamo(code: str) -> str:
    """Return the PLaMo-3-Translate language code for a public language tag.

    ``zh`` maps to Simplified Chinese (``zh-hans``) and ``fil`` to ``tl``;
    everything else is matched case-insensitively against the model's codes.
    Raises ``ValueError`` for an empty or unsupported code.
    """

    normalized = code.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("language code is empty")
    normalized = _PLAMO_ALIASES.get(normalized, normalized)
    if normalized in PLAMO_LANGUAGE_CODES:
        return normalized
    supported = ", ".join(sorted(PLAMO_LANGUAGE_CODES))
    raise ValueError(
        f"unsupported PLaMo language code {code!r}; supported codes: {supported}"
    )


def supported_languages(provider: str) -> list[str]:
    """Return the public language codes the configured MT provider accepts.

    The overlay builds its From/To menus from this list, so every returned code
    must round-trip through that provider's adapter.
    """

    if provider == "plamo_llama":
        return sorted(PLAMO_LANGUAGE_CODES)
    return sorted(ISO_TO_NLLB)


__all__ = [
    "ISO_TO_NLLB",
    "PLAMO_LANGUAGE_CODES",
    "supported_languages",
    "to_nllb",
    "to_plamo",
]
