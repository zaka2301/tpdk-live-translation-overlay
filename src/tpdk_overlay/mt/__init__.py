"""NLLB-200 translation adapter (FR-03).

``nllb.py`` implements :class:`tpdk_overlay.ports.Translator` with
``facebook/nllb-200-distilled-600M``. Wire language codes are ISO 639-1; map
``en -> eng_Latn`` and ``id -> ind_Latn`` inside the adapter. The tokenizer
needs ``src_lang`` and generation needs ``forced_bos_token_id`` for the target.
"""
