"""Gemini checkpoint/final summaries and windowing (FR-07, FR-08, NFR-06).

``gemini.py`` implements :class:`tpdk_overlay.ports.Summarizer` with the
``google-genai`` SDK. ``checkpoints.py`` owns the time windows, prompt assembly
from transcript + slide text, and the retry/rate-limit policy for the cloud
calls. :func:`create_summarizer` is the guarded factory used by the backend:
it returns ``None`` when summaries are disabled or no key is configured, so a
session runs transcript-only instead of failing to start.
"""

from __future__ import annotations

from tpdk_overlay.config import SummaryConfig
from tpdk_overlay.ports import Summarizer


def create_summarizer(config: SummaryConfig) -> Summarizer | None:
    """Build the configured summarizer, or ``None`` when it cannot run.

    Returns ``None`` when ``summary.enabled`` is false or the API key is
    missing. Raises ``ImportError`` when the ``summarize`` extra
    (``google-genai``) is not installed and ``ValueError`` for an unknown
    provider, so the caller can log the reason and keep summaries off.
    """

    if not config.enabled or not config.api_key():
        return None
    if config.provider == "gemini":
        try:
            import google.genai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed; install the 'summarize' extra"
            ) from exc
        from tpdk_overlay.summarize.gemini import GeminiSummarizer

        return GeminiSummarizer(config)
    raise ValueError(f"unsupported summary.provider: {config.provider!r}")


__all__ = ["create_summarizer"]
