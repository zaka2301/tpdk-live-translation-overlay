"""Gemini checkpoint/final summaries and windowing (FR-07, FR-08, NFR-06).

``gemini.py`` implements :class:`tpdk_overlay.ports.Summarizer` with the
``google-genai`` SDK. ``checkpoints.py`` owns the time windows, prompt assembly
from transcript + slide text, and final synthesis from partial summaries.
"""
