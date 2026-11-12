"""Typed ASR failures the orchestrator maps to user-visible notices."""

from __future__ import annotations


class AsrUnavailable(RuntimeError):
    """The ASR backend cannot serve a request (down, OOM, model load failure)."""


class AsrTimeout(AsrUnavailable):
    """The ASR backend did not answer in time."""


__all__ = ["AsrTimeout", "AsrUnavailable"]
