"""Shared test bootstrap.

The overlay widget tests instantiate Qt widgets; force the offscreen platform so
the suite never opens a window, even on a headless runner. No other test uses
Qt, so this is process-wide and harmless.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    # The offscreen platform does not ship fonts; use the system font folder so
    # widget-render tests draw real glyphs instead of tofu boxes.
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
