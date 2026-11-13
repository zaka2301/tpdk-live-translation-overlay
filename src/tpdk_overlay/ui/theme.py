"""Colors, fonts, and stylesheets shared by the overlay windows.

The caption window is frameless, so its chrome is painted from these constants:
one translucent panel color, one accent, and one typographic scale. Config
values (opacity, font family/size) are passed in by the caller; the defaults
here are only fallbacks for standalone use.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont

# Panel and text palette. Neutral graphite, no tinted chrome: the overlay is
# readable over video without looking like a colored widget. Saturated colors
# are reserved for meaning (preview deltas, warnings, errors). The caption panel
# uses ``BACKGROUND`` at ``CaptionConfig.opacity``; text stays fully opaque.
BACKGROUND = "#101010"
BORDER = "#2e2e2e"
SOURCE_TEXT = "#a8a8a8"
TRANSLATION_TEXT = "#f2f2f2"
STATUS_TEXT = "#8a8a8a"
ACCENT = "#e8e8e8"
ACCENT_HOVER = "#ffffff"
ACCENT_TEXT = "#0c0c0c"
PREVIEW_ADDED = "#8fbc8f"
WARNING = "#d9a441"
ERROR = "#d97a6c"

# Control surfaces (only visible while click-through is off).
CONTROL_BG = "#1a1a1a"
CONTROL_HOVER = "#262626"
CONTROL_PRESSED = "#303030"
CONTROL_TEXT = "#e6e6e6"

# History window.
HISTORY_BG = "#121212"
HISTORY_TEXT = "#e6e6e6"
HISTORY_MUTED = "#8a8a8a"


def rgba(color: str, alpha: float) -> str:
    """Return ``rgba(r, g, b, a)`` for a ``#rrggbb`` color and 0..1 alpha."""

    value = QColor(color)
    return f"rgba({value.red()}, {value.green()}, {value.blue()}, {alpha:.3f})"


def font(
    family: str,
    size: int,
    *,
    weight: QFont.Weight = QFont.Weight.Normal,
    italic: bool = False,
) -> QFont:
    """Build one overlay font; sizes are point sizes from the config."""

    result = QFont(family)
    result.setPointSize(max(6, size))
    result.setWeight(weight)
    result.setItalic(italic)
    return result


def controls_qss(family: str, size: int) -> str:
    """Stylesheet for the caption's control bar (scoped to ``#controlBar``)."""

    small = max(9, size - 7)
    control = max(9, size - 6)
    return f"""
    QWidget#controlBar {{
        background: {BACKGROUND};
        border: 1px solid {BORDER};
        border-radius: 4px;
    }}
    QWidget#controlBar QLabel {{
        color: {SOURCE_TEXT};
        font-family: '{family}';
        font-size: {small}pt;
        background: transparent;
    }}
    QWidget#controlBar QPushButton {{
        color: {CONTROL_TEXT};
        background: {CONTROL_BG};
        border: 1px solid {BORDER};
        border-radius: 3px;
        padding: 4px 10px;
        font-family: '{family}';
        font-size: {control}pt;
    }}
    QWidget#controlBar QPushButton:hover {{
        background: {CONTROL_HOVER};
    }}
    QWidget#controlBar QPushButton:pressed {{
        background: {CONTROL_PRESSED};
    }}
    QWidget#controlBar QPushButton:disabled {{
        color: {HISTORY_MUTED};
        background: transparent;
    }}
    QWidget#controlBar QPushButton#primaryButton {{
        color: {ACCENT_TEXT};
        background: {ACCENT};
        border: none;
        font-weight: 600;
    }}
    QWidget#controlBar QPushButton#primaryButton:hover {{
        background: {ACCENT_HOVER};
    }}
    QWidget#controlBar QComboBox,
    QWidget#controlBar QSpinBox {{
        color: {CONTROL_TEXT};
        background: {CONTROL_BG};
        border: 1px solid {BORDER};
        border-radius: 3px;
        padding: 3px 8px;
        font-family: '{family}';
        font-size: {control}pt;
    }}
    QWidget#controlBar QComboBox QAbstractItemView {{
        color: {HISTORY_TEXT};
        background: {HISTORY_BG};
        selection-background-color: {CONTROL_HOVER};
    }}
    """


def history_window_qss(family: str, size: int) -> str:
    """Stylesheet for the caption history window."""

    return f"""
    QWidget {{
        background: {HISTORY_BG};
        color: {HISTORY_TEXT};
        font-family: '{family}';
        font-size: {max(9, size)}pt;
    }}
    QTextBrowser {{
        background: {HISTORY_BG};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 10px;
        selection-background-color: {CONTROL_HOVER};
    }}
    QLabel {{
        background: transparent;
        color: {HISTORY_TEXT};
    }}
    QLabel#historyFooter {{
        color: {HISTORY_MUTED};
    }}
    QPushButton {{
        color: {CONTROL_TEXT};
        background: {CONTROL_BG};
        border: 1px solid {BORDER};
        border-radius: 3px;
        padding: 5px 14px;
        font-family: '{family}';
        font-size: {max(9, size)}pt;
    }}
    QPushButton:hover {{
        background: {CONTROL_HOVER};
    }}
    QPushButton:disabled {{
        color: {HISTORY_MUTED};
        background: transparent;
    }}
    """


def notice_color(level: str) -> str:
    """Status-line color for a notice level (``info``/``warning``/``error``)."""

    if level == "error":
        return ERROR
    if level == "warning":
        return WARNING
    return STATUS_TEXT


__all__ = [
    "ACCENT",
    "ACCENT_HOVER",
    "ACCENT_TEXT",
    "BACKGROUND",
    "BORDER",
    "CONTROL_BG",
    "CONTROL_HOVER",
    "CONTROL_PRESSED",
    "CONTROL_TEXT",
    "ERROR",
    "HISTORY_BG",
    "HISTORY_MUTED",
    "HISTORY_TEXT",
    "SOURCE_TEXT",
    "STATUS_TEXT",
    "TRANSLATION_TEXT",
    "WARNING",
    "controls_qss",
    "font",
    "history_window_qss",
    "notice_color",
    "rgba",
]
