"""Colors, fonts, and stylesheets shared by the overlay windows.

The caption window is frameless, so its chrome is painted from these constants:
one translucent panel color, one accent, and one typographic scale. Config
values (opacity, font family/size) are passed in by the caller; the defaults
here are only fallbacks for standalone use.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont

# Panel and text palette. The caption panel uses ``BACKGROUND`` at
# ``CaptionConfig.opacity``; text stays fully opaque so it reads cleanly over
# translucent video.
BACKGROUND = "#0b1017"
BORDER = "#2c3d4f"
SOURCE_TEXT = "#a9bccd"
TRANSLATION_TEXT = "#f3f8fd"
STATUS_TEXT = "#8497a9"
ACCENT = "#4cc2ff"
ACCENT_HOVER = "#6fd0ff"
WARNING = "#ffb454"
ERROR = "#ff6b6b"

# Control surfaces (only visible while click-through is off).
CONTROL_BG = "#15202b"
CONTROL_HOVER = "#22303f"
CONTROL_PRESSED = "#2d3f52"
CONTROL_TEXT = "#dbe6f0"

# History window.
HISTORY_BG = "#0d131a"
HISTORY_TEXT = "#dbe6f0"
HISTORY_MUTED = "#7f93a6"


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
        background: transparent;
    }}
    QWidget#controlBar QLabel {{
        color: {STATUS_TEXT};
        font-family: '{family}';
        font-size: {small}pt;
        background: transparent;
    }}
    QWidget#controlBar QPushButton {{
        color: {CONTROL_TEXT};
        background: {CONTROL_BG};
        border: 1px solid {BORDER};
        border-radius: 7px;
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
        color: #04121b;
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
        border-radius: 7px;
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
        border-radius: 8px;
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
        border-radius: 7px;
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
