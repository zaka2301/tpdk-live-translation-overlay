"""Keep overlay windows out of the slide capture (FR-05, FR-06).

The OCR loop grabs the picked region with ``mss`` (GDI ``BitBlt`` with
``CAPTUREBLT``), which composites every visible top-level window. Our overlays
are always-on-top, so a translation painted over the region would be re-OCR'd
as slide text on the next pass, and the slide-change detector would treat every
overlay repaint as a slide change.

Windows 10 2004+ can exclude a window from capture while keeping it visible on
the monitor: ``SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)``. It was
verified on this machine (Windows 11 build 26200, Qt 6.12 translucent layered
windows): an excluded window is absent from ``mss`` grabs and reappears when the
affinity is cleared. The same call also keeps the overlays out of screen shares,
which matches the trust boundary.

Always apply from ``showEvent``: the call is idempotent, and a re-applied
affinity covers a re-created native window as well as any reset while the
window stays visible.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

# winuser.h: "the window is displayed normally but is excluded from capture".
WDA_EXCLUDEFROMCAPTURE = 0x00000011

AffinitySetter = Callable[[int], bool]

_setter: AffinitySetter | None = None


def _windows_affinity(hwnd: int) -> bool:
    """Exclude ``hwnd`` from capture via user32; False when unsupported."""

    global _setter
    if sys.platform != "win32":
        return False
    if _setter is None:
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            setter = user32.SetWindowDisplayAffinity
        except (AttributeError, OSError):  # pragma: no cover - non-Windows
            return False
        setter.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        setter.restype = ctypes.c_bool
        _setter = setter
    return bool(_setter(ctypes.c_void_p(hwnd), WDA_EXCLUDEFROMCAPTURE))


def exclude_from_capture(
    widget: QWidget, *, setter: AffinitySetter | None = None
) -> bool:
    """Hide ``widget``'s native window from screen grabs; safe to re-call.

    Runs on every show because ``setWindowDisplayAffinity`` is cheap and
    idempotent, and the affinity can be reset without the native handle
    changing. Only the failure warning is deduplicated. ``setter`` is
    injectable for tests.
    """

    if setter is None and QGuiApplication.platformName() != "windows":
        return False
    hwnd = int(widget.winId())
    if hwnd == 0:
        return False
    if (setter or _windows_affinity)(hwnd):
        widget._capture_exclusion_warned = False
        logger.debug("excluded %s from screen capture", type(widget).__name__)
        return True
    if not getattr(widget, "_capture_exclusion_warned", False):
        widget._capture_exclusion_warned = True
        logger.warning(
            "could not exclude %s from screen capture; slide OCR may read "
            "its own overlay",
            type(widget).__name__,
        )
    return False


__all__ = ["WDA_EXCLUDEFROMCAPTURE", "exclude_from_capture"]
