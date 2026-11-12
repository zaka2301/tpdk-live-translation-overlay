"""Screen captures and slide-change detection (FR-05, NFR-03).

``capture.py`` implements :class:`tpdk_overlay.ports.ScreenSource` with ``mss``.
``change.py`` downsamples/grayscales consecutive frames and reports the fraction
of changed pixels so OCR only runs when the slide actually changed.
"""
