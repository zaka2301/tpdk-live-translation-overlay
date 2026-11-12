"""PySide6 overlay windows and WebSocket client (FR-04, FR-06).

Two windows:

* ``caption.py`` - frameless translucent always-on-top caption bar with a
  click-through toggle, drag/resize, and source/translation pairs.
* ``ocr_overlay.py`` - one transparent window over the captured region that
  draws each translated block at its original bounding box.

``client.py`` runs the WebSocket client on a Qt thread and reconnects without
blocking the UI.
"""
