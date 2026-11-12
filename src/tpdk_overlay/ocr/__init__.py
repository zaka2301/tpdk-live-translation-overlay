"""PP-OCRv6 adapter and block grouping (FR-05, FR-06).

``paddle.py`` implements :class:`tpdk_overlay.ports.OcrEngine` with PaddleOCR
3.7+ (``PP-OCRv6_small_det`` + ``PP-OCRv6_small_rec``), CPU by default to protect
the 4 GB GPU budget.

``grouping.py`` merges raw OCR lines into spatially coherent, reading-order
blocks so translation and per-block toggling operate on sentences, not fragments.
"""
