"""Slide-change detection keeps cursor noise below the OCR threshold."""

from __future__ import annotations

import numpy as np

from tpdk_overlay.config import ScreenConfig
from tpdk_overlay.screen.change import SlideChangeDetector


def _frame(height: int = 360, width: int = 640) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[100:140, 40:300] = 200  # some slide text-ish blocks
    image[160:180, 40:250] = 120
    return image


def test_first_frame_always_counts_as_changed() -> None:
    detector = SlideChangeDetector(ScreenConfig())

    assert detector.score(_frame()) == 1.0


def test_identical_frames_have_zero_change() -> None:
    detector = SlideChangeDetector(ScreenConfig())
    frame = _frame()
    detector.score(frame)

    assert detector.score(frame.copy()) == 0.0


def test_slide_change_scores_well_above_the_threshold() -> None:
    detector = SlideChangeDetector(ScreenConfig(change_threshold=0.02))
    frame = _frame()
    detector.score(frame)

    changed = frame.copy()
    changed[40:200, 320:620] = 255
    score = detector.score(changed)

    assert score > 0.02


def test_cursor_sized_change_is_ignored() -> None:
    detector = SlideChangeDetector(ScreenConfig(change_threshold=0.02))
    frame = _frame()
    detector.score(frame)

    with_cursor = frame.copy()
    with_cursor[50:62, 400:412] = 255  # a 12 px cursor/caret
    score = detector.score(with_cursor)

    assert score < 0.005


def test_reset_forces_the_next_frame_to_compare_as_changed() -> None:
    detector = SlideChangeDetector(ScreenConfig())
    frame = _frame()
    detector.score(frame)
    detector.score(frame)
    detector.reset()

    assert detector.score(frame) == 1.0


def test_shape_change_is_treated_as_a_new_slide() -> None:
    detector = SlideChangeDetector(ScreenConfig())
    detector.score(_frame())
    detector.score(_frame())

    assert detector.score(_frame(height=400)) == 1.0
