"""Slide-change detection keeps cursor noise below the OCR threshold."""

from __future__ import annotations

import numpy as np

from tpdk_overlay.config import ScreenConfig
from tpdk_overlay.contracts import BBox
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


def test_masked_overlay_repaint_is_not_a_slide_change() -> None:
    detector = SlideChangeDetector(ScreenConfig(change_threshold=0.02))
    frame = _frame()
    detector.score(frame)

    painted = frame.copy()
    painted[100:140, 40:300] = 20  # overlay panel over the first text block
    mask = BBox(x=40, y=100, w=260, h=40)

    assert detector.score(painted, mask=[mask]) == 0.0


def test_change_outside_the_mask_still_triggers() -> None:
    detector = SlideChangeDetector(ScreenConfig(change_threshold=0.02))
    frame = _frame()
    detector.score(frame)

    painted = frame.copy()
    painted[100:140, 40:300] = 20  # overlay panel (masked)
    painted[250:330, 400:620] = 255  # new slide content (not masked)
    mask = BBox(x=40, y=100, w=260, h=40)

    assert detector.score(painted, mask=[mask]) > 0.02


def test_fully_masked_frame_scores_zero() -> None:
    detector = SlideChangeDetector(ScreenConfig())
    frame = _frame()
    detector.score(np.zeros_like(frame))

    assert detector.score(frame, mask=[BBox(x=0, y=0, w=640, h=360)]) == 0.0


def test_remember_replaces_the_reference_frame() -> None:
    detector = SlideChangeDetector(ScreenConfig())
    frame = _frame()
    detector.score(frame)
    changed = frame.copy()
    changed[250:330, 400:620] = 255
    detector.remember(changed)

    assert detector.score(changed.copy()) == 0.0
