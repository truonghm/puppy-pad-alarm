"""Small checks for pad identity and pad-relative comparison."""

from __future__ import annotations

import cv2
import numpy as np

from puppy_pad_alarm.config import Settings
from puppy_pad_alarm.vision import find_candidates, locate_pads


def frame_with_pads(shift: int = 0, brown: bool = False) -> np.ndarray:
    """Draw two simple white pads with colored borders on a gray floor."""
    frame = np.full((600, 900, 3), 115, np.uint8)
    for left, border in ((60 + shift, (255, 0, 0)), (470 - shift, (180, 70, 210))):
        cv2.rectangle(frame, (left, 120), (left + 340, 480), border, -1)
        cv2.rectangle(frame, (left + 12, 132), (left + 328, 468), (245, 245, 245), -1)
    if brown:
        cv2.circle(frame, (230 + shift, 300), 20, (45, 85, 135), -1)
    return frame


def test_moving_pads_keep_identity_and_do_not_create_object() -> None:
    """Pad movement alone should leave both aligned interiors unchanged."""
    settings = Settings(search_region=[0, 0, 900, 600])
    baseline_pads, _ = locate_pads(frame_with_pads(), settings)
    moved_pads, reasons = locate_pads(frame_with_pads(shift=35), settings)
    assert set(baseline_pads) == set(moved_pads) == {"blue", "pink"}
    assert reasons == {}
    for color in ("blue", "pink"):
        candidates, reason = find_candidates(
            moved_pads[color], baseline_pads[color].image, None, settings
        )
        assert reason is None
        assert candidates == []


def test_solid_colored_block_is_not_a_pad() -> None:
    """A colored object without a white surface fails localization."""
    settings = Settings(search_region=[0, 0, 900, 600])
    frame = np.full((600, 900, 3), 115, np.uint8)
    cv2.rectangle(frame, (60, 120), (400, 480), (255, 0, 0), -1)
    pads, reasons = locate_pads(frame, settings)
    assert "blue" not in pads
    assert "blue" in reasons


def test_new_brown_region_is_found_after_pad_moves() -> None:
    """Comparison uses the current pad position after localization."""
    settings = Settings(search_region=[0, 0, 900, 600])
    baseline = locate_pads(frame_with_pads(), settings)[0]["blue"].image
    current = locate_pads(frame_with_pads(shift=35, brown=True), settings)[0]["blue"]
    candidates, reason = find_candidates(current, baseline, None, settings)
    assert reason is None
    assert len(candidates) == 1


def test_widespread_change_is_inconclusive() -> None:
    """A large alignment or color error must not make a poop claim."""
    settings = Settings(search_region=[0, 0, 900, 600])
    pad = locate_pads(frame_with_pads(), settings)[0]["blue"]
    baseline = pad.image.copy()
    pad.image[20:-20, 20:-20] = (45, 85, 135)
    candidates, reason = find_candidates(pad, baseline, None, settings)
    assert candidates == []
    assert reason == "Pad alignment or lighting is uncertain"
