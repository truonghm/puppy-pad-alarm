"""Small checks for pad identity and pad-relative comparison."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from puppy_pad_alarm.config import Settings, load_settings
from puppy_pad_alarm.vision import find_candidates, locate_pads


def frame_with_pads(
    shift: int = 0,
    object_color: tuple[int, int, int] | None = None,
    object_radius: int = 20,
) -> np.ndarray:
    """Draw two simple white pads with colored borders on a gray floor."""
    frame = np.full((600, 900, 3), 115, np.uint8)
    for left, border in ((60 + shift, (255, 0, 0)), (470 - shift, (180, 70, 210))):
        cv2.rectangle(frame, (left, 120), (left + 340, 480), border, -1)
        cv2.rectangle(frame, (left + 12, 132), (left + 328, 468), (245, 245, 245), -1)
    if object_color is not None:
        cv2.circle(frame, (230 + shift, 300), object_radius, object_color, -1)
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


def test_new_dark_regions_are_found_after_pad_moves() -> None:
    """Black and brown objects survive pad-relative comparison."""
    settings = Settings(search_region=[0, 0, 900, 600])
    baseline = locate_pads(frame_with_pads(), settings)[0]["blue"].image
    for color, radius in (((45, 85, 135), 20), ((10, 10, 10), 20), ((10, 10, 10), 60)):
        current = locate_pads(
            frame_with_pads(shift=35, object_color=color, object_radius=radius),
            settings,
        )[0]["blue"]
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


def test_old_brown_settings_do_not_break_saved_config(tmp_path: Path) -> None:
    """A saved config from the brown-only version still starts."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "brown_hsv_low: [5, 45, 20]\n"
        "brown_hsv_high: [28, 255, 210]\n"
        "min_object_area: 45\n"
        "max_object_area: 10000\n"
        "max_changed_fraction: 0.45\n"
    )
    settings = load_settings(path)
    assert settings.max_dark_value == 220
    assert settings.min_object_area == 20
    assert settings.max_object_area == 30000
    assert settings.max_changed_fraction == 0.65
