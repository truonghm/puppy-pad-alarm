"""Checks for one selected area's dark-object detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from puppy_pad_alarm.config import Settings, load_settings
from puppy_pad_alarm.vision import crop_region, find_candidates


def test_dark_rectangle_is_found_after_dog_visit() -> None:
    """A new black rectangle is a candidate in the selected area."""
    settings = Settings()
    region = [50, 40, 400, 250]
    baseline = np.full((360, 600, 3), 240, np.uint8)
    current = baseline.copy()
    cv2.rectangle(current, (200, 130), (245, 155), (10, 10, 10), -1)
    clean = crop_region(baseline, region)
    changed = crop_region(current, region)
    assert clean is not None and changed is not None
    candidates, reason = find_candidates(changed, clean, None, region, settings)
    assert reason is None
    assert len(candidates) == 1


def test_dog_covering_object_defers_detection() -> None:
    """An object hidden by the dog box is not confirmed yet."""
    settings = Settings()
    region = [50, 40, 400, 250]
    clean = np.full((250, 400, 3), 240, np.uint8)
    changed = clean.copy()
    cv2.rectangle(changed, (150, 90), (195, 115), (10, 10, 10), -1)
    candidates, _ = find_candidates(changed, clean, (190, 120, 260, 170), region, settings)
    assert candidates == []


def test_dark_dog_is_masked_but_separate_object_is_found() -> None:
    """A tight dog box does not turn dark fur into an alert candidate."""
    settings = Settings()
    region = [0, 0, 400, 250]
    clean = np.full((250, 400, 3), 240, np.uint8)
    changed = clean.copy()
    cv2.rectangle(changed, (70, 60), (160, 170), (10, 10, 10), -1)
    dog_box = (90, 80, 140, 150)
    candidates, reason = find_candidates(changed, clean, dog_box, region, settings)
    assert reason is None
    assert candidates == []

    cv2.rectangle(changed, (260, 100), (290, 130), (10, 10, 10), -1)
    candidates, reason = find_candidates(changed, clean, dog_box, region, settings)
    assert reason is None
    assert len(candidates) == 1


def test_example_config_loads_without_edits() -> None:
    """The copyable example uses only supported settings."""
    example = Path(__file__).resolve().parents[1] / "config.example.yaml"
    assert load_settings(example).camera_name == "C270 HD WEBCAM"
