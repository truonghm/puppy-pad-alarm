"""Find new dark regions inside one calibrated camera area."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings


@dataclass
class Candidate:
    """A new dark region in coordinates relative to the selected area."""

    center: tuple[float, float]
    contour: np.ndarray
    area: int


def crop_region(frame: np.ndarray, region: list[int]) -> np.ndarray | None:
    """Return the whole selected area when it fits inside the camera frame."""
    x, y, width, height = region
    frame_height, frame_width = frame.shape[:2]
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return None
    if x + width > frame_width or y + height > frame_height:
        return None
    return frame[y : y + height, x : x + width]


def find_candidates(
    current: np.ndarray,
    baseline: np.ndarray,
    dog_box: tuple[int, int, int, int] | None,
    region: list[int],
    settings: Settings,
) -> tuple[list[Candidate], str | None]:
    """Find new dark regions that differ from the clean area."""
    if baseline.shape != current.shape:
        return [], "Clean baseline does not match the selected area; press B"
    visible = np.full(current.shape[:2], 255, np.uint8)
    if dog_box is not None:
        x, y, _, _ = region
        x1, y1, x2, y2 = dog_box
        cv2.rectangle(visible, (x1 - x - 8, y1 - y - 8), (x2 - x + 8, y2 - y + 8), 0, -1)
    if cv2.countNonZero(visible) < visible.size * 0.25:
        return [], "Dog hides most of the selected area"
    current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB).astype(np.float32)
    difference = current_lab - base_lab
    difference[:, :, 0] -= float(np.median(difference[:, :, 0][visible > 0]))
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    changed = distance >= settings.min_delta_lab
    if float(np.mean(changed[visible > 0])) > settings.max_changed_fraction:
        return [], "Lighting or camera position changed too much; press B if clean"
    dark = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)[:, :, 2] <= settings.max_dark_value
    mask = np.where((visible > 0) & changed & dark, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Candidate] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if settings.min_object_area <= area <= settings.max_object_area:
            moments = cv2.moments(contour)
            if moments["m00"]:
                candidates.append(Candidate(
                    (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                    contour,
                    area,
                ))
    return candidates, None
