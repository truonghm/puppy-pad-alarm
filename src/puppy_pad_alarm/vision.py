"""Pad localization, alignment, and local change detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings

PAD_SIZE = (320, 240)
PAD_TARGET = np.array(
    [
        [0, 0],
        [PAD_SIZE[0] - 1, 0],
        [PAD_SIZE[0] - 1, PAD_SIZE[1] - 1],
        [0, PAD_SIZE[1] - 1],
    ],
    np.float32,
)


@dataclass
class Pad:
    """A currently visible pad and its frame-to-reference transform."""

    color: str
    corners: np.ndarray
    image: np.ndarray
    valid: np.ndarray


@dataclass
class Candidate:
    """A new dark region in pad coordinates."""

    center: tuple[float, float]
    contour: np.ndarray
    area: int


def locate_pads(
    frame: np.ndarray, settings: Settings
) -> tuple[dict[str, Pad], dict[str, str]]:
    """Find colored rectangular borders and warp their interiors."""
    height, width = frame.shape[:2]
    x, y, w, h = settings.search_region or [0, 0, width, height]
    x, y = max(0, x), max(0, y)
    crop = frame[y : min(y + h, height), x : min(x + w, width)]
    if crop.size == 0:
        return {}, {
            color: "Search region is outside the frame" for color in settings.border_hsv
        }
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pads: dict[str, Pad] = {}
    reasons: dict[str, str] = {}
    for color, bounds in settings.border_hsv.items():
        mask = cv2.inRange(
            hsv, np.array(bounds[0], np.uint8), np.array(bounds[1], np.uint8)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: tuple[float, np.ndarray] | None = None
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            _, (rw, rh), _ = rect
            area = rw * rh
            if (
                area < settings.min_pad_area
                or area > crop.shape[0] * crop.shape[1] * settings.max_pad_area_fraction
            ):
                continue
            if min(rw, rh) < 25 or max(rw, rh) / min(rw, rh) > 3.5:
                continue
            box = cv2.boxPoints(rect).astype(np.float32)
            polygon = np.zeros(mask.shape, np.uint8)
            cv2.fillConvexPoly(polygon, box.astype(np.int32), 255)
            border_count = cv2.countNonZero(cv2.bitwise_and(mask, polygon))
            if border_count < settings.min_border_pixels:
                continue
            trial = cv2.warpPerspective(
                crop,
                cv2.getPerspectiveTransform(order_corners(box), PAD_TARGET),
                PAD_SIZE,
            )
            interior = cv2.cvtColor(trial[20:-20, 20:-20], cv2.COLOR_BGR2HSV)
            white = (interior[:, :, 1] <= settings.white_saturation_max) & (
                interior[:, :, 2] >= settings.white_value_min
            )
            if float(np.mean(white)) < settings.min_white_fraction:
                continue
            score = border_count / max(area, 1)
            if best is None or score > best[0]:
                best = (score, box)
        if best is None:
            reasons[color] = "Colored pad border not visible"
            continue
        corners = order_corners(best[1]) + np.array([x, y], np.float32)
        transform = cv2.getPerspectiveTransform(corners, PAD_TARGET)
        image = cv2.warpPerspective(frame, transform, PAD_SIZE)
        inset = settings.border_exclusion_px
        valid = np.zeros((PAD_SIZE[1], PAD_SIZE[0]), np.uint8)
        valid[inset : PAD_SIZE[1] - inset, inset : PAD_SIZE[0] - inset] = 255
        pads[color] = Pad(color, corners, image, valid)
    return pads, reasons


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order rectangle points clockwise from the top left."""
    sums = corners.sum(axis=1)
    differences = np.diff(corners, axis=1).reshape(-1)
    return np.array(
        [
            corners[np.argmin(sums)],
            corners[np.argmin(differences)],
            corners[np.argmax(sums)],
            corners[np.argmax(differences)],
        ],
        np.float32,
    )


def dog_mask_on_pad(pad: Pad, dog_box: tuple[int, int, int, int] | None) -> np.ndarray:
    """Mask pixels covered by the current dog detection."""
    mask = np.zeros((PAD_SIZE[1], PAD_SIZE[0]), np.uint8)
    if dog_box is None:
        return mask
    x1, y1, x2, y2 = dog_box
    source = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    transform = cv2.getPerspectiveTransform(pad.corners, PAD_TARGET)
    warped = cv2.perspectiveTransform(source[None, :, :], transform)[0]
    cv2.fillConvexPoly(mask, warped.astype(np.int32), 255)
    return cv2.dilate(mask, np.ones((15, 15), np.uint8))


def find_candidates(
    pad: Pad,
    baseline: np.ndarray,
    dog_box: tuple[int, int, int, int] | None,
    settings: Settings,
) -> tuple[list[Candidate], str | None]:
    """Find new dark-region proposals relative to the clean reference."""
    if baseline.shape != pad.image.shape:
        return [], "Baseline dimensions do not match"
    occluded = dog_mask_on_pad(pad, dog_box)
    visible = cv2.bitwise_and(pad.valid, cv2.bitwise_not(occluded))
    if cv2.countNonZero(visible) < cv2.countNonZero(pad.valid) * 0.25:
        return [], "Dog hides most of the pad"
    current_lab = cv2.cvtColor(pad.image, cv2.COLOR_BGR2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB).astype(np.float32)
    difference = current_lab - base_lab
    # Remove the common brightness shift; chroma differences remain local.
    median_light = float(np.median(difference[:, :, 0][visible > 0]))
    difference[:, :, 0] -= median_light
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    changed_fraction = float(np.mean(distance[visible > 0] >= settings.min_delta_lab))
    if changed_fraction > settings.max_changed_fraction:
        return [], "Pad alignment or lighting is uncertain"
    hsv = cv2.cvtColor(pad.image, cv2.COLOR_BGR2HSV)
    dark = hsv[:, :, 2] <= settings.max_dark_value
    changed = np.uint8((distance >= settings.min_delta_lab) & dark) * 255
    changed = np.where(visible > 0, changed, 0).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if settings.min_object_area <= area <= settings.max_object_area:
            moments = cv2.moments(contour)
            if moments["m00"]:
                candidates.append(
                    Candidate(
                        (
                            moments["m10"] / moments["m00"],
                            moments["m01"] / moments["m00"],
                        ),
                        contour,
                        area,
                    )
                )
    return candidates, None
