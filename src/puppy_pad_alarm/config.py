"""Configuration and durable state for the local alarm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class Settings:
    """User adjustable camera and image processing settings."""

    camera_name: str = "C270 HD WEBCAM"
    camera_source: int | str | None = None
    search_region: list[int] | None = None
    output_dir: str = "alarm_data"
    model: str = "yolo11n.pt"
    inference_interval_s: float = 0.2
    dog_confidence: float = 0.35
    border_hsv: dict[str, list[list[int]]] = field(
        default_factory=lambda: {
            "blue": [[90, 55, 45], [135, 255, 255]],
            "pink": [[140, 40, 45], [179, 255, 255]],
        }
    )
    min_pad_area: int = 4000
    max_pad_area_fraction: float = 0.45
    min_border_pixels: int = 80
    white_value_min: int = 170
    white_saturation_max: int = 80
    min_white_fraction: float = 0.45
    border_exclusion_px: int = 9
    min_object_area: int = 20
    max_object_area: int = 30000
    min_delta_lab: float = 19.0
    max_changed_fraction: float = 0.65
    max_dark_value: int = 220
    min_persistence_frames: int = 3
    min_persistence_s: float = 0.5
    stable_distance_px: float = 25.0
    checking_timeout_s: float = 8.0
    clear_frames: int = 3
    dog_proximity_margin_px: int = 40
    pad_position_max_age_s: float = 10.0
    deterrent_cooldown_s: float = 3.0
    dog_away_confirm_s: float = 1.0
    save_debug_video: bool = False
    save_event_video: bool = True
    event_video_pre_roll_s: float = 10.0
    event_video_post_roll_s: float = 20.0
    event_video_fps: float = 10.0
    event_video_retention_days: int = 7
    event_video_max_gb: float = 5.0


def load_settings(path: Path) -> Settings:
    """Read settings and reject unknown keys to expose configuration mistakes."""
    if not path.exists():
        return Settings()
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    old_color_filter = "brown_hsv_low" in data or "brown_hsv_high" in data
    if old_color_filter:
        data.pop("brown_hsv_low", None)
        data.pop("brown_hsv_high", None)
        for key, old, new in (
            ("min_object_area", 45, 20),
            ("max_object_area", 10000, 30000),
            ("max_changed_fraction", 0.45, 0.65),
        ):
            if data.get(key) == old:
                data[key] = new
    unknown = set(data) - set(Settings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return Settings(**data)


def save_settings(path: Path, settings: Settings) -> None:
    """Save the full configuration after calibration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(settings), sort_keys=False))
