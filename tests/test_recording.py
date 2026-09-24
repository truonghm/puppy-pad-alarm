"""Checks for local dog visit recordings."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from puppy_pad_alarm.config import Settings
from puppy_pad_alarm.recording import EventRecorder


def frame(value: int) -> np.ndarray:
    """Make one small camera frame with a visible time marker."""
    return np.full((120, 160, 3), value, np.uint8)


def test_visit_video_has_pre_roll_and_post_roll(tmp_path: Path) -> None:
    """A short visit produces a replayable clip with surrounding frames."""
    settings = Settings(
        event_video_fps=1.0, event_video_pre_roll_s=2.0, event_video_post_roll_s=2.0
    )
    recorder = EventRecorder(tmp_path, settings, logging.getLogger("test_recording"))
    states = {"region": "READY"}
    for second in range(8):
        recorder.observe(frame(second * 25), float(second))
        recorder.visit(
            second in (3, 4),
            float(second),
            1_000.0 + second,
            0.8 if second in (3, 4) else 0.0,
            states,
        )
        if second == 4:
            recorder.mark_detection(
                4.0, 1_004.0, tmp_path / "event.jpg", states
            )
    clips = list(tmp_path.glob("visit_*.mp4"))
    assert len(clips) == 1
    capture = cv2.VideoCapture(str(clips[0]))
    assert capture.isOpened()
    values = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        values.append(float(np.mean(image)))
    capture.release()
    assert len(values) >= 6
    assert values[0] < 40
    assert values[-1] > 110
    metadata = json.loads(clips[0].with_suffix(".json").read_text())
    assert metadata["dog_visit_count"] == 1
    assert metadata["max_dog_confidence"] == 0.8
    assert metadata["detections"][0]["snapshot"] == "event.jpg"
    assert metadata["detections"][0]["snapshot"] == "event.jpg"
    assert metadata["true_label"] is None


def test_return_during_tail_extends_one_clip(tmp_path: Path) -> None:
    """Two close visits stay in one recording."""
    settings = Settings(
        event_video_fps=1.0, event_video_pre_roll_s=1.0, event_video_post_roll_s=2.0
    )
    recorder = EventRecorder(tmp_path, settings, logging.getLogger("test_recording"))
    states = {"region": "READY"}
    for second in range(8):
        recorder.observe(frame(second * 20), float(second))
        recorder.visit(second in (2, 4), float(second), 1_000.0 + second, 0.7, states)
    clips = list(tmp_path.glob("visit_*.mp4"))
    assert len(clips) == 1
    metadata = json.loads(clips[0].with_suffix(".json").read_text())
    assert metadata["dog_visit_count"] == 2


def test_retention_removes_oldest_clip_and_sidecar(tmp_path: Path) -> None:
    """The storage cap deletes only the oldest event video pair."""
    settings = Settings(event_video_max_gb=0.000001, event_video_retention_days=7)
    recorder = EventRecorder(tmp_path, settings, logging.getLogger("test_recording"))
    clips = []
    for index in range(3):
        clip = tmp_path / f"visit_{index}.mp4"
        clip.write_bytes(b"x" * 400)
        clip.with_suffix(".json").write_text("{}")
        os.utime(clip, (1_000.0 + index, 1_000.0 + index))
        clips.append(clip)
    recorder.prune(now=1_003.0)
    assert not clips[0].exists()
    assert not clips[0].with_suffix(".json").exists()
    assert clips[1].exists() and clips[2].exists()
    recorder.prune(now=1_003.0 + 8 * 86_400)
    assert not clips[1].exists() and not clips[2].exists()
