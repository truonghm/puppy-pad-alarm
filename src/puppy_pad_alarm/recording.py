"""Bounded local video capture around dog visits."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .config import Settings


@dataclass
class BufferedFrame:
    """One compressed frame held for a possible visit."""

    at: float
    image: bytes


class EventRecorder:
    """Save visit clips with pre-roll, post-roll, and bounded retention."""

    def __init__(
        self, directory: Path, settings: Settings, logger: logging.Logger
    ) -> None:
        if (
            settings.event_video_fps <= 0
            or settings.event_video_pre_roll_s < 0
            or settings.event_video_post_roll_s < 0
        ):
            raise ValueError(
                "Event video FPS must be positive and clip durations cannot be negative"
            )
        if settings.event_video_retention_days <= 0 or settings.event_video_max_gb <= 0:
            raise ValueError(
                "Event video retention days and storage limit must be positive"
            )
        self.directory = directory
        self.settings = settings
        self.logger = logger
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames: deque[BufferedFrame] = deque()
        self.last_sample_at: float | None = None
        self.writer: cv2.VideoWriter | None = None
        self.video_path: Path | None = None
        self.next_tick: float | None = None
        self.last_frame: np.ndarray | None = None
        self.last_interest_at: float | None = None
        self.in_visit = False
        self.disabled = False
        self.metadata: dict[str, object] = {}
        self.prune()

    def observe(self, frame: np.ndarray, now: float) -> None:
        """Keep a short compressed history and write sampled live frames."""
        if self.disabled:
            return
        interval = 1.0 / self.settings.event_video_fps
        if self.last_sample_at is not None and now - self.last_sample_at < interval:
            return
        self.last_sample_at = now
        try:
            encoded, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except cv2.error as error:
            self._disable(error)
            return
        if not encoded:
            self.logger.error("Could not buffer a camera frame for event video")
            return
        self.frames.append(BufferedFrame(now, image.tobytes()))
        while (
            self.frames
            and now - self.frames[0].at > self.settings.event_video_pre_roll_s
        ):
            self.frames.popleft()
        if self.writer is not None:
            self._write_at(now, frame)

    def visit(
        self,
        present: bool | None,
        now: float,
        wall_time: float,
        confidence: float,
        pad_states: dict[str, str],
    ) -> None:
        """Start or extend a clip when a dog overlaps the search region."""
        if self.disabled:
            return
        if present is True:
            if self.writer is None:
                self._start(now, wall_time, "dog_visit", pad_states)
            if self.writer is None:
                return
            if not self.in_visit:
                visit_count = self.metadata["dog_visit_count"]
                assert isinstance(visit_count, int)
                self.metadata["dog_visit_count"] = visit_count + 1
            max_confidence = self.metadata["max_dog_confidence"]
            assert isinstance(max_confidence, float)
            self.metadata["max_dog_confidence"] = max(max_confidence, confidence)
            self.last_interest_at = now
            self.in_visit = True
        elif present is False:
            self.in_visit = False
        if (
            self.writer is not None
            and self.last_interest_at is not None
            and now - self.last_interest_at >= self.settings.event_video_post_roll_s
        ):
            self.close(now, wall_time, pad_states)

    def mark_detection(
        self,
        color: str,
        now: float,
        wall_time: float,
        snapshot: Path | None,
        pad_states: dict[str, str],
    ) -> None:
        """Record a confirmed local event and extend its video tail."""
        if self.disabled:
            return
        if self.writer is None:
            self._start(now, wall_time, "detection", pad_states)
        if self.writer is None:
            return
        detections = self.metadata["detections"]
        assert isinstance(detections, list)
        detections.append(
            {
                "pad": color,
                "at_utc": self._utc(wall_time),
                "snapshot": snapshot.name if snapshot else None,
            }
        )
        self.last_interest_at = now

    def close(self, now: float, wall_time: float, pad_states: dict[str, str]) -> None:
        """Finish the active clip and write its review metadata."""
        if self.writer is None or self.video_path is None:
            return
        if self.last_frame is not None:
            self._write_at(now, self.last_frame)
        if self.writer is None:
            return
        try:
            self.writer.release()
        except (cv2.error, OSError) as error:
            self.logger.error(
                "Could not finalize event video %s: %s", self.video_path, error
            )
        self.writer = None
        video_path = self.video_path
        self.video_path = None
        self.next_tick = None
        self.last_frame = None
        self.last_interest_at = None
        self.in_visit = False
        self.metadata["ended_at_utc"] = self._utc(wall_time)
        self.metadata["final_pad_states"] = pad_states
        sidecar = video_path.with_suffix(".json")
        try:
            temporary = sidecar.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.metadata, indent=2))
            temporary.replace(sidecar)
            self.logger.info("Event video saved: %s", video_path)
        except OSError as error:
            self.logger.error(
                "Could not save event video metadata for %s: %s", video_path, error
            )
        self.prune(wall_time)

    def prune(self, now: float | None = None) -> None:
        """Remove only old event clips, oldest first, within the configured limits."""
        now = datetime.now(UTC).timestamp() if now is None else now
        clips = sorted(
            self.directory.glob("visit_*.mp4"), key=lambda path: path.stat().st_mtime
        )
        maximum_age = self.settings.event_video_retention_days * 86_400
        for clip in clips:
            if now - clip.stat().st_mtime > maximum_age:
                self._remove_pair(clip)
        clips = [clip for clip in clips if clip.exists()]
        maximum_bytes = int(self.settings.event_video_max_gb * 1_000_000_000)
        total = sum(self._pair_size(clip) for clip in clips)
        for clip in clips:
            if total <= maximum_bytes:
                break
            size = self._pair_size(clip)
            if self._remove_pair(clip):
                total -= size

    def _start(
        self, now: float, wall_time: float, trigger: str, pad_states: dict[str, str]
    ) -> None:
        """Open a clip and flush the buffered frames before the trigger."""
        stamp = datetime.fromtimestamp(wall_time, UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        path = self.directory / f"visit_{stamp}.mp4"
        first = (
            cv2.imdecode(
                np.frombuffer(self.frames[0].image, np.uint8), cv2.IMREAD_COLOR
            )
            if self.frames
            else None
        )
        if first is None:
            self.logger.error("Could not start event video: no buffered camera frame")
            return
        try:
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter.fourcc(*"mp4v"),
                self.settings.event_video_fps,
                (first.shape[1], first.shape[0]),
            )
        except (cv2.error, OSError) as error:
            self._disable(error)
            return
        if not writer.isOpened():
            writer.release()
            self.disabled = True
            self.logger.error(
                "Could not open event video writer at %s; recording is disabled for this run",
                path,
            )
            return
        self.writer = writer
        self.video_path = path
        self.next_tick = None
        self.last_frame = None
        self.metadata = {
            "started_at_utc": self._utc(wall_time - (now - self.frames[0].at)),
            "trigger": trigger,
            "dog_visit_count": 0,
            "max_dog_confidence": 0.0,
            "initial_pad_states": pad_states.copy(),
            "detections": [],
            "true_label": None,
            "video_fps": self.settings.event_video_fps,
        }
        for buffered in self.frames:
            image = cv2.imdecode(
                np.frombuffer(buffered.image, np.uint8), cv2.IMREAD_COLOR
            )
            if image is not None:
                self._write_at(buffered.at, image)

    def _write_at(self, now: float, frame: np.ndarray) -> None:
        """Write at a fixed frame rate, filling short processing gaps."""
        if self.writer is None:
            return
        if self.next_tick is None:
            self.next_tick = now
        interval = 1.0 / self.settings.event_video_fps
        while self.next_tick <= now + 1e-6:
            try:
                self.writer.write(frame)
            except (cv2.error, OSError) as error:
                self._disable(error)
                return
            self.next_tick += interval
        self.last_frame = frame.copy()

    def _disable(self, error: Exception) -> None:
        """Stop only event recording after a media write failure."""
        self.disabled = True
        self.logger.error(
            "Event recording failed and is disabled for this run: %s", error
        )
        if self.writer is not None:
            try:
                self.writer.release()
            except (cv2.error, OSError):
                pass
        self.writer = None
        self.video_path = None

    def _pair_size(self, clip: Path) -> int:
        """Count a clip and its metadata against the storage limit."""
        sidecar = clip.with_suffix(".json")
        return clip.stat().st_size + (sidecar.stat().st_size if sidecar.exists() else 0)

    def _remove_pair(self, clip: Path) -> bool:
        """Delete one expired clip and its matching sidecar."""
        try:
            clip.unlink()
            clip.with_suffix(".json").unlink(missing_ok=True)
            self.logger.info("Old event video removed: %s", clip)
            return True
        except OSError as error:
            self.logger.error("Could not remove old event video %s: %s", clip, error)
            return False

    @staticmethod
    def _utc(timestamp: float) -> str:
        """Format one Unix timestamp for event metadata."""
        return datetime.fromtimestamp(timestamp, UTC).isoformat()
