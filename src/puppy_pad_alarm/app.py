"""Live camera preview and event handling."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pygame
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .config import Settings, load_settings, save_settings
from .notify import Deterrent, send_pushover
from .recording import EventRecorder
from .state import AreaState, Phase
from .vision import Candidate, crop_region, find_candidates

WINDOW = "Puppy pad alarm"
DOG_DETECTION_GRACE_S = 1.0


def camera_source(settings: Settings) -> int | str:
    """Select the named camera or an explicit source."""
    if settings.camera_source is not None:
        if (
            isinstance(settings.camera_source, str)
            and settings.camera_source.isdecimal()
        ):
            return int(settings.camera_source)
        return settings.camera_source
    if sys.platform == "win32":
        return named_camera_index(windows_cameras(), settings.camera_name)
    devices = Path("/dev/v4l/by-id")
    if devices.exists():
        for device in sorted(devices.iterdir()):
            if settings.camera_name.lower().replace(" ", "_") in device.name.lower():
                return str(device.resolve())
        for device in sorted(devices.iterdir()):
            if "c270" in device.name.lower() and "video-index0" in device.name.lower():
                return str(device.resolve())
    return 0


def named_camera_index(devices: list[str], preferred: str) -> int:
    """Match the configured camera without falling back to another device."""
    wanted = preferred.casefold()
    for index, name in enumerate(devices):
        if wanted in name.casefold():
            return index
    if "c270" in wanted:
        for index, name in enumerate(devices):
            if "c270" in name.casefold():
                return index
    names = ", ".join(f"{index}: {name}" for index, name in enumerate(devices))
    raise RuntimeError(
        f"Camera {preferred!r} was not found. "
        f"Windows cameras: {names or 'none'}. "
        "Check the USB connection or set camera_source in config.yaml."
    )


def windows_cameras() -> list[str]:
    """List Windows cameras in DirectShow index order."""
    from pygrabber.dshow_graph import FilterGraph  # pyrefly: ignore[missing-import]

    return FilterGraph().get_input_devices()


def available_cameras() -> str:
    """Describe devices when the default camera cannot open."""
    if sys.platform == "win32":
        return "\n".join(
            f"{index}: {name}" for index, name in enumerate(windows_cameras())
        ) or "No Windows cameras were found"
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    devices = Path("/dev/v4l/by-id")
    return (
        "\n".join(str(path) for path in devices.iterdir())
        if devices.exists()
        else "No camera list is available on this platform"
    )


def overlaps(box: tuple[int, int, int, int], region: list[int]) -> bool:
    """Test whether a dog box intersects the calibrated region."""
    x, y, w, h = region
    return max(box[0], x) < min(box[2], x + w) and max(box[1], y) < min(box[3], y + h)


class Application:
    """Hold the camera, one clean reference, and one alert state."""

    def __init__(
        self,
        settings_path: Path,
        source_override: str | None = None,
        video: Path | None = None,
        expected: str | None = None,
    ) -> None:
        self.settings_path = settings_path
        self.settings = load_settings(settings_path)
        if source_override is not None:
            self.settings.camera_source = (
                int(source_override) if source_override.isdecimal() else source_override
            )
        self.video = video
        self.expected = expected
        self.output = (
            Path(self.settings.output_dir) / "replay"
            if video is not None
            else Path(self.settings.output_dir)
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("puppy_pad_alarm")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.FileHandler(self.output / "events.log"), logging.StreamHandler()):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.info("Monitor starting; output directory: %s", self.output)
        self.recorder = (
            EventRecorder(self.output / "event_videos", self.settings, self.logger)
            if video is None and self.settings.save_event_video
            else None
        )
        self.state = AreaState()
        self.baseline: np.ndarray | None = None
        self.baseline_region: list[int] | None = None
        self._load_references()
        self.last_dog_box: tuple[int, int, int, int] | None = None
        self.last_dog_mask_box: tuple[int, int, int, int] | None = None
        self.last_dog_seen_at: float | None = None
        self.last_dog_confidence = 0.0
        self.last_inference = 0.0
        self.dog_in_area = False
        self.detector_error: str | None = None
        self.notification_status = ""
        self.event_count = 0
        self.deterrent = Deterrent(
            Path(__file__).resolve().parents[2]
            / "assets/freesound_community-hey-42237.mp3"
        )
        self.delivery = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pushover")

    def _load_references(self) -> None:
        """Restore the selected-area reference and its latched event."""
        state_path = self.output / "state.json"
        saved = json.loads(state_path.read_text()) if state_path.exists() else {}
        image_path = self.output / "baseline_region.png"
        if self.settings.search_region is None or not image_path.exists():
            return
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (self.settings.search_region[3], self.settings.search_region[2]):
            self.logger.warning("Selected-area baseline is unreadable or has wrong dimensions")
            return
        if saved.get("baseline_region") != self.settings.search_region:
            self.logger.warning("Selected area changed; press B to save a new clean baseline")
            return
        self.baseline = image
        self.baseline_region = self.settings.search_region.copy()
        self.state.baseline_set()
        if saved.get("phase") == Phase.ALARM_LATCHED:
            self.state.phase = Phase.ALARM_LATCHED
            self.state.reason = "Event latched; clean area and press B"
            self.state.detected_at = float(saved.get("detected_at") or time.time())
            self.state.snapshot_name = saved.get("snapshot_name")
            center = saved.get("object_center")
            self.state.object_center = tuple(center) if center is not None else None

    def _save_state(self) -> None:
        """Persist latched state before external delivery."""
        path = self.output / "state.json"
        temporary = self.output / "state.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "region": self.settings.search_region,
                    "baseline_region": self.baseline_region,
                    "phase": self.state.phase.value,
                    "detected_at": self.state.detected_at,
                    "snapshot_name": self.state.snapshot_name,
                    "object_center": self.state.object_center,
                }
            )
        )
        temporary.replace(path)

    def _state_names(self) -> dict[str, str]:
        """Describe the selected area's state for video metadata."""
        return {"region": self.state.phase.value}

    def _dog_near_object(self) -> bool:
        """Test whether the dog approaches the saved event position."""
        if self.last_dog_box is None or self.settings.search_region is None:
            return False
        margin = self.settings.dog_proximity_margin_px
        center = self.state.object_center
        if center is not None:
            x, y, _, _ = self.settings.search_region
            point_x, point_y = center[0] + x, center[1] + y
            dog_x1, dog_y1, dog_x2, dog_y2 = self.last_dog_box
            nearest_x = min(max(point_x, dog_x1), dog_x2)
            nearest_y = min(max(point_y, dog_y1), dog_y2)
            return (point_x - nearest_x) ** 2 + (point_y - nearest_y) ** 2 <= margin**2
        x, y, width, height = self.settings.search_region
        return overlaps(
            self.last_dog_box,
            [x - margin, y - margin, width + 2 * margin, height + 2 * margin],
        )

    def _handle_dog_approach(self, now: float, wall_time: float) -> None:
        """Play once when the dog returns near a latched object."""
        state = self.state
        if state.phase != Phase.ALARM_LATCHED:
            return
        if not self._dog_near_object():
            if state.dog_away_since is None:
                state.dog_away_since = wall_time
            if wall_time - state.dog_away_since >= self.settings.dog_away_confirm_s:
                state.dog_near = False
            return
        state.dog_away_since = None
        approaching = not state.dog_near and wall_time - state.last_deterrent_at >= self.settings.deterrent_cooldown_s
        state.dog_near = True
        if approaching and self.video is None:
            state.last_deterrent_at = wall_time
            try:
                self.deterrent.play()
                self.logger.info("Dog approached latched object; deterrent played")
            except (OSError, RuntimeError, pygame.error) as error:
                self.logger.error("Deterrent playback failed: %s", error)

    def _detect_dog(self, model: YOLO, frame: np.ndarray, now: float) -> None:
        """Run the off-the-shelf detector at the configured interval."""
        if now - self.last_inference < self.settings.inference_interval_s:
            return
        self.last_inference = now
        self.last_dog_box = None
        self.last_dog_confidence = 0.0
        results = model.predict(
            frame, classes=[16], conf=self.settings.dog_confidence, verbose=False
        )
        for result in results:
            if not isinstance(result, Results) or result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence > self.last_dog_confidence:
                    self.last_dog_confidence = confidence
                    coordinates = box.xyxy[0]
                    self.last_dog_box = (
                        int(coordinates[0]),
                        int(coordinates[1]),
                        int(coordinates[2]),
                        int(coordinates[3]),
                    )
        self.detector_error = None

    def _set_baseline(self, frame: np.ndarray) -> None:
        """Capture the clean selected area on explicit request."""
        region = self.settings.search_region
        if region is None:
            self.notification_status = "Press C to select the area first"
            return
        if (
            self.last_dog_box is not None
            and overlaps(self.last_dog_box, region)
        ):
            self.notification_status = (
                "Wait until the dog leaves before setting a clean baseline"
            )
            return
        current = crop_region(frame, region)
        if current is None:
            self.notification_status = "Selected area is outside the camera image"
            return
        path = self.output / "baseline_region.png"
        if not cv2.imwrite(str(path), current):
            raise RuntimeError(f"Could not save clean baseline to {path}")
        self.baseline = current.copy()
        self.baseline_region = region.copy()
        self.state.baseline_set()
        self._save_state()
        self.notification_status = "Clean baseline saved"
        self.logger.info("Clean baseline saved for selected area")

    def _event(self, candidate: Candidate, frame: np.ndarray) -> None:
        """Save evidence, play deterrent, and send one phone notification."""
        self.event_count += 1
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        snapshot = self.output / f"{stamp}_event.jpg"
        annotated = frame.copy()
        assert self.settings.search_region is not None
        x, y, _, _ = self.settings.search_region
        contour = candidate.contour + np.array([[[x, y]]], np.int32)
        cv2.drawContours(annotated, [contour], -1, (0, 0, 255), 3)
        cv2.putText(
            annotated,
            "Possible poop in selected area",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        saved_snapshot = snapshot if cv2.imwrite(str(snapshot), annotated) else None
        state = self.state
        state.detected_at = time.time()
        state.snapshot_name = saved_snapshot.name if saved_snapshot else None
        state.object_center = candidate.center
        state.last_deterrent_at = state.detected_at
        state.dog_near = self._dog_near_object()
        self._save_state()
        self.logger.warning(
            "Possible poop in selected area; visit dog confidence %.3f; snapshot %s",
            state.visit_confidence,
            saved_snapshot or "unavailable",
        )
        if self.recorder is not None:
            self.recorder.mark_detection(
                time.monotonic(),
                time.time(),
                saved_snapshot,
                self._state_names(),
            )
        if self.video is not None:
            self.notification_status = "Replay event in selected area"
            return
        try:
            self.deterrent.play()
        except (OSError, RuntimeError, pygame.error) as error:
            self.logger.error("Deterrent playback failed: %s", error)
            self.notification_status = f"Sound failed: {error}"
        self.notification_status = "Sending Pushover"
        self.delivery.submit(send_pushover, saved_snapshot).add_done_callback(self._delivery_finished)

    def _delivery_finished(self, future: Future[None]) -> None:
        """Report the result of the background phone request."""
        label = "Pushover"
        error = future.exception()
        if error is None:
            self.notification_status = f"{label} sent"
            self.logger.info("%s sent", label)
        else:
            self.notification_status = f"{label} failed: {error}"
            self.logger.error("%s failed: %s", label, error)

    def run(self) -> None:
        """Process frames and keyboard controls until the user quits."""
        source = (
            str(self.video) if self.video is not None else camera_source(self.settings)
        )
        self.logger.info("Opening %s: %s", "video" if self.video is not None else "camera", source)
        camera = (
            cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if sys.platform == "win32" and isinstance(source, int) and self.video is None
            else cv2.VideoCapture(source)
        )
        if not camera.isOpened():
            if self.video is not None:
                raise RuntimeError(f"Could not open video {source!r}")
            raise RuntimeError(
                f"Could not open camera {source!r}. Available sources:\n{available_cameras()}"
            )
        writer: cv2.VideoWriter | None = None
        replay_complete = False
        try:
            model = YOLO(self.settings.model)
            cv2.namedWindow(WINDOW)
            failures = 0
            frame_number = 0
            fps = camera.get(cv2.CAP_PROP_FPS) or 30.0
            while True:
                ok, frame = camera.read()
                if not ok:
                    if self.video is not None:
                        replay_complete = True
                        break
                    failures += 1
                    if failures >= 30:
                        raise RuntimeError("Camera stopped returning frames")
                    time.sleep(0.1)
                    continue
                failures = 0
                frame_number += 1
                now = frame_number / fps if self.video is not None else time.monotonic()
                if self.recorder is not None:
                    self.recorder.observe(frame, now)
                if self.settings.save_debug_video and writer is None:
                    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                    writer = cv2.VideoWriter(
                        str(self.output / f"debug_{stamp}.mp4"),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                if self.settings.search_region is None:
                    self.notification_status = "Press C to mark the search region"
                else:
                    try:
                        self._detect_dog(model, frame, now)
                    except Exception as error:  # noqa: BLE001 - Keep monitoring if the model backend fails.
                        self.detector_error = str(error)
                        self.notification_status = f"Dog detector uncertain: {error}"
                        self.logger.error("Dog detector failed: %s", error)
                        self.last_dog_box = None
                    dog_detected = self.last_dog_box is not None and overlaps(
                        self.last_dog_box, self.settings.search_region
                    )
                    if dog_detected:
                        self.last_dog_seen_at = now
                        self.last_dog_mask_box = self.last_dog_box
                    dog_present = dog_detected or (
                        self.last_dog_seen_at is not None
                        and now - self.last_dog_seen_at < DOG_DETECTION_GRACE_S
                    )
                    dog_box_for_mask = self.last_dog_box if dog_detected else self.last_dog_mask_box if dog_present else None
                    if dog_present != self.dog_in_area:
                        self.dog_in_area = dog_present
                        self.logger.info("Dog %s selected area", "entered" if dog_present else "left")
                    if self.recorder is not None:
                        self.recorder.visit(
                            dog_present if self.detector_error is None else None,
                            now,
                            time.time(),
                            self.last_dog_confidence,
                            self._state_names(),
                        )
                    state = self.state
                    if self.detector_error is not None:
                        state.reason = f"Dog detector uncertain: {self.detector_error}"
                    else:
                        if dog_present:
                            state.visit_confidence = max(state.visit_confidence, self.last_dog_confidence)
                        current = crop_region(frame, self.settings.search_region)
                        candidates: list[Candidate] = []
                        reason = None if current is not None else "Selected area is outside the camera image"
                        if current is not None and self.baseline is not None and state.phase != Phase.READY:
                            candidates, reason = find_candidates(
                                current, self.baseline, dog_box_for_mask,
                                self.settings.search_region, self.settings,
                            )
                        old_reason = state.reason
                        confirmed = state.update(
                            dog_present=dog_present,
                            visible=current is not None,
                            candidates=candidates,
                            reason=reason,
                            now=now,
                            settings=self.settings,
                        )
                        if state.reason and state.reason != old_reason:
                            self.logger.info("Selected area: %s", state.reason)
                        if confirmed is not None:
                            self._event(confirmed, frame)
                    self._handle_dog_approach(now, time.time())
                    preview = self._draw(frame)
                    if writer is not None:
                        writer.write(preview)
                    cv2.imshow(WINDOW, preview)
                    key = (
                        cv2.waitKey(
                            max(1, int(1000 / fps)) if self.video is not None else 1
                        )
                        & 0xFF
                    )
                    if self._key(key, frame):
                        break
                    continue
                preview = self._draw(frame)
                if writer is not None:
                    writer.write(preview)
                cv2.imshow(WINDOW, preview)
                key = (
                    cv2.waitKey(
                        max(1, int(1000 / fps)) if self.video is not None else 1
                    )
                    & 0xFF
                )
                if self._key(key, frame):
                    break
        finally:
            if self.recorder is not None:
                self.recorder.close(time.monotonic(), time.time(), self._state_names())
            camera.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            self.deterrent.stop()
            self.delivery.shutdown(wait=True)
            if self.video is not None:
                print(f"Replay detections: {self.event_count}")
                if self.expected is not None and replay_complete and self.baseline is not None:
                    actual = self.event_count > 0
                    expected = self.expected == "poop"
                    missed = int(expected and not actual)
                    false = int(actual and not expected)
                    result = {
                        "clip": str(self.video),
                        "expected": self.expected,
                        "detected": "poop" if actual else "none",
                        "missed": missed,
                        "false": false,
                    }
                    results_path = self.output / "results.csv"
                    with results_path.open("a", newline="") as result_file:
                        csv_writer = csv.DictWriter(
                            result_file, fieldnames=list(result)
                        )
                        if result_file.tell() == 0:
                            csv_writer.writeheader()
                        csv_writer.writerow(result)
                    print(f"Expected: {self.expected}; missed={missed}; false={false}")
                elif self.expected is not None:
                    print("Replay ended early or lacked a clean baseline; no accuracy result was recorded")

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the selected area, state, and action keys."""
        image = frame.copy()
        if self.settings.search_region:
            x, y, w, h = self.settings.search_region
            cv2.rectangle(image, (x, y), (x + w, y + h), (220, 220, 220), 2)
        if self.last_dog_box:
            x1, y1, x2, y2 = self.last_dog_box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(
                image,
                f"dog {self.last_dog_confidence:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 255),
                1,
            )
        candidate = self.state.candidate
        if candidate is not None and self.settings.search_region is not None:
            x, y, _, _ = self.settings.search_region
            contour = candidate.contour + np.array([[[x, y]]], np.int32)
            cv2.drawContours(image, [contour], -1, (0, 255, 255), 2)
        lines = ["C calibrate | B clean baseline / clear event | Q quit"]
        lines.append(f"Area: {self.state.phase.value} {self.state.reason}")
        if self.notification_status:
            lines.append(self.notification_status[:100])
        for index, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (12, 25 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
        return image

    def _key(self, key: int, frame: np.ndarray) -> bool:
        """Handle documented keyboard controls."""
        if key in (ord("q"), 27):
            return True
        if key == ord("c"):
            rectangle = cv2.selectROI(WINDOW, frame, showCrosshair=True)
            if rectangle[2] > 0 and rectangle[3] > 0:
                self.settings.search_region = [value for value in rectangle]
                self.baseline = None
                self.baseline_region = None
                self.state = AreaState()
                save_settings(self.settings_path, self.settings)
                self._save_state()
                self.notification_status = "Search region saved"
        elif key == ord("b"):
            self._set_baseline(frame)
        return False


def main() -> None:
    """Start the local application."""
    parser = argparse.ArgumentParser(description="Puppy pad camera monitor")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--camera", help="Camera index or device path")
    parser.add_argument(
        "--video", type=Path, help="Replay a recorded video through the same analysis"
    )
    parser.add_argument(
        "--expected",
        choices=["none", "poop"],
        help="Expected event for a replay clip",
    )
    args = parser.parse_args()
    if args.expected is not None and args.video is None:
        parser.error("--expected requires --video")
    try:
        Application(args.config, args.camera, args.video, args.expected).run()
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
