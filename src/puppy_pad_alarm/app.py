"""Live camera preview and event handling."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
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
from .state import PadState, Phase
from .vision import PAD_SIZE, PAD_TARGET, Candidate, Pad, find_candidates, locate_pads

WINDOW = "Puppy pad alarm"
COLORS = {"blue": (255, 100, 20), "pink": (170, 100, 255)}


def camera_source(settings: Settings) -> int | str:
    """Resolve the preferred webcam name on Linux, then use an index."""
    if settings.camera_source is not None:
        if (
            isinstance(settings.camera_source, str)
            and settings.camera_source.isdecimal()
        ):
            return int(settings.camera_source)
        return settings.camera_source
    devices = Path("/dev/v4l/by-id")
    if devices.exists():
        for device in sorted(devices.iterdir()):
            if settings.camera_name.lower().replace(" ", "_") in device.name.lower():
                return str(device.resolve())
        for device in sorted(devices.iterdir()):
            if "c270" in device.name.lower() and "video-index0" in device.name.lower():
                return str(device.resolve())
    return 0


def available_cameras() -> str:
    """Describe devices when the default camera cannot open."""
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
    """Hold the camera, references, and two independent pad states."""

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
        logging.basicConfig(
            filename=self.output / "events.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        self.logger = logging.getLogger("puppy_pad_alarm")
        self.recorder = (
            EventRecorder(self.output / "event_videos", self.settings, self.logger)
            if video is None and self.settings.save_event_video
            else None
        )
        self.states = {color: PadState() for color in COLORS}
        self.baselines: dict[str, np.ndarray] = {}
        self._load_references()
        self.last_dog_box: tuple[int, int, int, int] | None = None
        self.last_dog_confidence = 0.0
        self.last_inference = 0.0
        self.detector_error: str | None = None
        self.notification_status = ""
        self.event_counts = {color: 0 for color in COLORS}
        self.last_pad_boxes: dict[str, tuple[int, int, int, int]] = {}
        self.last_pad_corners: dict[str, np.ndarray] = {}
        self.last_pad_seen: dict[str, float] = {}
        self.deterrent = Deterrent(
            Path(__file__).resolve().parents[2]
            / "assets/freesound_community-hey-42237.mp3"
        )
        self.delivery = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pushover")

    def _load_references(self) -> None:
        """Restore reference images and latched events after restart."""
        state_path = self.output / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        for color in COLORS:
            image_path = self.output / f"baseline_{color}.png"
            if image_path.exists():
                image = cv2.imread(str(image_path))
                if image is not None and image.shape[:2] == (PAD_SIZE[1], PAD_SIZE[0]):
                    self.baselines[color] = image
                    self.states[color].baseline_set()
                else:
                    self.logger.warning(
                        "%s baseline is unreadable or has wrong dimensions", color
                    )
            saved = state.get(color, {})
            phase = saved.get("phase") if isinstance(saved, dict) else saved
            if color in self.baselines and phase == Phase.ALARM_LATCHED:
                pad_state = self.states[color]
                pad_state.phase = Phase.ALARM_LATCHED
                pad_state.reason = "Event latched; clean pad and set baseline"
                pad_state.detected_at = (
                    float(saved.get("detected_at") or time.time())
                    if isinstance(saved, dict)
                    else time.time()
                )
                pad_state.last_reminder_index = (
                    int(saved.get("last_reminder_index", -1))
                    if isinstance(saved, dict)
                    else -1
                )
                pad_state.snapshot_name = (
                    saved.get("snapshot_name") if isinstance(saved, dict) else None
                )
                center = saved.get("object_center") if isinstance(saved, dict) else None
                pad_state.object_center = tuple(center) if center is not None else None

    def _save_state(self) -> None:
        """Persist latched state before external delivery."""
        path = self.output / "state.json"
        temporary = self.output / "state.tmp"
        temporary.write_text(
            json.dumps(
                {
                    color: {
                        "phase": state.phase.value,
                        "detected_at": state.detected_at,
                        "last_reminder_index": state.last_reminder_index,
                        "snapshot_name": state.snapshot_name,
                        "object_center": state.object_center,
                    }
                    for color, state in self.states.items()
                }
            )
        )
        temporary.replace(path)

    def _state_names(self) -> dict[str, str]:
        """Describe both pad states for event video metadata."""
        return {color: state.phase.value for color, state in self.states.items()}

    def _process_reminders(self, now: float) -> None:
        """Queue one due reminder per latched pad and persist its slot."""
        if self.video is not None:
            return
        for color, state in self.states.items():
            index = state.reminder_due(now)
            if index is None:
                continue
            self._save_state()
            snapshot = (
                self.output / state.snapshot_name if state.snapshot_name else None
            )
            self.logger.info("Pushover reminder %d queued for %s pad", index + 1, color)
            self.delivery.submit(
                send_pushover, color, snapshot, reminder=True
            ).add_done_callback(
                lambda future, event_color=color: self._delivery_finished(
                    event_color, future, reminder=True
                )
            )

    def _dog_near_pad(self, color: str, now: float) -> bool | None:
        """Use a recent pad position to decide if the dog is near it."""
        box = self.last_pad_boxes.get(color)
        if self.last_dog_box is None:
            return False
        if (
            box is None
            or now - self.last_pad_seen[color] > self.settings.pad_position_max_age_s
        ):
            return None
        margin = self.settings.dog_proximity_margin_px
        center = self.states[color].object_center
        if center is not None and color in self.last_pad_corners:
            transform = cv2.getPerspectiveTransform(
                PAD_TARGET, self.last_pad_corners[color]
            )
            point = cv2.perspectiveTransform(
                np.array([[center]], np.float32), transform
            )[0, 0]
            dog_x1, dog_y1, dog_x2, dog_y2 = self.last_dog_box
            nearest_x = min(max(float(point[0]), dog_x1), dog_x2)
            nearest_y = min(max(float(point[1]), dog_y1), dog_y2)
            return (point[0] - nearest_x) ** 2 + (
                point[1] - nearest_y
            ) ** 2 <= margin**2
        x1, y1, x2, y2 = box
        return overlaps(
            self.last_dog_box,
            [x1 - margin, y1 - margin, x2 - x1 + 2 * margin, y2 - y1 + 2 * margin],
        )

    def _handle_dog_approach(self, now: float, wall_time: float) -> None:
        """Play once when the dog returns near a latched pad."""
        approaching = []
        for color, state in self.states.items():
            if state.phase != Phase.ALARM_LATCHED:
                continue
            near = self._dog_near_pad(color, now)
            if near is None:
                continue
            if not near:
                if state.dog_away_since is None:
                    state.dog_away_since = wall_time
                if wall_time - state.dog_away_since >= self.settings.dog_away_confirm_s:
                    state.dog_near = False
                continue
            state.dog_away_since = None
            if (
                not state.dog_near
                and wall_time - state.last_deterrent_at
                >= self.settings.deterrent_cooldown_s
            ):
                approaching.append(color)
                state.last_deterrent_at = wall_time
            state.dog_near = True
        if approaching and self.video is None:
            try:
                self.deterrent.play()
                self.logger.info(
                    "Dog approached latched %s pad; deterrent played",
                    ", ".join(approaching),
                )
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

    def _set_baselines(self, pads: dict[str, Pad]) -> None:
        """Capture both visible clean pads on explicit request."""
        if set(pads) != set(COLORS):
            self.notification_status = (
                "Both pads must be visible to set clean baselines"
            )
            return
        if (
            self.last_dog_box is not None
            and self.settings.search_region is not None
            and overlaps(self.last_dog_box, self.settings.search_region)
        ):
            self.notification_status = (
                "Wait until the dog leaves before setting clean baselines"
            )
            return
        for color, pad in pads.items():
            path = self.output / f"baseline_{color}.png"
            reference = cv2.bitwise_and(pad.image, pad.image, mask=pad.valid)
            if not cv2.imwrite(str(path), reference):
                raise RuntimeError(f"Could not save clean baseline to {path}")
            self.baselines[color] = reference
            self.states[color].baseline_set()
        self._save_state()
        self.notification_status = "Clean baselines saved"
        self.logger.info("Clean baselines set for both pads")

    def _event(
        self, color: str, candidate: Candidate, pad: Pad, frame: np.ndarray
    ) -> None:
        """Save evidence, play deterrent, and send one phone notification."""
        self.event_counts[color] += 1
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        snapshot = self.output / f"{stamp}_{color}.jpg"
        annotated = frame.copy()
        transform = cv2.getPerspectiveTransform(
            np.array([[0, 0], [319, 0], [319, 239], [0, 239]], np.float32), pad.corners
        )
        contour = cv2.perspectiveTransform(
            candidate.contour.astype(np.float32), transform
        ).astype(np.int32)
        cv2.drawContours(annotated, [contour], -1, (0, 0, 255), 3)
        cv2.putText(
            annotated,
            f"Possible poop: {color} pad",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        saved_snapshot = snapshot if cv2.imwrite(str(snapshot), annotated) else None
        state = self.states[color]
        state.detected_at = time.time()
        state.snapshot_name = saved_snapshot.name if saved_snapshot else None
        state.object_center = candidate.center
        state.last_reminder_index = -1
        state.last_deterrent_at = state.detected_at
        state.dog_near = self._dog_near_pad(color, time.monotonic()) is True
        self._save_state()
        self.logger.warning(
            "Possible poop on %s pad; visit dog confidence %.3f; snapshot %s",
            color,
            state.visit_confidence,
            saved_snapshot or "unavailable",
        )
        if self.recorder is not None:
            self.recorder.mark_detection(
                color,
                time.monotonic(),
                time.time(),
                saved_snapshot,
                self._state_names(),
            )
        if self.video is not None:
            self.notification_status = f"Replay event: {color} pad"
            return
        try:
            self.deterrent.play()
        except (OSError, RuntimeError, pygame.error) as error:
            self.logger.error("Deterrent playback failed: %s", error)
            self.notification_status = f"Sound failed: {error}"
        self.notification_status = f"Sending Pushover: {color} pad"
        self.delivery.submit(send_pushover, color, saved_snapshot).add_done_callback(
            lambda future, event_color=color: self._delivery_finished(
                event_color, future
            )
        )

    def _delivery_finished(
        self, color: str, future: Future[None], *, reminder: bool = False
    ) -> None:
        """Report the result of the background phone request."""
        label = "Pushover reminder" if reminder else "Pushover"
        error = future.exception()
        if error is None:
            self.notification_status = f"{label} sent: {color} pad"
            self.logger.info("%s sent for %s pad", label, color)
        else:
            self.notification_status = f"{label} failed: {error}"
            self.logger.error("%s failed for %s pad: %s", label, color, error)

    def run(self) -> None:
        """Process frames and keyboard controls until the user quits."""
        source = (
            str(self.video) if self.video is not None else camera_source(self.settings)
        )
        camera = cv2.VideoCapture(source)
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
                self._process_reminders(time.time())
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
                    pads, reasons = locate_pads(frame, self.settings)
                    for color, pad in pads.items():
                        x1, y1 = pad.corners.min(axis=0).astype(int)
                        x2, y2 = pad.corners.max(axis=0).astype(int)
                        self.last_pad_boxes[color] = (x1, y1, x2, y2)
                        self.last_pad_corners[color] = pad.corners.copy()
                        self.last_pad_seen[color] = now
                    dog_present = self.last_dog_box is not None and overlaps(
                        self.last_dog_box, self.settings.search_region
                    )
                    if self.recorder is not None:
                        self.recorder.visit(
                            dog_present if self.detector_error is None else None,
                            now,
                            time.time(),
                            self.last_dog_confidence,
                            self._state_names(),
                        )
                    for color, state in self.states.items():
                        if self.detector_error is not None:
                            state.reason = (
                                f"Dog detector uncertain: {self.detector_error}"
                            )
                            continue
                        if dog_present:
                            state.visit_confidence = max(
                                state.visit_confidence, self.last_dog_confidence
                            )
                        pad = pads.get(color)
                        candidates: list[Candidate] = []
                        reason = reasons.get(color)
                        if (
                            pad is not None
                            and color in self.baselines
                            and state.phase != Phase.READY
                        ):
                            candidates, reason = find_candidates(
                                pad,
                                self.baselines[color],
                                self.last_dog_box,
                                self.settings,
                            )
                        old_reason = state.reason
                        confirmed = state.update(
                            dog_present=dog_present,
                            visible=pad is not None,
                            candidates=candidates,
                            reason=reason,
                            now=now,
                            settings=self.settings,
                        )
                        if state.reason and state.reason != old_reason:
                            self.logger.info("%s pad: %s", color, state.reason)
                        if confirmed is not None and pad is not None:
                            self._event(color, confirmed, pad, frame)
                    self._handle_dog_approach(now, time.time())
                    preview = self._draw(frame, pads)
                    if writer is not None:
                        writer.write(preview)
                    cv2.imshow(WINDOW, preview)
                    key = (
                        cv2.waitKey(
                            max(1, int(1000 / fps)) if self.video is not None else 1
                        )
                        & 0xFF
                    )
                    if self._key(key, frame, pads):
                        break
                    continue
                preview = self._draw(frame, {})
                if writer is not None:
                    writer.write(preview)
                cv2.imshow(WINDOW, preview)
                key = (
                    cv2.waitKey(
                        max(1, int(1000 / fps)) if self.video is not None else 1
                    )
                    & 0xFF
                )
                if self._key(key, frame, {}):
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
                print(
                    f"Replay detections: blue={self.event_counts['blue']}, pink={self.event_counts['pink']}"
                )
                if (
                    self.expected is not None
                    and replay_complete
                    and set(self.baselines) == set(COLORS)
                ):
                    actual = {
                        color for color, count in self.event_counts.items() if count > 0
                    }
                    expected = set() if self.expected == "none" else {self.expected}
                    missed = len(expected - actual)
                    false = len(actual - expected)
                    result = {
                        "clip": str(self.video),
                        "expected": self.expected,
                        "detected": ",".join(sorted(actual)) or "none",
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
                    print(
                        "Replay ended early or lacked clean baselines; no accuracy result was recorded"
                    )

    def _draw(self, frame: np.ndarray, pads: dict[str, Pad]) -> np.ndarray:
        """Overlay boundaries, state, and action keys on the preview."""
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
        for color, pad in pads.items():
            cv2.polylines(image, [pad.corners.astype(np.int32)], True, COLORS[color], 2)
            candidate = self.states[color].candidate
            if candidate is not None:
                target = cv2.getPerspectiveTransform(
                    np.array([[0, 0], [319, 0], [319, 239], [0, 239]], np.float32),
                    pad.corners,
                )
                contour = cv2.perspectiveTransform(
                    candidate.contour.astype(np.float32), target
                ).astype(np.int32)
                cv2.drawContours(image, [contour], -1, (0, 255, 255), 2)
        lines = ["C calibrate | B clean baseline / clear event | Q quit"]
        lines += [
            f"{color}: {state.phase.value} {state.reason}"
            for color, state in self.states.items()
        ]
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
                3,
            )
            cv2.putText(
                image,
                line,
                (12, 25 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
            )
        return image

    def _key(self, key: int, frame: np.ndarray, pads: dict[str, Pad]) -> bool:
        """Handle documented keyboard controls."""
        if key in (ord("q"), 27):
            return True
        if key == ord("c"):
            rectangle = cv2.selectROI(WINDOW, frame, showCrosshair=True)
            if rectangle[2] > 0 and rectangle[3] > 0:
                self.settings.search_region = [value for value in rectangle]
                save_settings(self.settings_path, self.settings)
                self.notification_status = "Search region saved"
        elif key == ord("b"):
            self._set_baselines(pads)
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
        choices=["none", "blue", "pink"],
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
