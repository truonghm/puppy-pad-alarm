# Puppy pad poop alarm: implementation plan

## Goal

Build a Python program that watches a laptop camera and sends a Pushover phone notification when a new poop becomes visible on either of two puppy pads after the dog visits them. Play `assets/freesound_community-hey-42237.mp3` locally to deter the dog. The program must avoid alarming for pee as far as the visible image allows. It must work without training a custom model or using a vision service. Use `uv` for dependency management.

The camera is available to the laptop. Its preferred device name is `C270 HD WEBCAM`. The two pads stay next to each other inside one user-selected camera area. The dog's body may hide a new poop for some time.

## Required behavior

1. On first run, show a camera preview and let the user mark one area containing both pads. Save this area in a local configuration file. Allow recalibration later.
2. Treat the selected area as one monitored surface. Do not identify or track individual pads.
3. When the area is clean, provide a `Set clean baseline` action. Save one reference image for the whole area and restore it after restart. Recalibration requires a new baseline.
4. Use an off-the-shelf dog detector to determine whether the dog enters the broad pad area. The detector needs only the `dog` class. Treat overlap between the dog's detected box and the search region as a visit. Do not attempt to infer pooping from posture in this version. Run detection often enough to capture a brief visit; make the inference interval configurable.
5. During and after a visit, compare the visible part of the selected area with its clean reference. Mask the dog. Do not wait for the dog to leave if a new object is visible beside her.
6. Find dark regions that are new relative to the clean baseline, including black objects. Favor fewer missed detections even if this causes more false alerts. Require a configurable minimum/maximum area and persistence at a stable position for multiple frames (initial default: at least three frames spanning 0.5 seconds). Reject broad exposure shifts. A dark region does not prove an object is poop.
7. Send one emergency-priority Pushover notification for a confirmed event, using `PUSHOVER_TOKEN` and `PUSHOVER_USER` from the environment. Pushover handles retries for up to 5 minutes. Play `assets/freesound_community-hey-42237.mp3` locally to deter the dog. Save a timestamped image with the detected region outlined and write the event to a local log. Keep the event latched across visits and restarts until the user cleans the area and manually sets a new clean baseline. A dog approach near the latched object plays the local deterrent again, once per approach.
8. If pad tracking, dog detection, or image alignment is uncertain, show the reason in the preview and log it. Do not announce a confirmed poop when the pad is hidden or the image is inconclusive. Recover automatically when a usable view returns.

## State and event rules

Use one state for the selected area: `NEEDS_BASELINE`, `READY`, `VISIT_ACTIVE`, `CHECKING`, and `ALARM_LATCHED`.

- `NEEDS_BASELINE`: no valid clean reference; do not make poop claims.
- `READY`: compare only after a dog visit; ignore marks already present in the clean reference.
- `VISIT_ACTIVE`: dog overlaps the selected area. Check the visible area.
- `CHECKING`: dog no longer overlaps the area; keep checking until a persistent new region appears or the timeout expires.
- `ALARM_LATCHED`: keep the object marked as active. Do not send another detection notification for a later visit. Play the deterrent when the dog approaches. Require an explicit clean-baseline reset after cleanup.

If a pad moves inside the selected area, the image comparison may treat that movement as a change. The user accepts some false alerts to reduce missed detections. A dog missed by the detector is a possible missed event; report detector confidence in saved diagnostics.

## Program interface and configuration

Create a small local application with a live preview. Draw the selected area, dog box, candidate region, and current state. Provide keyboard keys for calibration, setting the clean baseline, and quitting. On startup, try the camera device whose displayed name contains `C270`. Also support a configured camera index or device path. If selection fails, list available camera sources where the platform permits and exit with a useful error.

Use a `pyproject.toml` managed by `uv`, a short README with `uv sync` and `uv run` commands, and a sample configuration file. Keep video analysis local. Save event snapshots, event clips, and concise diagnostic logs by default; continuous video recording can be enabled for debugging. Put thresholds, inference interval, persistence time, search region, camera selection, and output paths in configuration. Do not hardcode color values inferred from an unseen camera feed.

Record a local video for every detected dog visit to support later evaluation. Include 10 seconds of buffered frames before entry and 20 seconds after departure; extend the clip when the dog returns during the tail. Save raw MP4 video and a sidecar with timestamps, dog confidence, pad states, detections, and an empty true-outcome label. Retain event videos for at most 7 days or 5 GB, removing the oldest pairs first. Keep these videos local and separate from Pushover attachments. A visit missed by the dog detector will not trigger event recording.

## Optional vision-model review

Keep this outside the alarm path in the first version. Provide a clean interface so a later implementation can send baseline/current pad crops for a second opinion when local rules are uncertain. A model's answer must not silently override an uncertain visual observation. A direct image-capable API is suitable for this optional feature; no Codex CLI integration is required for the initial implementation.

## Implementation order

1. Camera selection, preview, and area calibration.
2. Clean-baseline capture for the whole area.
3. Dog visit detection and the state machine.
4. New dark-region detection, persistence checks, and occlusion handling.
5. Emergency Pushover notification, local deterrent playback on detection and later approaches, manual clean reset, saved evidence, logs, and documented configuration.
6. Replay a small set of recorded clips through the same analysis functions as the live camera and tune exposed thresholds. Do not train a model.

## Acceptance checks

- With the dog absent, changes inside the selected area do not send a notification.
- A visit followed by a new visible dark object, including a black one, anywhere in the area sends one notification and plays the local deterrent. This can occur while the dog is still in the area if the object is visible.
- A visit that only wets a pad does not send a notification in the test clips.
- When the dog obscures the object, the display says the result is pending; a notification is sent after the object becomes visible and passes the persistence check.
- A brief shadow, exposure shift, or transient dark region does not send a notification in the test clips.
- The same object does not produce another detection notification. Pushover handles emergency retries, and a dog approach plays the deterrent again. Cleanup and `Set clean baseline` rearm detection.
- The app handles a missing camera, lost frames, pad localization failure, and missing baseline without crashing or making a false confirmed claim.

Use separate recorded clips for at least: empty pads, pad movement, a pee visit, a poop visit on each pad, a partially hidden poop, and lighting variation. Label them by observed outcome; measure both missed alarms and false alarms. The goal is a usable local prototype, not a guaranteed medical-grade or breed-independent classifier.

Recorded clip evaluation is deferred until footage is available. The current version includes replay and result logging for that later check.

## Notes

- Do not run linting or formatting or type checking during implementation. Leave this to the user.
- When adding new packages, add it exclusively via `uv add`. Do not modify `pyproject.toml` directly.
