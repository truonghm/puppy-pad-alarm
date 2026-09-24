# Puppy pad poop alarm: implementation plan

## Goal

Build a Python program that watches a laptop camera and sounds an alarm when a new poop becomes visible on either of two puppy pads after the dog visits them. The program must avoid alarming for pee as far as the visible image allows. It must work without training a custom model or using a vision service. Use `uv` for dependency management.

The camera is available to the laptop. Its preferred device name is `C270 HD WEBCAM`. The two pads remain within a general area of the image but can shift. They are white on a gray floor, with one blue border and one pink border. A brown toy of poop-like size will not be left on the pads. The dog's body may hide a new poop for some time.

## Required behavior

1. On first run, show a camera preview and let the user mark one broad search region that contains both pads at all their expected positions. Save this region in a local configuration file. Allow recalibration later. The region is only a search boundary; do not assume fixed pad coordinates.
2. Locate the blue-bordered and pink-bordered pads within that region whenever their surfaces are sufficiently visible. Use conventional image processing: floor/pad contrast, colored borders, contours, and size/shape checks. Identify pads by border color. Track each pad independently across frames. Expose thresholds in configuration. Do not require a pad detection model.
3. When both pads are visible and clean, provide a `Set clean baseline` action. Save a reference image for each pad in pad-relative coordinates, with its border excluded. Persist references across restarts. Never replace a reference automatically while a possible new object is present.
4. Use an off-the-shelf dog detector to determine whether the dog enters the broad pad area. The detector needs only the `dog` class. Treat overlap between the dog's detected box and the search region as a visit. Do not attempt to infer pooping from posture in this version. Run detection often enough to capture a brief visit; make the inference interval configurable.
5. During and after a visit, inspect every visible portion of each pad. Once a pad is found, align its current interior with its clean reference before comparing pixels. Mask the dog and any occluded parts. Do not wait for the dog to leave if a new object is already visible beside her. If the dog hides the surface, wait for a clear view and continue checking.
6. Find regions that are both new relative to the clean baseline and plausibly brown. Require a configurable minimum/maximum area, separation from the colored border, and persistence at a stable position for multiple frames (initial default: at least three frames spanning 0.5 seconds). Compare in a color space suited to brightness changes; reject simple global exposure shifts, shadows, and dark wet patches as well as practical rules allow. These rules are approximate; do not claim that color alone proves an object is poop.
7. Sound an audible repeating alarm and show which pad triggered it. Save a timestamped image with the detected region outlined and write the event to a local log. Provide an `Acknowledge` action that silences the alarm. Issue at most one alarm for a detected object. Keep the event latched across visits and restarts until the user clears the pad and sets a new clean baseline.
8. If pad tracking, dog detection, or image alignment is uncertain, show the reason in the preview and log it. Do not announce a confirmed poop when the pad is hidden or the image is inconclusive. Recover automatically when a usable view returns.

## State and event rules

Use explicit states: `NEEDS_BASELINE`, `READY`, `VISIT_ACTIVE`, `CHECKING`, and `ALARM_LATCHED`. Track state per pad where appropriate.

- `NEEDS_BASELINE`: no valid clean reference; do not make poop claims.
- `READY`: compare only after a dog visit; ignore old pad stains already present in the clean reference.
- `VISIT_ACTIVE`: dog overlaps the search region. Check any currently visible pad pixels.
- `CHECKING`: dog no longer overlaps the region; keep checking until both pads are visible long enough to decide whether a persistent new region appeared. Return to `READY` after a configurable timeout if neither pad has a candidate and both were inspected. If either pad remains hidden, continue waiting and report the obstruction.
- `ALARM_LATCHED`: alarm until acknowledged; keep the object marked as active and do not alert again for it. Require an explicit clean-baseline reset after cleanup.

The pad positions may change during a visit. Re-detect and align each pad before making a positive claim. Preserve the last known position as a hint only; do not compare images at stale coordinates. A dog missed by the detector is a possible missed event; report detector confidence in saved diagnostics.

## Program interface and configuration

Create a small local application with a live preview. Draw the search region, current outlines for both pads, dog box, candidate region, and current state. Provide visible controls or documented keyboard keys for calibration, setting clean baselines, acknowledging the alarm, and quitting. On startup, try the camera device whose displayed name is `C270 HD WEBCAM`. Also support a configured camera index or device path because camera-name lookup differs by operating system. If selection fails, list available camera sources where the platform permits and exit with a useful error.

Use a `pyproject.toml` managed by `uv`, a short README with `uv sync` and `uv run` commands, and a sample configuration file. Keep video analysis local. Save only event snapshots and concise diagnostic logs by default; continuous video recording can be enabled for debugging. Put thresholds, inference interval, persistence time, search region, camera selection, and output paths in configuration. Do not hardcode color values inferred from an unseen camera feed.

## Optional vision-model review

Keep this outside the alarm path in the first version. Provide a clean interface so a later implementation can send baseline/current pad crops for a second opinion when local rules are uncertain. A model's answer must not silently override an uncertain visual observation. A direct image-capable API is suitable for this optional feature; no Codex CLI integration is required for the initial implementation.

## Implementation order

1. Camera selection, preview, search-region calibration, and pad localization/identity.
2. Clean-baseline capture and pad-relative alignment as the pads move.
3. Dog visit detection and the state machine.
4. New-region detection, brown filtering, persistence checks, and occlusion handling.
5. Alarm, acknowledgement, saved evidence, logs, and documented configuration.
6. Replay a small set of recorded clips through the same analysis functions as the live camera and tune exposed thresholds. Do not train a model.

## Acceptance checks

- With the dog absent, shifting either pad within the marked search region does not itself raise an alarm; blue and pink identities remain correct.
- A visit followed by a new visible brown object on either pad raises one alarm. The alarm can occur while the dog is still in the area if the object is visible.
- A visit that only wets a pad does not raise an alarm in the test clips.
- When the dog obscures the object, the display says the result is pending; an alarm occurs after the object becomes visible and passes the persistence check.
- A brief shadow, exposure shift, or transient brown region does not raise an alarm in the test clips.
- Acknowledging an alarm stops the sound. The same object does not produce another alarm; cleanup and `Set clean baseline` rearm detection.
- The app handles a missing camera, lost frames, pad localization failure, and missing baseline without crashing or making a false confirmed claim.

Use separate recorded clips for at least: empty pads, pad movement, a pee visit, a poop visit on each pad, a partially hidden poop, and lighting variation. Label them by observed outcome; measure both missed alarms and false alarms. The goal is a usable local prototype, not a guaranteed medical-grade or breed-independent classifier.

## Notes

- Do not run linting or formatting or type checking during implementation. Leave this to the user.
- When adding new packages, add it exclusively via `uv add`. Do not modify `pyproject.toml` directly.