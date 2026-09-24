# Puppy pad monitor (vibe-coded!!!!)

This local camera app watches two puppy pads after a dog visit. It uses image rules to find a new brown region, sends a Pushover phone notification, and plays the supplied MP3 near the dog. A visual match is only a possible poop detection.

Install and run:

```sh
uv sync
export PUSHOVER_TOKEN='your-application-token'
export PUSHOVER_USER='your-user-key'
uv run python -m puppy_pad_alarm.app
```

The first run downloads the YOLO dog detector model. The camera video stays local. A confirmed event sends one saved snapshot to Pushover.

Press `C` and drag a rectangle around all expected pad positions. Press `B` when both pads are visible and clean. After an event, the pad stays latched, so a later visit cannot create another detection alert. Clean the pad and press `B` to save fresh baselines and clear the latched state. Press `Q` to quit. The app saves calibration to `config.yaml`, reference images and event snapshots to `alarm_data/`, and diagnostic messages to `alarm_data/events.log`.

While an event is latched, the app sends Pushover reminders each minute for 10 minutes, then every 10 minutes for another 5 hours. The schedule survives restarts and sends at most one overdue reminder after downtime. If the dog approaches a latched pad, the local MP3 plays again once per approach. The proximity margin and short playback cooldown are configurable. No image check is used to decide when the pad is clean; pressing `B` after cleanup clears the state.

The app tries the C270 camera first on Linux. Use `--camera 1` or `--camera /dev/video2` if needed. Copy `config.example.yaml` to `config.yaml` to adjust thresholds. Calibration can also create `config.yaml` automatically. If the environment variables are absent, detections remain latched and the preview and log show that phone delivery failed. The local MP3 still plays once for the event.

Each detected dog visit saves a raw MP4 under `alarm_data/event_videos/`. A clip includes up to 10 seconds before the visit and ends 20 seconds after the dog leaves. A return during that tail extends the same clip. The matching JSON file records times, dog confidence, pad states, and app detections; `true_label` is left empty for later review. Clips stay local and are not sent to Pushover. The app deletes the oldest event clips after 7 days or when the folder exceeds 5 GB. Set `save_event_video: false` to disable this recording, or adjust the `event_video_*` settings in the configuration. Event clips depend on dog detection, so a visit the detector misses will not be captured.

Use `--video clip.mp4 --expected blue` to replay a recording through the same analysis and compare its detections with the expected pad. Use `--expected none` for empty, pee, movement, and lighting clips. Replay saves snapshots and a `results.csv` file with missed and false alert counts under `alarm_data/replay/`. It does not send Pushover messages or play the MP3. The preview follows the clip's frame rate. Press `B` when the clean pads are visible at the start of a clip. Set `save_debug_video: true` in the configuration to save the annotated preview for review.

`save_debug_video` records the full annotated run and has no automatic storage limit. Leave it off for normal use.

The color ranges and area thresholds are starting values. Tune them using frames from your actual camera. Detection can miss a dog or confuse a brown object with poop; check the saved image before treating a notification as confirmed.
