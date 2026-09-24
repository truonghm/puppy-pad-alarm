# Puppy Pad Alarm (vibe-coded!!!!)

A local camera app for two puppy pads, one blue and one pink. After a dog visit, it looks for a new brown region on either pad. A possible detection plays a short deterrent sound near the dog and sends a Pushover notification to your phone.

The camera feed and event videos stay on your computer. Pushover receives a snapshot when an event is detected. A detection is a prompt to check the pad, not proof of poop.

## Start

You need Python 3.13+, [uv](https://docs.astral.sh/uv/), a camera, and [Pushover](https://pushover.net/api) credentials.

From the project directory:

```sh
uv sync
export PUSHOVER_TOKEN='your-application-token'
export PUSHOVER_USER='your-user-key'
uv run python -m puppy_pad_alarm.app
```

The first run downloads the YOLO dog detector model. If the app opens the wrong camera, use `--camera 1` or a device path such as `--camera /dev/video2`.

## Use the preview

| Key | Action |
| --- | --- |
| `C` | Drag a rectangle around all places where the pads can be. |
| `B` | Save clean reference images for both pads. After cleanup, this also clears an active alert. |
| `Q` | Quit. |

On first use, press `C`, then press `B` while both pads are visible and clean. You do not need to repeat this on a normal restart.

After a detection, the affected pad stays in an alert state until you clean it and press `B`. Another visit will not send a new detection alert while that state is active. The app sends reminders every minute for 10 minutes, then every 10 minutes for 5 hours. If the dog approaches the pad again, the deterrent sound plays again.

## Saved files

| Location | Contents |
| --- | --- |
| `config.yaml` | Saved camera search area and settings. |
| `alarm_data/` | Clean reference images, snapshots, alert state, and `events.log`. |
| `alarm_data/event_videos/` | Local videos of detected dog visits and matching JSON metadata. |

Visit videos include up to 10 seconds before the visit and 20 seconds after the dog leaves. The app keeps them for up to 7 days or 5 GB, whichever limit comes first. A visit missed by the dog detector will not produce a clip.

To change thresholds or recording limits, copy [config.example.yaml](config.example.yaml) to `config.yaml` and edit the values. Set `save_event_video: false` to turn off visit recording. Leave `save_debug_video: false` for normal use; debug video records the full run without a storage limit.

## Check recordings later

You can replay a clip without sending notifications or playing the sound:

```sh
uv run python -m puppy_pad_alarm.app --video clip.mp4 --expected blue
```

Use `--expected pink` for a pink-pad event or `--expected none` for a clip with no poop. Replay writes results under `alarm_data/replay/`. If the clip starts with clean, visible pads, press `B` to set its reference images.

## If something is wrong

- No phone alert: Check that `PUSHOVER_TOKEN` and `PUSHOVER_USER` are set in the terminal that starts the app. The preview and `alarm_data/events.log` show delivery errors.
- No detection or too many alerts: Check the saved snapshot and event video. The default image thresholds are starting values and may need adjustment for your camera and lighting.
- Restart appears to lose setup: Start the app from the project directory so it uses the same `config.yaml` and `alarm_data/` paths.
