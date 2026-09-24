# Puppy Pad Alarm (vibe-coded!!!!)

A local camera app for two puppy pads placed side by side. After a dog enters the selected area, it looks for a new dark object anywhere in that area. A possible detection plays a short deterrent sound near the dog and sends a Pushover notification to your phone.

The camera feed and event videos stay on your computer. Pushover receives a snapshot when an event is detected. A detection is a prompt to check the pad, not proof of poop.

## Start

You need Python 3.13+, [uv](https://docs.astral.sh/uv/), a camera, and [Pushover](https://pushover.net/api) credentials.

From the project directory in PowerShell on Windows:

```powershell
uv sync
$env:PUSHOVER_TOKEN = 'your-application-token'
$env:PUSHOVER_USER = 'your-user-key'
uv run python -m puppy_pad_alarm.app
```

On Linux:

```sh
uv sync
export PUSHOVER_TOKEN='your-application-token'
export PUSHOVER_USER='your-user-key'
uv run python -m puppy_pad_alarm.app
```

The first run downloads the YOLO dog detector model. On Windows, the app selects the Logitech C270 by name and reports the available camera names if it cannot find it. You can set `camera_source: 1` in `config.yaml` to select a camera index manually. On Linux, you can also use `--camera /dev/video2`.

## Use the preview

| Key | Action |
| --- | --- |
| `C` | Drag a rectangle around both pads, then press `Enter` or `Space` to save it. |
| `B` | Save one clean reference image for the whole area. After cleanup, this also clears an active alert. |
| `Q` | Quit. |

On first use, press `C`, then press `B` while both pads are visible and clean. You do not need to repeat this on a normal restart. If you change the selected area with `C`, press `B` again to save a new clean baseline.

After a detection, the whole area stays in an alert state until you clean it and press `B`. Another visit will not send a new detection alert while that state is active. If the dog approaches the detected object again, the deterrent sound plays again.

The app can alert while the dog is still in the area if a new dark object remains visible outside the dog's masked region. The mask includes a margin around the detected dog box. A brief missed dog detection is held for one second to avoid treating the dog as the object.

The phone alert uses Pushover emergency priority. Pushover repeats it every minute for up to 5 minutes unless you acknowledge it in the Pushover app. The app does not send separate reminders. Pressing `B` clears the app's alert state, but does not acknowledge an emergency message already sent to Pushover.

## Saved files

| Location | Contents |
| --- | --- |
| `config.yaml` | Saved camera search area and settings. |
| `alarm_data/` | Clean reference images, snapshots, alert state, and `events.log`. |
| `alarm_data/event_videos/` | Local videos of detected dog visits and matching JSON metadata. |

Visit videos include up to 10 seconds before the visit and 20 seconds after the dog leaves. The app keeps them for up to 7 days or 5 GB, whichever limit comes first. A visit missed by the dog detector will not produce a clip.

Copy [config.example.yaml](config.example.yaml) to `config.yaml` before starting. It is ready to use with the C270; `search_region: null` is filled when you press `C`. You can then edit thresholds or recording limits if needed. Set `save_event_video: false` to turn off visit recording. Leave `save_debug_video: false` for normal use; debug video records the full run without a storage limit.

## Check recordings later

You can replay a clip without sending notifications or playing the sound:

```sh
uv run python -m puppy_pad_alarm.app --video clip.mp4 --expected poop
```

Use `--expected none` for a clip with no poop. Replay writes results under `alarm_data/replay/`. If the clip starts with a clean area, press `B` to set its reference image.

## If something is wrong

- No phone alert: Check that `PUSHOVER_TOKEN` and `PUSHOVER_USER` are set in the terminal that starts the app. Events and errors appear in that terminal and in `alarm_data/events.log`. A network handshake timeout can still occur; the app allows up to 30 seconds for the request.
- No detection or too many alerts: Check the saved snapshot and event video. Detection accepts dark colors, including black, and can mistake shadows or dark wet patches for poop. Adjust `max_dark_value` and the other image thresholds for your camera and lighting.
- Restart appears to lose setup: Start the app from the project directory so it uses the same `config.yaml` and `alarm_data/` paths.
- C270 is not found on Windows: Check that Windows lists and enables the webcam. The startup error shows the DirectShow camera names and indices found by the app.
