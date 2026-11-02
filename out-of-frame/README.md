# Out of Frame — The Scouts @ Safety & Security Asia

A booth game. The camera takes 5 photos while its frame swings around at random.
Stay out of every photo to win a prize. At the end the photos are "developed".
Any body pixels found inside a photo get tinted red and stamped **CAUGHT**.

Two ways to run it, sharing the same code:

| | Mac (testing) | Drone (booth) |
|---|---|---|
| Camera | Webcam | SIYI A8 mini 4K over RTSP |
| "Gimbal" | A virtual box that glides over the fixed webcam image | The real gimbal swings to random yaw/pitch angles over the SIYI SDK (UDP 37260) |
| Detector | YOLO11n-seg on Apple GPU (MPS) | YOLO11n-seg TensorRT engine (FP16) |
| Config | `configs/mac.yaml` | `configs/jetson.yaml` |

## How a round works

1. **Attract screen.** Press `SPACE` to start.
2. **Get ready.** 3-second countdown.
3. **5 photos.** For each one, the frame moves through 1–3 random waypoints ("feints").
   Then it **locks on**: the box goes yellow, then flashes red, with a progress bar.
   Then it **snaps**. Every round gets faster and the box gets bigger.
4. **Developing photos.** The photos are revealed one at a time, each marked CLEAR or CAUGHT.
5. **Result.** A win shows a 4-character prize code. Every game is logged to `sessions/results.csv`,
   so staff can check codes. Each session folder also holds the raw, judged and full-frame images.

A person counts as caught when their segmentation mask covers at least `min_body_frac`
(0.1%) of the photo, or `min_body_px` pixels, whichever is larger. So even a hand or an elbow counts.

In **virtual mode** the wide webcam view is visible, so the game also requires a player to
be in view at every snap (`require_player_in_view`). Without this, walking off camera would
be an easy win. That case is marked "LEFT THE ARENA".

## Run on the Mac

```bash
cd out-of-frame
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m outofframe --config configs/mac.yaml
```

On the first run, macOS will ask for camera permission for your terminal. The model
(`yolo11n-seg.pt`, ~6 MB) downloads automatically.

Keys: `SPACE` start / skip reveal · `F` fullscreen · `D` show live person masks (debug) ·
`R` back to attract screen · `Q`/`Esc` quit.

Other sources: `--source 1` (a different webcam), `--source some_video.mp4`, `--source photo.jpg`.

## Run on a laptop + SIYI A8 mini (no Jetson)

The simplest booth setup. Plug the A8 mini's **Ethernet** lead into the laptop with a USB-C
Ethernet adapter. The camera isn't a USB webcam; it streams RTSP over Ethernet.

1. In System Settings → Network, set the adapter to **Manually**: IP `192.168.144.20`,
   subnet `255.255.255.0`, leave the router blank. Keep Wi-Fi on for the internet and phones.
2. Check the feed: `ffplay rtsp://192.168.144.25:8554/main.264` (or open it in VLC).
3. Run with `configs/mac_siyi.yaml`.
Gimbal control uses the same cable (UDP 37260). Calibrate the yaw/pitch play area in the config.

```bash
.venv/bin/python -m outofframe --config configs/mac_siyi.yaml
```

## Run on the Jetson Orin Nano + SIYI A8 mini

1. Wiring:
   - **Video always comes over the camera's Ethernet lead.** The A8 mini defaults to `192.168.144.25`.
     Give the Jetson an address on `192.168.144.0/24`. Check that
     `rtsp://192.168.144.25:8554/main.264` plays.
   - **Gimbal control** can go over that same Ethernet link (`framer.siyi.link: udp`, the default)
     or over the TTL serial cable (`link: serial`). On the serial link, use `serial_port: /dev/ttyTHS0`
     for the 40-pin header (pins 8/10), or `/dev/ttyUSB0` for a USB-serial adapter, at 115200 baud.
     Add your user to the `dialout` group so it can open the port.
   - **Monitor:** plug into the Orin Nano dev kit's DisplayPort. If the screen only has HDMI, use an
     *active* DP-to-HDMI adapter.
2. Python: JetPack 6 ships Python 3.10. Install PyTorch/torchvision from NVIDIA's Jetson wheels,
   then `pip install ultralytics pyyaml pyserial`. Use the system OpenCV (built with GStreamer). Do **not**
   pip-install `opencv-python` there, because it has no GStreamer support.
3. Build the TensorRT engine once:
   ```bash
   yolo export model=yolo11n-seg.pt format=engine half=True imgsz=640
   ```
4. Calibrate the play area. In `configs/jetson.yaml`, set `yaw_min/max` and `pitch_min/max` so that
   every angle the gimbal can pick still points at the floor where visitors stand.
5. Run:
   ```bash
   python3 -m outofframe --config configs/jetson.yaml
   ```

If the A8 mini is set to H.265, change `rtph264depay ! h264parse` to `rtph265depay ! h265parse`
in the camera pipeline.

## Tuning

All the knobs are in `outofframe/config.py`, with defaults and comments. Any of them can be overridden in the YAML.
The most useful ones:

- `game.shots`, `game.lock_s_first` / `lock_s_final`: number of photos and the warning time before each snap
- `framer.virtual.box_width_frac` / `box_width_frac_final`, `travel_s` / `travel_s_final`: difficulty
- `game.min_body_frac`: how much body inside a photo counts as caught
- `detector.conf`: raise it if visitors' bags or posters get detected as people

## Booth notes

- Passers-by in the background will "catch" the player. Point the camera so the play area
  has a backdrop (a wall or a banner) rather than the aisle.
- Mark the play area on the floor with tape.

## Tests

```bash
.venv/bin/python -m pytest -q
```
