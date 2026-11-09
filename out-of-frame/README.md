# Out of Frame — The Scouts @ Safety & Security Asia

A booth game. The camera takes 4 photos while its frame swings around at random.
Stay out of every photo to win a prize. Then comes a **bonus round**: copy the pose in a picture
for a souvenir photo. At the end the photos are "developed". Any body pixels found inside a photo get
tinted red and stamped **THREAT DETECTED**. The player can download a collage of the game on their phone.

Two ways to run it, sharing the same code:

| | Mac (testing) | Drone (booth) |
|---|---|---|
| Camera | Webcam | SIYI A8 mini 4K over RTSP |
| "Gimbal" | A virtual box that glides over the fixed webcam image | The real gimbal swings to random yaw/pitch angles over the SIYI SDK (UDP 37260) |
| Detector | YOLO11n-seg on Apple GPU (MPS) | YOLO11n-seg TensorRT engine (FP16) |
| Config | `configs/mac.yaml` | `configs/jetson.yaml` |

## How a round works

1. **Attract screen.** Shows QR codes for The Scouts' Instagram and LinkedIn (`display.socials`;
   regenerate the images in `assets/` with `tools/make_qr.py`). Press `SPACE` to start.
2. **Get ready.** 3-second countdown.
3. **4 photos** (`game.shots`). For each one, the frame moves through 1–3 random waypoints ("feints").
   Then it **locks on**: the box goes yellow, then flashes red, with a progress bar.
   Then it **snaps**. Every round gets faster and the box gets bigger.
4. **Bonus round** (`game.bonus_pose`). A random picture from `gallery/bonus.txt` (66 poses a visitor
   can copy standing up: no upside-down, airborne, floor or partner poses) is shown next to the live camera:
   "Copy this pose!". After `game.bonus_s` (6 s, with a 3-2-1) the camera takes one photo, cropped
   around the players. The pose isn't scored. Pictures don't repeat for the next 30 games.
5. **Developing photos.** The 4 photos are revealed one at a time, each stamped CLEAR, THREAT DETECTED or MISS
   (the collage says DODGED, THREAT DETECTED or MISSED).
6. **Result.** The collage (bonus photo on top with the copied picture inset, the 4 photos in a 2×2
   grid below) next to a **QR code** to download it, plus a 4-character prize code on a win. It stays
   up until `SPACE` goes back to the start screen (`SPACE` again starts a new game, `q` quits).
   Every game is logged to `sessions/results.csv`, so staff can check codes. Each session folder
   also holds the raw, judged and full-frame images, the bonus photo and `collage.jpg`.

### Collage download

Same flow as the drone booth: the QR opens a page served by the game itself (`outofframe/share.py`,
port **8081** so it can run next to the booth on 8080). The player enters name + email (stored in
`sessions/share/leads.csv`) and downloads or shares the collage (1080×1440, Instagram's 3:4 portrait).
The Jetson configs publish the page through a Cloudflare tunnel (`share.tunnel: cloudflare`) so any phone
can open it, on any network; see "Photo downloads from any phone" in the top-level README. The Mac
configs serve it on the local network only. Turn it off with `share.enabled: false`;
`share.require_details: false` makes the details optional.

The results screen stays up until `SPACE` (back to the start screen).

A person counts as caught when their segmentation mask covers at least `min_body_frac`
(0.1%) of the photo, or `min_body_px` pixels, whichever is larger. So even a hand or an elbow counts.

In **virtual mode** the wide view is visible, so a player must also be in view at every snap
(`require_player_in_view`). Otherwise walking off camera would be an easy win. If nobody is in
view, the photo is stamped **MISS** and still counts as a loss.

## Gallery and pose match

The bonus round picks from `gallery/`. Optionally (`pose.match_each_snap: true`, off by default), the
game also finds the player's pose after each snap and shows the `gallery/` picture that matches it best
("POSE MATCH!"), like the pose-match TikToks. The bundled gallery is 158 public-domain / CC0 photos
from Wikimedia Commons (athletes, dancers, martial artists, performers, statues, animals); see `gallery/CREDITS.md`.
The most similar picture is always shown, even if it was shown before.

- Drop images (`.jpg`/`.png`) into `gallery/`. Poses are read automatically on the next start and cached in
  `gallery/index.json`. Images with no person found are skipped.
- For a picture the pose model can't read (an animal, a letter), write its keypoints into `index.json`
  by hand with `"manual": true`.
- `gallery/bonus.txt` lists the pictures the bonus round may show, one file name per line. Add a new
  picture there only if anyone could copy it in the booth.
- Check a photo against the gallery: `python -m outofframe.pose --test photo.jpg`
- No picture is shown when the player is just standing still (`pose.max_neutral`): players are asked to
  strike a pose at every snap.
- Settings: `pose.show_s` (time on screen), `pose.min_score` (weakest match still shown), `pose.enabled`
  (or run with `--no-pose` to turn the gallery off for one run: no bonus round, no pose match).
- New pictures are only indexed when the pose model is loaded (`match_each_snap`, or
  `python -m outofframe.pose`); the bonus round uses the cached poses in `index.json`.

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

Jetson configs:

| Config | Gimbal | Screen |
|---|---|---|
| `jetson.yaml` | Swings to random angles; the whole image is the photo | Moves with the gimbal |
| `jetson_table.yaml` | Same, tuned for a camera on a desk | Moves with the gimbal |
| `jetson_virtual.yaml` | Holds still; a box glides over the view | Fixed |
| `jetson_follow.yaml` | Leans a few degrees toward the gliding box (for show only) | Moves slightly with the gimbal |
| `jetson_steady.yaml` | Same as follow | Held still by shifting each frame back (slight zoom, ~85% of the view) |

In the follow and steady configs, set the lean with `follow_yaw_deg` / `follow_pitch_deg`. If the camera turns
away from the box, flip `follow_invert_yaw` / `follow_invert_pitch`. If the steady picture jumps
when the gimbal moves, adjust `steady_latency_s`.

The A8 mini streams H.265 by default (even on the `main.264` path), so the pipeline uses
`rtph265depay ! h265parse`. If the camera is switched to H.264, change it to `rtph264depay ! h264parse`.

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

## Branding

Screens follow The Scouts brand (thescouts.io): Poppins type, the Scouts logo, and the logo palette
Ebony `#6d4641`, Dutch `#ffefca`, Powder `#9cadce` and Ink `#0e0e0e`. Fonts and logos are in
`assets/brand/` (Poppins is SIL Open Font License, see `assets/brand/fonts/OFL.txt`). The palette and the
Poppins/logo drawing helpers are in `outofframe/brand.py`. The same file is in both projects, so keep the two copies in sync.

## Tests

```bash
.venv/bin/python -m pytest -q
```
