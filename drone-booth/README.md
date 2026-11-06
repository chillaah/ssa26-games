# Caught by Scouts — the dronebooth

Part of The Scouts booth at Safety & Security Asia 2026. Its sibling game is in `../out-of-frame`.

A live "surveillance" view shows the camera feed, with an amber **SCANNING...** box on every person.
A visitor presses `SPACE`, gets a 3-2-1 countdown and a flash, and the souvenir photo is created:
1080×1440 (Instagram's 3:4 portrait), cropped from the camera image around the people in it.
The photo has a security-cam look: every person gets a red **THREAT DETECTED** box with a body
outline. It also carries a case number, a timestamp, the event name and thescouts.io.
The booth screen then shows the photo next to a **QR code** until `SPACE` (next visitor) or `R` (retake).

Scanning the QR opens a page on the visitor's phone. It shows a blurred preview and a form
(name, email, optional company, optional marketing consent). Submitting the form unlocks the
full photo, with Download and Share buttons. Share uses the phone's share sheet, so it works with
WhatsApp, Instagram and so on.
On the Jetson the page has a public address (Cloudflare tunnel), so it opens on any network; see
"Photo downloads from any phone" in the top-level README.

## Run on the Mac

```bash
cd drone-booth
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m dronebooth --config configs/mac.yaml
```

Keys: `SPACE` take photo / next visitor · `R` retake · `F` fullscreen · `Q` quit.

- **Data:** photos go in `booth_data/photos/` and leads in `booth_data/leads.csv`.
- **Optional details:** set `share.require_details: false` to make the form optional.
- **Network:** visitors' phones must be able to reach this machine. By default the QR points at
  `http://<this machine's LAN IP>:8080`.
  - The phones must be on the same network as the Mac. Conference Wi-Fi often blocks
    device-to-device traffic, so bring your own router or hotspot for the booth.
  - Alternatively, expose port 8080 through a tunnel (e.g. `cloudflared tunnel --url http://localhost:8080`)
    and set `share.public_base_url` to the tunnel URL.
  - macOS asks once to allow incoming connections for Python. Click Allow.

## Run on a laptop + SIYI A8 mini (no Jetson)

The simplest booth setup. Plug the A8 mini's **Ethernet** lead into the laptop with a USB-C
Ethernet adapter. The camera isn't a USB webcam; it streams RTSP over Ethernet.

1. In System Settings → Network, set the adapter to **Manually**: IP `192.168.144.20`,
   subnet `255.255.255.0`, leave the router blank. Keep Wi-Fi on for the internet and phones.
2. Check the feed: `ffplay rtsp://192.168.144.25:8554/main.264` (or open it in VLC).
3. Run with `configs/mac_siyi.yaml`.

```bash
.venv/bin/python -m dronebooth --config configs/mac_siyi.yaml
```

## Run on the Jetson Orin Nano + SIYI A8 mini

`configs/jetson.yaml` reads the A8 mini's RTSP stream with hardware decoding and uses a TensorRT
engine. The setup (network, PyTorch wheels, system OpenCV, `yolo export ... format=engine`) is the
same as in `../out-of-frame/README.md`. The booth doesn't move the gimbal. Point it at the
photo spot with the SIYI app.

```bash
python3 -m dronebooth --config configs/jetson.yaml
```

## Branding

Screens follow The Scouts brand (thescouts.io): Poppins type, the Scouts logo, and the logo palette
Ebony `#6d4641`, Dutch `#ffefca`, Powder `#9cadce` and Ink `#0e0e0e`. Fonts and logos are in
`assets/brand/` (Poppins is SIL Open Font License, see `assets/brand/fonts/OFL.txt`). The palette and the
Poppins/logo drawing helpers are in `dronebooth/brand.py`. The same file is in both projects, so keep the two copies in sync.
The phone download page uses the same look and serves the fonts and logo itself (`/brand/...`), so it works without internet.

## Tests

```bash
.venv/bin/python -m pytest -q
```
