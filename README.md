# SSA26 Games

Booth games for The Scouts at Safety & Security Asia 2026. Two self-contained Python projects that run
on a Mac with a webcam for testing, and on a Jetson Orin Nano with a SIYI A8 mini gimbal camera at the event.

| Folder | What it is |
|---|---|
| [`out-of-frame/`](out-of-frame/README.md) | **Out of Frame**, the game: dodge 5 photos from a randomly moving frame to win a prize |
| [`drone-booth/`](drone-booth/README.md) | **Caught by Scouts**, the photobooth: "THREAT DETECTED" photo + QR code to download it (collects name/email) |

## Quick start

```bash
cd out-of-frame            # or drone-booth
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m outofframe --config configs/mac.yaml   # drone-booth: python -m dronebooth --config configs/mac.yaml
```

Each project's README covers its configs (`mac`, `mac_siyi`, `jetson*`), SIYI A8 mini wiring and tuning.

Model weights (`*.pt`), captured sessions and visitor data are not tracked in git.

## Jetson desktop shortcuts

On the Jetson, `./jetson/install-shortcuts.sh` puts **Out of Frame** and **Drone Booth** icons on the
desktop. Out of Frame first asks which `configs/jetson*.yaml` to use (`jetson_steady` is pre-selected).
Each opens a terminal with the log, also saved to `/tmp/outofframe.log` / `/tmp/dronebooth.log`. The terminal closes
by itself when the game exits cleanly and stays open if it stopped with an error.

To close either game, tap the X at the top right twice (the first tap arms it), or press q / Esc.
Closing the window or terminal, Ctrl+C or `pkill` also shut it down cleanly: camera stopped, gimbal
centred, share server closed.

## Photo downloads from any phone

Both games show a QR code for the player's photo. The page behind it (name, email, company, then the
download) is served by the game itself: drone booth on port 8080, Out of Frame on 8081.

On the Jetson (`share.tunnel: cloudflare` in the `jetson*.yaml` configs) each game also starts a
**Cloudflare quick tunnel**, which gives the page a public `https://<random>.trycloudflare.com` address. Phones
on mobile data or any Wi-Fi can open it. There's no account or token, and no warning page like ngrok's free plan. The Jetson needs internet. The
address changes every time a game starts. The QR codes always use the current one, but links from before a
restart stop working. If the tunnel drops, the game restarts it. Without internet or `cloudflared`,
the QR codes fall back to the local-network address.

Install `cloudflared` once on the Jetson (one program, no sudo):

```bash
mkdir -p ~/.local/bin
curl -L -o ~/.local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64
chmod +x ~/.local/bin/cloudflared
```

For a fixed address (e.g. `photos.thescouts.io`), set up a named Cloudflare tunnel on a domain in
Cloudflare and put its address in `share.public_base_url`.

**Visitor details stay on the Jetson.** Each submission (time, photo id, name, email, company, marketing
consent) is appended to a CSV and written to disk at once:
`drone-booth/booth_data/leads.csv` and `out-of-frame/sessions/share/leads.csv`. Both folders are
git-ignored and left out of the Mac-to-Jetson sync, so they are never committed or overwritten. Copy them off
the Jetson after the event (`scp`). Photos are under `.../photos/` next to them.

## Branding

Both projects use The Scouts brand: Poppins, the Scouts logo, and the Ebony / Dutch / Powder palette.
Assets are in each project's `assets/brand/`.

## License

[MIT](LICENSE)
