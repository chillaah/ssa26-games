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

Each project's README covers configs (`mac`, `mac_siyi`, `jetson`), SIYI A8 mini wiring and tuning.

Model weights (`*.pt`), captured sessions and visitor data are not tracked in git.

## License

[MIT](LICENSE)
