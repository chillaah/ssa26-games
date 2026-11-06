"""Run the dronebooth:  python -m dronebooth --config configs/booth_mac.yaml"""
from __future__ import annotations

import argparse
import contextlib
import logging
import time

from .camera import Camera
from .detector import PersonDetector

from .booth import Booth, LiveScanner
from .config import load_booth_config
from .server import PhotoStore, ShareServer
from .window import GameWindow, quit_on_signals


def main() -> None:
    ap = argparse.ArgumentParser(description="Caught by Scouts - dronebooth")
    ap.add_argument("--config", default="configs/booth_mac.yaml")
    ap.add_argument("--source", help="override camera source (index, rtsp://, gst:..., file)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_booth_config(args.config)
    if args.source is not None:
        cfg.camera.source = int(args.source) if args.source.isdigit() else args.source
    if args.fullscreen:
        cfg.display.fullscreen = True

    quit_on_signals()
    # Everything started below is stopped again in reverse order, however the booth ends:
    # close button, q/Esc, closing the window, Ctrl+C, pkill or closing the terminal.
    with contextlib.ExitStack() as stack:
        try:
            detector = PersonDetector(cfg.detector)
            detector.warmup()
            store = PhotoStore(cfg.booth.data_dir)
            share = ShareServer(store, cfg.share).start()
            stack.callback(share.stop)
            print(f"\n  Visitors' phones open: {share.base_url}/p/<photo>\n")
            camera = Camera(cfg.camera).start()
            stack.callback(camera.stop)
            scanner = LiveScanner(camera, detector)
            stack.callback(scanner.stop)
            booth = Booth(cfg, scanner, store, share)

            window = GameWindow(cfg.display.window_name, cfg.display.fullscreen,
                                (cfg.display.width, cfg.display.height))
            stack.callback(window.destroy)
            while not (booth.quit or window.close_requested):
                frame, _ = camera.read()
                now = time.monotonic()
                window.show(booth.step(now, frame))
                key = window.poll()
                if key != 255:
                    booth.on_key(key, now)
        except KeyboardInterrupt:
            pass
        logging.info("Shutting down")
    logging.info("Closed cleanly")


if __name__ == "__main__":
    main()
