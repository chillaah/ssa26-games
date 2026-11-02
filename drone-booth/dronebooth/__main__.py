"""Run the dronebooth:  python -m dronebooth --config configs/booth_mac.yaml"""
from __future__ import annotations

import argparse
import logging
import time

import cv2

from .camera import Camera
from .detector import PersonDetector

from .booth import Booth, LiveScanner
from .config import load_booth_config
from .server import PhotoStore, ShareServer


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

    detector = PersonDetector(cfg.detector)
    detector.warmup()
    store = PhotoStore(cfg.booth.data_dir)
    share = ShareServer(store, cfg.share).start()
    print(f"\n  Visitors' phones open: {share.base_url}/p/<photo>\n")
    camera = Camera(cfg.camera).start()
    scanner = LiveScanner(camera, detector)
    booth = Booth(cfg, scanner, store, share)

    win = cfg.display.window_name
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fullscreen = cfg.display.fullscreen
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    if not fullscreen:
        cv2.resizeWindow(win, cfg.display.width // 2, cfg.display.height // 2)

    try:
        while not booth.quit:
            frame, _ = camera.read()
            now = time.monotonic()
            cv2.imshow(win, booth.step(now, frame))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key != 255:
                booth.on_key(key, now)
    except KeyboardInterrupt:
        pass
    finally:
        scanner.stop()
        camera.stop()
        share.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
