"""Run the game:  python -m outofframe --config configs/mac.yaml"""
from __future__ import annotations

import argparse
import contextlib
import logging
import time

from .camera import Camera
from .config import load_config
from .detector import PersonDetector
from .framer import FollowingFramer, make_framer
from .game import Game
from .pose import Gallery, PoseEstimator
from .window import GameWindow, quit_on_signals


def main() -> None:
    ap = argparse.ArgumentParser(description="Out of Frame - booth game")
    ap.add_argument("--config", default="configs/mac.yaml")
    ap.add_argument("--source", help="override camera source (index, rtsp://, gst:..., file)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-pose", action="store_true", help="skip pose matching (no pose model, no gallery)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.source is not None:
        cfg.camera.source = int(args.source) if args.source.isdigit() else args.source
    if args.fullscreen:
        cfg.display.fullscreen = True
    if args.no_pose:
        cfg.pose.enabled = False

    quit_on_signals()
    # Everything started below is stopped again in reverse order, however the game ends:
    # close button, q/Esc, closing the window, Ctrl+C, pkill or closing the terminal.
    with contextlib.ExitStack() as stack:
        try:
            detector = PersonDetector(cfg.detector)
            detector.warmup()
            camera = Camera(cfg.camera).start()
            stack.callback(camera.stop)
            framer = make_framer(cfg.framer, cfg.camera.mirror)
            stack.callback(framer.close)  # centres the gimbal
            steady = None
            if isinstance(framer, FollowingFramer) and cfg.framer.virtual.steady_view:
                from .steady import SteadyView

                steady = SteadyView(framer.gimbal, cfg.framer.virtual,
                                    (cfg.framer.siyi.home_yaw, cfg.framer.siyi.home_pitch),
                                    cfg.camera.mirror)
                steady.calibrate()
            poser = gallery = None
            if cfg.pose.enabled:
                poser = PoseEstimator(cfg.pose, cfg.detector.device, cfg.detector.half)
                gallery = Gallery.load(cfg.pose.gallery_dir, poser)
                if not gallery.items:
                    logging.warning("Pose match off: no usable images in %s", cfg.pose.gallery_dir)
            game = Game(cfg, framer, detector, poser=poser, gallery=gallery)

            window = GameWindow(cfg.display.window_name, cfg.display.fullscreen,
                                (cfg.display.width, cfg.display.height))
            stack.callback(window.destroy)
            while not (game.quit or window.close_requested):
                frame, frame_t = camera.read()
                if steady is not None and frame is not None:
                    frame = steady.apply(frame, frame_t)
                now = time.monotonic()
                window.show(game.step(now, frame))
                key = window.poll()
                if key != 255:
                    game.on_key(key, now)
        except KeyboardInterrupt:
            pass
        logging.info("Shutting down")
    logging.info("Closed cleanly")


if __name__ == "__main__":
    main()
