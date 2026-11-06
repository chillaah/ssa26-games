"""Run the game:  python -m outofframe --config configs/mac.yaml"""
from __future__ import annotations

import argparse
import logging
import time

import cv2

from .camera import Camera
from .config import load_config
from .detector import PersonDetector
from .framer import FollowingFramer, make_framer
from .game import Game
from .pose import Gallery, PoseEstimator


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

    detector = PersonDetector(cfg.detector)
    detector.warmup()
    camera = Camera(cfg.camera).start()
    framer = make_framer(cfg.framer, cfg.camera.mirror)
    steady = None
    if isinstance(framer, FollowingFramer) and cfg.framer.virtual.steady_view:
        from .steady import SteadyView

        steady = SteadyView(framer.gimbal, cfg.framer.virtual,
                            (cfg.framer.siyi.home_yaw, cfg.framer.siyi.home_pitch), cfg.camera.mirror)
        steady.calibrate()
    poser = gallery = None
    if cfg.pose.enabled:
        poser = PoseEstimator(cfg.pose, cfg.detector.device, cfg.detector.half)
        gallery = Gallery.load(cfg.pose.gallery_dir, poser)
        if not gallery.items:
            logging.warning("Pose match off: no usable images in %s", cfg.pose.gallery_dir)
    game = Game(cfg, framer, detector, poser=poser, gallery=gallery)

    win = cfg.display.window_name
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fullscreen = cfg.display.fullscreen
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    if not fullscreen:
        cv2.resizeWindow(win, cfg.display.width // 2, cfg.display.height // 2)

    try:
        while not game.quit:
            frame, frame_t = camera.read()
            if steady is not None and frame is not None:
                frame = steady.apply(frame, frame_t)
            now = time.monotonic()
            cv2.imshow(win, game.step(now, frame))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key != 255:
                game.on_key(key, now)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        framer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
