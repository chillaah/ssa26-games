"""Threaded frame grabber that always hands out the newest frame.

Network streams (the SIYI A8 mini RTSP feed) buffer frames; reading on a background
thread and keeping only the latest one keeps the on-screen box in sync with reality.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CameraCfg

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Camera:
    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_t = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._still: Optional[np.ndarray] = None

        src = cfg.source
        if isinstance(src, str) and src.lower().endswith(IMAGE_EXTS):
            # A still image source, used for testing without a camera.
            self._still = cv2.imread(src)
            if self._still is None:
                raise FileNotFoundError(src)

    def _open(self) -> cv2.VideoCapture:
        src = self.cfg.source
        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            cap = cv2.VideoCapture(int(src))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        elif src.startswith("gst:"):
            cap = cv2.VideoCapture(src[4:], cv2.CAP_GSTREAMER)
        elif src.startswith("rtsp://"):
            # TCP avoids smeared frames from dropped UDP packets; nobuffer keeps latency low.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(src)
        return cap

    def start(self) -> "Camera":
        if self._still is not None:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        is_file = isinstance(self.cfg.source, str) and not self.cfg.source.startswith(
            ("rtsp://", "gst:")
        ) and not self.cfg.source.isdigit()
        while self._running:
            cap = self._open()
            if not cap.isOpened():
                log.warning("Camera %r not available, retrying in 1s", self.cfg.source)
                time.sleep(1.0)
                continue
            log.info("Camera %r opened", self.cfg.source)
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    if is_file:  # loop test videos
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    log.warning("Camera read failed, reconnecting")
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_t = time.monotonic()
                if is_file:
                    time.sleep(1.0 / max(self.cfg.fps, 1))
            cap.release()

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (newest frame or None, capture timestamp)."""
        if self._still is not None:
            frame, t = self._still, time.monotonic()
        else:
            with self._lock:  # the reader thread swaps in a new array per frame, never edits one
                if self._frame is None:
                    return None, 0.0
                frame, t = self._frame, self._frame_t
        # Callers get their own copy (the flip makes one when mirroring).
        return (cv2.flip(frame, 1) if self.cfg.mirror else frame.copy()), t

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
