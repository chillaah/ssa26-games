"""Booth state machine: live scan -> countdown -> snap -> photo + QR code."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
import qrcode

from . import render as R
from .camera import Camera
from .detector import Person, PersonDetector

from .compose import SCAN, THREAT, PhotoMeta, compose_photo, draw_subjects, hud_frame, surveillance_grade
from .config import BoothConfig
from .server import PhotoStore, ShareServer

log = logging.getLogger(__name__)

LIVE, COUNTDOWN, FLASH, REVIEW = "live", "countdown", "flash", "review"


class LiveScanner:
    """Runs person detection on the newest camera frame in the background."""

    def __init__(self, camera: Camera, detector: PersonDetector):
        self.camera = camera
        self.detector = detector
        self.lock = threading.Lock()  # the YOLO model is not thread safe
        self.people: List[Person] = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="scanner", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        last_t = 0.0
        while self._running:
            frame, t = self.camera.read()
            if frame is None or t == last_t:
                time.sleep(0.005)
                continue
            last_t = t
            with self.lock:
                self.people = self.detector.detect(frame)

    def detect_now(self, frame: np.ndarray) -> List[Person]:
        with self.lock:
            return self.detector.detect(frame)

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)


def qr_image(url: str, size: int) -> np.ndarray:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = np.array(qr.make_image(fill_color="black", back_color="white").convert("RGB"))
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (size, size),
                      interpolation=cv2.INTER_NEAREST)


class Booth:
    def __init__(self, cfg: BoothConfig, scanner: LiveScanner, store: PhotoStore, share: ShareServer):
        self.cfg = cfg
        self.scanner = scanner
        self.store = store
        self.share = share
        self.state = LIVE
        self.state_t = 0.0
        self.quit = False
        self.photo: Optional[np.ndarray] = None
        self.qr: Optional[np.ndarray] = None
        self.url = ""
        self.notice = ""
        self.notice_until = 0.0

    def _set(self, state: str, now: float) -> None:
        self.state, self.state_t = state, now

    def on_key(self, key: int, now: float) -> None:
        if key in (ord("q"), 27):
            self.quit = True
        elif key in (ord(" "), 13):
            if self.state == LIVE:
                self._set(COUNTDOWN, now)
            elif self.state == REVIEW:
                self._set(LIVE, now)
        elif key == ord("r") and self.state == REVIEW:
            self._set(COUNTDOWN, now)  # retake

    def step(self, now: float, frame: Optional[np.ndarray]) -> np.ndarray:
        d = self.cfg.display
        b = self.cfg.booth
        canvas = np.zeros((d.height, d.width, 3), np.uint8)
        if self.state == REVIEW:
            if now - self.state_t >= b.review_s:
                self._set(LIVE, now)
            else:
                self._draw_review(canvas, now)
                return canvas
        if frame is None:
            R.text(canvas, "Waiting for camera...", (d.width // 2, d.height // 2), 1.5)
            return canvas

        dt = now - self.state_t
        if self.state == COUNTDOWN and dt >= b.countdown_s:
            self._capture(frame, now)
        elif self.state == FLASH and dt >= b.flash_s:
            self._set(REVIEW, now)
            self._draw_review(canvas, now)
            return canvas

        self._draw_live(canvas, frame, now, now - self.state_t)
        return canvas

    def _capture(self, frame: np.ndarray, now: float) -> None:
        people = self.scanner.detect_now(frame)
        if not people:
            self.notice, self.notice_until = "NO ONE DETECTED - STEP INTO THE ZONE", now + 2.5
            self._set(LIVE, now)
            return
        b = self.cfg.booth
        stamp = datetime.now()
        meta = PhotoMeta(
            case_id=stamp.strftime("%H%M%S")[-4:] + f"-{len(people):02d}",
            timestamp=stamp.strftime("%Y-%m-%d %H:%M:%S ") + b.timezone_label,
            event=b.event,
            website=b.website,
        )
        self.photo = compose_photo(frame, people, meta, b.max_subjects)
        pid = self.store.save(self.photo)
        self.url = self.share.url_for(pid)
        self.qr = qr_image(self.url, 440)
        log.info("photo %s: %d people -> %s", pid, len(people), self.url)
        self._set(FLASH, now)

    # ---- drawing ----------------------------------------------------------------
    def _draw_live(self, canvas: np.ndarray, frame: np.ndarray, now: float, dt: float) -> None:
        d = self.cfg.display
        b = self.cfg.booth
        W, H = d.width, d.height
        view = surveillance_grade(frame)
        people = self.scanner.people
        counting = self.state == COUNTDOWN
        draw_subjects(view, people, "LOCKING ON" if counting else "SCANNING...",
                      THREAT if counting else SCAN, b.max_subjects, outline=False)
        blink = int(now * 2) % 2 == 0
        hud_frame(view, "SCOUTS AI  //  LIVE", datetime.now().strftime("%H:%M:%S ") + b.timezone_label,
                  rec_on=blink)
        R.paste_fit(canvas, view, (0, 0, W, H))

        if self.state == LIVE:
            R.text(canvas, "CAUGHT BY SCOUTS", (W // 2, int(H * 0.82)), 2.6, (255, 255, 255), 6,
                   font=R.FONT_BOLD)
            msg = "PRESS SPACE TO GET CAUGHT"
            if self.notice and now < self.notice_until:
                msg = self.notice
            R.text(canvas, msg, (W // 2, int(H * 0.91)), 1.4, SCAN, 3)
        elif self.state == COUNTDOWN:
            left = max(1, int(np.ceil(b.countdown_s - dt)))
            R.text(canvas, str(left), (W // 2, H // 2), 10.0, THREAT, 18, font=R.FONT_BOLD)
            R.text(canvas, "STRIKE A POSE!", (W // 2, int(H * 0.82)), 2.2, (255, 255, 255), 5)
        elif self.state == FLASH:
            a = max(0.0, 1.0 - dt / b.flash_s)
            canvas[:] = cv2.addWeighted(canvas, 1 - a, np.full_like(canvas, 255), a, 0)

    def _draw_review(self, canvas: np.ndarray, now: float) -> None:
        d = self.cfg.display
        W, H = d.width, d.height
        canvas[:] = (22, 18, 16)
        R.text(canvas, "THREAT DETECTED", (60, 95), 2.4, THREAT, 6, anchor="left", font=R.FONT_BOLD)
        R.text(canvas, "You've been caught by The Scouts drone.", (60, 165), 1.2, (230, 230, 230), 2,
               anchor="left")
        if self.photo is not None:
            x1, y1, x2, y2 = R.paste_fit(canvas, self.photo, (60, 210, W - 560, H - 60))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), 2)

        px = W - 500
        cx = px + 220
        R.text(canvas, "SCAN TO GET", (cx, 270), 1.6, (255, 255, 255), 4)
        R.text(canvas, "YOUR PHOTO", (cx, 330), 1.6, (255, 255, 255), 4)
        if self.qr is not None:
            q = self.qr.shape[0]
            canvas[380:380 + q, cx - q // 2:cx + q // 2] = self.qr
            cv2.rectangle(canvas, (cx - q // 2 - 6, 374), (cx + q // 2 + 6, 386 + q), THREAT, 6)
        R.text(canvas, self.url, (cx, 870), 0.6, (170, 170, 170), 1)
        left = max(0, int(self.cfg.booth.review_s - (now - self.state_t)))
        R.text(canvas, f"Next visitor in {left}s", (cx, 940), 1.0, (200, 200, 200), 2)
        R.text(canvas, "SPACE: next   R: retake", (cx, 990), 0.8, (130, 130, 130), 2)
