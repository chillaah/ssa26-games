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
        if self.state == REVIEW:  # stays up until SPACE (next visitor) or R (retake)
            self._draw_review(canvas, now)
            return canvas
        if frame is None:
            canvas[:] = R.INK
            R.logo(canvas, "horizontal-white", (d.width // 2, d.height // 2 - 90), 110)
            R.text(canvas, "Waiting for camera...", (d.width // 2, d.height // 2 + 60), 1.3,
                   R.SILVER, font=R.FONT_REGULAR)
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
        # Keep the clock clear of the close button (top right of the screen, see window.py).
        hud_frame(view, "SCOUTS AUTONOMOUS DRONES  //  LIVE",
                  datetime.now().strftime("%H:%M:%S ") + b.timezone_label, rec_on=blink,
                  right_inset=int(150 * view.shape[1] / W))
        R.paste_fit(canvas, view, (0, 0, W, H))

        if self.state == LIVE:
            R.text(canvas, "Caught by Scouts", (W // 2, int(H * 0.80)), 2.8, R.WHITE, font=R.FONT_BOLD)
            if self.notice and now < self.notice_until:
                R.button(canvas, self.notice, (W // 2, int(H * 0.905)), 1.2, bg=THREAT)
            else:
                R.button(canvas, "Press SPACE to get caught", (W // 2, int(H * 0.905)), 1.3)
        elif self.state == COUNTDOWN:
            left = max(1, int(np.ceil(b.countdown_s - dt)))
            R.text(canvas, str(left), (W // 2, H // 2), 10.0, THREAT, font=R.FONT_BOLD)
            R.text(canvas, "Strike a pose!", (W // 2, int(H * 0.82)), 2.2, R.WHITE, font=R.FONT_BOLD)
        elif self.state == FLASH:
            a = max(0.0, 1.0 - dt / b.flash_s)
            canvas[:] = cv2.addWeighted(canvas, 1 - a, np.full_like(canvas, 255), a, 0)

    def _draw_review(self, canvas: np.ndarray, now: float) -> None:
        d = self.cfg.display
        W, H = d.width, d.height
        canvas[:] = R.INK
        R.text(canvas, "THREAT DETECTED", (60, 110), 2.4, THREAT, anchor="left", font=R.FONT_BOLD,
               shadow=False)
        R.text(canvas, "You've been caught by The Scouts drone.", (60, 170), 1.2, R.SILVER,
               anchor="left", font=R.FONT_REGULAR, shadow=False)
        if self.photo is not None:
            x1, y1, x2, y2 = R.paste_fit(canvas, self.photo, (60, 220, W - 560, H - 60))
            R.rounded_rect(canvas, (x1 - 2, y1 - 2, x2 + 2, y2 + 2), R.CHARCOAL, 6, thickness=3)

        px = W - 500
        cx = px + 220
        R.logo(canvas, "horizontal-white", (cx, 110), 84)
        R.text(canvas, "Scan to get", (cx, 255), 1.5, R.WHITE, font=R.FONT_SEMIBOLD, shadow=False)
        R.text(canvas, "your photo", (cx, 312), 1.5, R.DUTCH, font=R.FONT_SEMIBOLD, shadow=False)
        if self.qr is not None:
            q = self.qr.shape[0]
            R.rounded_rect(canvas, (cx - q // 2 - 22, 358, cx + q // 2 + 22, 402 + q), R.WHITE, 20)
            canvas[380:380 + q, cx - q // 2:cx + q // 2] = self.qr
        R.text(canvas, self.url, (cx, 880), 0.6, R.GREY, font=R.FONT_REGULAR, shadow=False)
        R.text(canvas, "SPACE: next visitor   ·   R: retake", (cx, 960), 0.85, R.SILVER,
               font=R.FONT_REGULAR, shadow=False)
