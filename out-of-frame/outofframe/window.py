"""The game window: fullscreen toggle, an on-screen close button, and clean shutdown.

The close button sits in the top-right corner. Tap it once to arm it, then again within a few
seconds to close, so a stray tap from a visitor doesn't end the game. Closing the window from the
desktop (when not fullscreen) also quits. The same file is in both projects; keep them in sync.
"""
from __future__ import annotations

import signal
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from . import brand as B

CONFIRM_S = 3.0  # second tap must come within this long


def quit_on_signals() -> None:
    """Turn Ctrl+C, SIGTERM and SIGHUP (pkill, closing the terminal) into one KeyboardInterrupt so
    the main loop's cleanup still runs: camera stopped, gimbal centred, window closed. Further
    signals are ignored so a second Ctrl+C can't cut the cleanup short."""
    sigs = [s for s in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)) if s]

    def handler(signum, frame):
        for s in sigs:
            signal.signal(s, signal.SIG_IGN)
        raise KeyboardInterrupt

    for s in sigs:
        signal.signal(s, handler)


class GameWindow:
    def __init__(self, name: str, fullscreen: bool, size: Tuple[int, int]):
        self.name = name
        self.fullscreen = fullscreen
        self.close_requested = False
        self._armed_until = 0.0
        self._btn: Optional[Tuple[int, int, int]] = None  # centre x, centre y, radius (image px)
        self._frames = 0
        self._can_detect_close = False
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        self._apply_fullscreen()
        if not fullscreen:
            cv2.resizeWindow(name, size[0] // 2, size[1] // 2)
        cv2.setMouseCallback(name, self._on_mouse)

    def _apply_fullscreen(self) -> None:
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

    @property
    def armed(self) -> bool:
        return time.monotonic() < self._armed_until

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self._btn is None:
            return
        cx, cy, r = self._btn
        if (x - cx) ** 2 + (y - cy) ** 2 > (1.6 * r) ** 2:  # generous hit area for fingers
            return
        if self.armed:
            self.close_requested = True
        else:
            self._armed_until = time.monotonic() + CONFIRM_S

    def _draw_close(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        r = max(16, int(w * 0.017))
        cx, cy = w - int(1.9 * r), int(1.9 * r)
        self._btn = (cx, cy, r)
        armed = self.armed
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), r, B.ALERT if armed else B.INK, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.9 if armed else 0.55, img, 0.1 if armed else 0.45, 0, dst=img)
        cv2.circle(img, (cx, cy), r, B.WHITE, max(1, r // 10), cv2.LINE_AA)
        d = int(r * 0.42)
        t = max(2, r // 7)
        cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), B.WHITE, t, cv2.LINE_AA)
        cv2.line(img, (cx - d, cy + d), (cx + d, cy - d), B.WHITE, t, cv2.LINE_AA)
        if armed:
            B.text(img, "Tap again to close", (cx - int(1.5 * r), cy + int(0.3 * r)), r / 40,
                   B.WHITE, anchor="right", font=B.SEMIBOLD)

    def show(self, img: np.ndarray) -> None:
        self._draw_close(img)
        cv2.imshow(self.name, img)

    def poll(self) -> int:
        """waitKey; handles f (fullscreen) itself and returns other keys (255 = none)."""
        key = cv2.waitKey(1) & 0xFF
        self._frames += 1
        visible = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE)
        if self._frames == 1:
            # Some OpenCV builds can't report this; only trust it if the window reads as open now.
            self._can_detect_close = visible >= 1
        elif self._can_detect_close and visible < 1:
            self.close_requested = True  # closed from the desktop (title bar / taskbar)
        if key == ord("f"):
            self.fullscreen = not self.fullscreen
            self._apply_fullscreen()
            return 255
        return key

    def destroy(self) -> None:
        cv2.destroyAllWindows()
        for _ in range(5):  # let the GUI backend actually tear the window down
            cv2.waitKey(1)
