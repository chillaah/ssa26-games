"""Keep the on-screen view still while the gimbal leans (FollowingFramer + steady_view).

Each frame is shifted back by however far the gimbal has turned from its resting angle, using
the gimbal's attitude readings, then zoomed slightly so the shifted edges never show.
"""
from __future__ import annotations

import bisect
import logging
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import VirtualFramerCfg

log = logging.getLogger(__name__)


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class SteadyView:
    def __init__(self, gimbal, cfg: VirtualFramerCfg, home: Tuple[float, float], mirror: bool):
        self.gimbal = gimbal
        self.cfg = cfg
        self.home = home
        self.mirror = mirror
        self.ref: Optional[Tuple[float, float]] = None
        self.history: deque = deque(maxlen=200)  # (t, yaw, pitch), oldest first
        self._last_att_t = 0.0

    def calibrate(self, timeout: float = 4.0) -> None:
        """Send the gimbal home and remember the attitude it settles at as 'no movement'."""
        self.gimbal.set_angles(*self.home)
        deadline = time.monotonic() + timeout
        prev = None
        while time.monotonic() < deadline:
            time.sleep(0.3)
            att = self.gimbal.attitude
            if att is not None and prev is not None and max(
                    abs(wrap_deg(att[0] - prev[0])), abs(wrap_deg(att[1] - prev[1]))) < 0.2:
                break
            prev = att
        att = self.gimbal.attitude
        if att is None:
            log.warning("No gimbal attitude; the view will not be steadied")
            return
        self.ref = (att[0], att[1])
        log.info("Steady view reference: yaw %.1f pitch %.1f", *self.ref)

    def _record(self) -> None:
        att, t = self.gimbal.attitude, self.gimbal.attitude_t
        if att is not None and t != self._last_att_t:
            self.history.append((t, att[0], att[1]))
            self._last_att_t = t

    def offset_at(self, t: float) -> Tuple[float, float]:
        """Gimbal (yaw, pitch) away from the reference at time t, interpolated between readings."""
        if self.ref is None or not self.history:
            return 0.0, 0.0
        times = [h[0] for h in self.history]
        i = bisect.bisect_left(times, t)
        if i <= 0:
            _, yaw, pitch = self.history[0]
        elif i >= len(self.history):
            _, yaw, pitch = self.history[-1]
        else:
            (t0, y0, p0), (t1, y1, p1) = self.history[i - 1], self.history[i]
            a = (t - t0) / max(t1 - t0, 1e-6)
            yaw = y0 + a * wrap_deg(y1 - y0)
            pitch = p0 + a * wrap_deg(p1 - p0)
        return wrap_deg(yaw - self.ref[0]), wrap_deg(pitch - self.ref[1])

    def apply(self, frame: np.ndarray, frame_t: float) -> np.ndarray:
        self._record()
        h, w = frame.shape[:2]
        k = self.cfg.steady_px_per_deg * w / 1280.0
        mx = k * (self.cfg.follow_yaw_deg + self.cfg.steady_margin_deg)
        my = k * (self.cfg.follow_pitch_deg + self.cfg.steady_margin_deg)
        scale = max(w / (w - 2 * mx), h / (h - 2 * my))
        dyaw, dpitch = self.offset_at(frame_t - self.cfg.steady_latency_s)
        # Turning the camera by +yaw / +pitch slides the (unmirrored) image right / down, or
        # left / up when follow_invert_yaw / follow_invert_pitch is set; slide it back.
        sx = k * dyaw * (1 if self.cfg.follow_invert_yaw else -1) * (-1 if self.mirror else 1)
        sy = k * dpitch * (1 if self.cfg.follow_invert_pitch else -1)
        sx, sy = max(-mx, min(mx, sx)), max(-my, min(my, sy))
        m = np.float32([[scale, 0, (1 - scale) * w / 2 + scale * sx],
                        [0, scale, (1 - scale) * h / 2 + scale * sy]])
        return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
