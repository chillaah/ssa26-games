"""Where is the camera "looking"? Two implementations behind one interface.

VirtualFramer:   a photo box glides around a fixed webcam image (Mac testing).
FollowingFramer: the same gliding box, while the SIYI gimbal leans a little toward it (show only).
SiyiFramer:      the SIYI A8 mini gimbal physically swings to random angles (drone).

Both only handle movement. The game decides when to lock on and snap.
"""
from __future__ import annotations

import logging
import math
import random
import time
from typing import List, Optional, Tuple

from .config import FramerCfg, SiyiFramerCfg, VirtualFramerCfg

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
Size = Tuple[int, int]  # width, height


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def difficulty(round_idx: int, n_rounds: int) -> float:
    return round_idx / max(n_rounds - 1, 1)


class Framer:
    def start_round(self, now: float, round_idx: int, n_rounds: int, size: Size) -> None:
        raise NotImplementedError

    def update(self, now: float, size: Size) -> Tuple[Box, bool]:
        """Return (photo box, finished_moving)."""
        raise NotImplementedError

    def idle(self, now: float, size: Size) -> Box:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VirtualFramer(Framer):
    def __init__(self, cfg: VirtualFramerCfg, rng: Optional[random.Random] = None):
        self.cfg = cfg
        self.rng = rng or random.Random()
        self.box_frac = cfg.box_width_frac
        self.center = (0.5, 0.5)  # normalised box centre
        self.path: List[Tuple[float, float]] = []
        self.seg_start = 0.0
        self.travel_s = cfg.travel_s

    def _limits(self) -> Tuple[float, float, float, float]:
        half = self.box_frac / 2 + self.cfg.margin_frac
        return half, 1 - half, half, 1 - half

    def _sample(self, prev: Tuple[float, float]) -> Tuple[float, float]:
        x0, x1, y0, y1 = self._limits()
        best, best_d = prev, -1.0
        for _ in range(30):
            p = (self.rng.uniform(x0, x1), self.rng.uniform(y0, y1))
            d = math.dist(p, prev)
            if d >= self.cfg.min_jump_frac:
                return p
            if d > best_d:
                best, best_d = p, d
        return best

    def start_round(self, now: float, round_idx: int, n_rounds: int, size: Size) -> None:
        d = difficulty(round_idx, n_rounds)
        self.box_frac = lerp(self.cfg.box_width_frac, self.cfg.box_width_frac_final, d)
        self.travel_s = lerp(self.cfg.travel_s, self.cfg.travel_s_final, d)
        x0, x1, y0, y1 = self._limits()
        start = (min(max(self.center[0], x0), x1), min(max(self.center[1], y0), y1))
        self.path = [start]
        for _ in range(self.rng.randint(self.cfg.feints_min, self.cfg.feints_max)):
            self.path.append(self._sample(self.path[-1]))
        self.seg_start = now

    def _box(self, size: Size) -> Box:
        w, h = size
        bw, bh = self.box_frac * w, self.box_frac * h
        cx, cy = self.center[0] * w, self.center[1] * h
        return int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)

    def update(self, now: float, size: Size) -> Tuple[Box, bool]:
        segments = len(self.path) - 1
        if segments <= 0:
            return self._box(size), True
        elapsed = (now - self.seg_start) / self.travel_s
        idx = min(int(elapsed), segments - 1)
        t = ease_in_out(elapsed - idx) if elapsed < segments else 1.0
        a, b = self.path[idx], self.path[idx + 1]
        self.center = (lerp(a[0], b[0], t), lerp(a[1], b[1], t))
        return self._box(size), elapsed >= segments

    def idle(self, now: float, size: Size) -> Box:
        self.box_frac = self.cfg.box_width_frac
        x0, x1, y0, y1 = self._limits()
        self.center = (
            lerp(x0, x1, 0.5 + 0.5 * math.sin(now * 0.45)),
            lerp(y0, y1, 0.5 + 0.5 * math.sin(now * 0.31 + 1.0)),
        )
        return self._box(size)


def open_gimbal(cfg: SiyiFramerCfg):
    from .siyi import SerialLink, SiyiGimbal, UdpLink

    if cfg.link == "serial":
        link = SerialLink(cfg.serial_port, cfg.baud)
    elif cfg.link == "udp":
        link = UdpLink(cfg.ip, cfg.port)
    else:
        raise ValueError(f"Unknown SIYI link: {cfg.link}")
    return SiyiGimbal(link).start()


class FollowingFramer(VirtualFramer):
    """VirtualFramer whose box also nudges the real gimbal. The game logic doesn't change:
    the box still decides who is caught; the gimbal lean is just for show."""

    def __init__(self, cfg: VirtualFramerCfg, siyi: SiyiFramerCfg, mirror: bool,
                 rng: Optional[random.Random] = None, gimbal=None):
        super().__init__(cfg, rng)
        self.siyi = siyi
        # Mirrored display: the box on the right of the screen is on the left of the camera image.
        self.yaw_sign = (1 if mirror else -1) * (-1 if cfg.follow_invert_yaw else 1)
        self.pitch_sign = -1 if cfg.follow_invert_pitch else 1
        self.gimbal = gimbal or open_gimbal(siyi)
        self.sent: Optional[Tuple[float, float]] = None
        self.sent_t = -math.inf

    def lean(self) -> Tuple[float, float]:
        """Gimbal angles for the current box centre: home, plus up to follow_*_deg toward the box."""
        x0, x1, y0, y1 = self._limits()
        u = max(-1.0, min(1.0, (self.center[0] - 0.5) / max(0.5 - x0, 1e-6)))  # -1 left .. +1 right
        v = max(-1.0, min(1.0, (self.center[1] - 0.5) / max(0.5 - y0, 1e-6)))  # -1 top .. +1 bottom
        return (self.siyi.home_yaw + self.yaw_sign * u * self.cfg.follow_yaw_deg,
                self.siyi.home_pitch - self.pitch_sign * v * self.cfg.follow_pitch_deg)

    def _follow(self, now: float) -> None:
        if now - self.sent_t < 1.0 / self.cfg.follow_hz:
            return
        target = self.lean()
        if self.sent is not None and math.dist(target, self.sent) < 0.2:
            return
        try:
            self.gimbal.set_angles(*target)
        except OSError as exc:
            log.warning("SIYI send failed: %s", exc)
        self.sent, self.sent_t = target, now

    def update(self, now: float, size: Size) -> Tuple[Box, bool]:
        result = super().update(now, size)
        self._follow(now)
        return result

    def idle(self, now: float, size: Size) -> Box:
        box = super().idle(now, size)
        self._follow(now)
        return box

    def close(self) -> None:
        try:
            self.gimbal.center()
        finally:
            self.gimbal.stop()


class SiyiFramer(Framer):
    def __init__(self, cfg: SiyiFramerCfg, rng: Optional[random.Random] = None):
        self.cfg = cfg
        self.rng = rng or random.Random()
        self.gimbal = open_gimbal(cfg)
        self.targets: List[Tuple[float, float]] = []
        self.target_idx = 0
        self.target_sent_t = 0.0
        self.arrived_t: Optional[float] = None
        self.last = (cfg.home_yaw, cfg.home_pitch)
        self._idle_sent = False

    def _box(self, size: Size) -> Box:
        w, h = size
        f = self.cfg.photo_box_frac
        mx, my = w * (1 - f) / 2, h * (1 - f) / 2
        return int(mx), int(my), int(w - mx), int(h - my)

    def _sample(self, prev: Tuple[float, float]) -> Tuple[float, float]:
        c = self.cfg
        for _ in range(30):
            p = (self.rng.uniform(c.yaw_min, c.yaw_max), self.rng.uniform(c.pitch_min, c.pitch_max))
            if math.dist(p, prev) >= c.min_jump_deg:
                return p
        return p

    def _go(self, now: float, target: Tuple[float, float]) -> None:
        self.gimbal.set_angles(*target)
        self.target_sent_t = now
        self.arrived_t = None

    def start_round(self, now: float, round_idx: int, n_rounds: int, size: Size) -> None:
        self._idle_sent = False
        n = self.rng.randint(self.cfg.feints_min, self.cfg.feints_max)
        self.targets = []
        prev = self.last
        for _ in range(n):
            prev = self._sample(prev)
            self.targets.append(prev)
        self.target_idx = 0
        self._go(now, self.targets[0])

    def _arrived(self, now: float, target: Tuple[float, float]) -> bool:
        att = self.gimbal.attitude
        fresh = att is not None and time.monotonic() - self.gimbal.attitude_t < 0.5
        if fresh:
            if math.dist((att[0], att[1]), target) <= self.cfg.settle_tolerance_deg:
                return True
        else:
            # No attitude feedback: assume ~90 deg/s slew.
            if now - self.target_sent_t >= math.dist(self.last, target) / 90.0 + 0.4:
                return True
        return now - self.target_sent_t >= self.cfg.move_timeout_s

    def update(self, now: float, size: Size) -> Tuple[Box, bool]:
        if self.target_idx >= len(self.targets):
            return self._box(size), True
        target = self.targets[self.target_idx]
        if self.arrived_t is None and self._arrived(now, target):
            self.arrived_t = now
            self.last = target
        if self.arrived_t is not None and now - self.arrived_t >= self.cfg.hold_s:
            self.target_idx += 1
            if self.target_idx >= len(self.targets):
                return self._box(size), True
            self._go(now, self.targets[self.target_idx])
        return self._box(size), False

    def idle(self, now: float, size: Size) -> Box:
        if not self._idle_sent:
            self.gimbal.set_angles(self.cfg.home_yaw, self.cfg.home_pitch)
            self.last = (self.cfg.home_yaw, self.cfg.home_pitch)
            self._idle_sent = True
        return self._box(size)

    def close(self) -> None:
        try:
            self.gimbal.center()
        finally:
            self.gimbal.stop()


def make_framer(cfg: FramerCfg, mirror: bool = False) -> Framer:
    if cfg.mode == "virtual":
        if cfg.virtual.follow_gimbal:
            return FollowingFramer(cfg.virtual, cfg.siyi, mirror)
        return VirtualFramer(cfg.virtual)
    if cfg.mode == "siyi":
        return SiyiFramer(cfg.siyi)
    raise ValueError(f"Unknown framer mode: {cfg.mode}")
