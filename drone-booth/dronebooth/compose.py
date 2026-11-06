"""Turn a camera frame + detected people into the "THREAT DETECTED" souvenir photo."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import render as R
from .detector import Person

THREAT = R.ALERT  # caught / locking on
SCAN = R.DUTCH  # scanning
HUD = (240, 240, 240)


def _on(color) -> tuple:
    """Text colour that reads on a fill of `color` (ink on light fills, white on dark ones)."""
    b, g, r = color
    return R.INK if 0.114 * b + 0.587 * g + 0.299 * r > 160 else R.WHITE


@dataclass
class PhotoMeta:
    case_id: str
    timestamp: str
    event: str
    website: str


def _u(img: np.ndarray) -> float:
    """Scale unit so overlays look the same at any resolution (1.0 at 1280 wide)."""
    return img.shape[1] / 1280.0


def _alpha_rect(img: np.ndarray, rect: Tuple[int, int, int, int], color, alpha: float) -> None:
    x1, y1, x2, y2 = (max(0, v) for v in rect)
    roi = img[y1:y2, x1:x2]
    if roi.size:
        roi[:] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


_GRADE_GAIN: dict = {}


def _grade_gain(h: int, w: int) -> np.ndarray:
    """Per-pixel gain (cool tint x scanlines x vignette) as uint8, 200 = 1.0. Built once per size."""
    if (h, w) not in _GRADE_GAIN:
        g = np.ones((h, w, 3), np.float32) * np.array([1.06, 1.03, 0.94], np.float32)
        g[::3] *= 0.9
        ys, xs = np.ogrid[:h, :w]
        d = ((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2
        g *= (1 - 0.28 * np.clip(d, 0, 1.6) / 1.6)[..., None]
        _GRADE_GAIN[(h, w)] = np.clip(np.round(g * 200), 0, 255).astype(np.uint8)
    return _GRADE_GAIN[(h, w)]


def surveillance_grade(img: np.ndarray) -> np.ndarray:
    """Cool, slightly desaturated security-camera look with scanlines and a vignette.
    Integer OpenCV ops with a cached gain map: ~10x faster than float math on the Jetson."""
    grey = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    mixed = cv2.addWeighted(img, 0.75, grey, 0.25, 0)
    return cv2.multiply(mixed, _grade_gain(*img.shape[:2]), scale=1 / 200)


def _brackets(img, box, color, t, frac=0.22) -> None:
    x1, y1, x2, y2 = box
    L = int(min(x2 - x1, y2 - y1) * frac)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, t // 3), cv2.LINE_AA)


def _tag(img, x: int, y: int, lines: Sequence[str], color, u: float, above: bool = True) -> None:
    """Label tag anchored at the box's top-left corner (drawn above it if there is room)."""
    scales = [0.72 * u, 0.5 * u]
    fonts = [R.FONT_SEMIBOLD, R.FONT_REGULAR]
    sizes = [R.text_size(s, scales[min(i, 1)], fonts[min(i, 1)]) for i, s in enumerate(lines)]
    pad, gap = int(10 * u), int(8 * u)
    w = max(s[0] for s in sizes) + pad * 2
    h = sum(s[1] for s in sizes) + gap * (len(lines) - 1) + pad * 2
    bar = int(56 * u)  # keep clear of the HUD status bar along the top (hud_frame)
    top = y - h - int(4 * u) if above and y - h - int(4 * u) >= bar else max(y, bar)
    x = min(max(x, 0), img.shape[1] - w)
    R.rounded_rect(img, (x, top, x + w, top + h), color, int(6 * u))
    fg = _on(color)
    cy = top + pad
    for i, (s, (sw, sh)) in enumerate(zip(lines, sizes)):
        cy += sh
        R.text(img, s, (x + pad, cy), scales[min(i, 1)], fg, anchor="left", font=fonts[min(i, 1)],
               shadow=False)
        cy += gap


def draw_subjects(img: np.ndarray, people: List[Person], title: str, color,
                  max_subjects: int = 6, outline: bool = True) -> None:
    u = _u(img)
    ranked = sorted(people, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]),
                    reverse=True)[:max_subjects]
    # number subjects left to right so the labels read naturally
    ranked.sort(key=lambda p: p.bbox[0])
    for i, p in enumerate(ranked, 1):
        if outline:
            contours, _ = cv2.findContours(p.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, max(1, int(2 * u)), cv2.LINE_AA)
        _brackets(img, p.bbox, color, max(2, int(4 * u)))
        _tag(img, p.bbox[0], p.bbox[1], [title, f"SUBJECT {i:02d}  |  PERSON {p.conf * 100:.0f}%"],
             color, u)


def hud_frame(img: np.ndarray, top_left: str, top_right: str, rec_on: bool = True,
              right_inset: Optional[int] = None) -> None:
    """Thin surveillance HUD: viewfinder corners and the top status bar. right_inset: room
    kept free right of top_right (default 40 px at 1920 wide)."""
    u = _u(img)
    h, w = img.shape[:2]
    m, L, t = int(22 * u), int(60 * u), max(2, int(3 * u))
    for (cx, cy, dx, dy) in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), HUD, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), HUD, t, cv2.LINE_AA)
    bar = int(52 * u)
    _alpha_rect(img, (0, 0, w, bar), (0, 0, 0), 0.45)
    y = bar // 2
    if rec_on:
        cv2.circle(img, (int(48 * u), y), int(8 * u), THREAT, -1, cv2.LINE_AA)
    R.text(img, top_left, (int(66 * u), y + int(9 * u)), 0.7 * u, HUD, max(1, int(2 * u)),
           anchor="left", shadow=False)
    inset = int(40 * u) if right_inset is None else right_inset
    R.text(img, top_right, (w - inset, y + int(9 * u)), 0.7 * u, HUD, max(1, int(2 * u)),
           anchor="right", shadow=False)


PHOTO_SIZE = (1080, 1440)  # Instagram's 3:4 portrait


def portrait_crop(frame: np.ndarray, people: List[Person],
                  size: Tuple[int, int] = PHOTO_SIZE) -> Tuple[np.ndarray, List[Person]]:
    """Crop the (landscape) frame to size's shape, full height, centred on the people, and scale
    it to size. The people's boxes and masks are moved into the new picture."""
    fh, fw = frame.shape[:2]
    out_w, out_h = size
    cw = min(fw, round(fh * out_w / out_h))
    ch = min(fh, round(cw * out_h / out_w))
    if people:
        cx = (min(p.bbox[0] for p in people) + max(p.bbox[2] for p in people)) / 2
    else:
        cx = fw / 2
    x0 = int(min(max(cx - cw / 2, 0), fw - cw))
    y0 = (fh - ch) // 2
    s = out_w / cw
    img = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], size, interpolation=cv2.INTER_CUBIC)
    moved = []
    for p in people:
        x1, y1, x2, y2 = p.bbox
        bx1, bx2 = max(x1 - x0, 0), min(x2 - x0, cw)
        if bx2 - bx1 < 0.5 * (x2 - x1):  # mostly outside the crop: not a subject of this photo
            continue
        box = (int(bx1 * s), int(max(y1 - y0, 0) * s), int(bx2 * s), int(min(y2 - y0, ch) * s))
        mask = cv2.resize(p.mask[y0:y0 + ch, x0:x0 + cw].astype(np.uint8), size,
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        moved.append(Person(box, p.conf, mask))
    return img, moved


# Souvenir layout (same frame as Out of Frame's collage): header, the photo panel, footer.
PANEL = (60, 222, 1020, 1290)  # x1, y1, x2, y2 of the camera shot on the PHOTO_SIZE canvas
_BACKGROUND: Optional[np.ndarray] = None


def background() -> np.ndarray:
    """Ink page with a soft Ebony glow behind the header (drawn small and scaled up: fast)."""
    global _BACKGROUND
    if _BACKGROUND is None:
        W, H = PHOTO_SIZE
        k = 8
        glow = np.zeros((H // k, W // k, 3), np.uint8)
        cv2.ellipse(glow, (W // k // 2, 0), (W // k, 420 // k), 0, 0, 360, R.EBONY_DEEP, -1)
        glow = cv2.resize(cv2.GaussianBlur(glow, (0, 0), 120 / k), (W, H),
                          interpolation=cv2.INTER_CUBIC)
        _BACKGROUND = cv2.addWeighted(np.full((H, W, 3), R.INK, np.uint8), 1.0, glow, 0.9, 0)
    return _BACKGROUND


def paste_rounded(canvas: np.ndarray, img: np.ndarray, org: Tuple[int, int], radius: int) -> None:
    x, y = org
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    r = min(radius, w // 2, h // 2)
    cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    roi = canvas[y:y + h, x:x + w]
    roi[:] = (img * a + roi * (1 - a)).astype(np.uint8)


def compose_photo(frame: np.ndarray, people: List[Person], meta: PhotoMeta,
                  max_subjects: int = 6, when: Optional[datetime] = None) -> np.ndarray:
    """The souvenir photo: PHOTO_SIZE (1080 x 1440, Instagram's 3:4 portrait)."""
    x1, y1, x2, y2 = PANEL
    shot, people = portrait_crop(frame, people, (x2 - x1, y2 - y1))
    shot = surveillance_grade(shot)
    u = _u(shot)
    draw_subjects(shot, people, "THREAT DETECTED", THREAT, max_subjects)
    hud_frame(shot, "SCOUTS AUTONOMOUS DRONES  //  DRONE CAM 01",
              f"{meta.timestamp}   CASE {meta.case_id}")
    n = min(len(people), max_subjects)
    status = f"THREAT DETECTED  \u00b7  {n} SUBJECT{'S' if n != 1 else ''} IDENTIFIED"
    R.button(shot, status, ((x2 - x1) // 2, (y2 - y1) - int(46 * u)), 0.72 * u, bg=THREAT,
             fg=R.WHITE)

    W, H = PHOTO_SIZE
    img = background().copy()
    R.logo(img, "horizontal-white", (W // 2, 60), 46)
    R.text(img, "Drone Booth", (W // 2, 136), 2.3, R.WHITE, font=R.FONT_BOLD)
    when = when or datetime.now()
    sub = "  \u00b7  ".join(x for x in (meta.event.title(), when.strftime("%-d %b %Y")) if x)
    R.text(img, sub, (W // 2, 188), 0.8, R.SILVER, font=R.FONT_REGULAR, shadow=False)
    paste_rounded(img, shot, (x1, y1), 24)
    R.text(img, "Scouts Drone Booth", (W // 2, (y2 + H - 46) // 2), 1.2, R.DUTCH, font=R.FONT_BOLD)
    R.text(img, f"{meta.website}  \u00b7  Scouts Autonomous Drones", (W // 2, H - 34), 0.72,
           R.SILVER, font=R.FONT, shadow=False)
    return img
