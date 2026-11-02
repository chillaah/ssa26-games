"""Turn a camera frame + detected people into the "THREAT DETECTED" souvenir photo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from . import render as R
from .detector import Person

THREAT = (40, 40, 235)  # BGR red
SCAN = (0, 215, 255)  # BGR amber
HUD = (235, 235, 235)


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


def surveillance_grade(img: np.ndarray) -> np.ndarray:
    """Cool, slightly desaturated security-camera look with scanlines and a vignette."""
    f = img.astype(np.float32)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    f = f * 0.75 + grey * 0.25
    f *= np.array([1.06, 1.03, 0.94], np.float32)
    f[::3] *= 0.9
    h, w = img.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    d = ((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2
    f *= (1 - 0.28 * np.clip(d, 0, 1.6) / 1.6)[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def _brackets(img, box, color, t, frac=0.22) -> None:
    x1, y1, x2, y2 = box
    L = int(min(x2 - x1, y2 - y1) * frac)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, t // 3), cv2.LINE_AA)


def _tag(img, x: int, y: int, lines: Sequence[str], color, u: float, above: bool = True) -> None:
    """Label tag anchored at the box's top-left corner (drawn above it if there is room)."""
    scales = [0.75 * u, 0.5 * u]
    thick = [max(1, int(2 * u)), max(1, int(1.4 * u))]
    sizes = [cv2.getTextSize(s, R.FONT, scales[min(i, 1)], thick[min(i, 1)])[0]
             for i, s in enumerate(lines)]
    pad, gap = int(8 * u), int(7 * u)
    w = max(s[0] for s in sizes) + pad * 2
    h = sum(s[1] for s in sizes) + gap * (len(lines) - 1) + pad * 2
    top = y - h if above and y - h >= 0 else y
    x = min(max(x, 0), img.shape[1] - w)
    cv2.rectangle(img, (x, top), (x + w, top + h), color, -1)
    cy = top + pad
    for i, (s, (sw, sh)) in enumerate(zip(lines, sizes)):
        cy += sh
        cv2.putText(img, s, (x + pad, cy), R.FONT, scales[min(i, 1)], (255, 255, 255),
                    thick[min(i, 1)], cv2.LINE_AA)
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


def hud_frame(img: np.ndarray, top_left: str, top_right: str, rec_on: bool = True) -> None:
    """Thin surveillance HUD: viewfinder corners and the top status bar."""
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
    R.text(img, top_right, (w - int(40 * u), y + int(9 * u)), 0.7 * u, HUD, max(1, int(2 * u)),
           anchor="right", shadow=False)


def compose_photo(frame: np.ndarray, people: List[Person], meta: PhotoMeta,
                  max_subjects: int = 6) -> np.ndarray:
    img = surveillance_grade(frame)
    u = _u(img)
    h, w = img.shape[:2]
    draw_subjects(img, people, "THREAT DETECTED", THREAT, max_subjects)
    hud_frame(img, "SCOUTS AI  //  DRONE CAM 01", f"{meta.timestamp}   CASE {meta.case_id}")

    # bottom ribbon
    rib = int(78 * u)
    _alpha_rect(img, (0, h - rib, w, h), (0, 0, 0), 0.6)
    cv2.rectangle(img, (0, h - rib), (int(10 * u), h), THREAT, -1)
    R.text(img, "CAUGHT BY SCOUTS", (int(34 * u), h - rib // 2 + int(14 * u)), 1.35 * u,
           (255, 255, 255), max(2, int(3 * u)), anchor="left", font=R.FONT_BOLD, shadow=False)
    R.text(img, meta.event, (w - int(34 * u), h - rib // 2 - int(6 * u)), 0.62 * u, HUD,
           max(1, int(2 * u)), anchor="right", shadow=False)
    R.text(img, meta.website, (w - int(34 * u), h - rib // 2 + int(24 * u)), 0.62 * u, SCAN,
           max(1, int(2 * u)), anchor="right", shadow=False)

    # status stamp under the top bar
    n = min(len(people), max_subjects)
    status = f"THREAT DETECTED  -  {n} SUBJECT{'S' if n != 1 else ''} IDENTIFIED"
    (tw, th), _ = cv2.getTextSize(status, R.FONT, 0.8 * u, max(2, int(2 * u)))
    x, y = (w - tw) // 2, int(70 * u)
    cv2.rectangle(img, (x - int(16 * u), y), (x + tw + int(16 * u), y + th + int(20 * u)), THREAT, -1)
    cv2.putText(img, status, (x, y + th + int(10 * u)), R.FONT, 0.8 * u, (255, 255, 255),
                max(2, int(2 * u)), cv2.LINE_AA)
    return img
