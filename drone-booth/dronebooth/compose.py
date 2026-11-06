"""Turn a camera frame + detected people into the "THREAT DETECTED" souvenir photo."""
from __future__ import annotations

from dataclasses import dataclass
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


def compose_photo(frame: np.ndarray, people: List[Person], meta: PhotoMeta,
                  max_subjects: int = 6) -> np.ndarray:
    """The souvenir photo, 3:4 portrait (PHOTO_SIZE)."""
    frame, people = portrait_crop(frame, people)
    img = surveillance_grade(frame)
    u = _u(img)
    h, w = img.shape[:2]
    draw_subjects(img, people, "THREAT DETECTED", THREAT, max_subjects)
    hud_frame(img, "SCOUTS AUTONOMOUS DRONES  //  DRONE CAM 01", f"{meta.timestamp}   CASE {meta.case_id}")

    # bottom ribbon: logo | Caught by Scouts ............ event / website
    rib = int(84 * u)
    _alpha_rect(img, (0, h - rib, w, h), R.INK, 0.72)
    cv2.rectangle(img, (0, h - rib), (w, h - rib + max(2, int(3 * u))), R.EBONY, -1)
    mid = h - rib // 2
    _, _, lx2, _ = R.logo(img, "horizontal-white", (int(30 * u), mid), int(46 * u), anchor="left")
    cv2.line(img, (lx2 + int(22 * u), mid - int(22 * u)), (lx2 + int(22 * u), mid + int(22 * u)),
             R.GREY, max(1, int(2 * u)), cv2.LINE_AA)
    R.text(img, "Caught by Scouts", (lx2 + int(44 * u), mid + int(14 * u)), 1.2 * u, R.WHITE,
           anchor="left", font=R.FONT_BOLD, shadow=False)
    R.text(img, meta.event, (w - int(30 * u), mid - int(5 * u)), 0.6 * u, HUD, anchor="right",
           font=R.FONT, shadow=False)
    R.text(img, meta.website, (w - int(30 * u), mid + int(25 * u)), 0.6 * u, R.DUTCH, anchor="right",
           font=R.FONT_SEMIBOLD, shadow=False)

    # status pill just above the ribbon, clear of the subject tags at the top
    n = min(len(people), max_subjects)
    status = f"THREAT DETECTED  \u00b7  {n} SUBJECT{'S' if n != 1 else ''} IDENTIFIED"
    R.button(img, status, (w // 2, h - rib - int(36 * u)), 0.72 * u, bg=THREAT, fg=R.WHITE)
    return img
