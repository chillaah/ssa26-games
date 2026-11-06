"""The souvenir collage a player downloads after the game (portrait, phone-story sized).

    Scouts logo / Out of Frame / event + date
    +--------------------------------------+
    |  BONUS POSE            the bonus     |
    |                        photo, with   |
    |                   [the picture they  |
    |                    copied, inset]    |
    +--------------------------------------+
    +-----------------+  +-----------------+
    |  dodge shot 1   |  |  dodge shot 2   |
    +-----------------+  +-----------------+
    +-----------------+  +-----------------+
    |  dodge shot 3   |  |  dodge shot 4   |
    +-----------------+  +-----------------+
    Result line / thescouts.io
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import brand as B

W, H = 1080, 1920
MARGIN = 60
BONUS_ASPECT = 960 / 760  # width / height of the bonus panel; photos are cropped to this
GREEN = B.CLEAR
RED = B.ALERT


_BACKGROUND: Optional[np.ndarray] = None


def background() -> np.ndarray:
    """Ink page with a soft Ebony glow behind the header. Built once: a full-size blur this wide
    takes seconds on the Jetson, so it's drawn at 1/8 size, blurred and scaled up."""
    global _BACKGROUND
    if _BACKGROUND is None:
        k = 8
        glow = np.zeros((H // k, W // k, 3), np.uint8)
        cv2.ellipse(glow, (W // k // 2, 0), (W // k, 520 // k), 0, 0, 360, B.EBONY_DEEP, -1)
        glow = cv2.resize(cv2.GaussianBlur(glow, (0, 0), 120 / k), (W, H),
                          interpolation=cv2.INTER_CUBIC)
        _BACKGROUND = cv2.addWeighted(np.full((H, W, 3), B.INK, np.uint8), 1.0, glow, 0.9, 0)
    return _BACKGROUND


def qr_image(url: str, size: int) -> np.ndarray:
    import qrcode

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = np.array(qr.make_image(fill_color="black", back_color="white").convert("RGB"))
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (size, size),
                      interpolation=cv2.INTER_NEAREST)


def cover(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Scale and centre-crop img to exactly fill size (w, h)."""
    w, h = size
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))), interpolation=cv2.INTER_AREA)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def contain(img: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    """Scale img to fit inside max_size (w, h), keeping its shape."""
    ih, iw = img.shape[:2]
    s = min(max_size[0] / iw, max_size[1] / ih)
    return cv2.resize(img, (max(1, round(iw * s)), max(1, round(ih * s))), interpolation=cv2.INTER_AREA)


def paste_rounded(canvas: np.ndarray, img: np.ndarray, org: Tuple[int, int], radius: int,
                  border: int = 0, border_color=B.WHITE) -> Tuple[int, int, int, int]:
    """Paste img with rounded corners at org (top-left), optionally framed. Returns its rect."""
    x, y = org
    h, w = img.shape[:2]
    if border:
        B.rounded_rect(canvas, (x - border, y - border, x + w + border, y + h + border), border_color,
                       radius + border)
    mask = np.zeros((h, w), np.uint8)
    r = min(radius, w // 2, h // 2)
    cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, -1, cv2.LINE_AA)
    cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, -1, cv2.LINE_AA)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    roi = canvas[y:y + h, x:x + w]
    roi[:] = (img * a + roi * (1 - a)).astype(np.uint8)
    return x, y, x + w, y + h


def pill(canvas: np.ndarray, label: str, org: Tuple[int, int], scale: float, bg, fg,
         anchor: str = "left") -> None:
    """Rounded label; org is its left (or right) end, vertically centred."""
    tw, th = B.text_size(label, scale, B.SEMIBOLD)
    px, py = int(th * 0.9), int(th * 0.55)
    x, y = org
    x1 = x if anchor == "left" else x - tw - 2 * px
    rect = (x1, y - th // 2 - py, x1 + tw + 2 * px, y + th // 2 + py)
    B.rounded_rect(canvas, rect, bg, (rect[3] - rect[1]) // 2)
    B.text(canvas, label, ((rect[0] + rect[2]) // 2, y), scale, fg, font=B.SEMIBOLD, shadow=False)


def shot_label(verdict) -> Tuple[str, tuple]:
    if verdict.no_player:
        return "MISSED", B.GREY
    if verdict.caught:
        return "CAUGHT", RED
    return "DODGED", GREEN


def players(boxes: Sequence[Tuple[int, int, int, int]], min_rel: float = 0.5):
    """The people posing: anyone at least min_rel as tall as the tallest person in view.
    Smaller (further away) people, e.g. someone at a desk behind, are left out of the framing."""
    if not boxes:
        return []
    tallest = max(b[3] - b[1] for b in boxes)
    return [b for b in boxes if b[3] - b[1] >= min_rel * tallest]


def bonus_crop(frame: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]],
               aspect: float = BONUS_ASPECT) -> np.ndarray:
    """Crop the bonus photo to the panel's shape, around the people in it (or the centre)."""
    fh, fw = frame.shape[:2]
    if boxes:
        x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Room around the people, but no tighter than 60% of the frame height.
        h = max((y2 - y1) * 1.3, (x2 - x1) * 1.2 / aspect, fh * 0.6)
    else:
        cx, cy, h = fw / 2, fh / 2, fh
    h = min(h, fh, fw / aspect)
    w = h * aspect
    if boxes and max(b[2] for b in boxes) - min(b[0] for b in boxes) > w:
        # The group doesn't fit: centre on the main player (the biggest person) instead.
        main = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cx = (main[0] + main[2]) / 2
    x = int(min(max(cx - w / 2, 0), fw - w))
    y = int(min(max(cy - h / 2, 0), fh - h))
    return frame[y:y + int(h), x:x + int(w)].copy()


def compose(shots: List, bonus_photo: Optional[np.ndarray], bonus_ref: Optional[np.ndarray],
            won: bool, event: str = "", logo: str = "horizontal-white",
            accent=B.DUTCH, when: Optional[datetime] = None) -> np.ndarray:
    """shots: the dodge Shot objects (photo + verdict). Returns the BGR collage (H x W)."""
    img = background().copy()

    B.logo(img, logo, (W // 2, 92), 64)
    B.text(img, "Out of Frame", (W // 2, 196), 2.9, B.WHITE, font=B.BOLD)
    when = when or datetime.now()
    sub = "  ·  ".join(x for x in (event, when.strftime("%-d %b %Y")) if x)
    B.text(img, sub, (W // 2, 262), 0.95, B.SILVER, font=B.REGULAR, shadow=False)

    y = 310
    inner = W - 2 * MARGIN
    if bonus_photo is not None:
        ph = int(inner / BONUS_ASPECT)
        x1, y1, x2, y2 = paste_rounded(img, cover(bonus_photo, (inner, ph)), (MARGIN, y), 28)
        pill(img, "BONUS POSE", (x1 + 26, y1 + 46), 0.9, accent, B.INK)
        if bonus_ref is not None:
            ref = contain(bonus_ref, (int(inner * 0.34), int(ph * 0.44)))
            rh, rw = ref.shape[:2]
            paste_rounded(img, ref, (x2 - rw - 28, y2 - rh - 28), 16, border=6)
            pill(img, "THE POSE", (x2 - 28, y2 - rh - 28 - 6 - 30), 0.7, B.WHITE, B.INK,
                 anchor="right")
        y = y2 + 40

    gap = 30
    cw = (inner - gap) // 2
    aspect = (shots[0].photo.shape[1] / max(shots[0].photo.shape[0], 1)) if shots else 16 / 9
    ch = int(cw / aspect)
    for i, shot in enumerate(shots[:4]):
        cx = MARGIN + (i % 2) * (cw + gap)
        cy = y + (i // 2) * (ch + gap)
        x1, y1, x2, y2 = paste_rounded(img, cover(shot.photo, (cw, ch)), (cx, cy), 20)
        label, color = shot_label(shot.verdict)
        pill(img, label, (x1 + 16, y2 - 34), 0.7, color,
             B.INK if color != B.GREY else B.WHITE)
    rows = (min(len(shots), 4) + 1) // 2
    y += rows * ch + max(rows - 1, 0) * gap

    dodged = sum(not s.verdict.caught for s in shots)
    if shots:
        headline = (f"Dodged all {len(shots)}!" if won else
                    f"Dodged {dodged} of {len(shots)}  ·  Caught by Scouts")
        B.text(img, headline, (W // 2, (y + H - 70) // 2), 1.45, accent, font=B.BOLD)
    B.text(img, "thescouts.io  ·  Scouts Autonomous Drones", (W // 2, H - 64), 0.85, B.SILVER,
           font=B.MEDIUM, shadow=False)
    return img
