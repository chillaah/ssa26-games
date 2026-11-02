"""OpenCV drawing helpers. Everything is drawn onto one canvas at display resolution."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_TRIPLEX

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (40, 40, 235)
GREEN = (90, 210, 60)
YELLOW = (0, 215, 255)
GREY = (150, 150, 150)

Color = Tuple[int, int, int]


def text(img: np.ndarray, s: str, org: Tuple[int, int], scale: float, color: Color = WHITE,
         thickness: int = 2, anchor: str = "center", font: int = FONT, shadow: bool = True) -> None:
    (tw, th), base = cv2.getTextSize(s, font, scale, thickness)
    x, y = org
    if anchor == "center":
        x, y = x - tw // 2, y + th // 2
    elif anchor == "right":
        x = x - tw
    if shadow:
        off = max(2, int(scale * 2))
        cv2.putText(img, s, (x + off, y + off), font, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def fit(src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> Tuple[float, int, int]:
    """Letterbox transform: (scale, x offset, y offset)."""
    sw, sh = src_size
    dw, dh = dst_size
    s = min(dw / sw, dh / sh)
    return s, int((dw - sw * s) / 2), int((dh - sh * s) / 2)


def paste_fit(canvas: np.ndarray, img: np.ndarray, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Paste img letterboxed into rect (x1, y1, x2, y2). Returns the pasted rect."""
    x1, y1, x2, y2 = rect
    s, ox, oy = fit((img.shape[1], img.shape[0]), (x2 - x1, y2 - y1))
    w, h = max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    px, py = x1 + ox, y1 + oy
    canvas[py:py + h, px:px + w] = resized
    return px, py, px + w, py + h


def dim_outside(canvas: np.ndarray, box: Tuple[int, int, int, int], amount: float = 0.55) -> None:
    x1, y1, x2, y2 = box
    inside = canvas[y1:y2, x1:x2].copy()
    canvas[:] = (canvas * (1 - amount)).astype(np.uint8)
    canvas[y1:y2, x1:x2] = inside


def brackets(img: np.ndarray, box: Tuple[int, int, int, int], color: Color, thickness: int = 6,
             length_frac: float = 0.18) -> None:
    x1, y1, x2, y2 = box
    L = int(min(x2 - x1, y2 - y1) * length_frac)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, thickness, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    # crosshair
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    c = max(10, L // 4)
    cv2.line(img, (mx - c, my), (mx + c, my), color, 2, cv2.LINE_AA)
    cv2.line(img, (mx, my - c), (mx, my + c), color, 2, cv2.LINE_AA)


def progress_bar(img: np.ndarray, rect: Tuple[int, int, int, int], frac: float, color: Color) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(img, (x1, y1), (x1 + int((x2 - x1) * max(0.0, min(1.0, frac))), y2), color, -1)


def tint(img: np.ndarray, mask: np.ndarray, color: Color, alpha: float = 0.45) -> np.ndarray:
    out = img.copy()
    overlay = np.zeros_like(img)
    overlay[:] = color
    out[mask] = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 3, cv2.LINE_AA)
    return out


def stamp(img: np.ndarray, label: str, color: Color, center: Tuple[int, int], scale: float,
          angle: float = -12.0) -> None:
    """Rubber-stamp style label, rotated."""
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, scale, max(2, int(scale * 2.5)))
    pad = int(th * 0.6)
    w, h = tw + pad * 2, th + pad * 2
    layer = np.zeros((h * 2, w * 2, 3), np.uint8)
    alpha = np.zeros((h * 2, w * 2), np.uint8)
    ox, oy = w // 2, h // 2
    t = max(3, int(scale * 3))
    for target, col in ((layer, color), (alpha, 255)):
        cv2.rectangle(target, (ox, oy), (ox + w, oy + h), col, t, cv2.LINE_AA)
        cv2.putText(target, label, (ox + pad, oy + pad + th), FONT_BOLD, scale, col,
                    max(2, int(scale * 2.5)), cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((w, h), angle, 1.0)
    layer = cv2.warpAffine(layer, M, (w * 2, h * 2))
    alpha = cv2.warpAffine(alpha, M, (w * 2, h * 2)).astype(np.float32)[..., None] / 255.0 * 0.9
    cx, cy = center
    x1, y1 = cx - w, cy - h
    # clip to image
    ix1, iy1 = max(x1, 0), max(y1, 0)
    ix2, iy2 = min(x1 + w * 2, img.shape[1]), min(y1 + h * 2, img.shape[0])
    if ix1 >= ix2 or iy1 >= iy2:
        return
    sl = (slice(iy1 - y1, iy2 - y1), slice(ix1 - x1, ix2 - x1))
    roi = img[iy1:iy2, ix1:ix2].astype(np.float32)
    img[iy1:iy2, ix1:ix2] = (roi * (1 - alpha[sl]) + layer[sl].astype(np.float32) * alpha[sl]).astype(np.uint8)


def grid_rects(n: int, area: Tuple[int, int, int, int], aspect: float, gap: int = 24):
    """Lay out n cards of a given aspect (w/h) in area, centred, 1-2 rows."""
    x1, y1, x2, y2 = area
    rows = 1 if n <= 3 else 2
    cols = math.ceil(n / rows)
    aw, ah = x2 - x1, y2 - y1
    cw = min((aw - gap * (cols - 1)) / cols, ((ah - gap * (rows - 1)) / rows) * aspect)
    ch = cw / aspect
    rects = []
    total_h = rows * ch + (rows - 1) * gap
    for r in range(rows):
        in_row = cols if r < rows - 1 else n - cols * (rows - 1)
        row_w = in_row * cw + (in_row - 1) * gap
        sx = x1 + (aw - row_w) / 2
        sy = y1 + (ah - total_h) / 2 + r * (ch + gap)
        for c in range(in_row):
            rx = int(sx + c * (cw + gap))
            rects.append((rx, int(sy), int(rx + cw), int(sy + ch)))
    return rects


def card(canvas: np.ndarray, img: Optional[np.ndarray], rect: Sequence[int], label: str,
         border: Color = WHITE) -> Tuple[int, int, int, int]:
    """Polaroid-ish card with a caption strip. Returns the image rect."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (245, 245, 245), -1)
    pad = max(6, (x2 - x1) // 30)
    cap = max(28, (y2 - y1) // 7)
    inner = (x1 + pad, y1 + pad, x2 - pad, y2 - cap)
    cv2.rectangle(canvas, inner[:2], inner[2:], (30, 30, 30), -1)
    shown = inner
    if img is not None:
        shown = paste_fit(canvas, img, inner)
    text(canvas, label, ((x1 + x2) // 2, y2 - cap // 2), max(0.5, cap / 45), (40, 40, 40), 2,
         shadow=False)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 4, cv2.LINE_AA)
    return shown
