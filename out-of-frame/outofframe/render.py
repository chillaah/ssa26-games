"""OpenCV drawing helpers. Everything is drawn onto one canvas at display resolution.
Text, colours and logos come from brand.py (The Scouts brand: Poppins, Ebony/Dutch/Powder)."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from . import brand as B
from .brand import (ALERT, BLACK, CHARCOAL, CLEAR, DUTCH, EBONY, EBONY_DEEP, GREY, INK, POWDER,
                    SILVER, WHITE, button, logo, rounded_rect, text, text_size)

FONT = B.MEDIUM
FONT_REGULAR = B.REGULAR
FONT_SEMIBOLD = B.SEMIBOLD
FONT_BOLD = B.BOLD

# Old names, mapped onto the brand palette.
RED = ALERT
GREEN = CLEAR
YELLOW = DUTCH

Color = Tuple[int, int, int]


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
    r = (y2 - y1) // 2
    rounded_rect(img, rect, CHARCOAL, r)
    fill = int((x2 - x1) * max(0.0, min(1.0, frac)))
    if fill > 0:
        rounded_rect(img, (x1, y1, x1 + max(fill, 2 * r), y2), color, r)


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
    """Rubber-stamp style label (rounded frame + bold Poppins), rotated."""
    mask, _, ox, oy, l, r, cap = B._glyphs(label, FONT_BOLD, B._px(scale))
    tw, th = r - l, cap
    pad = int(th * 0.6)
    t = max(3, int(scale * 3))
    w, h = tw + pad * 2 + t * 2, th + pad * 2 + t * 2
    alpha = np.zeros((h * 2, w * 2), np.uint8)
    ox0, oy0 = w // 2, h // 2
    frame = B._round_mask(w, h, int(h * 0.22))
    inner = B._round_mask(w - 2 * t, h - 2 * t, max(0, int(h * 0.22) - t))
    frame[t:h - t, t:w - t] = cv2.subtract(frame[t:h - t, t:w - t], inner)
    alpha[oy0:oy0 + h, ox0:ox0 + w] = frame
    # text: pen at (ox0 + t + pad - l, baseline oy0 + t + pad + th)
    px, py = ox0 + t + pad - l + ox, oy0 + t + pad + th + oy
    mh, mw = mask.shape
    region = alpha[py:py + mh, px:px + mw]
    np.maximum(region, mask[:region.shape[0], :region.shape[1]], out=region)
    M = cv2.getRotationMatrix2D((w, h), angle, 1.0)
    alpha = cv2.warpAffine(alpha, M, (w * 2, h * 2))
    B.blend_mask(img, alpha, center[0] - w, center[1] - h, color, 0.92)


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
    """Rounded photo card (thescouts.io style) with a caption strip. Returns the image rect."""
    x1, y1, x2, y2 = rect
    radius = max(8, (x2 - x1) // 28)
    rounded_rect(canvas, rect, WHITE, radius)
    pad = max(6, (x2 - x1) // 30)
    cap = max(28, (y2 - y1) // 7)
    inner = (x1 + pad, y1 + pad, x2 - pad, y2 - cap)
    rounded_rect(canvas, inner, CHARCOAL, max(4, radius - pad // 2))
    shown = inner
    if img is not None:
        shown = paste_fit(canvas, img, inner)
    text(canvas, label, ((x1 + x2) // 2, y2 - cap // 2), max(0.5, cap / 45), INK, font=FONT_SEMIBOLD,
         shadow=False)
    if border != WHITE:
        rounded_rect(canvas, rect, border, radius, thickness=5)
    return shown


def icon_label(img: np.ndarray, icon: Optional[str], label: str, center: Tuple[int, int],
               scale: float, color: Color) -> None:
    """A label with a small logo (assets/brand/logos/<icon>.png) in front, centred as a group."""
    tw, cap = text_size(label, scale, FONT_SEMIBOLD)
    size = int(cap * 1.6)
    gap = int(cap * 0.5) if icon else 0
    x = center[0] - (tw + (size + gap if icon else 0)) // 2
    if icon:
        logo(img, icon, (x, center[1]), size, anchor="left")
        x += size + gap
    text(img, label, (x, center[1] + cap // 2), scale, color, anchor="left", font=FONT_SEMIBOLD)
