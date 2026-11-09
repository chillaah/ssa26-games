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


def paste_fill(canvas: np.ndarray, img: np.ndarray, rect: Tuple[int, int, int, int],
               radius: int = 0, keep: Optional[Sequence[int]] = None) -> Tuple[int, int, int, int]:
    """Paste img filling rect (x1, y1, x2, y2): scaled to cover it and cropped, so there are no
    bars. The crop is centred, or on keep (x1, y1, x2, y2 in img pixels, e.g. a person); if keep
    can't fit in the crop, the whole img is shown on a blurred, darkened copy of itself. With
    radius, the corners stay rounded (what's under them shows through)."""
    x1, y1, x2, y2 = rect
    dw, dh = x2 - x1, y2 - y1
    ih, iw = img.shape[:2]
    s = max(dw / iw, dh / ih)
    cw, ch = min(iw, max(1, round(dw / s))), min(ih, max(1, round(dh / s)))  # source crop size
    cx, cy = (iw - cw) // 2, (ih - ch) // 2
    backdrop = False
    if keep is not None:
        kx1, ky1, kx2, ky2 = (int(v) for v in keep)
        if kx2 - kx1 > cw or ky2 - ky1 > ch:
            backdrop = True  # whole picture on a blurred copy of itself
        else:
            cx = min(max((kx1 + kx2 - cw) // 2, 0), iw - cw)
            cy = min(max((ky1 + ky2 - ch) // 2, 0), ih - ch)
    crop = img[cy:cy + ch, cx:cx + cw]
    resized = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if backdrop:
        k = max(31, (min(dw, dh) // 8) | 1)
        resized = cv2.convertScaleAbs(cv2.GaussianBlur(resized, (k, k), 0), alpha=0.45)
        paste_fit(resized, img, (0, 0, dw, dh))
    if radius > 0:
        mask = _corner_mask(dw, dh, radius)
        np.copyto(canvas[y1:y2, x1:x2], resized, where=mask)
    else:
        canvas[y1:y2, x1:x2] = resized
    return x1, y1, x2, y2


_corner_masks: dict = {}


def _corner_mask(w: int, h: int, r: int) -> np.ndarray:
    """(h, w, 1) bool mask of a rounded rectangle, cached (it's the same every frame)."""
    key = (w, h, r)
    if key not in _corner_masks:
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
        for c in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(m, c, r, 255, -1, cv2.LINE_AA)
        _corner_masks[key] = (m > 127)[:, :, None]
    return _corner_masks[key]


def card_photo_rect(rect: Sequence[int]) -> Tuple[int, int, int, int]:
    """The photo area card() draws in rect (inside the border, above the caption strip)."""
    x1, y1, x2, y2 = rect
    pad = max(6, (x2 - x1) // 30)
    cap = max(28, (y2 - y1) // 7)
    return x1 + pad, y1 + pad, x2 - pad, y2 - cap


def card_inner_radius(rect: Sequence[int]) -> int:
    """Corner radius of the photo area card() draws in rect."""
    x1, _, x2, _ = rect
    radius = max(8, (x2 - x1) // 28)
    return max(4, radius - max(6, (x2 - x1) // 30) // 2)


def dim_outside(canvas: np.ndarray, box: Tuple[int, int, int, int], amount: float = 0.55) -> None:
    x1, y1, x2, y2 = box
    inside = canvas[y1:y2, x1:x2].copy()
    # In place with OpenCV's SIMD scaling: ~5x faster than numpy float math on the Jetson.
    cv2.convertScaleAbs(canvas, dst=canvas, alpha=1 - amount)
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
         border: Color = WHITE, fill: bool = False,
         keep: Optional[Sequence[int]] = None) -> Tuple[int, int, int, int]:
    """Rounded photo card (thescouts.io style) with a caption strip. Returns the image rect.
    fill: crop img to fill the photo area (see paste_fill) instead of letterboxing it."""
    x1, y1, x2, y2 = rect
    radius = max(8, (x2 - x1) // 28)
    rounded_rect(canvas, rect, WHITE, radius)
    pad = max(6, (x2 - x1) // 30)
    cap = max(28, (y2 - y1) // 7)
    inner = card_photo_rect(rect)
    rounded_rect(canvas, inner, CHARCOAL, card_inner_radius(rect))
    shown = inner
    if img is not None:
        if fill:
            shown = paste_fill(canvas, img, inner, card_inner_radius(rect), keep)
        else:
            shown = paste_fit(canvas, img, inner)
    scale = max(0.5, cap / 45)
    room = (x2 - x1) - 4 * pad  # long names shrink to fit inside the card
    tw = text_size(label, scale, FONT_SEMIBOLD)[0]
    if tw > room:
        scale *= room / tw
    text(canvas, label, ((x1 + x2) // 2, y2 - cap // 2), scale, INK, font=FONT_SEMIBOLD,
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
