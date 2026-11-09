"""The Scouts brand for OpenCV screens: palette, Poppins text, logos, rounded shapes.

OpenCV's built-in fonts can't draw Poppins, so text is rendered with Pillow into a cached alpha
mask and blended onto the frame. Assets live in assets/brand/ (fonts are SIL OFL, see OFL.txt).
The same file is used by out-of-frame and drone-booth; keep the two copies in sync.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "brand"

Color = Tuple[int, int, int]

# Palette (BGR). Brand colours from the logo files, neutrals from thescouts.io.
INK = (14, 14, 14)  # #0e0e0e
CHARCOAL = (31, 31, 31)  # #1f1f1f
EBONY = (65, 70, 109)  # #6d4641
EBONY_DEEP = (36, 38, 56)  # #382624, dark panels
DUTCH = (202, 239, 255)  # #ffefca
POWDER = (206, 173, 156)  # #9cadce
GREY = (117, 117, 117)  # #757575
SILVER = (224, 224, 224)  # #e0e0e0
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# Status colours, kept functional (caught / clear) but toned to sit with the palette.
ALERT = (77, 72, 229)  # #e5484d
CLEAR = (138, 195, 76)  # #4cc38a

# Poppins weights
REGULAR, MEDIUM, SEMIBOLD, BOLD = "Regular", "Medium", "SemiBold", "Bold"
PX_PER_SCALE = 30  # text "scale" 1.0 is about as wide as OpenCV's Hershey font at 1.0


@lru_cache(maxsize=64)
def _font(weight: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(ASSETS / "fonts" / f"Poppins-{weight}.ttf"), px)


def _px(scale: float) -> int:
    return max(6, int(round(scale * PX_PER_SCALE)))


@lru_cache(maxsize=2048)
def _glyphs(s: str, weight: str, px: int):
    """(mask, shadow, ox, oy, ink_left, ink_right, cap) for s; ox/oy place the mask's top-left
    relative to the pen position on the baseline."""
    f = _font(weight, px)
    cap = -f.getbbox("H", anchor="ls")[1]
    l, t, r, b = f.getbbox(s, anchor="ls") if s else (0, 0, 0, 0)
    pad = int(px * 0.25) + 2
    im = Image.new("L", (max(1, r - l) + 2 * pad, max(1, b - t) + 2 * pad), 0)
    if s:
        ImageDraw.Draw(im).text((pad - l, pad - t), s, font=f, fill=255, anchor="ls")
    mask = np.asarray(im)
    shadow = cv2.GaussianBlur(mask, (0, 0), max(1.0, px * 0.08))
    return mask, shadow, l - pad, t - pad, l, r, cap


def text_size(s: str, scale: float, font: str = MEDIUM) -> Tuple[int, int]:
    """(ink width, cap height) in pixels, like cv2.getTextSize."""
    _, _, _, _, l, r, cap = _glyphs(s, font, _px(scale))
    return r - l, cap


def blend_mask(img: np.ndarray, mask: np.ndarray, x: int, y: int, color: Color,
               opacity: float = 1.0) -> None:
    """Paint color through an 8-bit mask whose top-left goes at (x, y). Clipped to img."""
    h, w = mask.shape[:2]
    ix1, iy1 = max(x, 0), max(y, 0)
    ix2, iy2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if ix1 >= ix2 or iy1 >= iy2:
        return
    a = mask[iy1 - y:iy2 - y, ix1 - x:ix2 - x].astype(np.float32)[..., None] * (opacity / 255.0)
    roi = img[iy1:iy2, ix1:ix2].astype(np.float32)
    img[iy1:iy2, ix1:ix2] = (roi + (np.array(color, np.float32) - roi) * a).astype(np.uint8)


def text(img: np.ndarray, s: str, org: Tuple[int, int], scale: float, color: Color = WHITE,
         thickness: Optional[int] = None, anchor: str = "center", font: str = MEDIUM,
         shadow: bool = True, opacity: float = 1.0) -> None:
    """Draw s in Poppins. anchor "center" centres on org; "left"/"right" put the baseline at org.
    thickness is ignored (weight comes from font); kept so old cv2-style calls still work."""
    mask, soft, ox, oy, l, r, cap = _glyphs(s, font, _px(scale))
    x, y = org
    if anchor == "center":
        x, y = x - (l + r) // 2, y + cap // 2
    elif anchor == "right":
        x = x - r
    if shadow:
        off = max(1, _px(scale) // 16)
        blend_mask(img, soft, x + ox, y + oy + off, BLACK, 0.75 * opacity)
    blend_mask(img, mask, x + ox, y + oy, color, opacity)


# ---- shapes ---------------------------------------------------------------------
def _round_mask(w: int, h: int, radius: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    r = max(0, min(radius, w // 2, h // 2))
    if r == 0:
        m[:] = 255
        return m
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def rounded_rect(img: np.ndarray, rect: Tuple[int, int, int, int], color: Color, radius: int = 16,
                 thickness: int = -1, opacity: float = 1.0) -> None:
    """Filled (thickness < 0) or outlined rounded rectangle, anti-aliased, optionally translucent."""
    x1, y1, x2, y2 = (int(v) for v in rect)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    m = _round_mask(w, h, radius)
    if thickness > 0:
        t = thickness
        inner = np.zeros_like(m)
        if w > 2 * t and h > 2 * t:
            inner[t:h - t, t:w - t] = _round_mask(w - 2 * t, h - 2 * t, max(0, radius - t))
        m = cv2.subtract(m, inner)
    blend_mask(img, m, x1, y1, color, opacity)


def button(img: np.ndarray, label: str, center: Tuple[int, int], scale: float,
           bg: Color = EBONY, fg: Color = WHITE, font: str = SEMIBOLD, opacity: float = 1.0) -> Tuple[int, int, int, int]:
    """thescouts.io-style button: solid fill, 12px-ish corners, white label. Returns its rect."""
    w, cap = text_size(label, scale, font)
    px = _px(scale)
    padx, pady = int(px * 1.1), int(px * 0.7)
    cx, cy = center
    rect = (cx - w // 2 - padx, cy - cap // 2 - pady, cx + w // 2 + padx, cy + cap // 2 + pady)
    rounded_rect(img, rect, bg, int(px * 0.6), opacity=opacity)
    text(img, label, center, scale, fg, font=font, shadow=False, opacity=opacity)
    return rect


# ---- logos ----------------------------------------------------------------------
@lru_cache(maxsize=32)
def _logo_src(name: str) -> Optional[np.ndarray]:
    return cv2.imread(str(ASSETS / "logos" / f"{name}.png"), cv2.IMREAD_UNCHANGED)


@lru_cache(maxsize=64)
def _logo(name: str, height: int) -> Optional[np.ndarray]:
    src = _logo_src(name)
    if src is None:
        return None
    w = max(1, round(src.shape[1] * height / src.shape[0]))
    return cv2.resize(src, (w, height), interpolation=cv2.INTER_AREA)


def logo(img: np.ndarray, name: str, org: Tuple[int, int], height: int, anchor: str = "center",
         opacity: float = 1.0) -> Tuple[int, int, int, int]:
    """Paste a logo from assets/brand/logos/<name>.png (e.g. "horizontal-white", "logomark-dutch").
    anchor: "center", "left"/"right" (org is the vertical middle of that edge). Returns its rect."""
    lg = _logo(name, max(1, int(height)))
    if lg is None:
        return (org[0], org[1], org[0], org[1])
    h, w = lg.shape[:2]
    x, y = org[0] - w // 2, org[1] - h // 2
    if anchor == "left":
        x = org[0]
    elif anchor == "right":
        x = org[0] - w
    ix1, iy1 = max(x, 0), max(y, 0)
    ix2, iy2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if ix1 < ix2 and iy1 < iy2:
        part = lg[iy1 - y:iy2 - y, ix1 - x:ix2 - x].astype(np.float32)
        a = part[..., 3:4] * (opacity / 255.0)
        roi = img[iy1:iy2, ix1:ix2].astype(np.float32)
        img[iy1:iy2, ix1:ix2] = (roi + (part[..., :3] - roi) * a).astype(np.uint8)
    return x, y, x + w, y + h
