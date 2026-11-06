"""OpenCV drawing helpers. Everything is drawn onto one canvas at display resolution.
Text, colours and logos come from brand.py (The Scouts brand: Poppins, Ebony/Dutch/Powder)."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from . import brand as B
from .brand import (ALERT, BLACK, CHARCOAL, DUTCH, EBONY, EBONY_DEEP, GREY, INK, POWDER, SILVER,
                    WHITE, button, logo, rounded_rect, text, text_size)

FONT = B.MEDIUM
FONT_REGULAR = B.REGULAR
FONT_SEMIBOLD = B.SEMIBOLD
FONT_BOLD = B.BOLD

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
