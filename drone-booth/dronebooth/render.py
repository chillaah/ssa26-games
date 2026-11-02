"""OpenCV drawing helpers. Everything is drawn onto one canvas at display resolution."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_TRIPLEX

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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

