"""Draw the Instagram and LinkedIn icons shown next to the social QR codes, into both projects:
assets/brand/logos/instagram.png and linkedin.png (256 px, transparent corners).

    python tools/make_social_icons.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = [ROOT / "out-of-frame/assets/brand/logos", ROOT / "drone-booth/assets/brand/logos"]
FONT = ROOT / "out-of-frame/assets/brand/fonts/Poppins-Bold.ttf"
S = 256  # final size
K = 4  # supersampling for smooth edges
N = S * K


def rounded_mask(radius: float) -> Image.Image:
    m = Image.new("L", (N, N), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, N - 1, N - 1), radius=int(N * radius), fill=255)
    return m


def instagram() -> Image.Image:
    # Brand gradient: yellow/orange at the bottom left to purple/blue at the top right.
    stops = [(0.0, (254, 218, 117)), (0.25, (250, 126, 30)), (0.5, (214, 41, 118)),
             (0.75, (150, 47, 191)), (1.0, (79, 91, 213))]
    y, x = np.mgrid[0:N, 0:N].astype(np.float32) / (N - 1)
    t = np.clip((x + (1 - y)) / 2, 0, 1)
    rgb = np.zeros((N, N, 3), np.float32)
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        sel = (t >= t0) & (t <= t1)
        a = ((t - t0) / (t1 - t0))[sel][:, None]
        rgb[sel] = np.array(c0) * (1 - a) + np.array(c1) * a
    img = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(img)
    w = int(N * 0.075)  # stroke
    m = N * 0.2
    d.rounded_rectangle((m, m, N - m, N - m), radius=int(N * 0.17), outline="white", width=w)
    r = N * 0.15
    c = N / 2
    d.ellipse((c - r, c - r, c + r, c + r), outline="white", width=w)
    dr = N * 0.045
    dx, dy = N * 0.69, N * 0.31
    d.ellipse((dx - dr, dy - dr, dx + dr, dy + dr), fill="white")
    img.putalpha(rounded_mask(0.22))
    return img


def linkedin() -> Image.Image:
    img = Image.new("RGBA", (N, N), (10, 102, 194, 255))  # #0A66C2
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT), int(N * 0.72))
    l, t, r, b = d.textbbox((0, 0), "in", font=font)
    d.text(((N - (r - l)) / 2 - l + N * 0.02, (N - (b - t)) / 2 - t + N * 0.03), "in",
           font=font, fill="white")
    img.putalpha(rounded_mask(0.18))
    return img


for name, icon in (("instagram", instagram()), ("linkedin", linkedin())):
    small = icon.resize((S, S), Image.LANCZOS)
    for out in OUT:
        small.save(out / f"{name}.png")
        print("wrote", out / f"{name}.png")
