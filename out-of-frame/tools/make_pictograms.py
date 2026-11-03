"""Generate the starter pose gallery: pictogram figures with exact (manual) keypoints.

    python tools/make_pictograms.py gallery

Each figure is built from joint angles: an arm preset per side and a leg preset per side, plus
an optional torso lean. Angles are in degrees from straight down; positive swings away from the
body's centre line, 180 is straight up. Mirror images aren't needed (matching tries both).
"""
from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np

W, H = 600, 800
BODY = (90, 50, 20)  # BGR navy
BG = (235, 240, 245)
UPPER_ARM, FOREARM, THIGH, SHIN = 95, 90, 140, 140
TORSO, SHOULDER_W, HIP_W, NECK = 185, 100, 60, 80

ARMS = {  # (upper arm, forearm)
    "down": (8, 4), "low": (40, 40), "out": (90, 90), "y": (135, 140), "up": (172, 176),
    "hip": (40, -70), "flex": (90, 180), "fold": (25, -95), "head": (150, -110), "wave": (100, 160),
}
LEGS = {  # (thigh, shin)
    "stand": (3, 0), "wide": (22, 22), "squat": (62, -8), "lunge": (48, 2), "reach": (35, 35),
    "knee": (105, 5), "kick": (82, 86), "cross": (-12, -12),
}
ARM_PAIRS = [  # (figure's left, figure's right)
    ("down", "down"), ("low", "low"), ("out", "out"), ("y", "y"), ("up", "up"), ("hip", "hip"),
    ("flex", "flex"), ("fold", "fold"), ("head", "head"), ("up", "down"), ("out", "down"),
    ("y", "hip"), ("flex", "down"), ("up", "hip"), ("wave", "down"), ("head", "out"),
    ("up", "out"), ("y", "down"),
]
LEG_PAIRS = [("stand", "stand"), ("wide", "wide"), ("squat", "squat"), ("lunge", "reach"),
             ("knee", "stand"), ("kick", "stand"), ("cross", "cross")]
LEANS = {"": 0, "lean": 18}

ARM_NAMES = {
    ("down", "down"): "", ("low", "low"): "ARMS LOW", ("out", "out"): "ARMS OUT", ("y", "y"): "ARMS Y",
    ("up", "up"): "ARMS UP", ("hip", "hip"): "HANDS ON HIPS", ("flex", "flex"): "DOUBLE FLEX",
    ("fold", "fold"): "ARMS FOLDED", ("head", "head"): "HANDS ON HEAD", ("up", "down"): "DISCO",
    ("out", "down"): "POINT", ("y", "hip"): "SASSY", ("flex", "down"): "FLEX",
    ("up", "hip"): "HAND RAISED", ("wave", "down"): "WAVE", ("head", "out"): "SALUTE",
    ("up", "out"): "DAB", ("y", "down"): "CHEER",
}
LEG_NAMES = {("stand", "stand"): "", ("wide", "wide"): "WIDE", ("squat", "squat"): "SQUAT",
             ("lunge", "reach"): "LUNGE", ("knee", "stand"): "KNEE UP", ("kick", "stand"): "KICK",
             ("cross", "cross"): "CROSSED LEGS"}
SPECIAL = {
    (("down", "down"), ("stand", "stand"), ""): "STANDING",
    (("out", "out"), ("stand", "stand"), ""): "T POSE",
    (("y", "y"), ("wide", "wide"), ""): "STAR JUMP",
    (("hip", "hip"), ("wide", "wide"), ""): "SUPERHERO",
    (("flex", "flex"), ("wide", "wide"), ""): "STRONGMAN",
    (("up", "up"), ("knee", "stand"), ""): "TREE POSE",
    (("out", "out"), ("knee", "stand"), ""): "FLAMINGO",
    (("out", "out"), ("kick", "stand"), ""): "KARATE KICK",
    (("out", "out"), ("lunge", "reach"), ""): "WARRIOR",
    (("fold", "fold"), ("stand", "stand"), ""): "BOUNCER",
    (("up", "up"), ("stand", "stand"), ""): "HANDS UP",
    (("hip", "hip"), ("stand", "stand"), ""): "TEAPOT",
    (("out", "out"), ("squat", "squat"), ""): "SUMO",
    (("up", "down"), ("wide", "wide"), ""): "DISCO FEVER",
}


def limb(start, angle_deg, length, side):
    """side +1 = figure's left (image right), -1 = figure's right (image left)."""
    a = math.radians(angle_deg)
    return start[0] + side * math.sin(a) * length, start[1] + math.cos(a) * length


def figure(arms, legs, lean_deg):
    """COCO-17 keypoints (dict index -> (x, y)) for a figure, feet resting near the bottom."""
    hip_c = np.array([0.0, 0.0])
    t = math.radians(lean_deg)  # lean toward the figure's left
    up = np.array([math.sin(t), -math.cos(t)])
    across = np.array([math.cos(t), math.sin(t)])  # toward figure's left, perpendicular to torso
    sh_c = hip_c + up * TORSO
    k = {5: sh_c + across * SHOULDER_W / 2, 6: sh_c - across * SHOULDER_W / 2,
         11: hip_c + across * HIP_W / 2, 12: hip_c - across * HIP_W / 2,
         0: sh_c + up * NECK}
    for side, (s, e, w), name in ((1, (5, 7, 9), arms[0]), (-1, (6, 8, 10), arms[1])):
        ua, fa = ARMS[name]
        k[e] = np.array(limb(k[s], ua + side * lean_deg, UPPER_ARM, side))
        k[w] = np.array(limb(k[e], fa + side * lean_deg, FOREARM, side))
    for side, (h, kn, an), name in ((1, (11, 13, 15), legs[0]), (-1, (12, 14, 16), legs[1])):
        th, sh = LEGS[name]
        k[kn] = np.array(limb(k[h], th, THIGH, side))
        k[an] = np.array(limb(k[kn], sh, SHIN, side))
    pts = np.array(list(k.values()))
    lo, hi = pts.min(0), pts.max(0)
    scale = min((W - 140) / (hi[0] - lo[0] + 1), (H - 160) / (hi[1] - lo[1] + 1), 1.0)
    off = np.array([W / 2 - (lo[0] + hi[0]) / 2 * scale, H - 50 - hi[1] * scale])
    return {i: tuple(int(v) for v in p * scale + off) for i, p in k.items()}, scale


def draw(k, scale):
    img = np.full((H, W, 3), BG, np.uint8)
    th = max(18, int(40 * scale))
    torso = np.array([k[5], k[6], k[12], k[11]], np.int32)
    cv2.fillPoly(img, [torso], BODY, cv2.LINE_AA)
    cv2.polylines(img, [torso], True, BODY, th, cv2.LINE_AA)
    for a, b in [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]:
        cv2.line(img, k[a], k[b], BODY, th, cv2.LINE_AA)
        cv2.circle(img, k[b], th // 2, BODY, -1, cv2.LINE_AA)
    neck = ((k[5][0] + k[6][0]) // 2, (k[5][1] + k[6][1]) // 2)
    cv2.line(img, neck, k[0], BODY, int(th * 0.75), cv2.LINE_AA)
    cv2.circle(img, k[0], int(52 * scale), BODY, -1, cv2.LINE_AA)
    return img


def name_for(arms, legs, lean):
    if (arms, legs, lean) in SPECIAL:
        return SPECIAL[(arms, legs, lean)]
    parts = [ARM_NAMES[arms], LEG_NAMES[legs], "LEAN" if lean else ""]
    return " + ".join(p for p in parts if p) or "STANDING"


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    index = {}
    for arms, legs, (lean_name, lean) in product(ARM_PAIRS, LEG_PAIRS, LEANS.items()):
        if lean and legs not in (("stand", "stand"), ("wide", "wide")):
            continue  # leaning only with simple stances
        name = name_for(arms, legs, lean_name)
        fn = name.lower().replace(" + ", "__").replace(" ", "_") + ".png"
        k, scale = figure(arms, legs, lean)
        cv2.imwrite(str(out / fn), draw(k, scale))
        xs = [p[0] for p in k.values()]
        ys = [p[1] for p in k.values()]
        index[fn] = {"manual": True,
                     "kpts": [list(k.get(i, (0, 0))) for i in range(17)],
                     "conf": [1.0 if i in k else 0.0 for i in range(17)],
                     "bbox": [min(xs) - 30, min(ys) - 60, max(xs) + 30, max(ys) + 20]}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} pictograms in {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "gallery"))
