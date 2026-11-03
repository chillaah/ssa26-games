"""Pose match: find the gallery image whose pose looks most like the player's.

Poses are COCO-17 keypoints from a YOLO pose model. Two poses are compared by the directions of
their limbs, so position, size and body proportions don't matter. Each gallery pose is also tried
mirrored, so a raised left arm can match a picture with a raised right arm.

Gallery: drop images into gallery/. Their poses are cached in gallery/index.json; an entry can
be hand-written with "manual": true for pictures the pose model can't read (an animal, a letter).

Index the gallery (also done automatically at startup when images change):
    python -m outofframe.pose --gallery gallery
Try a photo against it:
    python -m outofframe.pose --gallery gallery --test photo.jpg
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import PoseCfg

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# COCO-17: 0 nose, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles.
# 17 = mid-shoulder, 18 = mid-hip (added below).
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
LIMBS = [  # (from, to, weight)
    (5, 7, 1.0), (7, 9, 1.0), (6, 8, 1.0), (8, 10, 1.0),  # arms
    (11, 13, 1.0), (13, 15, 1.0), (12, 14, 1.0), (14, 16, 1.0),  # legs
    (18, 17, 2.0),  # torso lean
    (17, 0, 0.5),  # head tilt
]
ANGLE_TOLERANCE_DEG = 35.0
UNSEEN_WEIGHT = 0.3
# Limb directions of someone standing straight with arms down (image y points down).
NEUTRAL = np.array([[0.0, 1.0]] * 8 + [[0.0, -1.0], [0.0, -1.0]])
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6)]


@dataclass
class Pose:
    kpts: np.ndarray  # (17, 2) pixels
    conf: np.ndarray  # (17,)
    bbox: Tuple[int, int, int, int]

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(x2 - x1, 0) * max(y2 - y1, 0)


def _with_midpoints(kpts: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k = np.vstack([kpts, (kpts[5] + kpts[6]) / 2, (kpts[11] + kpts[12]) / 2])
    c = np.concatenate([conf, [min(conf[5], conf[6]), min(conf[11], conf[12])]])
    return k, c


def limb_vectors(kpts: np.ndarray, conf: np.ndarray, min_conf: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unit direction of each limb, and whether both of its ends were seen."""
    k, c = _with_midpoints(np.asarray(kpts, float), np.asarray(conf, float))
    vecs = np.zeros((len(LIMBS), 2))
    ok = np.zeros(len(LIMBS), bool)
    for i, (a, b, _) in enumerate(LIMBS):
        d = k[b] - k[a]
        n = np.hypot(*d)
        if c[a] >= min_conf and c[b] >= min_conf and n > 1e-6:
            vecs[i], ok[i] = d / n, True
    return vecs, ok


def visible_limbs(pose: Pose, min_conf: float = 0.3) -> int:
    return int(limb_vectors(pose.kpts, pose.conf, min_conf)[1].sum())


def mirror(kpts: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k = np.asarray(kpts, float)[FLIP_IDX].copy()
    k[:, 0] = -k[:, 0]
    return k, np.asarray(conf, float)[FLIP_IDX]


def similarity(a: Pose, b: Pose, min_conf: float = 0.3) -> float:
    """0..1. Weighted mean agreement of limb directions. Only limbs seen in `a` (the player) count,
    so a player cut off by the frame is matched on what is visible; limbs of `a` that `b` lacks
    lower the score."""
    va, oka = limb_vectors(a.kpts, a.conf, min_conf)
    vb, okb = limb_vectors(b.kpts, b.conf, min_conf)
    w = np.array([l[2] for l in LIMBS])
    both = oka & okb
    if not both.any():
        return 0.0
    agree = _agree(va, vb)
    score = (agree * w)[both].sum() / w[both].sum()
    score *= w[both].sum() / w[oka].sum()
    # Limbs the player doesn't show (e.g. legs below the frame) are assumed to be in a plain
    # standing position, weakly, so an unseen leg doesn't match a kick as well as a straight leg.
    unseen = okb & ~oka
    if unseen.any():
        prior = _agree(NEUTRAL[unseen], vb[unseen])
        pw = UNSEEN_WEIGHT * w[unseen]
        score = (score * w[oka].sum() + (prior * pw).sum()) / (w[oka].sum() + pw.sum())
    return float(score)


def _agree(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    angle = np.degrees(np.arccos(np.clip((va * vb).sum(1), -1.0, 1.0)))
    return np.exp(-(angle / ANGLE_TOLERANCE_DEG) ** 2)  # 1 when parallel, ~0.37 at the tolerance


@dataclass
class GalleryItem:
    path: Path
    pose: Pose

    @property
    def name(self) -> str:
        return self.path.stem.replace("__", " + ").replace("_", " ").replace("-", " ").upper()


class PoseEstimator:
    def __init__(self, cfg: PoseCfg, device: Optional[str] = None, half: bool = False):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.device, self.half = device, half

    def detect(self, image: np.ndarray) -> List[Pose]:
        kw = dict(conf=self.cfg.conf, verbose=False)
        if self.half:
            kw["half"] = True
        if self.device is not None:
            kw["device"] = self.device
        r = self.model.predict(image, **kw)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []
        kp = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kp.shape[:2])
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        return [Pose(kp[i], kc[i], tuple(int(v) for v in boxes[i])) for i in range(len(boxes))]

    def player(self, image: np.ndarray) -> Optional[Pose]:
        """The biggest person in view is taken to be the player."""
        poses = self.detect(image)
        return max(poses, key=lambda p: p.area) if poses else None


class Gallery:
    INDEX = "index.json"

    def __init__(self, folder: Path, items: List[GalleryItem]):
        self.folder = folder
        self.items = items

    @classmethod
    def load(cls, folder, estimator: Optional[PoseEstimator]) -> "Gallery":
        """Read the cached poses; (re)index new or changed images if an estimator is given."""
        folder = Path(folder)
        index_path = folder / cls.INDEX
        cache: Dict[str, dict] = json.loads(index_path.read_text()) if index_path.exists() else {}
        images = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS) \
            if folder.is_dir() else []
        changed = False
        items: List[GalleryItem] = []
        for path in images:
            entry = cache.get(path.name)
            mtime = path.stat().st_mtime
            # "manual" entries hold hand-placed keypoints (e.g. a bear or a letter) and are kept as is.
            stale_entry = entry is None or (not entry.get("manual") and entry.get("mtime") != mtime)
            if stale_entry and estimator is not None:
                img = cv2.imread(str(path))
                pose = estimator.player(img) if img is not None else None
                entry = {"mtime": mtime, "kpts": pose.kpts.tolist(), "conf": pose.conf.tolist(),
                         "bbox": list(pose.bbox)} if pose else {"mtime": mtime, "kpts": None}
                cache[path.name] = entry
                changed = True
                log.info("Gallery: %s %s", path.name, "indexed" if pose else "NO PERSON FOUND, skipped")
            if entry and entry.get("kpts"):
                items.append(GalleryItem(path, Pose(np.array(entry["kpts"]), np.array(entry["conf"]),
                                                    tuple(entry["bbox"]))))
        stale = set(cache) - {p.name for p in images}
        for name in stale:
            del cache[name]
        if (changed or stale) and folder.is_dir():
            index_path.write_text(json.dumps(cache, indent=1))
        log.info("Gallery: %d usable images in %s", len(items), folder)
        return cls(folder, items)

    def best(self, pose: Pose, min_conf: float = 0.3) -> Tuple[Optional[GalleryItem], float, bool]:
        """(item, score, mirrored) for the closest gallery pose."""
        best, best_s, best_m = None, -1.0, False
        mk, mc = mirror(pose.kpts, pose.conf)
        flipped = Pose(mk, mc, pose.bbox)
        for item in self.items:
            for p, m in ((pose, False), (flipped, True)):
                s = similarity(p, item.pose, min_conf)
                if s > best_s:
                    best, best_s, best_m = item, s, m
        return best, best_s, best_m


def draw_skeleton(img: np.ndarray, pose: Pose, color=(0, 255, 255), thickness: int = 4,
                  min_conf: float = 0.3, offset: Tuple[int, int] = (0, 0), scale: float = 1.0) -> None:
    pts = [(int(offset[0] + x * scale), int(offset[1] + y * scale)) for x, y in pose.kpts]
    for a, b in SKELETON:
        if pose.conf[a] >= min_conf and pose.conf[b] >= min_conf:
            cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if pose.conf[i] >= min_conf:
            cv2.circle(img, p, thickness + 2, color, -1, cv2.LINE_AA)


def main() -> None:
    from .config import load_config

    ap = argparse.ArgumentParser(description="Index the pose gallery / test a photo against it")
    ap.add_argument("--config", default="configs/mac.yaml")
    ap.add_argument("--gallery", help="override pose.gallery_dir")
    ap.add_argument("--test", help="photo to match against the gallery")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    est = PoseEstimator(cfg.pose, cfg.detector.device, cfg.detector.half)
    gallery = Gallery.load(args.gallery or cfg.pose.gallery_dir, est)
    if args.test:
        pose = est.player(cv2.imread(args.test))
        if pose is None:
            print("No person found in", args.test)
            return
        ranked = sorted(((max(similarity(pose, i.pose), similarity(Pose(*mirror(pose.kpts, pose.conf),
                         pose.bbox), i.pose)), i.path.name) for i in gallery.items), reverse=True)
        for s, name in ranked[:5]:
            print(f"{s:.2f}  {name}")


if __name__ == "__main__":
    main()
