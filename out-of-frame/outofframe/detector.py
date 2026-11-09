"""Person segmentation (Ultralytics YOLO) and the in-frame verdict."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DetectorCfg

log = logging.getLogger(__name__)

PERSON = 0


def fp16_kwargs() -> dict:
    """predict() arguments for FP16. Newer Ultralytics renamed half=True to quantize=16 and warns
    on every call that still uses half."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}


@dataclass
class Person:
    bbox: Tuple[int, int, int, int]
    conf: float
    mask: np.ndarray  # bool HxW in full-frame coordinates


@dataclass
class ShotVerdict:
    caught: bool
    people_in_box: int
    people_in_view: int
    no_player: bool  # nobody visible at all (virtual mode anti-cheat)
    coverage: List[float] = field(default_factory=list)  # per person: fraction of box covered
    caught_idx: List[int] = field(default_factory=list)  # people that count as in the photo


class PersonDetector:
    def __init__(self, cfg: DetectorCfg):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.is_seg = self.model.task == "segment"
        if not self.is_seg:
            log.warning("Model %s has no masks; falling back to bounding boxes", cfg.model)

    def warmup(self) -> None:
        self.detect(np.zeros((480, 640, 3), np.uint8))

    def detect(self, frame: np.ndarray) -> List[Person]:
        kw = dict(classes=[PERSON], conf=self.cfg.conf, imgsz=self.cfg.imgsz, verbose=False)
        if self.cfg.half:
            kw.update(fp16_kwargs())
        if self.cfg.device is not None:
            kw["device"] = self.cfg.device
        result = self.model.predict(frame, **kw)[0]
        h, w = frame.shape[:2]
        people: List[Person] = []
        if result.boxes is None:
            return people
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        polys = result.masks.xy if (self.is_seg and result.masks is not None) else None
        for i, (bbox, conf) in enumerate(zip(boxes, confs)):
            mask = np.zeros((h, w), np.uint8)
            if polys is not None and len(polys[i]) >= 3:
                cv2.fillPoly(mask, [polys[i].astype(np.int32)], 1)
            else:
                x1, y1, x2, y2 = bbox
                mask[max(y1, 0):y2, max(x1, 0):x2] = 1
            people.append(Person(tuple(int(v) for v in bbox), float(conf), mask.astype(bool)))
        return people


def judge(people: List[Person], box: Tuple[int, int, int, int], min_body_frac: float,
          min_body_px: int, require_player_in_view: bool) -> ShotVerdict:
    x1, y1, x2, y2 = box
    box_area = max((x2 - x1) * (y2 - y1), 1)
    threshold = max(min_body_px, min_body_frac * box_area)
    coverage, caught_idx = [], []
    for i, p in enumerate(people):
        px = int(p.mask[max(y1, 0):y2, max(x1, 0):x2].sum())
        coverage.append(px / box_area)
        if px >= threshold:
            caught_idx.append(i)
    no_player = require_player_in_view and not people
    return ShotVerdict(
        caught=bool(caught_idx) or no_player,
        people_in_box=len(caught_idx),
        people_in_view=len(people),
        no_player=no_player,
        coverage=coverage,
        caught_idx=caught_idx,
    )


def union_mask(people: List[Person], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not people:
        return None
    m = np.zeros(shape, bool)
    for p in people:
        m |= p.mask
    return m
