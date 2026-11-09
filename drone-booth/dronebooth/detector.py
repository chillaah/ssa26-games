"""Person detection (Ultralytics YOLO). Only boxes are used; a -seg model works too (its masks
are ignored)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

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


class PersonDetector:
    def __init__(self, cfg: DetectorCfg):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.model)

    def warmup(self) -> None:
        self.detect(np.zeros((480, 640, 3), np.uint8))

    def detect(self, frame: np.ndarray) -> List[Person]:
        kw = dict(classes=[PERSON], conf=self.cfg.conf, imgsz=self.cfg.imgsz, verbose=False)
        if self.cfg.half:
            kw.update(fp16_kwargs())
        if self.cfg.device is not None:
            kw["device"] = self.cfg.device
        result = self.model.predict(frame, **kw)[0]
        people: List[Person] = []
        if result.boxes is None:
            return people
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for bbox, conf in zip(boxes, confs):
            people.append(Person(tuple(int(v) for v in bbox), float(conf)))
        return people
