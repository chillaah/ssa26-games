"""The game state machine. Pure logic + drawing; the main loop feeds it frames and keys."""
from __future__ import annotations

import csv
import logging
import random
import string
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import render as R
from .config import Config
from .detector import PersonDetector, ShotVerdict, judge, union_mask
from .framer import Box, Framer, lerp, difficulty
from .pose import Gallery, GalleryItem, Pose, PoseEstimator, draw_skeleton, visible_limbs

log = logging.getLogger(__name__)

ATTRACT, GET_READY, MOVING, LOCKING, FLASH, POSE, REVEAL, RESULTS = (
    "attract", "get_ready", "moving", "locking", "flash", "pose", "reveal", "results")


@dataclass
class PoseMatch:
    item: GalleryItem
    score: float
    mirrored: bool  # the gallery image is shown flipped so it faces the same way as the player
    pose: Pose
    frame: np.ndarray  # full frame at the snap


@dataclass
class Shot:
    photo: np.ndarray  # raw crop of the box
    annotated: np.ndarray  # crop with detected bodies tinted
    verdict: ShotVerdict


class Game:
    def __init__(self, cfg: Config, framer: Framer, detector: PersonDetector,
                 rng: Optional[random.Random] = None, poser: Optional[PoseEstimator] = None,
                 gallery: Optional[Gallery] = None):
        self.cfg = cfg
        self.framer = framer
        self.detector = detector
        self.poser = poser if gallery is not None and gallery.items else None
        self.gallery = gallery
        self.match: Optional[PoseMatch] = None
        self.used_pictures = set()  # each gallery picture at most once per game
        self._gallery_imgs = {}
        self.rng = rng or random.Random()
        self.state = ATTRACT
        self.state_t = 0.0
        self.round = 0
        self.shots: List[Shot] = []
        self.box: Box = (0, 0, 1, 1)
        self.lock_s = cfg.game.lock_s_first
        self.session_dir: Optional[Path] = None
        self.win_code = ""
        self.show_live = cfg.display.show_live_detections
        self.quit = False
        self._live_people = []

    # ---- input ------------------------------------------------------------------
    def on_key(self, key: int, now: float) -> None:
        if key in (ord("q"), 27):
            self.quit = True
        elif key == ord("d"):
            self.show_live = not self.show_live
        elif key == ord("r"):
            self._set(ATTRACT, now)
        elif key in (ord(" "), 13):
            if self.state in (ATTRACT, RESULTS):
                self.start(now)
            elif self.state == REVEAL:
                self._set(RESULTS, now)

    def start(self, now: float) -> None:
        self.round = 0
        self.shots = []
        self.win_code = ""
        self.match = None
        self.used_pictures = set()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = Path(self.cfg.game.save_dir) / stamp
        self._set(GET_READY, now)

    def _set(self, state: str, now: float) -> None:
        log.debug("state %s -> %s", self.state, state)
        self.state, self.state_t = state, now

    # ---- main tick --------------------------------------------------------------
    def step(self, now: float, frame: Optional[np.ndarray]) -> np.ndarray:
        d = self.cfg.display
        canvas = np.zeros((d.height, d.width, 3), np.uint8)
        if frame is None:
            R.text(canvas, "Waiting for camera...", (d.width // 2, d.height // 2), 1.5)
            return canvas

        fh, fw = frame.shape[:2]
        size = (fw, fh)
        dt = now - self.state_t
        g = self.cfg.game

        if self.state == ATTRACT:
            self.box = self.framer.idle(now, size)
        elif self.state == GET_READY:
            self.box = self.framer.idle(now, size)
            if dt >= g.get_ready_s:
                self._begin_round(now, size)
        elif self.state == MOVING:
            self.box, done = self.framer.update(now, size)
            if done:
                self._set(LOCKING, now)
        elif self.state == LOCKING:
            self.box, _ = self.framer.update(now, size)
            if dt >= self.lock_s:
                self._snap(frame)
                self._set(FLASH, now)
        elif self.state == FLASH:
            if dt >= g.flash_s:
                if self.match is not None:
                    self._set(POSE, now)
                else:
                    self._next_round(now, size)
        elif self.state == POSE:
            if dt >= self.cfg.pose.show_s:
                self._next_round(now, size)
        elif self.state == REVEAL:
            if dt >= g.reveal_per_shot_s * (len(self.shots) + 1):
                self._set(RESULTS, now)
        elif self.state == RESULTS:
            if dt >= g.results_s:
                self._set(ATTRACT, now)

        if self.show_live and self.state not in (REVEAL, RESULTS):
            self._live_people = self.detector.detect(frame)

        self._draw(canvas, frame, now, now - self.state_t)
        return canvas

    def _next_round(self, now: float, size: Tuple[int, int]) -> None:
        self.round += 1
        if self.round < self.cfg.game.shots:
            self._begin_round(now, size)
        else:
            self._finish()
            self._set(REVEAL, now)

    def _begin_round(self, now: float, size: Tuple[int, int]) -> None:
        g = self.cfg.game
        self.framer.start_round(now, self.round, g.shots, size)
        self.lock_s = lerp(g.lock_s_first, g.lock_s_final, difficulty(self.round, g.shots))
        self._set(MOVING, now)

    def _snap(self, frame: np.ndarray) -> None:
        g = self.cfg.game
        people = self.detector.detect(frame)
        require = g.require_player_in_view and self.cfg.framer.mode == "virtual"
        verdict = judge(people, self.box, g.min_body_frac, g.min_body_px, require)
        x1, y1, x2, y2 = self.box
        photo = frame[y1:y2, x1:x2].copy()
        annotated = photo
        mask = union_mask([people[i] for i in verdict.caught_idx], frame.shape[:2])
        if mask is not None:
            crop_mask = mask[y1:y2, x1:x2]
            if crop_mask.any():
                annotated = R.tint(photo, crop_mask, R.RED)
        self.shots.append(Shot(photo, annotated, verdict))
        log.info("shot %d: caught=%s in_box=%d in_view=%d", len(self.shots), verdict.caught,
                 verdict.people_in_box, verdict.people_in_view)
        if self.session_dir:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            n = len(self.shots)
            cv2.imwrite(str(self.session_dir / f"shot{n}.jpg"), photo)
            cv2.imwrite(str(self.session_dir / f"shot{n}_judged.jpg"), annotated)
            cv2.imwrite(str(self.session_dir / f"shot{n}_full.jpg"), frame)
        self.match = self._match_pose(frame)

    def _match_pose(self, frame: np.ndarray) -> Optional[PoseMatch]:
        if self.poser is None:
            return None
        pose = self.poser.player(frame)
        if pose is None or visible_limbs(pose, self.cfg.pose.conf) < self.cfg.pose.min_limbs:
            return None
        item, score, mirrored = self.gallery.best(pose, self.cfg.pose.conf, self.used_pictures,
                                                  self.rng, self.cfg.pose.tie)
        log.info("shot %d: pose match %s (%.2f%s)", len(self.shots), item.path.name if item else "-",
                 score, ", mirrored" if mirrored else "")
        if item is None or score < self.cfg.pose.min_score:
            return None
        self.used_pictures.add(item.path)
        return PoseMatch(item, score, mirrored, pose, frame)

    def _gallery_image(self, item: GalleryItem) -> Optional[np.ndarray]:
        if item.path not in self._gallery_imgs:
            self._gallery_imgs[item.path] = cv2.imread(str(item.path))
        return self._gallery_imgs[item.path]

    @property
    def won(self) -> bool:
        return bool(self.shots) and not any(s.verdict.caught for s in self.shots)

    def _finish(self) -> None:
        if self.won:
            self.win_code = "".join(self.rng.choices(string.ascii_uppercase.replace("O", "")
                                                      .replace("I", "") + "23456789", k=4))
        caught = sum(s.verdict.caught for s in self.shots)
        log.info("game over: %s (%d/%d caught) code=%s", "WIN" if self.won else "LOSE",
                 caught, len(self.shots), self.win_code)
        log_path = Path(self.cfg.game.save_dir) / "results.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new = not log_path.exists()
        with open(log_path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "session", "won", "caught_shots", "shots", "win_code"])
            w.writerow([datetime.now().isoformat(timespec="seconds"),
                        self.session_dir.name if self.session_dir else "", int(self.won),
                        caught, len(self.shots), self.win_code])

    # ---- drawing ----------------------------------------------------------------
    def _draw(self, canvas: np.ndarray, frame: np.ndarray, now: float, dt: float) -> None:
        if self.state in (REVEAL, RESULTS):
            self._draw_results(canvas, now, dt)
            return
        if self.state == POSE:
            self._draw_pose(canvas, dt)
            return

        d = self.cfg.display
        W, H = d.width, d.height
        view = frame
        if self.show_live and self._live_people:
            m = union_mask(self._live_people, frame.shape[:2])
            view = R.tint(frame, m, R.YELLOW, 0.3)
        vx1, vy1, vx2, vy2 = R.paste_fit(canvas, view, (0, 0, W, H))
        s = (vx2 - vx1) / frame.shape[1]
        bx1, by1, bx2, by2 = self.box
        box = (int(vx1 + bx1 * s), int(vy1 + by1 * s), int(vx1 + bx2 * s), int(vy1 + by2 * s))
        cx = W // 2

        if self.state == ATTRACT:
            canvas[:] = (canvas * 0.55).astype(np.uint8)
            R.brackets(canvas, box, d.accent, 4)
            pulse = 0.5 + 0.5 * np.sin(now * 3)
            R.text(canvas, "OUT OF FRAME", (cx, int(H * 0.30)), 4.0, d.accent, 8, font=R.FONT_BOLD)
            R.text(canvas, "The camera takes %d photos. Stay out of every one." % self.cfg.game.shots,
                   (cx, int(H * 0.46)), 1.5, R.WHITE, 3)
            R.text(canvas, "Dodge them all and WIN A PRIZE!", (cx, int(H * 0.54)), 1.5, R.YELLOW, 3)
            col = tuple(int(c * (0.5 + 0.5 * pulse)) for c in R.WHITE)
            R.text(canvas, "PRESS SPACE TO PLAY", (cx, int(H * 0.75)), 2.0, col, 4)
        elif self.state == GET_READY:
            canvas[:] = (canvas * 0.6).astype(np.uint8)
            left = self.cfg.game.get_ready_s - dt
            R.text(canvas, "GET READY TO DODGE!", (cx, int(H * 0.25)), 2.4, R.WHITE, 5)
            R.text(canvas, str(max(1, int(np.ceil(left)))), (cx, int(H * 0.55)), 9.0, d.accent, 16,
                   font=R.FONT_BOLD)
            R.text(canvas, "Stay in view of the camera, but out of the box", (cx, int(H * 0.82)),
                   1.3, R.WHITE, 2)
        elif self.state in (MOVING, LOCKING, FLASH):
            R.dim_outside(canvas, box, 0.5)
            if self.state == MOVING:
                R.brackets(canvas, box, d.accent, 6)
            elif self.state == LOCKING:
                frac = min(1.0, dt / self.lock_s)
                blink = int(now * 12) % 2 == 0
                color = R.RED if frac > 0.5 and blink else R.YELLOW
                inset = int((1 - frac) * 25)
                R.brackets(canvas, (box[0] - inset, box[1] - inset, box[2] + inset, box[3] + inset),
                           color, 8)
                R.progress_bar(canvas, (box[0], box[3] + 14, box[2], box[3] + 26), frac, color)
                R.text(canvas, "LOCKING ON", ((box[0] + box[2]) // 2, box[1] - 30), 1.2, color, 3)
            else:  # FLASH
                a = max(0.0, 1.0 - dt / self.cfg.game.flash_s)
                canvas[:] = cv2.addWeighted(canvas, 1 - a, np.full_like(canvas, 255), a, 0)
                R.text(canvas, "SNAP!", (cx, H // 2), 5.0, R.BLACK, 12, font=R.FONT_BOLD,
                       shadow=False)
            self._draw_hud(canvas)

    def _draw_pose(self, canvas: np.ndarray, dt: float) -> None:
        """The player (with their skeleton) next to the gallery picture that matches their pose."""
        d = self.cfg.display
        W, H = d.width, d.height
        m = self.match
        canvas[:] = (25, 22, 20)
        # The player, cropped around them with some room.
        x1, y1, x2, y2 = m.pose.bbox
        fh, fw = m.frame.shape[:2]
        pad = int(0.15 * max(x2 - x1, y2 - y1))
        cx1, cy1, cx2, cy2 = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, fw), min(y2 + pad, fh)
        crop = m.frame[cy1:cy2, cx1:cx2].copy()
        shifted = Pose(m.pose.kpts - (cx1, cy1), m.pose.conf, m.pose.bbox)
        draw_skeleton(crop, shifted, d.accent, max(2, crop.shape[0] // 120), self.cfg.pose.conf)
        top = int(H * 0.2)
        R.card(canvas, crop, (60, top, W // 2 - 30, H - 50), "YOU", R.WHITE)
        pic = self._gallery_image(m.item)
        if pic is not None:
            if m.mirrored:
                pic = cv2.flip(pic, 1)
            R.card(canvas, pic, (W // 2 + 30, top, W - 60, H - 50), m.item.name, d.accent)
        pop = min(1.0, dt / 0.25)
        R.text(canvas, "POSE MATCH!", (W // 2, int(H * 0.1)), 2.4 + 0.8 * (1 - pop), d.accent, 7,
               font=R.FONT_BOLD)
        R.text(canvas, f"{round(m.score * 100)}% match", (W // 2, int(H * 0.16)), 1.2, R.WHITE, 2)

    def _draw_hud(self, canvas: np.ndarray) -> None:
        d = self.cfg.display
        W, H = d.width, d.height
        shown = min(self.round + 1, self.cfg.game.shots)
        R.text(canvas, f"PHOTO {shown}/{self.cfg.game.shots}", (40, 70), 1.8, R.WHITE, 4, anchor="left")
        R.text(canvas, d.brand, (W - 40, 70), 1.2, d.accent, 3, anchor="right")
        # Taken photos as face-down cards, verdicts are revealed at the end.
        n = self.cfg.game.shots
        cw, ch, gap = 110, 80, 14
        x0 = W - 40 - n * cw - (n - 1) * gap
        for i in range(n):
            x = x0 + i * (cw + gap)
            rect = (x, H - ch - 30, x + cw, H - 30)
            if i < len(self.shots):
                R.paste_fit(canvas, self.shots[i].photo, rect)
                cv2.rectangle(canvas, rect[:2], rect[2:], R.WHITE, 3)
            else:
                cv2.rectangle(canvas, rect[:2], rect[2:], R.GREY, 2)
                R.text(canvas, str(i + 1), ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), 1.0,
                       R.GREY, 2)

    def _draw_results(self, canvas: np.ndarray, now: float, dt: float) -> None:
        d = self.cfg.display
        g = self.cfg.game
        W, H = d.width, d.height
        canvas[:] = (25, 22, 20)
        revealed = len(self.shots) if self.state == RESULTS else int(dt / g.reveal_per_shot_s)
        if self.shots:
            p = self.shots[0].photo
            aspect = p.shape[1] / max(p.shape[0], 1)
        else:
            aspect = 4 / 3
        rects = R.grid_rects(len(self.shots), (60, int(H * 0.22), W - 60, H - 60), aspect * 0.92)
        for i, (shot, rect) in enumerate(zip(self.shots, rects)):
            if i < revealed:
                v = shot.verdict
                border = R.RED if v.caught else R.GREEN
                label = (f"#{i + 1}  LEFT THE ARENA" if v.no_player else
                         f"#{i + 1}  {v.people_in_box} CAUGHT" if v.caught else f"#{i + 1}  CLEAR")
                img_rect = R.card(canvas, shot.annotated, rect, label, border)
                cxy = ((img_rect[0] + img_rect[2]) // 2, (img_rect[1] + img_rect[3]) // 2)
                # Nobody in view still counts as a loss, but it wasn't really a catch.
                word = "MISS" if v.no_player else "CAUGHT" if v.caught else "CLEAR"
                R.stamp(canvas, word, border, cxy,
                        max(0.8, (img_rect[2] - img_rect[0]) / 260))
            else:
                R.card(canvas, None, rect, f"#{i + 1}  ?")

        if self.state == REVEAL:
            R.text(canvas, "DEVELOPING PHOTOS...", (W // 2, int(H * 0.11)), 2.4, R.WHITE, 5)
            return
        if self.won:
            R.text(canvas, "YOU WIN!", (W // 2, int(H * 0.09)), 3.6, R.GREEN, 9, font=R.FONT_BOLD)
            R.text(canvas, f"Not in a single photo. Show code {self.win_code} to claim your prize!",
                   (W // 2, int(H * 0.18)), 1.3, R.WHITE, 3)
        else:
            caught = sum(s.verdict.caught for s in self.shots)
            R.text(canvas, "CAUGHT BY SCOUTS!", (W // 2, int(H * 0.09)), 3.2, R.RED, 8,
                   font=R.FONT_BOLD)
            R.text(canvas, f"You were spotted in {caught} of {len(self.shots)} photos. Try again!",
                   (W // 2, int(H * 0.18)), 1.3, R.WHITE, 3)
