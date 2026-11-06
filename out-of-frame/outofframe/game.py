"""The game state machine. Pure logic + drawing; the main loop feeds it frames and keys."""
from __future__ import annotations

import csv
import logging
import random
import string
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import collage as C
from . import render as R
from .config import Config
from .detector import PersonDetector, ShotVerdict, judge, union_mask
from .framer import Box, Framer, lerp, difficulty
from .pose import (Gallery, GalleryItem, Pose, PoseEstimator, draw_skeleton, neutral_score,
                   visible_limbs)

log = logging.getLogger(__name__)

ATTRACT, GET_READY, MOVING, LOCKING, FLASH, POSE, BONUS, BONUS_FLASH, REVEAL, RESULTS = (
    "attract", "get_ready", "moving", "locking", "flash", "pose", "bonus", "bonus_flash", "reveal",
    "results")


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
                 gallery: Optional[Gallery] = None, share=None):
        self.cfg = cfg
        self.share = share  # share.ShareServer: results QR -> phone download page
        self.framer = framer
        self.detector = detector
        has_gallery = gallery is not None and bool(gallery.items)
        self.poser = poser if has_gallery and cfg.pose.match_each_snap else None
        self.gallery = gallery if has_gallery else None
        self.match: Optional[PoseMatch] = None
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
        # Bonus round: a gallery picture to copy, then one photo of the player (no scoring).
        self.bonus_item: Optional[GalleryItem] = None
        self.bonus_photo: Optional[np.ndarray] = None
        self._recent_bonus: deque = deque(maxlen=30)  # don't repeat pictures for a while
        self.collage: Optional[np.ndarray] = None
        self.share_url = ""
        self.share_qr: Optional[np.ndarray] = None
        self._bonus_bg: Optional[Tuple[np.ndarray, Tuple[int, int, int, int], int]] = None
        C.background()  # build the collage backdrop now, not at the end of the first game
        self._socials = []  # (social, QR image) shown on the attract screen
        for soc in cfg.display.socials:
            qr = cv2.imread(soc["qr"])
            if qr is None:
                log.warning("QR image %s not found; showing %s without it", soc["qr"], soc["name"])
            self._socials.append((soc, qr))

    # ---- input ------------------------------------------------------------------
    def on_key(self, key: int, now: float) -> None:
        if key in (ord("q"), 27):
            self.quit = True
        elif key == ord("d"):
            self.show_live = not self.show_live
        elif key == ord("r"):
            self._set(ATTRACT, now)
        elif key in (ord(" "), 13):
            if self.state == ATTRACT:
                self.start(now)
            elif self.state == RESULTS:
                self._set(ATTRACT, now)
            elif self.state == REVEAL:
                self._set(RESULTS, now)

    def start(self, now: float) -> None:
        self.round = 0
        self.shots = []
        self.win_code = ""
        self.match = None
        self.bonus_item = self.bonus_photo = self.collage = self.share_qr = None
        self.share_url = ""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = Path(self.cfg.game.save_dir) / stamp
        self._set(GET_READY, now)

    def reset(self, now: float) -> None:
        """Back to the start screen (used after an unexpected error)."""
        self.match = None
        self._set(ATTRACT, now)

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
        elif self.state == BONUS:
            self.box = self.framer.idle(now, size)
            if dt >= g.bonus_s:
                self._snap_bonus(frame)
                self._set(BONUS_FLASH, now)
        elif self.state == BONUS_FLASH:
            if dt >= g.flash_s:
                self._end_game(now)
        elif self.state == REVEAL:
            if dt >= g.reveal_per_shot_s * (len(self.shots) + 1):
                self._set(RESULTS, now)
        # RESULTS stays up (collage + QR) until SPACE goes back to the start screen.

        if self.show_live and self.state not in (REVEAL, RESULTS, BONUS, BONUS_FLASH):
            self._live_people = self.detector.detect(frame)

        self._draw(canvas, frame, now, now - self.state_t)
        return canvas

    def _next_round(self, now: float, size: Tuple[int, int]) -> None:
        self.round += 1
        if self.round < self.cfg.game.shots:
            self._begin_round(now, size)
        elif self.cfg.game.bonus_pose and self.gallery is not None and self.gallery.bonus_items:
            self.bonus_item = self._pick_bonus()
            self._bonus_bg = None
            log.info("bonus round: %s", self.bonus_item.path.name)
            self._set(BONUS, now)
        else:
            self._end_game(now)

    def _end_game(self, now: float) -> None:
        self._finish()
        self._make_collage()
        self._set(REVEAL, now)

    def _pick_bonus(self) -> GalleryItem:
        pool = self.gallery.bonus_items
        fresh = [i for i in pool if i.path not in self._recent_bonus]
        item = self.rng.choice(fresh or pool)
        self._recent_bonus.append(item.path)
        return item

    def _snap_bonus(self, frame: np.ndarray) -> None:
        people = self.detector.detect(frame)
        self.bonus_photo = C.bonus_crop(frame, C.players([p.bbox for p in people]))
        if self.session_dir:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.session_dir / "bonus.jpg"), self.bonus_photo)
            cv2.imwrite(str(self.session_dir / "bonus_full.jpg"), frame)

    def _make_collage(self) -> None:
        d = self.cfg.display
        ref = self._gallery_image(self.bonus_item) if self.bonus_item is not None else None
        self.collage = C.compose(self.shots, self.bonus_photo, ref, self.won, d.event, d.logo,
                                 d.accent)
        if self.session_dir:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.session_dir / "collage.jpg"), self.collage)
        if self.share is not None:
            try:
                pid = self.share.store.save(self.collage)
                self.share_url = self.share.url_for(pid)
                self.share_qr = C.qr_image(self.share_url, 400)
                log.info("collage %s at %s", pid, self.share_url)
            except Exception:  # the game must go on without the download
                log.exception("Could not publish the collage")

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
        if neutral_score(pose, self.cfg.pose.conf) > self.cfg.pose.max_neutral:
            log.info("shot %d: no pose match, player is standing still", len(self.shots))
            return None
        item, score, mirrored = self.gallery.best(pose, self.cfg.pose.conf)
        log.info("shot %d: pose match %s (%.2f%s)", len(self.shots), item.path.name if item else "-",
                 score, ", mirrored" if mirrored else "")
        if item is None or score < self.cfg.pose.min_score:
            return None
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
        spotted, missed, dodged = C.tally(self.shots)
        log.info("game over: %s (%d spotted, %d out of view, %d dodged) code=%s",
                 "WIN" if self.won else "LOSE", spotted, missed, dodged, self.win_code)
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
        if self.state in (BONUS, BONUS_FLASH):
            self._draw_bonus(canvas, frame, now, dt)
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
            canvas[:] = (canvas * 0.45).astype(np.uint8)
            R.brackets(canvas, box, d.accent, 4)
            pulse = 0.5 + 0.5 * np.sin(now * 3)
            R.logo(canvas, d.logo, (cx, int(H * 0.075)), int(H * 0.07))
            R.text(canvas, "Out of Frame", (cx, int(H * 0.19)), 3.6, R.WHITE, font=R.FONT_BOLD)
            R.text(canvas, "The camera takes %d photos with a moving box." % self.cfg.game.shots,
                   (cx, int(H * 0.285)), 1.3, R.WHITE, font=R.FONT_REGULAR)
            R.text(canvas, "Stay out of the box but in frame when the photo clicks.",
                   (cx, int(H * 0.345)), 1.3, R.WHITE, font=R.FONT_REGULAR)
            bx1, by1, bx2, by2 = R.button(canvas, "Press SPACE to play", (cx, int(H * 0.455)), 1.4)
            g = int(6 + 6 * pulse)
            R.rounded_rect(canvas, (bx1 - g, by1 - g, bx2 + g, by2 + g), d.accent, 28 + g, thickness=3,
                           opacity=1.0 - 0.6 * pulse)
            self._draw_socials(canvas)
        elif self.state == GET_READY:
            canvas[:] = (canvas * 0.6).astype(np.uint8)
            left = self.cfg.game.get_ready_s - dt
            R.text(canvas, "Get ready to dodge!", (cx, int(H * 0.25)), 2.4, R.WHITE, font=R.FONT_BOLD)
            R.text(canvas, str(max(1, int(np.ceil(left)))), (cx, int(H * 0.55)), 9.0, d.accent,
                   font=R.FONT_BOLD)
            R.text(canvas, "Stay in view of the camera, but out of the box", (cx, int(H * 0.82)),
                   1.3, R.WHITE, font=R.FONT_REGULAR)
            if self.poser is not None:
                R.text(canvas, "Strike a pose at every snap!", (cx, int(H * 0.89)), 1.3, d.accent,
                       font=R.FONT_SEMIBOLD)
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
                R.text(canvas, "LOCKING ON", ((box[0] + box[2]) // 2, box[1] - 30), 1.2, color,
                       font=R.FONT_SEMIBOLD)
            else:  # FLASH
                a = max(0.0, 1.0 - dt / self.cfg.game.flash_s)
                canvas[:] = cv2.addWeighted(canvas, 1 - a, np.full_like(canvas, 255), a, 0)
                R.text(canvas, "SNAP!", (cx, H // 2), 5.0, R.INK, font=R.FONT_BOLD, shadow=False)
            self._draw_hud(canvas)

    def _draw_socials(self, canvas: np.ndarray) -> None:
        """QR codes for the Scouts' social accounts along the bottom of the screen."""
        if not self._socials:
            return
        W, H = self.cfg.display.width, self.cfg.display.height
        R.text(canvas, "Follow Scouts Socials", (W // 2, int(H * 0.585)), 1.2, R.WHITE, font=R.FONT_SEMIBOLD)
        q = int(H * 0.22)  # QR side
        p = int(q * 0.07)  # white card around the code
        gap = int(W * 0.12)
        n = len(self._socials)
        x = (W - n * q - (n - 1) * gap) // 2
        y = int(H * 0.64)
        for soc, qr in self._socials:
            if qr is not None:
                R.rounded_rect(canvas, (x - p, y - p, x + q + p, y + q + p), R.WHITE, p * 2)
                canvas[y:y + q, x:x + q] = cv2.resize(qr, (q, q), interpolation=cv2.INTER_NEAREST)
            R.icon_label(canvas, soc.get("icon"), soc["name"], (x + q // 2, y + q + p + 34), 1.0,
                         self.cfg.display.accent)
            R.text(canvas, soc["handle"], (x + q // 2, y + q + p + 72), 0.85, R.WHITE,
                   font=R.FONT_REGULAR)
            x += q + gap

    def _draw_pose(self, canvas: np.ndarray, dt: float) -> None:
        """The player (with their skeleton) next to the gallery picture that matches their pose."""
        d = self.cfg.display
        W, H = d.width, d.height
        m = self.match
        canvas[:] = R.INK
        R.logo(canvas, d.logo, (60, 70), 56, anchor="left")
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
        R.text(canvas, "POSE MATCH!", (W // 2, int(H * 0.09)), 2.4 + 0.8 * (1 - pop), d.accent,
               font=R.FONT_BOLD)
        R.text(canvas, f"{round(m.score * 100)}% match", (W // 2, int(H * 0.155)), 1.2, R.WHITE,
               font=R.FONT_REGULAR)

    def _draw_bonus(self, canvas: np.ndarray, frame: np.ndarray, now: float, dt: float) -> None:
        """The picture to copy next to the live camera, with a countdown to the photo."""
        d = self.cfg.display
        W, H = d.width, d.height
        g = self.cfg.game
        if self._bonus_bg is None:
            # Everything but the live camera and the countdown stays put: draw it once
            # (the big rounded cards cost ~100 ms a frame on the Jetson).
            bg = np.empty_like(canvas)
            bg[:] = R.INK
            R.logo(bg, d.logo, (60, 70), 56, anchor="left")
            R.text(bg, "Copy this pose for your souvenir photo!", (W // 2, int(H * 0.155)), 1.3,
                   R.WHITE, font=R.FONT_REGULAR)
            top = int(H * 0.2)
            R.card(bg, self._gallery_image(self.bonus_item), (60, top, W // 2 - 30, H - 50),
                   self.bonus_item.name, d.accent, fill=True, keep=self.bonus_item.pose.bbox)
            you = (W // 2 + 30, top, W - 60, H - 50)
            inner = R.card(bg, None, you, "YOU", R.WHITE)
            self._bonus_bg = (bg, inner, R.card_inner_radius(you))
        bg, inner, radius = self._bonus_bg
        np.copyto(canvas, bg)
        # Fill the card (sides cropped) instead of letterboxing: no black bars above and below.
        live = R.paste_fill(canvas, frame, inner, radius)
        pop = min(1.0, dt / 0.3) if self.state == BONUS else 1.0
        R.text(canvas, "BONUS ROUND", (W // 2, int(H * 0.085)), 2.6 + 0.8 * (1 - pop), d.accent,
               font=R.FONT_BOLD)
        if self.state == BONUS:
            left = g.bonus_s - dt
            cx, cy = (live[0] + live[2]) // 2, (live[1] + live[3]) // 2
            if left <= 3:
                R.text(canvas, str(max(1, int(np.ceil(left)))), (cx, cy), 8.0, d.accent,
                       font=R.FONT_BOLD)
            R.progress_bar(canvas, (live[0], live[3] - 14, live[2], live[3] - 4),
                           min(1.0, dt / g.bonus_s), d.accent)
        else:  # BONUS_FLASH
            a = max(0.0, 1.0 - dt / g.flash_s)
            canvas[:] = cv2.addWeighted(canvas, 1 - a, np.full_like(canvas, 255), a, 0)
            R.text(canvas, "SNAP!", (W // 2, H // 2), 5.0, R.INK, font=R.FONT_BOLD, shadow=False)

    def _draw_hud(self, canvas: np.ndarray) -> None:
        d = self.cfg.display
        W, H = d.width, d.height
        shown = min(self.round + 1, self.cfg.game.shots)
        R.text(canvas, f"PHOTO {shown}/{self.cfg.game.shots}", (40, 78), 1.8, R.WHITE, anchor="left",
               font=R.FONT_SEMIBOLD)
        R.logo(canvas, d.logo, (W - 130, 62), 64, anchor="right")  # clear of the close button
        # Taken photos as face-down cards, verdicts are revealed at the end.
        n = self.cfg.game.shots
        cw, ch, gap = 110, 80, 14
        x0 = W - 40 - n * cw - (n - 1) * gap
        for i in range(n):
            x = x0 + i * (cw + gap)
            rect = (x, H - ch - 30, x + cw, H - 30)
            if i < len(self.shots):
                R.paste_fit(canvas, self.shots[i].photo, rect)
                R.rounded_rect(canvas, rect, R.WHITE, 10, thickness=3)
            else:
                R.rounded_rect(canvas, rect, R.INK, 10, opacity=0.45)
                R.rounded_rect(canvas, rect, R.SILVER, 10, thickness=2, opacity=0.7)
                R.text(canvas, str(i + 1), ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), 1.0,
                       R.SILVER, font=R.FONT_SEMIBOLD)

    def _draw_results(self, canvas: np.ndarray, now: float, dt: float) -> None:
        d = self.cfg.display
        g = self.cfg.game
        W, H = d.width, d.height
        canvas[:] = R.INK
        R.logo(canvas, d.logo, (60, 70), 56, anchor="left")
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
                # Nobody in view still counts as a loss, but it wasn't really a catch.
                word = "MISS" if v.no_player else "THREAT DETECTED" if v.caught else "CLEAR"
                img_rect = R.card(canvas, shot.annotated, rect, word, border)
                cxy = ((img_rect[0] + img_rect[2]) // 2, (img_rect[1] + img_rect[3]) // 2)
                # Stamp as big as "CAUGHT" used to be, but never wider than ~80% of the photo.
                iw = img_rect[2] - img_rect[0]
                tw = R.text_size(word, 1.0, R.FONT_BOLD)[0]
                R.stamp(canvas, word, border, cxy, max(0.6, min(iw / 260, 0.8 * iw / tw)))
            else:
                R.card(canvas, None, rect, "?")

        if self.state == REVEAL:
            R.text(canvas, "Developing photos...", (W // 2, int(H * 0.11)), 2.4, R.WHITE,
                   font=R.FONT_BOLD)
            return
        self._draw_final(canvas, now, dt)

    def _draw_final(self, canvas: np.ndarray, now: float, dt: float) -> None:
        """The 3:4 collage on the left; result, prize code and the download QR in a column on
        the right (same layout as the drone booth's review screen)."""
        d = self.cfg.display
        W, H = d.width, d.height
        canvas[:] = R.INK
        if self.collage is not None:
            x1, y1, x2, y2 = R.paste_fit(canvas, self.collage, (90, 60, 810, H - 60))
            R.rounded_rect(canvas, (x1 - 3, y1 - 3, x2 + 3, y2 + 3), R.CHARCOAL, 8, thickness=3)
        cx = (900 + W - 90) // 2
        if self.won:
            R.text(canvas, "YOU WIN!", (cx, 140), 2.8, R.GREEN, font=R.FONT_BOLD)
            R.text(canvas, "Not in a single photo. Show this code for your prize:", (cx, 205), 1.0,
                   R.SILVER, font=R.FONT_REGULAR, shadow=False)
            R.button(canvas, self.win_code, (cx, 280), 1.4, bg=R.GREEN)
        else:
            spotted, _, _ = C.tally(self.shots)
            R.text(canvas, "Caught by Scouts!" if spotted else "Out of view!", (cx, 175), 2.6,
                   R.RED, font=R.FONT_BOLD)
            R.text(canvas, C.result_line(self.shots), (cx, 245), 1.1, R.SILVER,
                   font=R.FONT_REGULAR, shadow=False)
        if self.share_qr is not None:
            q = self.share_qr.shape[0]
            top = 400
            R.text(canvas, "Scan to get your photo", (cx, top), 1.35, d.accent, font=R.FONT_SEMIBOLD,
                   shadow=False)
            y = top + 40
            R.rounded_rect(canvas, (cx - q // 2 - 24, y, cx + q // 2 + 24, y + q + 48), R.WHITE, 22)
            canvas[y + 24:y + 24 + q, cx - q // 2:cx - q // 2 + q] = self.share_qr
            R.text(canvas, "Point your phone camera at the code", (cx, y + q + 98), 0.8, R.GREY,
                   font=R.FONT_REGULAR, shadow=False)
        R.text(canvas, "SPACE: back to start   ·   Q: quit", (cx, H - 70), 0.85, R.SILVER,
               font=R.FONT_REGULAR, shadow=False)
