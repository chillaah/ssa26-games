"""Threaded frame grabber that always hands out the newest frame.

Network streams (the SIYI A8 mini RTSP feed) buffer frames; reading on a background
thread and keeping only the latest one keeps the on-screen box in sync with reality.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CameraCfg

try:  # GStreamer's Python bindings: present on the Jetson, usually not on a Mac
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
except (ImportError, ValueError):
    Gst = None

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Longest the picture holds on the last clean frame while a GOP is broken. One lost keyframe
# costs about 1 s; if keyframes stay away longer, the live view moves on (smeared) rather than
# freezing into FrameGuard's "Waiting for camera..." (2 s).
MAX_HOLD_S = 1.5


class GopGuard:
    """Tells which compressed frames will decode cleanly.

    The A8 mini sends a keyframe every 25 frames but now and then drops one at the source
    (its RTSP server never sends it, so TCP can't help). The decoder then builds the next
    second of frames on the wrong picture, which shows up as blocky smears. Feed every
    compressed frame to see() in order; it returns False from the first frame of a broken
    GOP until the next keyframe. A GOP counts as broken when:
      - no keyframe has arrived yet (joined mid-GOP),
      - more frames than the usual keyframe interval go by without one (keyframe lost),
      - the timestamps jump by more than two frame times (frames lost).
    """

    def __init__(self) -> None:
        self.since_key = -1  # frames since the last keyframe; -1 = none seen yet
        self.clean = False
        self.intervals: List[int] = []  # recent keyframe intervals, to learn the GOP length
        self.frame_dur: Optional[int] = None
        self.last_pts: Optional[int] = None
        self.bad = 0  # frames reported broken, for the log

    @property
    def gop(self) -> Optional[int]:
        """The usual keyframe interval once two recent ones agree, else None."""
        if len(self.intervals) < 2:
            return None
        best = max(set(self.intervals), key=self.intervals.count)
        return best if self.intervals.count(best) >= 2 else None

    def see(self, keyframe: bool, pts: Optional[int]) -> bool:
        if pts is not None and self.last_pts is not None and pts > self.last_pts:
            step = pts - self.last_pts
            if self.frame_dur is not None and step > 2.5 * self.frame_dur:
                self.clean = False  # frames went missing
            if self.frame_dur is None or step < 1.5 * self.frame_dur:
                self.frame_dur = step if self.frame_dur is None else min(step, self.frame_dur)
        if pts is not None:
            self.last_pts = pts
        if keyframe:
            if self.since_key > 0:
                self.intervals = (self.intervals + [self.since_key + 1])[-8:]
            self.since_key = 0
            self.clean = True
        else:
            if self.since_key >= 0:
                self.since_key += 1
            gop = self.gop
            if gop is not None and self.since_key >= gop:
                self.clean = False  # the keyframe that should have come never did
        if not self.clean:
            self.bad += 1
        return self.clean


class GstReader:
    """Runs a GStreamer pipeline ending in an appsink and publishes only clean frames.

    A probe after the H.264/H.265 parser feeds each compressed frame to a GopGuard and notes
    the timestamps of broken ones; decoded frames with those timestamps are skipped, so the
    picture holds on the last good frame (up to MAX_HOLD_S) instead of smearing.
    """

    def __init__(self, pipeline: str, publish, running) -> None:
        self.desc, self.publish, self.running = pipeline, publish, running
        self._bad_pts: set = set()
        self._plock = threading.Lock()

    def _probe(self, guard: GopGuard):
        def probe(_pad, info):
            buf = info.get_buffer()
            pts = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else None
            ok = guard.see(not buf.has_flags(Gst.BufferFlags.DELTA_UNIT), pts)
            if not ok and pts is not None:
                with self._plock:
                    self._bad_pts.add(pts)
            return Gst.PadProbeReturn.OK

        return probe

    def run(self) -> None:
        while self.running():
            try:
                pipe = Gst.parse_launch(self.desc)
            except Exception as e:  # a typo in the config: same as OpenCV failing to open
                log.warning("Camera pipeline won't build (%s), retrying in 1s", e)
                time.sleep(1.0)
                continue
            sink, guard = None, GopGuard()
            elements, it = [], pipe.iterate_elements()
            while True:  # PyGObject on the Jetson can't loop over a Gst.Iterator directly
                res, el = it.next()
                if res != Gst.IteratorResult.OK:
                    break
                elements.append(el)
            for el in elements:
                name = el.get_factory().get_name()
                if name == "appsink":
                    sink = el
                elif name in ("h264parse", "h265parse"):
                    el.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._probe(guard))
            if sink is None:
                log.error("Camera pipeline has no appsink: %s", self.desc)
                return
            with self._plock:
                self._bad_pts.clear()
            pipe.set_state(Gst.State.PLAYING)
            log.info("Camera %r opened (GStreamer, broken-GOP frames are skipped)", self.desc)
            try:
                self._pull(pipe, sink, guard)
            finally:
                pipe.set_state(Gst.State.NULL)
            if self.running():
                time.sleep(1.0)

    def _pull(self, pipe, sink, guard: GopGuard) -> None:
        bus, reported, last_pub = pipe.get_bus(), 0, time.monotonic()
        while self.running():
            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is not None:
                detail = msg.parse_error()[0].message if msg.type == Gst.MessageType.ERROR else "end of stream"
                log.warning("Camera stream stopped (%s), reconnecting", detail)
                return
            sample = sink.emit("try-pull-sample", 500 * Gst.MSECOND)
            if sample is None:
                if sink.get_property("eos"):
                    log.warning("Camera stream ended, reconnecting")
                    return
                continue
            buf = sample.get_buffer()
            with self._plock:
                bad = buf.pts in self._bad_pts
                # forget timestamps the sink has passed (it drops frames when we're slow)
                self._bad_pts = {p for p in self._bad_pts if p > buf.pts}
            if bad and time.monotonic() - last_pub < MAX_HOLD_S:
                continue
            frame = self._to_array(sample)
            if frame is not None:
                self.publish(frame)
                last_pub = time.monotonic()
            if guard.bad - reported >= 25:
                log.info("Camera: skipped %d frames from broken GOPs so far", guard.bad)
                reported = guard.bad

    @staticmethod
    def _to_array(sample) -> Optional[np.ndarray]:
        s = sample.get_caps().get_structure(0)
        w, h = s.get_value("width"), s.get_value("height")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            stride = info.size // h
            arr = np.frombuffer(info.data, np.uint8, count=stride * h).reshape(h, stride)
            return arr[:, : w * 3].reshape(h, w, 3).copy()
        finally:
            buf.unmap(info)


class Camera:
    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_t = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._still: Optional[np.ndarray] = None

        src = cfg.source
        if isinstance(src, str) and src.lower().endswith(IMAGE_EXTS):
            # A still image source, used for testing without a camera.
            self._still = cv2.imread(src)
            if self._still is None:
                raise FileNotFoundError(src)

    def _open(self) -> cv2.VideoCapture:
        src = self.cfg.source
        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            cap = cv2.VideoCapture(int(src))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        elif src.startswith("gst:"):
            cap = cv2.VideoCapture(src[4:], cv2.CAP_GSTREAMER)
        elif src.startswith("rtsp://"):
            # TCP avoids smeared frames from dropped UDP packets; nobuffer keeps latency low.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(src)
        return cap

    def _publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._frame_t = time.monotonic()

    def start(self) -> "Camera":
        if self._still is not None:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        is_file = isinstance(self.cfg.source, str) and not self.cfg.source.startswith(
            ("rtsp://", "gst:")
        ) and not self.cfg.source.isdigit()
        if isinstance(self.cfg.source, str) and self.cfg.source.startswith("gst:") and Gst is not None:
            # GStreamer pipelines with an H.264/H.265 parser go through GstReader, which drops
            # frames decoded from a broken GOP instead of handing out smeared pictures.
            GstReader(self.cfg.source[4:], self._publish, lambda: self._running).run()
            return
        while self._running:
            cap = self._open()
            if not cap.isOpened():
                log.warning("Camera %r not available, retrying in 1s", self.cfg.source)
                time.sleep(1.0)
                continue
            log.info("Camera %r opened", self.cfg.source)
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    if is_file:  # loop test videos
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    log.warning("Camera read failed, reconnecting")
                    break
                self._publish(frame)
                if is_file:
                    time.sleep(1.0 / max(self.cfg.fps, 1))
            cap.release()

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (newest frame or None, capture timestamp)."""
        if self._still is not None:
            frame, t = self._still, time.monotonic()
        else:
            with self._lock:  # the reader thread swaps in a new array per frame, never edits one
                if self._frame is None:
                    return None, 0.0
                frame, t = self._frame, self._frame_t
        # Callers get their own copy (the flip makes one when mirroring).
        return (cv2.flip(frame, 1) if self.cfg.mirror else frame.copy()), t

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
