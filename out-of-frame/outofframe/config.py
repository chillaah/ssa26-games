"""Typed configuration loaded from a YAML file (see configs/)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class CameraCfg:
    # int -> webcam index, "rtsp://..." -> network stream, "gst:<pipeline>" -> GStreamer,
    # anything else -> video/image file path (handy for testing).
    source: Union[int, str] = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True  # selfie view so dodging feels natural on a webcam


@dataclass
class VirtualFramerCfg:
    # The "gimbal" is simulated: a photo box glides around the fixed webcam view.
    box_width_frac: float = 0.42  # box width as a fraction of the frame width (round 1)
    box_width_frac_final: float = 0.50  # box grows each round -> harder
    margin_frac: float = 0.02
    travel_s: float = 1.6  # time to glide between waypoints (round 1)
    travel_s_final: float = 0.9
    min_jump_frac: float = 0.30  # each waypoint must be at least this far from the last
    feints_min: int = 1
    feints_max: int = 3
    # Optional: the real SIYI gimbal leans a little toward the box while the game itself is
    # unchanged. Link settings and the resting angle (home_yaw/home_pitch) come from framer.siyi.
    follow_gimbal: bool = False
    follow_yaw_deg: float = 6.0  # lean when the box is as far left/right as it can go
    follow_pitch_deg: float = 4.0  # lean when the box is as far up/down as it can go
    follow_hz: float = 10.0  # max angle commands per second
    follow_invert_yaw: bool = False  # flip if the camera leans away from the box
    follow_invert_pitch: bool = False
    # With follow_gimbal: shift each frame back by the gimbal's movement so the view looks still.
    # Costs a slight zoom (the edges the camera can't always see are cropped).
    steady_view: bool = False
    steady_px_per_deg: float = 13.6  # image shift per degree at 1280 px wide (measured, A8 mini)
    steady_latency_s: float = 0.075  # video lags the attitude readings by about this much (measured)
    steady_margin_deg: float = 0.5  # extra crop beyond the lean, for overshoot


@dataclass
class SiyiFramerCfg:
    # The real gimbal moves; the photo is (a centred box of) the whole camera image.
    # Gimbal control link. Video always comes over Ethernet (see camera.source).
    link: str = "udp"  # "udp" (Ethernet) or "serial" (TTL UART)
    ip: str = "192.168.144.25"
    port: int = 37260
    serial_port: str = "/dev/ttyTHS0"  # Orin Nano 40-pin header UART; USB adapter = /dev/ttyUSB0
    baud: int = 115200
    yaw_min: float = -60.0
    yaw_max: float = 60.0
    pitch_min: float = -40.0
    pitch_max: float = 0.0
    min_jump_deg: float = 20.0
    settle_tolerance_deg: float = 1.5
    move_timeout_s: float = 3.0
    hold_s: float = 0.3  # pause at each waypoint so it reads as deliberate
    feints_min: int = 1
    feints_max: int = 2
    photo_box_frac: float = 1.0  # 1.0 = whole image is the photo
    home_yaw: float = 0.0
    home_pitch: float = -15.0


@dataclass
class FramerCfg:
    mode: str = "virtual"  # "virtual" (webcam) or "siyi" (SIYI A8 mini gimbal)
    virtual: VirtualFramerCfg = field(default_factory=VirtualFramerCfg)
    siyi: SiyiFramerCfg = field(default_factory=SiyiFramerCfg)


@dataclass
class DetectorCfg:
    model: str = "yolo11n-seg.pt"  # use a TensorRT .engine on the Jetson
    device: Optional[str] = None  # None = auto, "mps", "cpu", "0" (CUDA)
    imgsz: int = 640
    conf: float = 0.30
    half: bool = False


@dataclass
class GameCfg:
    shots: int = 5
    get_ready_s: float = 3.0
    lock_s_first: float = 1.2  # "locking on" warning before each snap (round 1)
    lock_s_final: float = 0.7
    flash_s: float = 0.35
    reveal_per_shot_s: float = 1.1
    results_s: float = 15.0
    # A person counts as "in the photo" once this much of the box is covered by body pixels.
    min_body_frac: float = 0.001
    min_body_px: int = 80
    # Virtual mode only: someone must still be visible in the wide view at each snap,
    # otherwise walking off camera would be an easy win.
    require_player_in_view: bool = True
    save_dir: str = "sessions"


@dataclass
class DisplayCfg:
    window_name: str = "OUT OF FRAME - The Scouts"
    fullscreen: bool = False
    width: int = 1920
    height: int = 1080
    brand: str = "THE SCOUTS"
    # BGR
    accent: tuple = (255, 200, 0)
    show_live_detections: bool = False  # debug overlay, toggle with "d"


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    framer: FramerCfg = field(default_factory=FramerCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    game: GameCfg = field(default_factory=GameCfg)
    display: DisplayCfg = field(default_factory=DisplayCfg)


def _merge(obj: Any, data: dict, path: str = "") -> Any:
    names = {f.name: f for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in names:
            raise KeyError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge(current, value, f"{path}{key}.")
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    cfg = Config()
    if path:
        with open(path) as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    return cfg
