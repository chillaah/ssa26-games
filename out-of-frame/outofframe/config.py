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
class PoseCfg:
    # The gallery feeds the bonus round (a picture to copy). Turns itself off if the gallery
    # folder has no usable images.
    enabled: bool = True
    # Also show the gallery picture closest to the player's pose after each dodge snap.
    match_each_snap: bool = False
    model: str = "yolo11n-pose.pt"
    conf: float = 0.30
    gallery_dir: str = "gallery"
    show_s: float = 1.6  # how long the match is on screen
    min_score: float = 0.6  # 0..1; below this no match is shown
    min_limbs: int = 4  # of 10 (arms, legs, torso, head); fewer visible -> no match shown
    max_neutral: float = 0.68  # player standing still (closer than this to arms-down) -> no match


@dataclass
class GameCfg:
    shots: int = 4  # dodge rounds (the box moves); the bonus pose round comes after them
    bonus_pose: bool = True  # final round: copy a gallery picture's pose for the souvenir photo
    bonus_s: float = 6.0  # time to get into the pose before the photo
    get_ready_s: float = 3.0
    lock_s_first: float = 0.9  # "locking on" warning before each snap (round 1)
    lock_s_final: float = 0.5  # (the bonus round uses bonus_s instead)
    flash_s: float = 0.35
    reveal_per_shot_s: float = 1.1
    # A person counts as "in the photo" once this much of the box is covered by body pixels.
    min_body_frac: float = 0.001
    min_body_px: int = 80
    # Virtual mode only: someone must still be visible in the wide view at each snap,
    # otherwise walking off camera would be an easy win.
    require_player_in_view: bool = True
    save_dir: str = "sessions"


@dataclass
class DisplayCfg:
    window_name: str = "Out of Frame - The Scouts"
    fullscreen: bool = False
    width: int = 1920
    height: int = 1080
    logo: str = "horizontal-white"  # assets/brand/logos/<logo>.png, shown in the HUD
    event: str = "Safety & Security Asia 2026"  # printed on the souvenir collage
    # BGR. Default is the brand's Dutch (#ffefca); see outofframe/brand.py for the palette.
    accent: tuple = (202, 239, 255)
    show_live_detections: bool = False  # debug overlay, toggle with "d"
    # Shown with QR codes on the attract screen. Regenerate the QR images with tools/make_qr.py;
    # icon = assets/brand/logos/<icon>.png (tools/make_social_icons.py).
    socials: list = field(default_factory=lambda: [
        {"name": "Instagram", "handle": "@thescouts.io",
         "url": "https://www.instagram.com/thescouts.io/", "qr": "assets/qr_instagram.png",
         "icon": "instagram"},
        {"name": "LinkedIn", "handle": "Scouts - Autonomous Drones",
         "url": "https://www.linkedin.com/company/scoutsapp/posts/", "qr": "assets/qr_linkedin.png",
         "icon": "linkedin"},
    ])


@dataclass
class ShareCfg:
    # Results QR -> phone page where the player leaves name + email and downloads the collage.
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8081  # the drone booth uses 8080, so both can run on one machine
    # Leave empty to use this machine's LAN address. Set it when phones reach the
    # game through something else, e.g. a tunnel: "https://game.example.com".
    public_base_url: str = ""
    # "cloudflare": publish the download page at a public https address (Cloudflare quick
    # tunnel, needs internet and the cloudflared program) so any phone can open the QR code.
    # "" = local network only. Ignored when public_base_url is set.
    tunnel: str = ""
    cloudflared: str = "cloudflared"  # path to the program; ~/.local/bin is also searched
    tunnel_wait_s: float = 20.0  # at start, wait this long for the public address
    # True: players must enter name + email before they can download the photo.
    require_details: bool = True
    data_dir: str = "sessions/share"  # collages + leads.csv
    marketing_consent_text: str = (
        "Yes, The Scouts may email me about drone security products and events."
    )
    # True: the share link uses a Cloudflare quick tunnel, whose address changes each start.
    privacy_text: str = "Your photo link expires after the event."


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    framer: FramerCfg = field(default_factory=FramerCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    pose: PoseCfg = field(default_factory=PoseCfg)
    game: GameCfg = field(default_factory=GameCfg)
    display: DisplayCfg = field(default_factory=DisplayCfg)
    share: ShareCfg = field(default_factory=ShareCfg)


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
