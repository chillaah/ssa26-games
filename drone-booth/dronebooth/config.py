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
    mirror: bool = True  # selfie view on a webcam


@dataclass
class DetectorCfg:
    model: str = "yolo11n-seg.pt"  # use a TensorRT .engine on the Jetson
    device: Optional[str] = None  # None = auto, "mps", "cpu", "0" (CUDA)
    imgsz: int = 640
    conf: float = 0.55  # bags scored 48%, real people 59%+ in booth tests
    half: bool = False


@dataclass
class DisplayCfg:
    window_name: str = "Drone Booth - The Scouts"
    fullscreen: bool = False
    width: int = 1920
    height: int = 1080
    # QR codes for The Scouts' socials, in the bottom corners of the live screen (same as
    # Out of Frame's attract screen). icon = assets/brand/logos/<icon>.png.
    socials: list = field(default_factory=lambda: [
        {"name": "Instagram", "handle": "@thescouts.io",
         "url": "https://www.instagram.com/thescouts.io/", "qr": "assets/qr_instagram.png",
         "icon": "instagram"},
        {"name": "LinkedIn", "handle": "Scouts - Autonomous Drones",
         "url": "https://www.linkedin.com/company/scoutsapp/posts/", "qr": "assets/qr_linkedin.png",
         "icon": "linkedin"},
    ])


@dataclass
class BoothCfg:
    countdown_s: float = 3.0
    flash_s: float = 0.4
    max_subjects: int = 6  # label at most this many people per photo
    event: str = "SAFETY & SECURITY ASIA 2026"
    website: str = "thescouts.io"
    timezone_label: str = "SGT"
    data_dir: str = "booth_data"


@dataclass
class ShareCfg:
    host: str = "0.0.0.0"
    port: int = 8080
    # Leave empty to use this machine's LAN address. Set it when phones reach the
    # booth through something else, e.g. a tunnel: "https://booth.example.com".
    public_base_url: str = ""
    # "cloudflare": publish the download page at a public https address (Cloudflare quick
    # tunnel, needs internet and the cloudflared program) so any phone can open the QR code.
    # "" = local network only. Ignored when public_base_url is set.
    tunnel: str = ""
    cloudflared: str = "cloudflared"  # path to the program; ~/.local/bin is also searched
    tunnel_wait_s: float = 20.0  # at start, wait this long for the public address
    # True: visitors must enter name + email before they can download the photo.
    require_details: bool = True
    marketing_consent_text: str = (
        "Yes, The Scouts may email me about drone security products and events."
    )
    # True: the share link uses a Cloudflare quick tunnel, whose address changes each start.
    privacy_text: str = "Your photo link expires after the event."


@dataclass
class BoothConfig:
    camera: CameraCfg = field(default_factory=CameraCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    display: DisplayCfg = field(default_factory=DisplayCfg)
    booth: BoothCfg = field(default_factory=BoothCfg)
    share: ShareCfg = field(default_factory=ShareCfg)


def _merge(obj: Any, data: dict, path: str = "") -> Any:
    names = {f.name for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in names:
            raise KeyError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)
    return obj


def load_booth_config(path: Optional[Union[str, Path]] = None) -> BoothConfig:
    cfg = BoothConfig()
    if path:
        with open(path) as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    return cfg
