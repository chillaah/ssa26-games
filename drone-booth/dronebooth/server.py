"""Tiny web server the QR code points at: visitor enters name + email, gets their photo.

Standard library only. Photos live in <data_dir>/photos/<id>.jpg, leads in <data_dir>/leads.csv.
"""
from __future__ import annotations

import csv
import html
import logging
import os
import re
import secrets
import socket
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

import cv2
import numpy as np

from .brand import ASSETS
from .config import ShareCfg
from .tunnel import CloudflareTunnel

log = logging.getLogger(__name__)

PHOTO_ID = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_BODY = 4096


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets are sent for UDP connect
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class PhotoStore:
    def __init__(self, data_dir: str):
        self.dir = Path(data_dir)
        self.photos = self.dir / "photos"
        self.photos.mkdir(parents=True, exist_ok=True)
        self.leads_path = self.dir / "leads.csv"
        self._lock = threading.Lock()

    def save(self, photo: np.ndarray) -> str:
        pid = secrets.token_urlsafe(6)
        cv2.imwrite(str(self.photos / f"{pid}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # Blurred teaser shown above the form, so the full photo needs the details.
        small = cv2.resize(photo, (photo.shape[1] // 4, photo.shape[0] // 4))
        teaser = cv2.GaussianBlur(small, (0, 0), 6)
        cv2.imwrite(str(self.photos / f"{pid}_teaser.jpg"), teaser, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return pid

    def path(self, pid: str, teaser: bool = False) -> Optional[Path]:
        if not PHOTO_ID.match(pid):
            return None
        p = self.photos / (f"{pid}_teaser.jpg" if teaser else f"{pid}.jpg")
        return p if p.exists() else None

    def add_lead(self, pid: str, name: str, email: str, company: str, marketing: bool) -> None:
        with self._lock:
            new = not self.leads_path.exists()
            with open(self.leads_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["time", "photo_id", "name", "email", "company", "marketing_consent"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), pid, name, email,
                            company, int(marketing)])
                fh.flush()
                os.fsync(fh.fileno())  # on disk now, even if the power goes


BRAND_FILE = re.compile(r"^/brand/(fonts|logos)/([A-Za-z0-9-]+\.(ttf|png))$")
BRAND_TYPES = {"ttf": "font/ttf", "png": "image/png"}

# Styled after thescouts.io: white page, Poppins (served from assets/brand so it works offline),
# black headings, grey body copy, light-grey rounded cards and Ebony buttons.
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#ffffff">
<title>Scouts Drone Booth</title>
<style>
  @font-face {{ font-family: Poppins; font-weight: 400; font-display: swap;
               src: url(/brand/fonts/Poppins-Regular.ttf) format("truetype"); }}
  @font-face {{ font-family: Poppins; font-weight: 500; font-display: swap;
               src: url(/brand/fonts/Poppins-Medium.ttf) format("truetype"); }}
  @font-face {{ font-family: Poppins; font-weight: 600; font-display: swap;
               src: url(/brand/fonts/Poppins-SemiBold.ttf) format("truetype"); }}
  @font-face {{ font-family: Poppins; font-weight: 700; font-display: swap;
               src: url(/brand/fonts/Poppins-Bold.ttf) format("truetype"); }}
  :root {{ --ink: #0e0e0e; --ebony: #6d4641; --ebony-dark: #573834; --dutch: #ffefca;
          --powder: #9cadce; --muted: #757575; --card: #f2f2f2; --line: #e0e0e0; --alert: #e5484d; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #fff; color: var(--ink);
         font: 16px/1.5 Poppins, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width: 560px; margin: 0 auto; padding: 20px 18px 40px; }}
  header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 22px; }}
  header img {{ height: 34px; width: auto; border: 0; border-radius: 0; }}
  .badge {{ display: inline-block; background: var(--alert); color: #fff; font-weight: 600;
           letter-spacing: .04em; padding: 5px 12px; font-size: 12px; border-radius: 999px; }}
  h1 {{ margin: 6px 0 4px; font-size: 32px; line-height: 1.15; font-weight: 700; letter-spacing: -0.03em; }}
  p {{ color: var(--muted); margin: 6px 0 18px; }}
  .photo {{ width: 100%; display: block; border-radius: 16px; }}
  form, .card {{ background: var(--card); border-radius: 16px; padding: 18px; margin-top: 18px; }}
  label {{ display: block; font-size: 14px; font-weight: 500; color: var(--ink); margin: 14px 0 6px; }}
  input[type=text], input[type=email] {{ width: 100%; padding: 13px 14px; font: inherit;
           background: #fff; color: var(--ink); border: 1px solid var(--line); border-radius: 12px; }}
  input[type=text]:focus, input[type=email]:focus {{ outline: 2px solid var(--ebony); border-color: transparent; }}
  .check {{ display: flex; gap: 10px; align-items: flex-start; margin-top: 16px; color: var(--muted);
           font-size: 14px; }}
  .check label {{ font-weight: 400; color: var(--muted); }}
  .check input {{ margin-top: 3px; width: 18px; height: 18px; accent-color: var(--ebony); }}
  button, .btn {{ display: block; width: 100%; margin-top: 18px; padding: 14px 18px; font: inherit;
           font-size: 16px; font-weight: 600; letter-spacing: .01em; color: #fff; background: var(--ebony);
           border: 0; border-radius: 12px; text-align: center; text-decoration: none; cursor: pointer; }}
  button:active, .btn:active {{ background: var(--ebony-dark); }}
  .btn.secondary {{ background: #fff; color: var(--ink); border: 1px solid var(--line); }}
  .err {{ color: var(--alert); font-weight: 600; margin-top: 0; }}
  small {{ color: var(--muted); display: block; margin-top: 14px; font-size: 12px; }}
  footer {{ margin-top: 32px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted);
           font-size: 13px; text-align: center; }}
  footer a {{ color: var(--ink); font-weight: 600; text-decoration: none; }}
</style></head>
<body><main>
<header><img src="/brand/logos/horizontal-black.png" alt="Scouts"><span class="badge">THREAT DETECTED</span></header>
{body}
<footer>Scouts &middot; Autonomous Drones &middot; <a href="https://thescouts.io">thescouts.io</a></footer>
</main></body></html>"""


def form_page(pid: str, cfg: ShareCfg, error: str = "", values: Optional[dict] = None) -> str:
    v = {k: html.escape(x) for k, x in (values or {}).items()}
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    required = "required" if cfg.require_details else ""
    heading = ("Enter your details to download your photo." if cfg.require_details
               else "Your photo is ready. Leave your details to stay in touch.")
    return PAGE.format(body=f"""
<h1>Scouts Drone Booth</h1>
<p>{heading}</p>
<img class="photo" src="/img/{pid}_teaser.jpg" alt="Blurred preview of your photo">
<form method="post" action="/p/{pid}">
  {err}
  <label for="name">Name</label>
  <input id="name" name="name" type="text" autocomplete="name" maxlength="100" {required}
         value="{v.get('name', '')}">
  <label for="email">Email</label>
  <input id="email" name="email" type="email" autocomplete="email" maxlength="200" {required}
         value="{v.get('email', '')}">
  <label for="company">Company (optional)</label>
  <input id="company" name="company" type="text" autocomplete="organization" maxlength="100"
         value="{v.get('company', '')}">
  <div class="check"><input id="marketing" name="marketing" type="checkbox" value="1">
    <label for="marketing" style="margin:0">{html.escape(cfg.marketing_consent_text)}</label></div>
  <button type="submit">Get my photo</button>
  <small>{html.escape(cfg.privacy_text)}</small>
</form>""")


def photo_page(pid: str, name: str = "") -> str:
    hello = f"Thanks, {html.escape(name.split()[0])}!" if name.strip() else "Here you go!"
    return PAGE.format(body=f"""
<h1>{hello}</h1>
<p>Press and hold the photo to save it, or use the buttons below.</p>
<img class="photo" src="/img/{pid}.jpg" alt="Your Scouts Drone Booth photo">
<button id="share" type="button">Share</button>
<a class="btn secondary" href="/img/{pid}.jpg" download="scouts-drone-booth.jpg">Download</a>
<script>
  const btn = document.getElementById('share');
  btn.addEventListener('click', async () => {{
    try {{
      const blob = await (await fetch('/img/{pid}.jpg')).blob();
      const file = new File([blob], 'scouts-drone-booth.jpg', {{ type: 'image/jpeg' }});
      if (navigator.canShare && navigator.canShare({{ files: [file] }})) {{
        await navigator.share({{ files: [file], title: 'Scouts Drone Booth',
                                 text: 'I got captured by The Scouts drone at Safety & Security Asia!' }});
      }} else {{ window.location = '/img/{pid}.jpg'; }}
    }} catch (e) {{ /* share sheet dismissed */ }}
  }});
</script>""")


def message_page(title: str, text: str) -> str:
    return PAGE.format(body=f"<h1>{html.escape(title)}</h1><p>{html.escape(text)}</p>")


def make_handler(store: PhotoStore, cfg: ShareCfg):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Dronebooth/1.0"

        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8",
                  cache: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _html(self, code: int, page: str) -> None:
            self._send(code, page.encode())

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            m = BRAND_FILE.match(path)
            if m:
                f = ASSETS / m.group(1) / m.group(2)
                if not f.is_file():
                    return self._html(404, message_page("Not found", ""))
                return self._send(200, f.read_bytes(), BRAND_TYPES[m.group(3)], cache=True)
            if path.startswith("/img/") and path.endswith(".jpg"):
                stem = path[5:-4]
                teaser = stem.endswith("_teaser")
                p = store.path(stem[:-7] if teaser else stem, teaser=teaser)
                if p is None:
                    return self._html(404, message_page("Not found", "That photo doesn't exist."))
                return self._send(200, p.read_bytes(), "image/jpeg", cache=True)
            if path.startswith("/p/"):
                pid = path[3:].strip("/")
                if store.path(pid) is None:
                    return self._html(404, message_page("Photo not found",
                                                        "Scan the QR code on the booth screen again."))
                return self._html(200, form_page(pid, cfg))
            if path == "/":
                return self._html(200, message_page("Scouts Drone Booth",
                                                    "Visit The Scouts booth to get captured by our drone."))
            self._html(404, message_page("Not found", ""))

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            pid = path[3:].strip("/") if path.startswith("/p/") else ""
            if not pid or store.path(pid) is None:
                return self._html(404, message_page("Photo not found", ""))
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._html(413, message_page("Too much data", ""))
            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            values = {k: (form.get(k, [""])[0]).strip()[:200] for k in ("name", "email", "company")}
            marketing = form.get("marketing", [""])[0] == "1"
            name, email = values["name"], values["email"]
            given = bool(name or email)
            if cfg.require_details or given:
                if not name:
                    return self._html(400, form_page(pid, cfg, "Please enter your name.", values))
                if not EMAIL.match(email):
                    return self._html(400, form_page(pid, cfg, "Please enter a valid email.", values))
                store.add_lead(pid, name, email, values["company"], marketing)
                log.info("lead captured for photo %s", pid)
            self._html(200, photo_page(pid, name))

    return Handler


class ShareServer:
    def __init__(self, store: PhotoStore, cfg: ShareCfg):
        self.cfg = cfg
        self.httpd = ThreadingHTTPServer((cfg.host, cfg.port), make_handler(store, cfg))
        self.httpd.daemon_threads = True
        port = self.httpd.server_address[1]
        self._fixed_url = cfg.public_base_url.rstrip("/")
        self._lan_url = f"http://{lan_ip()}:{port}"
        # share.tunnel: "cloudflare" -> a public https address so phones on any network work.
        self.tunnel = (CloudflareTunnel(port, cfg.cloudflared)
                       if cfg.tunnel == "cloudflare" and not self._fixed_url else None)
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="share", daemon=True)

    @property
    def base_url(self) -> str:
        """public_base_url if set, else the tunnel's address once it is up, else the LAN address."""
        if self._fixed_url:
            return self._fixed_url
        if self.tunnel is not None and self.tunnel.url:
            return self.tunnel.url
        return self._lan_url

    def start(self) -> "ShareServer":
        self._thread.start()
        if self.tunnel is not None:
            self.tunnel.start()
            if not self.tunnel.wait(self.cfg.tunnel_wait_s):
                log.warning("No public address yet; QR codes use %s until the tunnel is up",
                            self._lan_url)
        log.info("Share server at %s", self.base_url)
        return self

    def url_for(self, pid: str) -> str:
        return f"{self.base_url}/p/{pid}"

    def stop(self) -> None:
        if self.tunnel is not None:
            self.tunnel.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
