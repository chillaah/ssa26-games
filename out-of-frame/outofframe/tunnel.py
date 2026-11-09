"""Public HTTPS address for the photo download page, so phones on any network (mobile data,
venue Wi-Fi) can open the QR code.

Runs a Cloudflare quick tunnel: `cloudflared tunnel --url http://127.0.0.1:<port>`. No account
or token is needed, and visitors see no warning page. The address (https://<random>.trycloudflare.com)
is read from cloudflared's log and changes each time the tunnel starts; QR codes are made from
the current address, so that's fine, but links from before a restart stop working. If
cloudflared exits it is restarted. The machine needs internet; without it the LAN address is used.

For a fixed address on your own domain, run a named Cloudflare tunnel (or any reverse proxy)
yourself and set share.public_base_url instead. The same file is in both projects; keep them in sync.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import threading
from typing import Optional

log = logging.getLogger(__name__)

URL = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")


def find_cloudflared(binary: str = "cloudflared") -> Optional[str]:
    for cand in (binary, os.path.expanduser("~/.local/bin/cloudflared"), "/usr/local/bin/cloudflared"):
        path = shutil.which(os.path.expanduser(cand))
        if path:
            return path
    return None


def _die_with_parent() -> None:
    """Linux: have the kernel stop cloudflared if the game dies without cleaning up."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


class CloudflareTunnel:
    def __init__(self, port: int, binary: str = "cloudflared"):
        self.port = port
        self.binary = binary
        self.url: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tunnel", daemon=True)

    def start(self) -> "CloudflareTunnel":
        self._thread.start()
        return self

    def wait(self, timeout: float) -> Optional[str]:
        """Wait up to timeout seconds for the public address; returns it or None."""
        self._ready.wait(timeout)
        return self.url

    def _run(self) -> None:
        path = find_cloudflared(self.binary)
        if path is None:
            log.error("cloudflared not found; the QR codes will use the local network address. "
                      "Install it (see README) or set share.tunnel: \"\"")
            self._ready.set()
            return
        backoff = 2.0
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    [path, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                    start_new_session=True, preexec_fn=_die_with_parent if os.name == "posix" else None)
            except OSError as exc:
                log.error("Could not start cloudflared: %s", exc)
                self._ready.set()
                return
            for line in self._proc.stdout:
                m = URL.search(line)
                if m and m.group(0) != self.url:
                    self.url = m.group(0)
                    log.info("Public photo address: %s", self.url)
                    self._ready.set()
                    backoff = 2.0
            self._proc.wait()
            self.url = None
            if self._stop.is_set():
                break
            log.warning("cloudflared stopped (exit %s); restarting in %.0fs", self._proc.returncode, backoff)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 60.0)
        self._ready.set()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._thread.join(timeout=2)
