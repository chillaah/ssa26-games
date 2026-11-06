import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from dronebooth.compose import PhotoMeta, compose_photo
from dronebooth.config import ShareCfg, load_booth_config
from dronebooth.detector import Person
from dronebooth.server import PhotoStore, ShareServer


def _person():
    mask = np.zeros((720, 1280), bool)
    mask[200:700, 500:700] = True
    return Person((500, 200, 700, 700), 0.9, mask)


def test_configs_load():
    assert load_booth_config("configs/mac.yaml").share.require_details is True
    assert load_booth_config("configs/jetson.yaml").detector.model.endswith(".engine")


def test_compose_keeps_frame_size():
    frame = np.full((720, 1280, 3), 120, np.uint8)
    meta = PhotoMeta("0001-01", "2026-11-04 10:00:00 SGT", "EVENT", "thescouts.io")
    assert compose_photo(frame, [_person()], meta).shape == frame.shape


def test_share_flow(tmp_path):
    store = PhotoStore(str(tmp_path))
    server = ShareServer(store, ShareCfg(host="127.0.0.1", port=0)).start()
    try:
        pid = store.save(np.zeros((720, 1280, 3), np.uint8))
        base = f"http://127.0.0.1:{server.httpd.server_address[1]}"
        assert "Get my photo" in urllib.request.urlopen(f"{base}/p/{pid}").read().decode()

        def post(data):
            body = urllib.parse.urlencode(data).encode()
            try:
                return urllib.request.urlopen(f"{base}/p/{pid}", body).status
            except urllib.error.HTTPError as e:
                return e.code

        assert post({"name": "Ann", "email": "nope"}) == 400
        assert post({"name": "Ann Tan", "email": "ann@example.com"}) == 200
        assert "ann@example.com" in (tmp_path / "leads.csv").read_text()
        assert urllib.request.urlopen(f"{base}/img/{pid}.jpg").headers["Content-Type"] == "image/jpeg"
        font = urllib.request.urlopen(f"{base}/brand/fonts/Poppins-Bold.ttf")
        assert font.headers["Content-Type"] == "font/ttf" and len(font.read()) > 10000
        assert "horizontal-black.png" in urllib.request.urlopen(f"{base}/p/{pid}").read().decode()
        for bad in ("/brand/fonts/../../dronebooth/config.py", "/brand/fonts/Nope.ttf"):
            try:
                urllib.request.urlopen(base + bad)
                assert False, "expected 404"
            except urllib.error.HTTPError as e:
                assert e.code == 404
        try:
            urllib.request.urlopen(f"{base}/p/unknown123")
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.stop()
