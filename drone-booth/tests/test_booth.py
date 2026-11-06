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


def test_compose_is_3_by_4_portrait_around_the_people():
    from dronebooth.compose import portrait_crop

    frame = np.full((720, 1280, 3), 120, np.uint8)
    meta = PhotoMeta("0001-01", "2026-11-04 10:00:00 SGT", "EVENT", "thescouts.io")
    assert compose_photo(frame, [_person()], meta).shape == (1440, 1080, 3)
    # Someone near the right edge stays in the picture, box and mask moved with the crop.
    mask = np.zeros((720, 1280), bool)
    mask[100:700, 1100:1250] = True
    img, moved = portrait_crop(frame, [Person((1100, 100, 1250, 700), 0.9, mask)])
    assert img.shape == (1440, 1080, 3) and len(moved) == 1
    x1, y1, x2, y2 = moved[0].bbox
    assert 0 <= x1 < x2 <= 1080 and moved[0].mask[(y1 + y2) // 2, (x1 + x2) // 2]


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


def test_review_stays_up_until_space():
    from dronebooth.booth import LIVE, REVIEW, Booth

    class Scanner:
        people = []

    booth = Booth(load_booth_config("configs/mac.yaml"), Scanner(), None, None)
    booth.photo = np.zeros((1440, 1080, 3), np.uint8)
    booth.url, booth.qr = "http://example/p/x", np.zeros((440, 440, 3), np.uint8)
    booth._set(REVIEW, 0.0)
    frame = np.zeros((720, 1280, 3), np.uint8)
    booth.step(3600.0, frame)  # an hour later: no timer
    assert booth.state == REVIEW
    booth.on_key(ord(" "), 3601.0)
    assert booth.state == LIVE
