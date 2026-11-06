import random

import numpy as np

from outofframe.config import VirtualFramerCfg, load_config
from outofframe.detector import Person, judge
from outofframe.framer import VirtualFramer
from outofframe.siyi import build_packet, parse_packet


def test_siyi_center_packet_matches_sdk_doc():
    # "Auto centering" example from the SIYI SDK protocol document.
    assert build_packet(0x08, b"\x01", seq=0).hex(" ") == "55 66 01 01 00 00 00 08 01 d1 12"


def test_siyi_roundtrip():
    pkt = build_packet(0x0E, b"\x2c\x01\x9c\xff", seq=7)
    assert parse_packet(pkt) == (0x0E, b"\x2c\x01\x9c\xff")
    assert parse_packet(pkt[:-1] + b"\x00") is None


def _person(mask):
    return Person((0, 0, 1, 1), 0.9, mask)


def test_judge_caught_and_clear():
    m = np.zeros((100, 100), bool)
    m[:, 0:20] = True  # body on the left edge
    assert judge([_person(m)], (50, 0, 100, 100), 0.003, 10, True).caught is False
    assert judge([_person(m)], (10, 0, 60, 100), 0.003, 10, True).caught is True


def test_judge_empty_scene_counts_as_caught_when_required():
    v = judge([], (0, 0, 50, 50), 0.003, 10, True)
    assert v.caught and v.no_player
    assert judge([], (0, 0, 50, 50), 0.003, 10, False).caught is False


def test_virtual_framer_stays_in_bounds_and_finishes():
    f = VirtualFramer(VirtualFramerCfg(), random.Random(1))
    size = (1280, 720)
    for r in range(5):
        f.start_round(0.0, r, 5, size)
        t, done = 0.0, False
        while not done:
            (x1, y1, x2, y2), done = f.update(t, size)
            assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720
            t += 0.05
        assert t < 10


def test_configs_load():
    assert load_config("configs/mac.yaml").framer.mode == "virtual"
    assert load_config("configs/jetson.yaml").framer.mode == "siyi"
    follow = load_config("configs/jetson_follow.yaml").framer
    assert follow.mode == "virtual" and follow.virtual.follow_gimbal


class _RecordingGimbal:
    def __init__(self):
        self.sent = []

    def set_angles(self, yaw, pitch):
        self.sent.append((yaw, pitch))

    def center(self):
        pass

    def stop(self):
        pass


def test_following_framer_moves_box_exactly_like_virtual_and_leans_gimbal():
    from outofframe.config import SiyiFramerCfg
    from outofframe.framer import FollowingFramer

    cfg, size = VirtualFramerCfg(), (1280, 720)
    plain = VirtualFramer(cfg, random.Random(3))
    gimbal = _RecordingGimbal()
    follow = FollowingFramer(cfg, SiyiFramerCfg(home_yaw=0, home_pitch=-10), mirror=True,
                             rng=random.Random(3), gimbal=gimbal)
    for f in (plain, follow):
        f.start_round(0.0, 0, 5, size)
    t = 0.0
    while t < 6:
        assert plain.update(t, size) == follow.update(t, size)  # the game sees the same box
        t += 0.02
    assert 10 <= len(gimbal.sent) <= 6 * cfg.follow_hz + 1  # rate-limited
    for yaw, pitch in gimbal.sent:
        assert abs(yaw) <= cfg.follow_yaw_deg + 1e-9
        assert abs(pitch + 10) <= cfg.follow_pitch_deg + 1e-9


def test_following_framer_lean_direction():
    from outofframe.config import SiyiFramerCfg
    from outofframe.framer import FollowingFramer

    def lean(center, mirror, **kw):
        f = FollowingFramer(VirtualFramerCfg(**kw), SiyiFramerCfg(home_yaw=0, home_pitch=0),
                            mirror=mirror, gimbal=_RecordingGimbal())
        f.center = center
        return f.lean()

    assert lean((0.5, 0.5), True) == (0.0, 0.0)
    right_mirrored, right_raw = lean((0.9, 0.5), True), lean((0.9, 0.5), False)
    assert right_mirrored[0] == 6.0 and right_raw[0] == -6.0
    assert lean((0.9, 0.5), True, follow_invert_yaw=True)[0] == -6.0
    assert lean((0.5, 0.9), True)[1] == -4.0  # box low on screen -> tilt down


def test_split_packets_handles_fragments_and_noise():
    from outofframe.siyi import split_packets

    a, b = build_packet(0x0D, b"\x01\x02\x03\x04\x05\x06"), build_packet(0x08, b"\x01")
    stream = b"\x00\x55junk" + a + b
    buf, got = bytearray(), []
    for i in range(0, len(stream), 3):  # arrive in 3-byte chunks, like a slow UART
        buf += stream[i:i + 3]
        got += split_packets(buf)
    assert got == [a, b]


def _fake_gimbal(read, write):
    """Answers attitude requests with the last commanded angles."""
    import struct
    import threading

    from outofframe.siyi import split_packets

    state = {"att": (0, 0)}

    def run():
        buf = bytearray()
        while True:
            chunk = read()
            if chunk is None:
                return
            buf += chunk
            for pkt in split_packets(buf):
                cmd, data = parse_packet(pkt)
                if cmd == 0x0E:
                    state["att"] = struct.unpack("<hh", data)
                elif cmd == 0x0D:
                    write(build_packet(0x0D, struct.pack("<hhhhhh", *state["att"], 0, 0, 0, 0),
                                       need_ack=False))

    threading.Thread(target=run, daemon=True).start()


def _drive(framer):
    import time

    framer.start_round(time.monotonic(), 0, 3, (1280, 720))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, done = framer.update(time.monotonic(), (1280, 720))
        if done:
            return framer.gimbal.attitude
        time.sleep(0.01)
    raise AssertionError("gimbal never settled")


def test_siyi_framer_over_udp():
    import socket

    from outofframe.config import SiyiFramerCfg
    from outofframe.framer import SiyiFramer

    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    peer = {}

    def read():
        data, peer["addr"] = srv.recvfrom(1024)
        return data

    _fake_gimbal(read, lambda pkt: srv.sendto(pkt, peer["addr"]))
    f = SiyiFramer(SiyiFramerCfg(link="udp", ip="127.0.0.1", port=srv.getsockname()[1]),
                   random.Random(0))
    try:
        assert math_close(_drive(f)[:2], f.targets[-1])
    finally:
        f.close()


def test_siyi_framer_over_serial():
    import os
    import tty

    from outofframe.config import SiyiFramerCfg
    from outofframe.framer import SiyiFramer

    master, slave = os.openpty()
    tty.setraw(master)
    tty.setraw(slave)
    _fake_gimbal(lambda: os.read(master, 256), lambda pkt: os.write(master, pkt))
    f = SiyiFramer(SiyiFramerCfg(link="serial", serial_port=os.ttyname(slave)), random.Random(0))
    try:
        assert math_close(_drive(f)[:2], f.targets[-1])
    finally:
        f.close()


def math_close(a, b, tol=0.15):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_steady_view_cancels_gimbal_shift():
    import cv2

    from outofframe.steady import SteadyView

    class Gimbal:
        attitude, attitude_t = (-2.4, 180.0, 0.0), 1.0

    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.random((720, 1280)).astype(np.float32), (15, 15), 4)
    def steadied(cfg, dyaw, dpitch, mirror):
        k = cfg.steady_px_per_deg
        g = Gimbal()
        view = SteadyView(g, cfg, (0, 0), mirror)
        view.ref = (-2.4, 180.0)
        g.attitude = (-2.4 + dyaw, wrap(180.0 + dpitch), 0.0)
        # +yaw/+pitch slides the unmirrored image right/down, or left/up when inverted
        sy = 1 if cfg.follow_invert_pitch else -1
        sx = 1 if cfg.follow_invert_yaw else -1
        raw = np.roll(scene, (-sy * round(k * dpitch), -sx * round(k * dyaw)), axis=(0, 1))
        img = cv2.flip(raw, 1) if mirror else raw
        return view.apply(img, 1.0)

    def wrap(a):
        return (a + 180) % 360 - 180

    for invert in (False, True):
        cfg = VirtualFramerCfg(steady_latency_s=0.0, follow_invert_yaw=invert, follow_invert_pitch=invert)
        for mirror in (False, True):
            still = steadied(cfg, 0, 0, mirror)
            for dyaw, dpitch in [(5, 0), (-4, 3), (0, -3.5)]:
                (dx, dy), _ = cv2.phaseCorrelate(still, steadied(cfg, dyaw, dpitch, mirror))
                assert abs(dx) < 2 and abs(dy) < 2, (invert, mirror, dyaw, dpitch, dx, dy)


def _pose_kpts(**over):
    """COCO-17 keypoints of a standing figure, with some joints moved."""
    k = {0: (300, 150), 5: (350, 235), 6: (250, 235), 7: (372, 330), 8: (228, 330), 9: (378, 420),
         10: (222, 420), 11: (330, 420), 12: (270, 420), 13: (335, 560), 14: (265, 560),
         15: (340, 700), 16: (260, 700)}
    k.update({int(i[1:]): v for i, v in over.items()})
    return np.array([k.get(i, (0, 0)) for i in range(17)], float), \
        np.array([1.0 if i in k else 0.0 for i in range(17)])


POSES = {
    "t_pose": dict(j7=(445, 235), j8=(155, 235), j9=(540, 235), j10=(60, 235)),
    "hands_up": dict(j7=(385, 140), j8=(215, 140), j9=(395, 45), j10=(205, 45)),
    "star_jump": dict(j7=(430, 150), j8=(170, 150), j9=(505, 70), j10=(95, 70), j13=(395, 555),
                      j14=(205, 555), j15=(455, 690), j16=(145, 690)),
    "squat": dict(j11=(340, 500), j12=(260, 500), j13=(440, 560), j14=(160, 560), j15=(410, 700),
                  j16=(190, 700)),
    "disco": dict(j8=(215, 140), j10=(170, 45), j7=(420, 330), j9=(345, 410)),
}


def test_pose_similarity_and_gallery_pick(tmp_path):
    import json

    import cv2

    from outofframe.pose import Gallery, Pose, mirror, similarity

    index = {}
    for name, over in POSES.items():
        k, c = _pose_kpts(**over)
        cv2.imwrite(str(tmp_path / f"{name}.png"), np.zeros((8, 8, 3), np.uint8))
        index[f"{name}.png"] = {"manual": True, "kpts": k.tolist(), "conf": c.tolist(),
                                "bbox": [0, 0, 600, 800]}
    (tmp_path / "index.json").write_text(json.dumps(index))

    def pose(name):
        return Pose(*_pose_kpts(**POSES[name]), (0, 0, 600, 800))

    t, up = pose("t_pose"), pose("hands_up")
    assert similarity(t, t) > 0.99
    assert similarity(t, up) < 0.8
    assert similarity(t, Pose(t.kpts * 2 + 100, t.conf, t.bbox)) > 0.99  # position/size don't matter

    gallery = Gallery.load(tmp_path, None)
    assert len(gallery.items) == len(POSES)
    for name in ("star_jump", "squat", "disco"):
        p = pose(name)
        noisy = Pose(p.kpts + np.random.default_rng(0).normal(0, 6, p.kpts.shape), p.conf, p.bbox)
        item, score, _ = gallery.best(noisy)
        assert item.path.stem == name and score > 0.9
    d = pose("disco")
    item, _, mirrored = gallery.best(Pose(*mirror(d.kpts, d.conf), d.bbox))
    assert item.path.stem == "disco" and mirrored
    # Excluded pictures are never picked again.
    item, _, _ = gallery.best(d, exclude={gallery.folder / "disco.png"})
    assert item.path.stem != "disco"


def test_brand_text_and_logo_draw():
    from outofframe import brand as B

    img = np.zeros((200, 600, 3), np.uint8)
    w, cap = B.text_size("OUT OF FRAME", 1.0, B.BOLD)
    assert 150 < w < 300 and 15 < cap < 30  # about as wide as the old Hershey text
    B.text(img, "OUT OF FRAME", (300, 60), 1.0, B.DUTCH, font=B.BOLD)
    assert img[40:80, 150:450].max() > 200
    B.text(img, "edge", (-50, -50), 2.0)  # off-canvas text is clipped, not an error
    x1, y1, x2, y2 = B.logo(img, "horizontal-white", (300, 150), 60)
    assert y2 - y1 == 60 and img[120:180, x1:x2].max() > 200


def test_close_button_needs_two_taps():
    import cv2

    from outofframe import window as W

    win = W.GameWindow.__new__(W.GameWindow)  # no real window needed for the tap logic
    win.close_requested, win._armed_until = False, 0.0
    img = np.zeros((720, 1280, 3), np.uint8)
    win._draw_close(img)
    cx, cy, r = win._btn
    assert cx > 1200 and cy < 80 and img[cy, cx].any()
    win._on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)  # elsewhere: ignored
    assert not win.armed and not win.close_requested
    win._on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0, None)  # first tap arms
    assert win.armed and not win.close_requested
    win._on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0, None)  # second tap closes
    assert win.close_requested


def test_four_dodge_rounds_then_bonus_pose_then_collage_and_qr(tmp_path):
    import json
    import urllib.request

    import cv2

    from outofframe import game as G
    from outofframe.config import Config
    from outofframe.pose import Gallery
    from outofframe.share import PhotoStore, ShareServer

    gal = tmp_path / "gallery"
    gal.mkdir()
    k, c = _pose_kpts()
    for name in ("ninja_1", "robot_1"):
        cv2.imwrite(str(gal / f"{name}.jpg"), np.full((400, 300, 3), 200, np.uint8))
    (gal / "index.json").write_text(json.dumps({
        f"{n}.jpg": {"manual": True, "kpts": k.tolist(), "conf": c.tolist(), "bbox": [0, 0, 300, 400]}
        for n in ("ninja_1", "robot_1")}))

    class Detector:  # one person standing far left, never in the box's middle
        def detect(self, frame):
            mask = np.zeros(frame.shape[:2], bool)
            mask[200:600, 0:40] = True
            return [Person((0, 200, 40, 600), 0.9, mask)]

    cfg = Config()
    cfg.game.save_dir = str(tmp_path / "sessions")
    cfg.share.data_dir = str(tmp_path / "share")
    cfg.share.port = 0  # any free port
    share = ShareServer(PhotoStore(cfg.share.data_dir), cfg.share).start()
    try:
        game = G.Game(cfg, VirtualFramer(cfg.framer.virtual, random.Random(1)), Detector(),
                      rng=random.Random(2), gallery=Gallery.load(gal, None), share=share)
        frame = np.full((720, 1280, 3), 90, np.uint8)
        game.start(0.0)
        seen, t = [], 0.0
        while game.state not in (G.RESULTS,) and t < 120:
            t += 0.05
            game.step(t, frame)
            if not seen or seen[-1] != game.state:
                seen.append(game.state)
            if game.state == G.REVEAL:
                game.on_key(ord(" "), t)
        assert len(game.shots) == 4
        assert seen.count(G.LOCKING) == 4 and G.POSE not in seen
        assert seen.index(G.BONUS) > max(i for i, s in enumerate(seen) if s == G.LOCKING)
        assert game.bonus_item is not None and game.bonus_photo is not None
        assert game.collage.shape == (1920, 1080, 3)
        assert game.share_qr is not None and game.share_url.startswith(share.base_url)
        pid = game.share_url.rsplit("/", 1)[1]
        assert share.store.path(pid) is not None
        page = urllib.request.urlopen(f"http://127.0.0.1:{share.httpd.server_address[1]}/p/{pid}")
        assert b"Out of Frame" in page.read()
        assert (game.session_dir / "collage.jpg").exists()
        game.step(t + 0.05, frame)  # results screen draws
        game.step(t + 600, frame)  # ...and stays up, no timer
        assert game.state == G.RESULTS
        game.on_key(ord(" "), t + 601)  # SPACE: back to the start screen
        assert game.state == G.ATTRACT
        game.on_key(ord(" "), t + 602)  # SPACE again: new game
        assert game.state == G.GET_READY
        game.on_key(ord("q"), t + 603)
        assert game.quit
    finally:
        share.stop()


def test_bonus_crop_frames_the_players_not_the_background():
    from outofframe.collage import BONUS_ASPECT, bonus_crop, players

    near, also_near, far = (900, 100, 1200, 720), (600, 150, 880, 720), (100, 300, 180, 450)
    assert players([near, also_near, far]) == [near, also_near]
    frame = np.zeros((720, 1280, 3), np.uint8)
    crop = bonus_crop(frame, players([near, far]))
    assert abs(crop.shape[1] / crop.shape[0] - BONUS_ASPECT) < 0.01
    # A group wider than the photo: centred on the biggest person (marked white here).
    frame[:, 600:1270] = 255
    crop = bonus_crop(frame, [(560, 0, 1280, 720), (0, 280, 190, 650)])
    assert crop[:, -50:].mean() > 200


def test_bonus_round_only_picks_listed_doable_pictures(tmp_path):
    import json

    import cv2

    from outofframe import game as G
    from outofframe.config import Config
    from outofframe.pose import Gallery

    k, c = _pose_kpts()
    names = ("tree_pose_1", "gymnast_2", "scarecrow_1")
    for n in names:
        cv2.imwrite(str(tmp_path / f"{n}.jpg"), np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "index.json").write_text(json.dumps({f"{n}.jpg": {
        "manual": True, "kpts": k.tolist(), "conf": c.tolist(), "bbox": [0, 0, 8, 8]} for n in names}))
    (tmp_path / "bonus.txt").write_text("# doable\ntree_pose_1.jpg\n\nscarecrow_1.jpg\n")
    gallery = Gallery.load(tmp_path, None)
    assert len(gallery.items) == 3
    assert {i.path.stem for i in gallery.bonus_items} == {"tree_pose_1", "scarecrow_1"}
    game = G.Game(Config(), VirtualFramer(VirtualFramerCfg()), None, rng=random.Random(0),
                  gallery=gallery)
    assert {game._pick_bonus().path.stem for _ in range(20)} == {"tree_pose_1", "scarecrow_1"}


def test_shipped_bonus_list_matches_the_gallery():
    from pathlib import Path

    from outofframe.pose import Gallery

    gallery = Gallery.load(Path(__file__).resolve().parent.parent / "gallery", None)
    assert 40 <= len(gallery.bonus_items) < len(gallery.items)
    assert not any("gymnast" in i.path.stem or "handstand" in i.path.stem for i in gallery.bonus_items)
