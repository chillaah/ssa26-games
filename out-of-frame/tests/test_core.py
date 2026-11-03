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
    cfg = VirtualFramerCfg(steady_latency_s=0.0)
    k = cfg.steady_px_per_deg

    def steadied(dyaw, dpitch, mirror):
        g = Gimbal()
        view = SteadyView(g, cfg, (0, 0), mirror)
        view.ref = (-2.4, 180.0)
        g.attitude = (-2.4 + dyaw, wrap(180.0 + dpitch), 0.0)
        # +yaw/+pitch slides the unmirrored image left/up
        raw = np.roll(scene, (-round(k * dpitch), -round(k * dyaw)), axis=(0, 1))
        img = cv2.flip(raw, 1) if mirror else raw
        return view.apply(img, 1.0)

    def wrap(a):
        return (a + 180) % 360 - 180

    for mirror in (False, True):
        still = steadied(0, 0, mirror)
        for dyaw, dpitch in [(5, 0), (-4, 3), (0, -3.5)]:
            (dx, dy), _ = cv2.phaseCorrelate(still, steadied(dyaw, dpitch, mirror))
            assert abs(dx) < 2 and abs(dy) < 2, (mirror, dyaw, dpitch, dx, dy)


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
