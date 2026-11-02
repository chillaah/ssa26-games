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
