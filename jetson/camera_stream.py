#!/usr/bin/env python3
"""Check / set the SIYI A8 mini main video stream (what both games read over RTSP).

    python3 jetson/camera_stream.py                 # show it
    python3 jetson/camera_stream.py 1920 1080 4000  # make sure it is 1080p H.265 at 4000 kbps

Standard library only. Talks to the camera's SDK port (UDP 37260); the setting is only sent
if it differs, so running it before every game start is harmless. Exits 0 even if the camera
doesn't answer (the game itself shows "Waiting for camera..." then).
"""
import socket
import struct
import sys
import time

CAMERA = ("192.168.144.25", 37260)
GET_ENCODING, SET_ENCODING = 0x20, 0x21
MAIN_STREAM, H265 = 1, 2


def crc16_xmodem(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else crc << 1
            crc &= 0xFFFF
    return crc


def packet(cmd: int, data: bytes) -> bytes:
    head = struct.pack("<BBBHHB", 0x55, 0x66, 1, len(data), 0, cmd) + data
    return head + struct.pack("<H", crc16_xmodem(head))


def ask(sock: socket.socket, cmd: int, data: bytes):
    sock.sendto(packet(cmd, data), CAMERA)
    end = time.time() + 2.0
    while time.time() < end:
        try:
            buf = sock.recv(1024)
        except socket.timeout:
            break
        if len(buf) >= 10 and buf[:2] == b"\x55\x66" and buf[7] == cmd:
            return buf[8:8 + struct.unpack("<H", buf[3:5])[0]]
    return None


def main() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    cur = ask(sock, GET_ENCODING, bytes([MAIN_STREAM]))
    if cur is None or len(cur) < 8:
        print("camera_stream: no answer from the camera at %s:%d" % CAMERA)
        return 0
    w, h = struct.unpack("<HH", cur[2:6])
    kbps = struct.unpack("<H", cur[6:8])[0]
    print("camera_stream: main stream %dx%d, %d kbps, %s" % (w, h, kbps, "H.265" if cur[1] == 2 else "H.264"))
    if len(sys.argv) == 4:
        want = tuple(int(x) for x in sys.argv[1:4])
        if (w, h, kbps) != want or cur[1] != H265:
            ask(sock, SET_ENCODING, struct.pack("<BBHHHB", MAIN_STREAM, H265, *want, 0))
            time.sleep(2.0)  # the camera restarts the stream
            new = ask(sock, GET_ENCODING, bytes([MAIN_STREAM]))
            if new and len(new) >= 8:
                print("camera_stream: set to %dx%d, %d kbps" % (struct.unpack("<HH", new[2:6])
                                                                 + (struct.unpack("<H", new[6:8])[0],)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
