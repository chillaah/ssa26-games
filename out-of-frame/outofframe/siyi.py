"""Minimal SIYI gimbal SDK client (A8 mini) over UDP (Ethernet) or a serial UART.

Packet layout (little endian), per the SIYI SDK protocol doc:
    STX(0x55 0x66) | CTRL(1) | DATA_LEN(2) | SEQ(2) | CMD_ID(1) | DATA(n) | CRC16(2)
CRC16 is CRC-16/XMODEM (poly 0x1021, init 0) over everything before it.
The same packets are used over Ethernet and over the TTL serial port (115200 8N1).
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import List, Optional, Tuple, Union

log = logging.getLogger(__name__)

CMD_FIRMWARE = 0x01
CMD_CENTER = 0x08
CMD_ATTITUDE = 0x0D
CMD_SET_ANGLES = 0x0E

# Physical limits of the A8 mini
YAW_LIMITS = (-135.0, 135.0)
PITCH_LIMITS = (-90.0, 25.0)


def crc16_xmodem(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def build_packet(cmd_id: int, data: bytes = b"", seq: int = 0, need_ack: bool = True) -> bytes:
    head = struct.pack("<BBBHHB", 0x55, 0x66, 1 if need_ack else 0, len(data), seq & 0xFFFF, cmd_id)
    body = head + data
    return body + struct.pack("<H", crc16_xmodem(body))


def parse_packet(pkt: bytes) -> Optional[Tuple[int, bytes]]:
    """Return (cmd_id, data) for a valid packet, else None."""
    if len(pkt) < 10 or pkt[0] != 0x55 or pkt[1] != 0x66:
        return None
    (data_len,) = struct.unpack_from("<H", pkt, 3)
    end = 8 + data_len
    if len(pkt) < end + 2:
        return None
    (crc,) = struct.unpack_from("<H", pkt, end)
    if crc != crc16_xmodem(pkt[:end]):
        return None
    return pkt[7], pkt[8:end]


def split_packets(buf: bytearray) -> List[bytes]:
    """Pull complete packets off the front of a byte stream (serial delivers partial reads)."""
    out = []
    while True:
        start = buf.find(b"\x55\x66")
        if start < 0:
            del buf[:max(len(buf) - 1, 0)]  # keep a trailing 0x55 in case 0x66 follows
            return out
        del buf[:start]
        if len(buf) < 10:
            return out
        (data_len,) = struct.unpack_from("<H", buf, 3)
        total = 10 + data_len
        if data_len > 512:  # corrupt length: skip this header
            del buf[:2]
            continue
        if len(buf) < total:
            return out
        out.append(bytes(buf[:total]))
        del buf[:total]


class UdpLink:
    def __init__(self, ip: str, port: int, timeout: float = 0.05):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def send(self, pkt: bytes) -> None:
        self.sock.sendto(pkt, self.addr)

    def recv(self) -> bytes:
        try:
            return self.sock.recvfrom(1024)[0]
        except (socket.timeout, OSError):
            return b""

    def close(self) -> None:
        self.sock.close()


class SerialLink:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05):
        import serial  # pyserial, only needed for the UART connection

        self.ser = serial.Serial(port, baud, timeout=timeout)

    def send(self, pkt: bytes) -> None:
        self.ser.write(pkt)

    def recv(self) -> bytes:
        return self.ser.read(max(1, self.ser.in_waiting))

    def close(self) -> None:
        self.ser.close()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SiyiGimbal:
    def __init__(self, link: Union[UdpLink, SerialLink]):
        self.link = link
        self._seq = 0
        self._lock = threading.Lock()
        self.attitude: Optional[Tuple[float, float, float]] = None  # yaw, pitch, roll (deg)
        self.attitude_t = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _send(self, cmd_id: int, data: bytes = b"") -> None:
        with self._lock:
            self._seq = (self._seq + 1) & 0xFFFF
            pkt = build_packet(cmd_id, data, self._seq)
        self.link.send(pkt)

    # ---- commands -------------------------------------------------------------
    def center(self) -> None:
        self._send(CMD_CENTER, b"\x01")

    def set_angles(self, yaw: float, pitch: float) -> None:
        yaw = clamp(yaw, *YAW_LIMITS)
        pitch = clamp(pitch, *PITCH_LIMITS)
        self._send(CMD_SET_ANGLES, struct.pack("<hh", int(round(yaw * 10)), int(round(pitch * 10))))

    def request_attitude(self) -> None:
        self._send(CMD_ATTITUDE)

    # ---- background attitude polling -------------------------------------------
    def start(self, poll_hz: float = 20.0) -> "SiyiGimbal":
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(poll_hz,), name="siyi", daemon=True)
        self._thread.start()
        return self

    def _loop(self, poll_hz: float) -> None:
        period = 1.0 / poll_hz
        next_poll = 0.0
        buf = bytearray()
        while self._running:
            now = time.monotonic()
            if now >= next_poll:
                try:
                    self.request_attitude()
                except OSError as exc:
                    log.warning("SIYI send failed: %s", exc)
                next_poll = now + period
            try:
                buf += self.link.recv()
            except OSError as exc:
                log.warning("SIYI receive failed: %s", exc)
                time.sleep(0.1)
                continue
            for pkt in split_packets(buf):
                parsed = parse_packet(pkt)
                if parsed and parsed[0] == CMD_ATTITUDE and len(parsed[1]) >= 6:
                    yaw, pitch, roll = struct.unpack_from("<hhh", parsed[1])
                    self.attitude = (yaw / 10.0, pitch / 10.0, roll / 10.0)
                    self.attitude_t = time.monotonic()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.link.close()
