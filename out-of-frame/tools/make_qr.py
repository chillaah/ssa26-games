"""Write the QR code images shown on the attract screen (needs OpenCV >= 4.8 for the QR encoder).

    python tools/make_qr.py
"""
import cv2

from outofframe.config import DisplayCfg

MODULE_PX = 10
QUIET = 4  # white modules around the code, needed by phone scanners

for s in DisplayCfg().socials:
    q = cv2.QRCodeEncoder.create().encode(s["url"])
    q = cv2.copyMakeBorder(q, QUIET, QUIET, QUIET, QUIET, cv2.BORDER_CONSTANT, value=255)
    q = cv2.resize(q, None, fx=MODULE_PX, fy=MODULE_PX, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(s["qr"], q)
    print(s["qr"], q.shape)
