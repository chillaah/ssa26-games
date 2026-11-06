"""Write the QR code images shown on the attract screen (needs OpenCV >= 4.8 for the QR encoder).

    python tools/make_qr.py
"""
import cv2

from outofframe.config import DisplayCfg

MODULE_PX = 10
# OpenCV's encoder already leaves 2 white modules around the code; the white card the screens
# draw around it adds ~2.5 more, which makes the 4-module quiet zone phone scanners need.
# (Adding more here only made the codes look small on a big white square.)

for s in DisplayCfg().socials:
    q = cv2.QRCodeEncoder.create().encode(s["url"])
    q = cv2.resize(q, None, fx=MODULE_PX, fy=MODULE_PX, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(s["qr"], q)
    print(s["qr"], q.shape)
