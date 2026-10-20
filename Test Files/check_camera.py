#!/usr/bin/env python3
"""Quick check that the Raspberry Pi camera works: detects it, grabs a frame, saves a JPEG."""
import sys
from picamera2 import Picamera2

OUT = "/home/ssr-loco/Desktop/Irene/camera_test.jpg"

cams = Picamera2.global_camera_info()
if not cams:
    sys.exit("FAIL: no camera detected (check ribbon cable / config)")
print(f"Detected: {cams}")

picam2 = Picamera2()
try:
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    frame = picam2.capture_array()
    picam2.capture_file(OUT)
finally:
    picam2.close()

print(f"OK: captured frame shape={frame.shape}, mean brightness={frame.mean():.1f}")
print(f"Saved image to {OUT}")
if frame.mean() < 2:
    print("WARNING: image is nearly black - lens cap on or sensor issue?")
