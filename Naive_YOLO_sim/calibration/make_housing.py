#!/usr/bin/env python3
"""Build assets/housing.png: the static camera-housing ring that surrounds the fisheye window in real frames.

The housing does not move relative to the camera, so the per-pixel median over many real frames keeps it and averages
the scene away. The simulator pastes this image outside the image circle so YOLO sees the same borders as on the robot.

    ../yolo_env_mac/bin/python calibration/make_housing.py "/path/to/Dillon Training Data" [--every 5]
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("--every", type=int, default=5)
args = p.parse_args()
files = sorted(glob.glob(str(Path(args.folder) / "*" / "*.jpg")))[::args.every]
ims = np.stack([cv2.imread(f) for f in files]).astype(np.float32)
med = np.median(ims, 0).astype(np.uint8)
std = ims.std(0).mean(2)
ys, xs = np.nonzero(std > 25)
print(f"{len(files)} frames; moving-content bbox x {xs.min()}..{xs.max()}  y {ys.min()}..{ys.max()}")
out = Path(__file__).resolve().parent.parent / "assets" / "housing.png"
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), med)
print("saved", out)
