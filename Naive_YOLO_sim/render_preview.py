#!/usr/bin/env python3
"""Look at what the simulated camera sees (and what YOLO makes of it) without running a whole simulation.
Use this to tune the renderer / camera against your real images.

    ../yolo_env_mac/bin/python render_preview.py                      # montage of poses -> preview/montage.png
    ../yolo_env_mac/bin/python render_preview.py --real a.jpg b.jpg   # run the detector on real frames, print + save

Montage columns are lateral offsets from the line (m); rows are heading errors (deg).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from cli_common import add_common_args, build_config
from sim.detector import YoloDetector
from sim.simulation import build_scenario
from sim.viz import draw_camera_panel


def annotate(frame, dets, label=""):
    det = max(dets, key=lambda d: d[5]) if dets else None
    img = draw_camera_panel(frame, dict(det=det, dets=dets))
    txt = label + (f" conf={det[5]:.2f} cx={det[0]:.0f}" if det else " no detection")
    cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--real", nargs="*", default=None, help="real image files to run through the detector")
    p.add_argument("--at", type=float, default=3.0, help="arc length along the transect for the montage, m")
    p.add_argument("--out", default="preview")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    cfg = build_config(args, seed=0)
    detector = YoloDetector(args.model, args.conf, args.iou)

    if args.real:
        for f in args.real:
            bgr = cv2.imread(f)
            rgb = cv2.cvtColor(cv2.resize(bgr, (cfg.camera.width, cfg.camera.height)), cv2.COLOR_BGR2RGB)
            dets, ms = detector.detect(rgb)
            print(f"{f}: {len(dets)} detections ({ms:.0f} ms)", [f"{d[5]:.2f}" for d in dets[:3]])
            cv2.imwrite(str(out / f"real_{Path(f).stem}.png"), cv2.cvtColor(annotate(rgb, dets, "real"), cv2.COLOR_RGB2BGR))
        return

    cfg.scenario.randomize_start = False
    sc = build_scenario(cfg)
    x, y, hdg = sc.transect.pose_at(args.at)
    rows = []
    for dh in (-20, 0, 20):
        cols = []
        for lat in (-0.5, -0.25, 0.0, 0.25, 0.5):
            psi = hdg + np.radians(dh)
            px, py = x - lat * np.sin(hdg), y + lat * np.cos(hdg)  # lat > 0: robot left of the line
            frame = sc.renderer.render(px, py, psi, 0.0)
            dets, _ = detector.detect(frame)
            cols.append(annotate(frame, dets, f"lat={lat:+.2f} dh={dh:+d}"))
        rows.append(np.concatenate(cols, axis=1))
    montage = np.concatenate(rows, axis=0)
    montage = cv2.resize(montage, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "montage.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"saved {out / 'montage.png'}")


if __name__ == "__main__":
    main()
