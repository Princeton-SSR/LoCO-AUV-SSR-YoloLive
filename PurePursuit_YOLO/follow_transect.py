#!/usr/bin/env python3
"""Follow a transect line (tape measure on the pool floor) using YOLO-OBB + pure pursuit + PID.

Pipeline per frame:  camera -> YOLO OBB -> pick best box -> line in robot frame -> TransectController
                     -> Command (surge/sway/yaw) -> vehicle (MAVLink MANUAL_CONTROL)

Examples:
    # bench test: real camera + model, commands only printed, annotated frames saved
    python3 follow_transect.py --model best.onnx --dry-run --save-every 5

    # in the water (arms the vehicle itself after the start delay)
    python3 follow_transect.py --model best.onnx --arm --max-seconds 60 --log run1.csv

Ctrl+C always sends neutral (and disarms if this script armed).
"""
import argparse
import csv
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

# reuse the preprocessing / OBB decoding from the baseline script
from yolo_live import letterbox, postprocess, read_metadata, box_corners
from transect_controller import (ControllerConfig, GroundMap, TransectController,
                                 line_from_obb, pick_detection)
from vehicle import DryRunVehicle, MavVehicle


def parse_args():
    """Define and parse the command-line options (model, geometry, control gains, vehicle, debug)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cam-size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    p.add_argument("--min-aspect", type=float, default=2.0, help="reject OBBs with long/short side below this")

    g = p.add_argument_group("geometry")
    g.add_argument("--ref", type=float, nargs=2, default=[0.5, 0.85], metavar=("X", "Y"),
                   help="image point (fraction of width/height) that should end up over the line")
    g.add_argument("--calib", help="JSON with image_pts / robot_pts for a metric ground homography (optional)")

    g = p.add_argument_group("control")
    g.add_argument("--lookahead", type=float, help="default 0.7 (half-widths) or 1.0 (m) with --calib")
    g.add_argument("--surge", type=float, default=0.30, help="max forward command, 0..1")
    g.add_argument("--yaw-pid", type=float, nargs=3, default=[1.0, 0.05, 0.10], metavar=("KP", "KI", "KD"))
    g.add_argument("--yaw-limit", type=float, default=0.5)
    g.add_argument("--yaw-sign", type=float, default=1.0, help="use -1 if the robot yaws away from the line")
    g.add_argument("--lost-hold", type=float, default=1.0, help="seconds to creep forward after losing the line")
    g.add_argument("--lost-timeout", type=float, default=10.0, help="stop the run after this long without the line")

    g = p.add_argument_group("vehicle")
    g.add_argument("--dry-run", action="store_true", help="don't touch MAVLink, just print commands")
    g.add_argument("--port", default="/dev/ttyAMA0")
    g.add_argument("--baud", type=int, default=921600)
    g.add_argument("--mode", default="STABILIZE")
    g.add_argument("--arm", action="store_true", help="arm after the start delay (and disarm at the end)")
    g.add_argument("--start-delay", type=float, default=10.0)
    g.add_argument("--max-seconds", type=float, default=0.0, help="0 = run until Ctrl+C")

    g = p.add_argument_group("debug")
    g.add_argument("--save-every", type=int, default=0, help="save annotated last_frame.jpg every N frames")
    g.add_argument("--log", help="CSV file with per-frame telemetry (for tuning gains)")
    return p.parse_args()


def annotate(frame, det, gmap, tel, cmd, path="last_frame.jpg"):
    """Save a debug image: OBB (green), robot reference point (blue), lookahead point/line (red) and a status line."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    if det is not None:
        d.polygon(box_corners(*det[:5]), outline=(0, 255, 0), width=2)
    ru, rv = gmap.ref_uv
    d.ellipse([ru - 5, rv - 5, ru + 5, rv + 5], outline=(0, 160, 255), width=2)
    if tel.lookahead_pt is not None:
        lu, lv = gmap.to_pixel(*tel.lookahead_pt)
        d.line([ru, rv, lu, lv], fill=(255, 0, 0), width=2)
        d.ellipse([lu - 5, lv - 5, lu + 5, lv + 5], fill=(255, 0, 0))
    d.text((8, 8), f"{tel.state} e={tel.e:+.2f} a={np.degrees(tel.alpha):+.0f}deg "
                   f"surge={cmd.surge:+.2f} yaw={cmd.yaw:+.2f}", fill=(255, 255, 0))
    img.save(path)


def main():
    """Set up model, controller, vehicle and camera, then run the capture -> detect -> control -> command loop."""
    args = parse_args()

    # ---- model
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    in_size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
    task, _ = read_metadata(session)
    if task != "obb":
        sys.exit(f"Model task is '{task}', but this module needs an OBB model.")

    # ---- control
    W, H = args.cam_size
    gmap = GroundMap(W, H, ref_frac=tuple(args.ref), calib_path=args.calib)
    cfg = ControllerConfig(
        lookahead=args.lookahead or (1.0 if gmap.metric else 0.7),
        surge_max=args.surge, yaw_pid=tuple(args.yaw_pid),
        yaw_limit=args.yaw_limit,
        yaw_sign=args.yaw_sign, lost_hold=args.lost_hold)
    ctrl = TransectController(cfg)

    # ---- vehicle
    vehicle = DryRunVehicle() if args.dry_run else MavVehicle(args.port, args.baud, args.mode)
    vehicle.connect()

    # ---- camera
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": (W, H), "format": "XBGR8888"}))
    cam.start()
    time.sleep(1)
    session.run(None, {inp.name: np.zeros((1, 3, in_size, in_size), dtype=np.float32)})  # warm-up

    logf = open(args.log, "w", newline="") if args.log else None
    logw = csv.writer(logf) if logf else None
    if logw:
        logw.writerow(["t", "state", "score", "e", "alpha_rad", "surge", "yaw", "infer_ms"])

    try:
        print(f"Starting in {args.start_delay:.0f} s (Ctrl+C aborts)...")
        time.sleep(args.start_delay)
        if args.arm:
            vehicle.arm()

        t_start = last_seen = time.monotonic()
        n = 0
        print("Following transect...")
        while True:
            now = time.monotonic()
            if args.max_seconds and now - t_start > args.max_seconds:
                print("\nmax-seconds reached")
                break
            if now - last_seen > args.lost_timeout:
                print("\nline lost for too long, stopping")
                break

            frame = np.ascontiguousarray(cam.capture_array()[..., :3])
            img, scale, pad = letterbox(frame, in_size)
            blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            t0 = time.perf_counter()
            out = session.run(None, {inp.name: blob})[0]
            infer_ms = 1e3 * (time.perf_counter() - t0)
            dets = postprocess(out, "obb", scale, pad, frame.shape[:2], args.conf, args.iou)

            det = pick_detection(dets)
            line = line_from_obb(det, gmap, args.min_aspect) if det is not None else None
            if line is not None:
                last_seen = time.monotonic()
            cmd, tel = ctrl.update(line, time.monotonic())
            vehicle.set_command(cmd)

            n += 1
            if args.save_every and n % args.save_every == 0:
                annotate(frame, det, gmap, tel, cmd)
            if logw:
                logw.writerow([f"{time.monotonic()-t_start:.3f}", tel.state, f"{det[5]:.3f}" if det else "",
                               f"{tel.e:.4f}", f"{tel.alpha:.4f}", f"{cmd.surge:.3f}",
                               f"{cmd.yaw:.3f}", f"{infer_ms:.0f}"])
            if not args.dry_run:
                sys.stdout.write(f"\r{tel.state:5s} e={tel.e:+.2f} alpha={np.degrees(tel.alpha):+5.0f}deg "
                                 f"surge={cmd.surge:+.2f} yaw={cmd.yaw:+.2f} "
                                 f"infer={infer_ms:.0f}ms   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        try:
            vehicle.close()  # sends neutral, disarms if we armed
        finally:
            cam.close()
            if logf:
                logf.close()
            print("Done.")


if __name__ == "__main__":
    main()
