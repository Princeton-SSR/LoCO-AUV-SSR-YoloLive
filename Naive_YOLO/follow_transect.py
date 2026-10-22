#!/usr/bin/env python3
"""Naive transect follower: YOLO-OBB finds the line; yaw toward whichever side of the image the box is on.

Pipeline per frame:  camera -> YOLO OBB -> best box -> horizontal offset from image centre
                     -> NaiveController -> Command (surge/yaw) -> vehicle (MAVLink MANUAL_CONTROL)

Examples:
    # bench test: real camera + model, commands only printed, annotated frames saved
    python3 follow_transect.py --model ../best_dillon.onnx --dry-run --save-every 5

    # in the water (arms the vehicle itself after the start delay)
    python3 follow_transect.py --model ../best_dillon.onnx --arm --max-seconds 60

Ctrl+C always sends neutral (and disarms if this script armed).
"""
import argparse
import sys
import time

import numpy as np
import onnxruntime as ort

# reuse the preprocessing / OBB decoding from the baseline script
from detection import letterbox, postprocess, read_metadata
from controller import NaiveConfig, NaiveController, box_offset, pick_detection
from vehicle import DryRunVehicle, MavVehicle
from stream import Streamer, annotate
from log import LOG_DIR, RunLogger


def parse_args():
    """Define and parse the command-line options (model, control, vehicle, debug)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--webcam", type=int, nargs="?", const=0, default=None, metavar="INDEX",
                   help="use an OpenCV webcam (default index 0) instead of the Pi camera, for testing on a laptop")
    p.add_argument("--cam-size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))

    g = p.add_argument_group("control")
    g.add_argument("--surge", type=float, default=0.25, help="constant forward command while the line is visible")
    g.add_argument("--yaw", type=float, default=0.30, help="yaw command magnitude (max, if --proportional)")
    g.add_argument("--deadband", type=float, default=0.10,
                   help="no yaw while the box centre is within this fraction of half the image width")
    g.add_argument("--proportional", action="store_true", help="scale yaw with the offset instead of fixed +-yaw")
    g.add_argument("--smooth", type=float, default=0.5, help="weight of the newest offset in the moving average")
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
    g.add_argument("--log-dir", default=LOG_DIR, help="folder for the per-run CSV logs (named by start time)")
    g.add_argument("--no-log", action="store_true", help="don't write a run log")
    g.add_argument("--stream", type=int, nargs="?", const=8000, default=0, metavar="PORT",
                   help="serve live video + status page at http://<ip>:PORT/ (see stream.py). "
                        "Bare --stream uses port 8000; omit the flag (or pass 0) to turn it off")
    g.add_argument("--params", default="params.json",
                   help="JSON file for values edited on the web page: loaded at start (overrides the flags "
                        "above for those values) and rewritten on every change")
    g.add_argument("--stream-fps", type=float, default=10.0, help="max frames/s pushed to the stream")
    return p.parse_args()


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
    ctrl = NaiveController(NaiveConfig(
        surge=args.surge, yaw_cmd=args.yaw, deadband=args.deadband, proportional=args.proportional,
        smooth=args.smooth, yaw_sign=args.yaw_sign, lost_hold=args.lost_hold))

    # ---- vehicle
    vehicle = DryRunVehicle() if args.dry_run else MavVehicle(args.port, args.baud, args.mode)
    vehicle.connect()

    # ---- camera
    if args.webcam is not None:
        # This part is for testing it on my computer 
        import cv2

        class WebcamCamera:
            """OpenCV webcam with the same capture_array()/close() interface as Picamera2 (returns RGB)."""
            def __init__(self, index):
                self.cap = cv2.VideoCapture(index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
                if not self.cap.isOpened():
                    sys.exit(f"Could not open webcam {index}")

            def capture_array(self):
                ok, bgr = self.cap.read()
                if not ok:
                    raise RuntimeError("webcam read failed")
                return cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB)

            def close(self):
                self.cap.release()

        cam = WebcamCamera(args.webcam)
    else:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"size": (W, H), "format": "XBGR8888"}))
        cam.start()
    time.sleep(1)
    session.run(None, {inp.name: np.zeros((1, 3, in_size, in_size), dtype=np.float32)})  # warm-up

    logger = None if args.no_log else RunLogger(args.log_dir, ctrl.cfg)  # new timestamped CSV for every run

    streamer = (Streamer(args.stream, args.deadband, args.stream_fps, dry_run=args.dry_run,
                         cfg=ctrl.cfg, params_path=args.params)
                if args.stream else None)
    if streamer:
        print(f"Live view: http://<pi-ip>:{args.stream}/")

    try:
        print(f"Starting in {args.start_delay:.0f} s (Ctrl+C aborts)...")
        time.sleep(args.start_delay)
        if args.arm:
            vehicle.arm()

        t_start = last_seen = time.monotonic()
        n = 0
        print("Following transect...")
        while True:
            # --- stop conditions: time limit, or line missing for too long
            now = time.monotonic()
            if args.max_seconds and now - t_start > args.max_seconds:
                print("\nmax-seconds reached")
                break
            if now - last_seen > args.lost_timeout:
                print("\nline lost for too long, stopping")
                break

            # --- grab a frame and prepare it for the model
            frame = np.ascontiguousarray(cam.capture_array()[..., :3])
            img, scale, pad = letterbox(frame, in_size)
            blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            # --- run YOLO and decode the oriented boxes
            t0 = time.perf_counter()
            out = session.run(None, {inp.name: blob})[0]
            infer_ms = 1e3 * (time.perf_counter() - t0)
            dets = postprocess(out, "obb", scale, pad, frame.shape[:2], args.conf, args.iou)

            # --- best box -> horizontal offset from image centre -> yaw/surge command
            det = pick_detection(dets)
            offset = box_offset(det, W) if det is not None else None
            if offset is not None:
                last_seen = time.monotonic()
            cmd, tel = ctrl.update(offset, time.monotonic())
            vehicle.set_command(cmd)  # the vehicle thread keeps streaming this to the autopilot

            # --- optional debug output: annotated image, CSV log, console status
            n += 1
            if args.save_every and n % args.save_every == 0:
                annotate(frame, det, tel, cmd, args.deadband)
            if streamer:
                streamer.update(frame, det, tel, cmd, offset, infer_ms)
            if logger:
                logger.log(time.monotonic() - t_start, tel.state, det, offset, tel.offset, cmd, infer_ms)
            if not args.dry_run:
                sys.stdout.write(f"\r{tel.state:5s} offset={tel.offset:+.2f} surge={cmd.surge:+.2f} "
                                 f"yaw={cmd.yaw:+.2f} infer={infer_ms:.0f}ms   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        try:
            vehicle.close()  # sends neutral, disarms if we armed
        finally:
            cam.close()
            if streamer:
                streamer.close()
            if logger:
                logger.close()
            print("Done.")


if __name__ == "__main__":
    main()
