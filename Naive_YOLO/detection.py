#!/usr/bin/env python3
"""Baseline: run a YOLOv8/YOLO11 ONNX model (detect or OBB) on the live Pi camera feed, printing live FPS.

Usage:
    python3 detection.py --model best.onnx
    python3 detection.py --model best.onnx --show          # live window (run on the Pi desktop)
    python3 detection.py --model best.onnx --conf 0.4 --save-every 30

Deps: picamera2 (system), numpy, pillow, onnxruntime
Stop with Ctrl+C.
"""
import argparse
import ast
import sys
import time
from collections import deque

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to YOLO .onnx file")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--cam-size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"),
                   help="camera capture resolution")
    p.add_argument("--threads", type=int, default=4, help="onnxruntime intra-op threads")
    p.add_argument("--show", action="store_true",
                   help="open a live preview window with boxes + FPS (needs a desktop session)")
    p.add_argument("--save-every", type=int, default=0,
                   help="save annotated frame to last_frame.jpg every N frames (0 = never)")
    return p.parse_args()


def read_metadata(session):
    """Ultralytics-exported ONNX files store task and class names in the metadata."""
    meta = session.get_modelmeta().custom_metadata_map
    task = meta.get("task", "detect")
    names = None
    if "names" in meta:
        try:
            d = ast.literal_eval(meta["names"])
            names = [d[i] for i in sorted(d)]
        except Exception:
            pass
    return task, names


def letterbox(img, size):
    """Resize keeping aspect ratio, pad to a square `size`. Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, scale, (px, py)


def nms_axis_aligned(boxes, scores, iou_thr):
    """Plain numpy NMS on xyxy boxes. Returns kept indices."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def _gauss_params(xywhr):
    """Covariance terms of the Gaussian that models each rotated box (as in Ultralytics' ProbIoU)."""
    w, h, r = xywhr[..., 2], xywhr[..., 3], xywhr[..., 4]
    a, b = w ** 2 / 12, h ** 2 / 12
    cos, sin = np.cos(r), np.sin(r)
    return a * cos ** 2 + b * sin ** 2, a * sin ** 2 + b * cos ** 2, (a - b) * cos * sin


def probiou(box, others, eps=1e-7):
    """Approximate IoU between one rotated box (cx,cy,w,h,r) and many, via Gaussian Bhattacharyya distance."""
    x1, y1 = box[0], box[1]
    x2, y2 = others[:, 0], others[:, 1]
    a1, b1, c1 = _gauss_params(box)
    a2, b2, c2 = _gauss_params(others)
    den = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / den * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / den * 0.5
    t3 = np.log(den / (4 * np.sqrt(np.clip(a1 * b1 - c1 ** 2, 0, None) *
                                   np.clip(a2 * b2 - c2 ** 2, 0, None)) + eps) + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    return 1 - np.sqrt(1 - np.exp(-bd) + eps)


def nms_rotated(xywhr, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[probiou(xywhr[i], xywhr[rest]) <= iou_thr]
    return keep


def postprocess(output, task, scale, pad, orig_shape, conf_thr, iou_thr):
    """Decode raw output into detections.

    Returns list of (cx, cy, w, h, angle_rad, score, class_id) in original-image coords.
    detect output: (1, 4+nc, N)    obb output: (1, 4+nc+1, N) with the angle (radians) as the last channel.
    """
    preds = output[0].T  # (N, channels)
    obb = task == "obb"
    class_scores = preds[:, 4:-1] if obb else preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(preds)), class_ids]

    mask = scores >= conf_thr
    preds, scores, class_ids = preds[mask], scores[mask], class_ids[mask]
    if len(preds) == 0:
        return []

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    angle = preds[:, -1] if obb else np.zeros_like(cx)

    # undo letterbox (angle is unaffected by uniform scaling)
    cx, cy = (cx - pad[0]) / scale, (cy - pad[1]) / scale
    w, h = w / scale, h / scale
    xywhr = np.stack([cx, cy, w, h, angle], axis=1)

    # class-aware NMS: shift each class far apart so classes never suppress each other
    shifted = xywhr.copy()
    shifted[:, 0] += class_ids * 100000.0
    if obb:
        keep = nms_rotated(shifted, scores, iou_thr)
    else:
        xyxy = np.stack([shifted[:, 0] - w / 2, cy - h / 2, shifted[:, 0] + w / 2, cy + h / 2], axis=1)
        keep = nms_axis_aligned(xyxy, scores, iou_thr)
    return [(*xywhr[i], scores[i], int(class_ids[i])) for i in keep]


def box_corners(cx, cy, w, h, r):
    cos, sin = np.cos(r), np.sin(r)
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])
    return [(float(cx + x * cos - y * sin), float(cy + x * sin + y * cos)) for x, y in zip(dx, dy)]


def draw_dets(draw, dets, names):
    for cx, cy, w, h, r, score, cid in dets:
        pts = box_corners(cx, cy, w, h, r)
        draw.polygon(pts, outline=(0, 255, 0, 255), width=2)
        draw.text((pts[0][0] + 2, pts[0][1] + 2), f"{class_name(names, cid)} {score:.2f}",
                  fill=(255, 255, 0, 255))


def annotate(frame, dets, names):
    img = Image.fromarray(frame)
    draw_dets(ImageDraw.Draw(img), dets, names)
    img.save("last_frame.jpg")


def make_overlay(shape, dets, names, fps):
    """Transparent RGBA image with boxes + FPS, for picamera2's live preview."""
    h, w = shape
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_dets(draw, dets, names)
    draw.text((8, 8), f"FPS {fps:.1f}  det {len(dets)}", fill=(255, 255, 255, 255))
    return np.asarray(img)


def class_name(names, cid):
    return names[cid] if 0 <= cid < len(names) else str(cid)


def main():
    args = parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    in_size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
    task, names = read_metadata(session)
    if names is None:
        n_extra = 5 if task == "obb" else 4
        names = [str(i) for i in range(out.shape[1] - n_extra)]
    print(f"Model: {args.model} | task {task} | input {inp.shape} | output {out.shape} | classes: {names}")

    from picamera2 import Picamera2, Preview  # imported here so other scripts can reuse this module off-Pi
    cam = Picamera2()
    # "XBGR8888" gives an (H, W, 4) array in R,G,B,X order. 32-bit formats are what the GL preview supports.
    cam.configure(cam.create_video_configuration(
        main={"size": tuple(args.cam_size), "format": "XBGR8888"}))
    if args.show:
        cam.start_preview(Preview.QTGL)  # use Preview.QT here if QTGL gives GL errors
    cam.start()
    time.sleep(1)  # let auto-exposure settle

    # warm-up so the first timed frames aren't skewed by one-time setup costs
    dummy = np.zeros((1, 3, in_size, in_size), dtype=np.float32)
    session.run(None, {inp.name: dummy})

    window = deque(maxlen=30)  # rolling window of frame times for smoothed FPS
    frame_count = 0
    last_print = 0.0
    print("Running... Ctrl+C to stop.")
    try:
        while True:
            t0 = time.perf_counter()
            frame = np.ascontiguousarray(cam.capture_array()[..., :3])  # drop padding byte -> (H, W, 3) RGB uint8
            t1 = time.perf_counter()

            img, scale, pad = letterbox(frame, in_size)
            blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            t2 = time.perf_counter()

            output = session.run(None, {inp.name: blob})[0]
            t3 = time.perf_counter()

            dets = postprocess(output, task, scale, pad, frame.shape[:2], args.conf, args.iou)
            t4 = time.perf_counter()

            frame_count += 1
            window.append(t4 - t0)
            if args.save_every and frame_count % args.save_every == 0:
                annotate(frame, dets, names)

            fps = len(window) / sum(window)
            if args.show:
                cam.set_overlay(make_overlay(frame.shape[:2], dets, names, fps))

            if t4 - last_print >= 0.25:  # refresh the status line 4x/sec
                last_print = t4
                labels = ", ".join(f"{class_name(names, d[6])}:{d[5]:.2f}" for d in dets[:3])
                sys.stdout.write(
                    f"\rFPS {fps:5.1f} | cap {1e3*(t1-t0):5.1f}ms pre {1e3*(t2-t1):5.1f}ms "
                    f"infer {1e3*(t3-t2):6.1f}ms post {1e3*(t4-t3):5.1f}ms | "
                    f"{len(dets):2d} det {labels:<40}")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if window:
            print(f"\nStopped after {frame_count} frames. Avg FPS (last {len(window)}): "
                  f"{len(window)/sum(window):.1f}")


if __name__ == "__main__":
    main()
