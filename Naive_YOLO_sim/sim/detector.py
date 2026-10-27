"""The REAL YOLO-OBB model in the loop: same preprocessing, ONNX inference and decoding as follow_transect.py."""
import time

import numpy as np
import onnxruntime as ort

from .robot_code import detection


class YoloDetector:
    def __init__(self, model_path, conf=0.35, iou=0.45, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
        self.task, self.names = detection.read_metadata(self.session)
        if self.task != "obb":
            raise SystemExit(f"Model task is '{self.task}', but the transect follower needs an OBB model.")
        self.conf, self.iou = conf, iou
        self.session.run(None, {self.input_name: np.zeros((1, 3, self.in_size, self.in_size), np.float32)})  # warm-up

    needs_pose = False

    def detect(self, frame_rgb, pose=None):
        """frame_rgb: (H, W, 3) uint8. Returns (dets, infer_ms); dets entries are (cx, cy, w, h, angle, score, cls)."""
        img, scale, pad = detection.letterbox(frame_rgb, self.in_size)
        blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0]
        infer_ms = 1e3 * (time.perf_counter() - t0)
        dets = detection.postprocess(out, "obb", scale, pad, frame_rgb.shape[:2], self.conf, self.iou)
        return dets, infer_ms


class OracleDetector:
    """Stand-in 'perfect' detector: projects the true tape into the image and boxes it (no neural network).

    Use it to validate the loop / controller / dynamics independently of the YOLO sim-to-real gap (`--oracle`).
    Gaps in the tape are respected; `dropout` randomly misses frames; `noise_px` jitters the box centre."""
    needs_pose = True
    ignores_frame = True  # lets batch runs skip rendering entirely

    def __init__(self, scenario, max_range=5.0, dropout=0.0, noise_px=0.0, seed=0):
        self.sc, self.max_range, self.dropout, self.noise_px = scenario, max_range, dropout, noise_px
        self.rng = np.random.default_rng(seed)

    def detect(self, frame_rgb, pose=None):
        import cv2
        x, y, psi = pose
        tr, cam = self.sc.transect, self.sc.camera
        s_now, _ = tr.project(x, y)
        mask = (tr.s >= s_now - 0.5) & (tr.s <= s_now + self.max_range + 1.0)
        for g0, g1 in tr.gaps:
            mask &= ~((tr.s >= g0) & (tr.s <= g1))
        pts = tr.pts[mask]
        if len(pts) < 2 or self.rng.random() < self.dropout:
            return [], 0.0
        u, v = cam.world_to_pixel(pts[:, 0], pts[:, 1], x, y, psi)
        cx_, cy_ = cam.camera_position(x, y, psi)
        near = np.hypot(pts[:, 0] - cx_, pts[:, 1] - cy_) < self.max_range
        ok = near & np.isfinite(u) & (np.hypot(u - cam.cx, v - cam.cy) < cam.R * 0.98) & (v > cam.cy)
        if ok.sum() < 2:
            return [], 0.0
        (bx, by), (bw, bh), ang = cv2.minAreaRect(np.column_stack([u[ok], v[ok]]).astype(np.float32))
        bx += self.noise_px * self.rng.standard_normal()
        return [(bx, by, max(bw, 4.0), max(bh, 4.0), np.radians(ang), 0.9, 0)], 0.0
