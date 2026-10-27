"""Geometry / dynamics sanity tests (no ONNX model needed).  Run from Naive_YOLO_sim/:  python3 tests/test_sim.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.camera import FisheyeCamera
from sim.config import CameraConfig, DynamicsConfig, ScenarioConfig, RenderConfig
from sim.controllers import Command
from sim.dynamics import Robot
from sim.render import Renderer
from sim.world import Transect, build_floor, make_transect


def test_camera_roundtrip():
    cam = FisheyeCamera(CameraConfig(pitch_deg=10.0, k1=-0.02))
    X, Y, slant, valid = cam.floor_hits(1.0, 0.5, 0.3, 8.0)
    ys, xs = np.nonzero(valid)
    i = len(ys) // 3
    u, v = cam.world_to_pixel(X[ys[i], xs[i]], Y[ys[i], xs[i]], 1.0, 0.5, 0.3)
    assert abs(u - xs[i]) < 0.1 and abs(v - ys[i]) < 0.1, (u, v, xs[i], ys[i])
    print("ok  camera ray <-> pixel round trip")


def test_straight_ahead_floor_point():
    cam = FisheyeCamera(CameraConfig(height_m=1.0, pitch_deg=0.0, forward_offset_m=0.0))
    u, v = cam.world_to_pixel(2.0, 0.0, 0.0, 0.0, 0.0)  # 2 m straight ahead on the floor
    assert abs(u - cam.cx) < 1e-6 and v > cam.cy              # centred horizontally, below the horizon
    assert abs((v - cam.cy) - cam.f * np.arctan2(1.0, 2.0)) < 1e-6  # equidistant: r = f * theta
    u2, _ = cam.world_to_pixel(2.0, -0.5, 0.0, 0.0, 0.0)       # to the robot's right (-y) -> right of centre
    assert u2 > cam.cx
    print("ok  forward point projection, right = +u")


def test_transect_project():
    t = Transect(np.column_stack([np.linspace(0, 10, 201), np.zeros(201)]))
    s, cte = t.project(4.0, 0.3)
    assert abs(s - 4.0) < 1e-6 and abs(cte - 0.3) < 1e-6        # + = left of the heading
    assert t.project(4.0, -0.2)[1] < 0
    print("ok  transect projection / signed cte")


def test_rendered_tape_is_visible_and_on_correct_side():
    cfg = ScenarioConfig(path="straight", length_m=8.0, view_range_m=6.0, floor_res_m=0.01)
    rng = np.random.default_rng(0)
    tr = make_transect("straight", 8.0, rng)
    floor = build_floor(tr, cfg, rng)
    cam = FisheyeCamera(CameraConfig())
    rend = Renderer(cam, floor, RenderConfig(noise_sigma=0, exposure_jitter=0, caustics=0), 6.0)
    # robot 0.3 m to the LEFT of the line (y = +0.3), heading +x: the line is on its right -> yellow right of centre
    img = rend.render(0.5, 0.3, 0.0, 0.0).astype(float)
    yellowness = (img[..., 0] + img[..., 1] - 2 * img[..., 2]) * cam.circle_mask  # ignore the housing ring
    ys, xs = np.nonzero(yellowness > np.percentile(yellowness, 99.8))
    assert len(xs) > 5 and xs.mean() > cam.cx, xs.mean()
    img2 = rend.render(0.5, -0.3, 0.0, 0.0).astype(float)
    y2 = (img2[..., 0] + img2[..., 1] - 2 * img2[..., 2]) * cam.circle_mask
    _, xs2 = np.nonzero(y2 > np.percentile(y2, 99.8))
    assert xs2.mean() < cam.cx, xs2.mean()
    print("ok  tape renders on the correct side of the image")


def test_dynamics_directions():
    r = Robot(DynamicsConfig(drift_noise_mps=0, yaw_noise_dps=0), seed=0)
    for _ in range(200):
        r.step(Command(0.5, 0.5), 0.02)
    assert r.x > 0 and r.psi < 0 and r.y < 0, (r.x, r.y, r.psi)  # yaw + = clockwise: heading decreases, curves right
    print("ok  +yaw turns clockwise")


if __name__ == "__main__":
    test_camera_roundtrip()
    test_straight_ahead_floor_point()
    test_transect_project()
    test_rendered_tape_is_visible_and_on_correct_side()
    test_dynamics_directions()
