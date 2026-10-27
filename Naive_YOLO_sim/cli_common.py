"""Command-line options shared by run.py, batch.py and render_preview.py."""
import argparse
from pathlib import Path

from sim.config import Config
from sim.world import PATH_KINDS

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent / "best_dillon.onnx"


def add_common_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("scenario")
    g.add_argument("--path", default="gentle_s", choices=PATH_KINDS, help="transect shape")
    g.add_argument("--length", type=float, default=12.0, help="transect length, m")
    g.add_argument("--gaps", type=int, default=0, help="number of gaps in the tape")
    g.add_argument("--floor-image", default=None, help="top-down photo of the pool floor to tile instead of the "
                                                         "procedural floor")
    g.add_argument("--floor-image-width", type=float, default=2.0, help="metres covered by the width of --floor-image")
    g.add_argument("--tile", type=float, default=0.0, help="draw a tile grid with this pitch in m (0 = off)")
    g.add_argument("--tape-width", type=float, default=0.025, help="tape width, m")

    g = p.add_argument_group("camera")
    g.add_argument("--cam-height", type=float, default=1.0, help="camera height above the floor, m")
    g.add_argument("--cam-pitch", type=float, default=0.0, help="camera pitch down, degrees")
    g.add_argument("--fov", type=float, default=140.0, help="fisheye field of view across the image circle, degrees")
    g.add_argument("--supersample", type=int, default=1, help="render at Nx resolution for anti-aliasing (slow)")

    g = p.add_argument_group("dynamics / timing")
    g.add_argument("--max-speed", type=float, default=0.5, help="forward speed (m/s) at surge=1")
    g.add_argument("--max-yaw-rate", type=float, default=60.0, help="yaw rate (deg/s) at yaw=1")
    g.add_argument("--current", type=float, default=0.0, help="steady current speed, m/s")
    g.add_argument("--current-dir", type=float, default=90.0, help="direction the current flows to, degrees")
    g.add_argument("--fps", type=float, default=8.0, help="perception loop rate (frames/s) as on the Pi")
    g.add_argument("--latency", type=float, default=0.2, help="frame capture -> thrusters delay, s")
    g.add_argument("--duration", type=float, default=90.0, help="max simulated seconds")

    g = p.add_argument_group("detector / controller")
    g.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO OBB .onnx")
    g.add_argument("--oracle", action="store_true",
                   help="replace YOLO by a perfect geometric detector (validates loop/controller without the model)")
    g.add_argument("--conf", type=float, default=0.35)
    g.add_argument("--iou", type=float, default=0.45)
    g.add_argument("--controller", default="naive", help="naive | naive_p | pd (see sim/controllers.py)")
    g.add_argument("--params", default=None, help="params JSON (default: ../Naive_YOLO/params.json)")


def build_config(args, seed=None) -> Config:
    cfg = Config()
    s = cfg.scenario
    s.path, s.length_m, s.n_gaps = args.path, args.length, args.gaps
    s.floor_image, s.floor_image_width_m, s.tile_m, s.tape_width_m = (args.floor_image, args.floor_image_width,
                                                                       args.tile, args.tape_width)
    c = cfg.camera
    c.height_m, c.pitch_deg, c.fov_deg, c.supersample = args.cam_height, args.cam_pitch, args.fov, args.supersample
    d = cfg.dynamics
    d.max_speed_mps, d.max_yaw_rate_dps = args.max_speed, args.max_yaw_rate
    d.current_speed_mps, d.current_dir_deg = args.current, args.current_dir
    m = cfg.sim
    m.perception_dt, m.command_latency_s, m.duration_s = 1.0 / args.fps, args.latency, args.duration
    m.conf, m.iou = args.conf, args.iou
    if seed is not None:
        s.seed = m.seed = seed
    return cfg
