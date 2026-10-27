#!/usr/bin/env python3
"""Run ONE closed-loop simulation of the transect follower (real YOLO + real controller, simulated camera/robot).

From this folder, with the Mac venv:
    ../yolo_env_mac/bin/python run.py --show                         # live window (Space = pause, Esc = quit)
    ../yolo_env_mac/bin/python run.py --path sine --seed 3 --video runs/sine3.mp4
    ../yolo_env_mac/bin/python run.py --controller pd --set kd=0.5 --set yaw_cmd=0.3
    ../yolo_env_mac/bin/python run.py --path turn90 --gaps 1 --current 0.1

Outputs a metrics summary and a per-frame CSV in runs/ (same column names as the robot's own logs, plus ground truth
x, y, psi_deg, cte, progress), so simulated and real runs can be plotted together.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from cli_common import add_common_args, build_config
from sim.controllers import TUNABLES, load_params, make_controller
from sim.detector import OracleDetector, YoloDetector
from sim.simulation import Simulation, build_scenario
from sim.viz import PANEL_H, PANEL_W, TopDown, Window, compose


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = (v.lower() == "true") if v.lower() in ("true", "false") else float(v)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--seed", type=int, default=0, help="seed for path, floor, start pose, noise")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a controller param (repeatable)")
    p.add_argument("--show", action="store_true", help="live pygame window")
    p.add_argument("--realtime", action="store_true", help="slow the window down to simulated real time")
    p.add_argument("--video", default=None, help="write the composite view to this .mp4")
    p.add_argument("--save-frames", default=None, metavar="DIR", help="save every raw camera frame as PNG to DIR")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()

    cfg = build_config(args, seed=args.seed)
    params = load_params(args.params, parse_set(args.set))
    controller = make_controller(args.controller, params)

    t0 = time.time()
    print("building scenario ...")
    scenario = build_scenario(cfg)
    detector = OracleDetector(scenario, seed=args.seed) if args.oracle else YoloDetector(args.model, cfg.sim.conf, cfg.sim.iou)
    sim = Simulation(scenario, detector, controller, cfg)
    print(f"path={cfg.scenario.path} length={scenario.transect.length:.1f}m seed={args.seed} "
          f"controller={args.controller} params={params}")

    topdown = TopDown(scenario) if (args.show or args.video) else None
    window = Window((2 * PANEL_W, PANEL_H), controller, TUNABLES) if args.show else None
    writer = None
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / cfg.sim.perception_dt,
                                 (2 * PANEL_W, PANEL_H))
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)

    def on_frame(ctx):
        if args.save_frames:
            cv2.imwrite(f"{args.save_frames}/frame_{ctx['row']['frame']:05d}.png", cv2.cvtColor(ctx["frame"], cv2.COLOR_RGB2BGR))
        if topdown is not None:
            live = controller.get_params()  # current slider values
            img = compose(ctx, topdown, cfg.camera, live["deadband"], live["deadband_center"])
            if writer:
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if window and not window.show(img, ctx):
                return False
            if args.realtime:
                time.sleep(cfg.sim.perception_dt)
        elif ctx["row"]["frame"] % 25 == 0:
            r = ctx["row"]
            print(f"  t={r['t']:5.1f}s {r['state']:5s} cte={r['cte']:+.2f} progress={r['progress']:.1f}m")
        return True

    try:
        result = sim.run(on_frame)
    finally:
        if writer:
            writer.release()
        if window:
            window.close()

    print("\nresult:")
    for k, v in result.metrics.items():
        print(f"  {k:15s} {v}")
    if not args.no_log:
        out = Path(__file__).parent / "runs"
        out.mkdir(exist_ok=True)
        name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{cfg.scenario.path}_s{args.seed}_{args.controller}.csv"
        result.write_csv(out / name)
        print(f"log: runs/{name}")
    print(f"wall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
