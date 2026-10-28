#!/usr/bin/env python3
"""Batch evaluation: many seeds x paths x controllers x parameter values, results to a CSV and a summary table.

    ../yolo_env_mac/bin/python batch.py --seeds 10 --paths gentle_s sine turn90 \
        --controllers naive pd --sweep deadband=0.1,0.2,0.33 --sweep yaw=150,300

Scenarios (floor texture etc.) are built once per (path, seed) and shared by all controller / parameter combinations,
so every combination faces exactly the same world.
"""
import argparse
import csv
import itertools
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from cli_common import add_common_args, build_config
from sim.controllers import load_params, make_controller
from sim.detector import OracleDetector, YoloDetector
from sim.simulation import Simulation, build_scenario
from sim.world import PATH_KINDS


def parse_sweeps(items):
    sweeps = {}
    for it in items or []:
        k, vals = it.split("=", 1)
        sweeps[k] = [float(v) for v in vals.split(",")]
    return sweeps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0 .. N-1)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--paths", nargs="+", choices=PATH_KINDS, default=None, help="default: just --path")
    p.add_argument("--controllers", nargs="+", default=None, help="default: just --controller")
    p.add_argument("--sweep", action="append", metavar="KEY=V1,V2,..", help="controller param values to sweep")
    p.add_argument("--out", default=None, help="CSV path (default runs/batch_<time>.csv)")
    args = p.parse_args()

    paths = args.paths or [args.path]
    controllers = args.controllers or [args.controller]
    sweeps = parse_sweeps(args.sweep)
    combos = [dict(zip(sweeps, vals)) for vals in itertools.product(*sweeps.values())] or [{}]
    base_params = load_params(args.params)

    yolo = None if args.oracle else YoloDetector(args.model, args.conf, args.iou)
    rows = []
    n_total = len(paths) * args.seeds * len(controllers) * len(combos)
    done, t0 = 0, time.time()
    for path in paths:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            args.path = path
            cfg = build_config(args, seed=seed)
            scenario = build_scenario(cfg)
            detector = OracleDetector(scenario, seed=seed) if args.oracle else yolo
            for cname in controllers:
                for combo in combos:
                    params = {**base_params, **combo}
                    res = Simulation(scenario, detector, make_controller(cname, params), cfg).run()
                    rows.append(dict(path=path, seed=seed, controller=cname,
                                     params=";".join(f"{k}={v}" for k, v in combo.items()), **res.metrics))
                    done += 1
                    print(f"[{done}/{n_total}] {path} s{seed} {cname} {combo} -> {res.metrics['end_reason']} "
                          f"progress={res.metrics['progress_frac']} rms_cte={res.metrics['rms_cte']}  "
                          f"({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out) if args.out else Path(__file__).parent / "runs" / f"batch_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["path"], r["controller"], r["params"])].append(r)
    print(f"\n{'path':10s} {'controller':10s} {'params':34s} {'n':>3s} {'finish%':>8s} {'progress':>9s} "
          f"{'rms_cte':>8s} {'max_cte':>8s} {'lost%':>6s}")
    for (path, c, prm), g in sorted(groups.items()):
        m = lambda k: float(np.mean([r[k] for r in g]))
        print(f"{path:10s} {c:10s} {prm:34s} {len(g):3d} {100 * m('finished'):8.0f} {m('progress_frac'):9.2f} "
              f"{m('rms_cte'):8.3f} {m('max_abs_cte'):8.3f} {100 * m('frac_lost'):6.1f}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
