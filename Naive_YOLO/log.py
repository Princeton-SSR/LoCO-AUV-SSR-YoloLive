"""Per-run logging for follow_transect.py.

Each run creates its own CSV in `log_dir`, named by the start time, e.g. logs/2026-10-20_16-54-03.csv.
One row per control-loop iteration:

    wall_time   ISO timestamp of the frame
    t           seconds since the run started
    frame       frame counter
    state       TRACK / COAST / LOST
    detected    1 if a box was picked this frame, else 0
    score       detection confidence (blank if none)
    box_cx, box_cy, box_w, box_h, box_angle   picked oriented box in image pixels (angle in radians; blank if none)
    raw_offset  box-centre offset from image centre, -1..1 (blank if none)
    offset      smoothed offset used by the controller
    surge, yaw  commands sent to the vehicle, -1..1
    infer_ms    model inference time
    param_*     controller parameters in effect for that frame (surge, yaw_cmd, deadband, deadband_center, smooth, lost_hold,
                proportional, yaw_sign); they change mid-run if edited on the web page

Usage:
    logger = RunLogger(cfg=cfg)     # opens a new file; cfg's current values are logged on every row
    logger.log(t, state, det, raw_offset, offset, cmd, infer_ms)   # once per frame
    logger.close()
"""
import csv
import os
from datetime import datetime

# Always the `logs` folder next to this file, no matter which directory the script is launched from
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Controller parameters recorded on every row (columns are named param_<name>). Logged per row, not once,
# because they can be changed live from the web page during a run.
PARAM_FIELDS = ["surge", "yaw_cmd", "deadband", "deadband_center", "smooth", "lost_hold", "proportional", "yaw_sign"]

COLUMNS = ["wall_time", "t", "frame", "state", "detected", "score",
           "box_cx", "box_cy", "box_w", "box_h", "box_angle",
           "raw_offset", "offset", "surge", "yaw", "infer_ms"] + ["param_" + p for p in PARAM_FIELDS]


class RunLogger:
    """Writes one CSV per run, named by the start timestamp. Flushes every row so a crash loses nothing."""

    def __init__(self, log_dir=LOG_DIR, cfg=None):
        """cfg: the live NaiveConfig; its current values are written on every row."""
        self.cfg = cfg
        os.makedirs(log_dir, exist_ok=True)
        self.start = datetime.now()
        self.path = os.path.join(log_dir, self.start.strftime("%Y-%m-%d_%H-%M-%S") + ".csv")
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)
        self._n = 0
        print(f"Logging to {self.path}")

    def log(self, t, state, det, raw_offset, offset, cmd, infer_ms):
        """Record one frame. det is (cx, cy, w, h, angle, score, cls) or None; cmd has .surge and .yaw."""
        self._n += 1
        if det is not None:
            box = [f"{det[0]:.1f}", f"{det[1]:.1f}", f"{det[2]:.1f}", f"{det[3]:.1f}", f"{det[4]:.4f}"]
            score = f"{det[5]:.3f}"
        else:
            box, score = [""] * 5, ""
        self._w.writerow([
            datetime.now().isoformat(timespec="milliseconds"), f"{t:.3f}", self._n, state,
            int(det is not None), score, *box,
            "" if raw_offset is None else f"{raw_offset:.4f}", f"{offset:.4f}",
            f"{cmd.surge:.3f}", f"{cmd.yaw:.3f}", f"{infer_ms:.0f}",
            *[self._param(p) for p in PARAM_FIELDS]])
        self._f.flush()

    def _param(self, name):
        """Current value of a config field as a CSV cell (blank if there is no config)."""
        if self.cfg is None:
            return ""
        v = getattr(self.cfg, name)
        return int(v) if isinstance(v, bool) else v

    def close(self):
        if not self._f.closed:
            self._f.close()
