"""Pluggable controllers. Every controller implements  step(obs) -> (Command, Telemetry).

`obs` carries everything a controller could use: the best detection (cx, cy, w, h, angle, score, cls) or None, its
horizontal offset (-1..1, as computed by the real box_offset) or None, the image size, and the simulated time.

To compare your own algorithm, subclass / write a class with the same step() and add it to REGISTRY.
Parameters come from Naive_YOLO/params.json (the file the web page on the robot edits) plus any overrides.
"""
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from .robot_code import controller as rc

Command, Telemetry = rc.Command, rc.Telemetry

DEFAULT_PARAMS = Path(__file__).resolve().parent.parent.parent / "Naive_YOLO" / "params.json"


# Parameters offered as live sliders in the sim window. Mirrors TUNABLES in Naive_YOLO/stream.py (the web page):
# (name, label, kind, min, max, step); kind "sign" is a checkbox mapping checked -> -1.
TUNABLES = [
    ("surge", "Surge (-1000..1000)", "float", -1000, 1000, 10),
    ("yaw", "Yaw command (0..1000)", "float", 0, 1000, 10),
    ("deadband", "Deadband", "float", 0.0, 0.5, 0.01),
    ("deadband_center", "Deadband centre", "float", -0.5, 0.5, 0.01),
    ("smooth", "Smoothing (1 = none)", "float", 0.05, 1.0, 0.05),
    ("lost_hold", "Lost hold (s)", "float", 0.0, 5.0, 0.1),
    ("proportional", "Proportional yaw", "bool", 0, 1, 1),
    ("yaw_sign", "Invert yaw", "sign", -1, 1, 1),
]
TUNABLE_DEFAULTS = {"surge": 250, "yaw": 300, "deadband": 0.1, "deadband_center": 0.0, "smooth": 0.5,
                    "lost_hold": 1.0, "proportional": False, "yaw_sign": 1.0}


def _coerce(name, value):
    kind = next(t[2] for t in TUNABLES if t[0] == name)
    if kind == "float":
        t = next(t for t in TUNABLES if t[0] == name)
        return max(t[3], min(t[4], float(value)))
    if kind == "bool":
        return bool(value)
    return -1.0 if float(value) < 0 else 1.0


@dataclass
class Observation:
    det: Optional[tuple]
    offset: Optional[float]
    img_w: int
    img_h: int
    t: float


class NaiveAdapter:
    """The robot's own NaiveController, unchanged."""
    name = "naive"

    def __init__(self, params):
        # right_bias compensates the real vehicle's thrusters only; keep the sim symmetric
        known = {f.name for f in fields(rc.NaiveConfig)} - {"right_bias"}
        self.cfg = rc.NaiveConfig(**{k: v for k, v in params.items() if k in known})
        self.ctrl = rc.NaiveController(self.cfg)

    def step(self, obs):
        return self.ctrl.update(obs.offset, obs.t)

    # live parameter access (sliders): the NaiveController reads self.cfg on every update, so edits apply at once
    def get_params(self):
        return {n: getattr(self.cfg, n, TUNABLE_DEFAULTS[n]) for n, *_ in TUNABLES}

    def set_param(self, name, value):
        if hasattr(self.cfg, name):
            setattr(self.cfg, name, _coerce(name, value))


class NaiveProportionalAdapter(NaiveAdapter):
    """Same controller with proportional yaw switched on."""
    name = "naive_p"

    def __init__(self, params):
        super().__init__({**params, "proportional": True})


class PDAdapter:
    """Example alternative: proportional + derivative on the offset, surge reduced while turning hard.

    params used: surge, yaw (max yaw), deadband, smooth, lost_hold, yaw_sign, kp (default 1.0), kd (default 0.3)."""
    name = "pd"

    def __init__(self, params):
        self.p = dict(params)
        self.off = None
        self.prev_off, self.prev_t = None, None
        self.last_seen = None

    def step(self, obs):
        p = self.p
        surge, yaw_max = p.get("surge", 250), p.get("yaw", 300)
        if obs.offset is None:
            if self.last_seen is not None and obs.t - self.last_seen < p.get("lost_hold", 1.0):
                return Command(surge * 0.5), Telemetry("COAST", self.off or 0.0)
            self.off = self.prev_off = None
            return Command(), Telemetry("LOST")
        a = p.get("smooth", 0.5)
        self.off = obs.offset if self.off is None else self.off + a * (obs.offset - self.off)
        d = 0.0
        if self.prev_off is not None and obs.t > self.prev_t:
            d = (self.off - self.prev_off) / (obs.t - self.prev_t)
        self.prev_off, self.prev_t, self.last_seen = self.off, obs.t, obs.t
        err = self.off - p.get("deadband_center", 0.0)
        e = 0.0 if abs(err) < p.get("deadband", 0.1) else err
        yaw = max(-yaw_max, min(yaw_max, p.get("kp", 1.0) * e * yaw_max + p.get("kd", 0.3) * d * yaw_max))
        return Command(surge * (1.0 - 0.5 * abs(self.off)), p.get("yaw_sign", 1.0) * yaw), Telemetry("TRACK", self.off)

    def get_params(self):
        return {n: self.p.get(n, TUNABLE_DEFAULTS[n]) for n, *_ in TUNABLES}

    def set_param(self, name, value):
        self.p[name] = _coerce(name, value)


REGISTRY = {c.name: c for c in (NaiveAdapter, NaiveProportionalAdapter, PDAdapter)}


def load_params(path=None, overrides=None):
    params = {}
    path = Path(path) if path else DEFAULT_PARAMS
    if path.exists():
        params = json.loads(path.read_text())
    params.update(overrides or {})
    return params


def make_controller(name, params):
    if name not in REGISTRY:
        raise SystemExit(f"unknown controller '{name}', choose from {sorted(REGISTRY)}")
    return REGISTRY[name](params)
