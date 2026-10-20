"""Pure-math core of the transect follower (no camera, no MAVLink -> easy to unit test).

Frames
------
Robot frame: +x = forward, +y = right. The origin is the point you want centred over the line.
All "commands" are normalised to [-1, 1]; the vehicle layer scales them to MANUAL_CONTROL units.

Control law
-----------
  * Pure pursuit: aim at the point on the line a distance `lookahead` ahead of the robot.
    alpha = bearing to that point (rad, + = to the right)  ->  PID  ->  yaw command.
  * The vehicle is non-holonomic (surge + yaw only, no sway), so centring on the line is done purely by steering:
    the lookahead bearing already encodes both heading error and cross-track offset. The PID's integral term
    removes steady bias (e.g. thruster asymmetry). Cross-track e is still computed for telemetry.
  * Surge: constant, scaled down while |alpha| is large (and a bit when far off the line).
"""
from dataclasses import dataclass, field
import json
import math

import numpy as np


# ----------------------------------------------------------------------------- PID
class PID:
    """PID on a scalar error, output clamped to +-out_limit.

    Derivative is low-pass filtered; the integral is clamped and frozen while the output is saturated (anti-windup).
    """

    def __init__(self, kp, ki, kd, out_limit, i_limit=None, d_smooth=0.3):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit = out_limit
        self.i_limit = i_limit if i_limit is not None else out_limit
        self.d_smooth = d_smooth  # 0..1, weight of the newest derivative sample
        self.reset()

    def reset(self):
        """Forget all history (integral, derivative, last timestamp), e.g. after losing the line."""
        self._t = None
        self._prev = 0.0
        self._i = 0.0
        self._d = 0.0

    def update(self, err, t):
        """Return the control output for error `err` measured at monotonic time `t` (seconds)."""
        dt = None if self._t is None else t - self._t
        if dt is None or dt <= 0 or dt > 1.0:  # first sample or long gap: no I/D this step
            self._t, self._prev = t, err
            out = self.kp * err
            return float(np.clip(out, -self.out_limit, self.out_limit))

        raw_d = (err - self._prev) / dt
        self._d += self.d_smooth * (raw_d - self._d)
        unsat = self.kp * err + self.ki * self._i + self.kd * self._d
        # conditional integration: don't wind up while saturated and still pushing the same way
        if not (abs(unsat) > self.out_limit and err * unsat > 0):
            self._i = float(np.clip(self._i + err * dt, -self.i_limit, self.i_limit))
        out = self.kp * err + self.ki * self._i + self.kd * self._d
        self._t, self._prev = t, err
        return float(np.clip(out, -self.out_limit, self.out_limit))


# ----------------------------------------------------------------------------- geometry
def homography_from_points(src, dst):
    """Fit a 3x3 homography H with dst ~ H @ src from >= 4 point pairs (direct linear transform via SVD).

    Used to map image pixels onto the floor plane so distances/angles are real, not perspective-distorted.
    """
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


class GroundMap:
    """Image pixel <-> robot-frame (forward, right) coordinates.

    Default (no calibration): pseudo-metric. The reference pixel `ref_uv` is the robot origin, and units are
    half-image-widths, so forward = (ref_v - v)/half_w, right = (u - ref_u)/half_w. Perspective is ignored, which
    is OK for a first test but under-estimates distances far up the image.

    Calibrated: pass a JSON file {"image_pts": [[u,v],...], "robot_pts": [[fwd,right],...]} (>= 4 pairs, e.g. tape
    marks on the floor measured from the point under the robot) and everything is in real units (metres).
    """

    def __init__(self, img_w, img_h, ref_frac=(0.5, 0.85), calib_path=None):
        """Set the robot-origin pixel (`ref_frac` of the image size) and optionally load a ground calibration JSON."""
        self.half_w = img_w / 2.0
        self.ref_uv = (ref_frac[0] * img_w, ref_frac[1] * img_h)
        self.H = self.Hinv = None
        if calib_path:
            with open(calib_path) as f:
                c = json.load(f)
            self.H = homography_from_points(c["image_pts"], c["robot_pts"])
            self.Hinv = np.linalg.inv(self.H)
        self.metric = self.H is not None

    def to_robot(self, u, v):
        """Convert pixel (u, v) to robot-frame (forward, right) coordinates."""
        if self.H is None:
            return np.array([(self.ref_uv[1] - v) / self.half_w, (u - self.ref_uv[0]) / self.half_w])
        p = self.H @ np.array([u, v, 1.0])
        return p[:2] / p[2]

    def to_pixel(self, fwd, right):
        """Inverse of to_robot: robot-frame (forward, right) back to pixel (u, v). Used for debug drawing."""
        if self.H is None:
            return (self.ref_uv[0] + right * self.half_w, self.ref_uv[1] - fwd * self.half_w)
        p = self.Hinv @ np.array([fwd, right, 1.0])
        return (p[0] / p[2], p[1] / p[2])


@dataclass
class Line:
    """Visible segment of the line in the robot frame. p_far has the larger forward (x) coordinate."""
    p_near: np.ndarray
    p_far: np.ndarray


def line_from_obb(det, gmap, min_aspect=2.0):
    """Turn an OBB detection (cx, cy, w, h, angle_rad, ...) into a robot-frame Line, or None if too stubby.

    A tape measure is long and thin, so the OBB's long axis is the line direction, and the box's far ends
    are the ends of the visible segment.
    """
    cx, cy, w, h, r = det[:5]
    long_, short_ = max(w, h), min(w, h)
    if short_ <= 0 or long_ / short_ < min_aspect:
        return None  # direction would be unreliable
    d = np.array([math.cos(r), math.sin(r)]) if w >= h else np.array([-math.sin(r), math.cos(r)])
    a = gmap.to_robot(cx - d[0] * long_ / 2, cy - d[1] * long_ / 2)
    b = gmap.to_robot(cx + d[0] * long_ / 2, cy + d[1] * long_ / 2)
    return Line(a, b) if a[0] <= b[0] else Line(b, a)


def pick_detection(dets):
    """Highest-confidence detection (dets entries: cx, cy, w, h, r, score, cls)."""
    return max(dets, key=lambda d: d[5]) if dets else None


# ----------------------------------------------------------------------------- controller
@dataclass
class Command:
    """Normalised actuator request, each in [-1, 1]: surge (+ forward) and yaw (+ clockwise / right)."""
    surge: float = 0.0
    yaw: float = 0.0


@dataclass
class ControllerConfig:
    """Tunable parameters of TransectController (lookahead, speeds, PID gains, limits, lost-line behaviour)."""
    lookahead: float = 0.7          # same units as GroundMap (half-widths, or metres if calibrated)
    surge_max: float = 0.30         # fraction of full MANUAL_CONTROL range
    surge_min_frac: float = 0.3     # surge never drops below this fraction of surge_max while tracking
    alpha_slow: float = 0.6         # rad: bearing at which surge has dropped to its minimum
    yaw_pid: tuple = (1.0, 0.05, 0.10)  # kp, ki, kd on alpha (rad) -> yaw command
    yaw_limit: float = 0.5
    lost_hold: float = 1.0          # seconds to keep creeping forward after losing the line
    lost_surge_frac: float = 0.5
    yaw_sign: float = 1.0           # flip if the robot turns away from the line


@dataclass
class Telemetry:
    """Per-update diagnostics: state (TRACK / COAST / LOST), cross-track e, bearing alpha, lookahead point."""
    found: bool = False
    e: float = 0.0
    alpha: float = 0.0
    lookahead_pt: tuple = None  # robot frame (fwd, right)
    state: str = "LOST"


class TransectController:
    """Pure pursuit + PID line follower for a surge/yaw-only vehicle. Call update() once per vision frame."""

    def __init__(self, cfg=None):
        self.cfg = cfg or ControllerConfig()
        c = self.cfg
        self.yaw_pid = PID(*c.yaw_pid, out_limit=c.yaw_limit)
        self._last_seen = None

    def update(self, line, t):
        """Compute the next command from the latest line observation.

        line: Line in the robot frame, or None if the line was not detected. t: monotonic seconds.
        Returns (Command, Telemetry). Steps: find the lookahead point on the line, take its bearing alpha,
        PID alpha -> yaw, and reduce surge when alpha is large.
        """
        c = self.cfg
        if line is None:
            return self._lost(t)

        D = line.p_far - line.p_near
        n = np.linalg.norm(D)
        if n < 1e-9:
            return self._lost(t)
        D = D / n
        F = line.p_near + (-line.p_near @ D) * D       # foot of the perpendicular from the origin
        e = float(F[1])                                 # + = line is on the robot's right
        # circle (radius = lookahead) / line intersection, forward one
        along = math.sqrt(max(c.lookahead ** 2 - float(F @ F), 0.0))
        along = min(along, float((line.p_far - F) @ D))  # don't aim past the visible end of the line
        LA = F + along * D
        alpha = math.atan2(LA[1], LA[0])                # + = lookahead point is to the right

        yaw = c.yaw_sign * self.yaw_pid.update(alpha, t)
        slow = float(np.clip(1.0 - abs(alpha) / c.alpha_slow, c.surge_min_frac, 1.0))
        self._last_seen = t
        return (Command(c.surge_max * slow, yaw),
                Telemetry(True, e, alpha, (float(LA[0]), float(LA[1])), "TRACK"))

    def _lost(self, t):
        """No usable line: creep forward for `lost_hold` seconds (COAST), then stop and reset the PID (LOST)."""
        c = self.cfg
        if self._last_seen is not None and t - self._last_seen < c.lost_hold:
            return Command(c.surge_max * c.lost_surge_frac), Telemetry(state="COAST")
        self.yaw_pid.reset()
        return Command(), Telemetry(state="LOST")
