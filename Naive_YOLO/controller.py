"""Very simple transect follower logic (no camera, no MAVLink -> easy to unit test).

Rule: look at where the detected line box sits horizontally in the image.
    box centre right of image centre  -> yaw right
    box centre left of image centre   -> yaw left
    close to the centre (deadband)    -> don't yaw
The robot always creeps forward at a constant surge while it can see the line.

No lookahead, no line direction, no PID: just the horizontal offset of the box centre.
"""
from dataclasses import dataclass


@dataclass
class Command:
    """Normalised actuator request, each in [-1, 1]: surge (+ forward) and yaw (+ clockwise / right)."""
    surge: float = 0.0
    yaw: float = 0.0


@dataclass
class NaiveConfig:
    """Tunable parameters of NaiveController."""
    surge: float = 0.25         # constant forward command while the line is visible
    yaw_cmd: float = 0.30       # yaw magnitude: fixed in bang-bang mode, the maximum in proportional mode
    deadband: float = 0.10      # |offset| below this -> no yaw (offset is in half-image-widths, -1..1)
    proportional: bool = False  # False: fixed +-yaw_cmd (bang-bang). True: yaw_cmd * offset / 1.0
    smooth: float = 0.5         # 0..1, weight of the newest offset in the moving average (1 = no smoothing)
    yaw_sign: float = 1.0       # use -1 if the robot turns away from the line
    lost_hold: float = 1.0      # seconds to keep creeping forward after losing the line
    lost_surge_frac: float = 0.5


@dataclass
class Telemetry:
    """Diagnostics from one update: state (TRACK / COAST / LOST) and the smoothed horizontal offset."""
    state: str = "LOST"
    offset: float = 0.0


def box_offset(det, img_w):
    """Horizontal offset of the OBB centre from the image centre, in half-image-widths (-1 left ... +1 right)."""
    return (det[0] - img_w / 2.0) / (img_w / 2.0)


def pick_detection(dets):
    """Highest-confidence detection (dets entries: cx, cy, w, h, r, score, cls)."""
    return max(dets, key=lambda d: d[5]) if dets else None


class NaiveController:
    """Steers toward the line by turning whichever way the box centre is off to. Call update() once per frame."""

    def __init__(self, cfg=None):
        self.cfg = cfg or NaiveConfig()
        self._off = None
        self._last_seen = None

    def update(self, offset, t):
        """offset: box-centre offset (-1..1) or None if no line was detected. t: monotonic seconds.

        Returns (Command, Telemetry).
        """
        c = self.cfg
        if offset is None:
            return self._lost(t)

        # moving average so a single noisy detection doesn't flip the yaw direction
        self._off = offset if self._off is None else self._off + c.smooth * (offset - self._off)
        self._last_seen = t

        if abs(self._off) < c.deadband:
            yaw = 0.0
        elif c.proportional:
            yaw = c.yaw_cmd * max(-1.0, min(1.0, self._off))
        else:
            yaw = c.yaw_cmd if self._off > 0 else -c.yaw_cmd
        return Command(c.surge, c.yaw_sign * yaw), Telemetry("TRACK", self._off)

    def _lost(self, t):
        """No line: creep forward for `lost_hold` seconds (COAST), then stop and forget the smoothed offset (LOST)."""
        c = self.cfg
        if self._last_seen is not None and t - self._last_seen < c.lost_hold:
            return Command(c.surge * c.lost_surge_frac), Telemetry("COAST", self._off or 0.0)
        self._off = None
        return Command(), Telemetry("LOST")
