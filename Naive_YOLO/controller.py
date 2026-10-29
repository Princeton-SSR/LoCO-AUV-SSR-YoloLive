"""Very simple transect follower logic (no camera, no MAVLink -> easy to unit test).

Rule: look at where the detected line box sits horizontally in the image.
    box centre right of image centre  -> yaw right
    box centre left of image centre   -> yaw left
    close to the centre (deadband)    -> don't yaw
    (the "centre" is image centre + deadband_center, so the deadband can be shifted left/right)
The robot always creeps forward at a constant surge while it can see the line.

No lookahead, no line direction, no PID: just the horizontal offset of the box centre.
"""
from dataclasses import dataclass


@dataclass
class Command:
    """Actuator request in MANUAL_CONTROL units, each in [-1000, 1000]: surge (+ forward) and yaw (+ clockwise / right)."""
    surge: float = 0.0
    yaw: float = 0.0


@dataclass
class NaiveConfig:
    """Tunable parameters of NaiveController."""
    surge: float = 250          # constant forward command while the line is visible, -1000..1000
    yaw: float = 300        # yaw magnitude: fixed in bang-bang mode, the maximum in proportional mode, 0..1000
    deadband: float = 0.10      # |offset - deadband_center| below this -> no yaw (offset is in half-image-widths, -1..1)
    deadband_center: float = 0.0  # where the deadband sits: 0 = image centre, +0.2 = 20% of the half-width to the right.
                                  # The controller steers so the line ends up at this offset
    proportional: bool = False  # False: fixed +-yaw (bang-bang). True: yaw * offset (offset is -1..1)
    right_bias: float = 0.0     # extra yaw magnitude added only when turning right (line right of the deadband), 0..1000.
                                # Applied before yaw_sign, so "right" always means toward the line on the right.
                                # Compensates a vehicle that turns right more weakly than left. 0 = symmetric (the sim leaves it at 0)
    smooth: float = 0.5         # 0..1, weight of the newest offset in the moving average (1 = no smoothing)
    yaw_sign: float = 1.0       # use -1 if the robot turns away from the line
    lost_hold: float = 0.0      # seconds to keep creeping forward after losing the line (0 = stop immediately: LOST, both thrusters 0)
    lost_surge_frac: float = 0.5
    lost_timeout: float = 10.0  # seconds without the line (counted from the last detection, so it includes lost_hold) after which
                                # follow_transect.py ends the run and sends neutral. 0 = never stop. Not used by the controller itself
    lost_spin: float = 0.0      # 0 = off. Otherwise, in LOST (after lost_hold) spin in place (surge 0) at this yaw command, 0..1000,
                                # toward the side of the image centre where the line was last seen, until it is found again


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
        self._last_side = 1.0   # +1 / -1: side of the image centre the line was last seen on; starts at +1 (right) so a run that
                                # begins without the line spins right when lost_spin is set

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
        self._last_side = 1.0 if self._off >= 0 else -1.0  # relative to the image centre, not deadband_center

        err = self._off - c.deadband_center  # offset relative to where the line should sit
        if abs(err) < c.deadband:
            yaw = 0.0
        elif c.proportional:
            yaw = c.yaw * max(-1.0, min(1.0, err))
        else:
            yaw = c.yaw if err > 0 else -c.yaw
        if yaw > 0:  # turning right: add the right-side bias, then clamp to the MANUAL_CONTROL range
            yaw = min(1000.0, yaw + c.right_bias)
        return Command(c.surge, c.yaw_sign * yaw), Telemetry("TRACK", self._off)

    def _lost(self, t):
        """No line: creep forward for `lost_hold` seconds (COAST), then LOST: stop, or if `lost_spin` is set spin in
        place toward the side the line was last seen on. The smoothed offset is forgotten."""
        c = self.cfg
        if self._last_seen is not None and t - self._last_seen < c.lost_hold:
            return Command(c.surge * c.lost_surge_frac), Telemetry("COAST", self._off or 0.0)
        self._off = None
        if c.lost_spin:
            # side +1 = line was on the right -> yaw right (+ before yaw_sign), same convention as tracking
            return Command(0.0, c.yaw_sign * self._last_side * min(1000.0, abs(c.lost_spin))), Telemetry("LOST")
        return Command(), Telemetry("LOST")
