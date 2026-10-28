"""All tunable simulator parameters in one place (plain dataclasses, easy to override from the CLI).

Conventions used everywhere in the simulator
    world frame : x, y in metres, floor is z = 0, heading psi in radians, counter-clockwise positive (math convention)
    robot       : forward = (cos psi, sin psi), right = (sin psi, -cos psi); a +yaw command (clockwise) DEcreases psi
    camera      : OpenCV style (x right, y down, z forward), mounted on the robot centreline, `forward_offset_m`
                  ahead of the rotation centre, `height_m` above the floor, pitched down by `pitch_deg`
    cte         : signed cross-track error in metres, positive when the robot is to the LEFT of the transect
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CameraConfig:
    """Fisheye window as seen in the Dillon frames: a ~325 px wide image circle, centred at (352, 252) of 640x480.
    Equidistant model r = f * (theta + k1 theta^3 + k2 theta^5). fov_deg, height_m and pitch_deg are estimates."""
    width: int = 640
    height: int = 480
    fov_deg: float = 140.0          # field of view across the image-circle diameter (estimate)
    circle_radius_px: float = 162.0 # image circle radius in px; 0 -> height / 2
    center_dx: float = 32.5         # image-circle centre offset from the frame centre, px (measured)
    center_dy: float = 12.5
    k1: float = 0.0                 # extra radial distortion terms (0 = ideal equidistant)
    k2: float = 0.0
    height_m: float = 1.0           # camera height above the pool floor
    pitch_deg: float = 0.0          # 0 = looks straight ahead (parallel to the floor), + = tilted down
    forward_offset_m: float = 0.25  # camera distance ahead of the robot's rotation centre
    supersample: int = 1            # render at N x resolution then downsample (anti-aliasing, N^2 slower)


@dataclass
class ScenarioConfig:
    """The transect and the pool floor."""
    path: str = "gentle_s"          # straight | gentle_s | sine | arc | turn90 | random
    length_m: float = 12.0
    seed: int = 0
    n_gaps: int = 0                 # number of places where the tape is missing
    gap_length_m: Tuple[float, float] = (0.3, 0.8)
    tape_width_m: float = 0.025
    tape_color: Tuple[int, int, int] = (225, 215, 145)  # RGB albedo: pale yellow-green, low contrast on the floor
    floor_color: Tuple[int, int, int] = (232, 232, 196)  # RGB albedo of the very bright, pale pool floor
    floor_contrast: float = 0.25    # strength of the floor mottling (1 = strong); the real floor is nearly uniform
    floor_res_m: float = 0.006      # metres per texture pixel (smaller = sharper tape, more RAM)
    tile_m: float = 0.0             # >0 draws a tile-grid with this pitch (0 = plain floor)
    floor_image: Optional[str] = None   # optional real top-down floor photo to tile instead of the procedural floor
    floor_image_width_m: float = 2.0    # how many metres the width of `floor_image` covers
    view_range_m: float = 8.0       # beyond this slant range the floor fades fully into the water colour
    # initial pose relative to the start of the transect (which starts at the origin heading +x)
    start_lateral_m: float = 0.15   # max |lateral offset|; sampled uniformly in +- this when randomize_start
    start_heading_deg: float = 10.0  # max |heading error|; sampled uniformly in +- this when randomize_start
    randomize_start: bool = True


@dataclass
class RenderConfig:
    """Underwater look: veiling light, colour attenuation, lighting, lens and sensor effects."""
    water_color: Tuple[float, float, float] = (0.08, 0.74, 0.74)    # veiling light at the horizon (RGB 0..1)
    water_color_up: Tuple[float, float, float] = (0.23, 0.75, 0.64)  # veiling light when looking up towards the surface
    attenuation_per_m: Tuple[float, float, float] = (0.15, 0.03, 0.02)  # per-channel extinction (red dies first)
    exposure: float = 1.0
    exposure_jitter: float = 0.08   # +- fractional per-run exposure variation, drifting slowly over time
    caustics: float = 0.04          # amplitude of the moving light pattern on the floor (0 = off)
    caustic_scale_m: float = 0.5
    blur_sigma_px: float = 1.0      # lens + video-compression softness
    vignette: float = 0.15          # darkening towards the edge of the image circle
    noise_sigma: float = 0.015      # sensor noise (fraction of full scale)
    outside_circle_color: Tuple[float, float, float] = (0.16, 0.04, 0.10)  # used only if no housing image
    housing_image: Optional[str] = "assets/housing.png"  # static camera-housing ring pasted outside the circle


@dataclass
class DynamicsConfig:
    """Planar vehicle model: first-order lag from command to speed / yaw rate, plus a wandering current."""
    max_speed_mps: float = 0.5      # forward speed at surge = 1000
    max_yaw_rate_dps: float = 60.0  # yaw rate at |yaw| = 1000
    tau_speed_s: float = 0.8
    tau_yaw_s: float = 0.5
    current_speed_mps: float = 0.0  # steady current
    current_dir_deg: float = 90.0   # direction the water moves towards (world frame)
    drift_noise_mps: float = 0.02   # random-walk (Ornstein-Uhlenbeck) velocity disturbance
    drift_tau_s: float = 3.0
    yaw_noise_dps: float = 0.5      # white noise on the yaw rate
    cmd_deadzone: float = 0.0       # |command| below this does nothing, in MANUAL_CONTROL units


@dataclass
class SimConfig:
    """Run-level settings."""
    duration_s: float = 90.0
    physics_dt: float = 0.02
    perception_dt: float = 0.12     # frame period (capture + inference), the real loop runs ~8-12 Hz on the Pi
    command_latency_s: float = 0.2  # frame capture -> thrusters (inference + MAVLink + thruster spin-up)
    lost_timeout_s: float = 10.0    # stop if the line was not detected for this long (as follow_transect.py)
    finish_margin_m: float = 0.3    # run counts as finished this close to the end of the transect
    diverge_cte_m: float = 4.0      # abort if the robot gets this far from the transect
    conf: float = 0.35
    iou: float = 0.45
    seed: int = 0


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    sim: SimConfig = field(default_factory=SimConfig)
