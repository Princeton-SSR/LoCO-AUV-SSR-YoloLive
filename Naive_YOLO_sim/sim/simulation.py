"""Closed loop: render frame -> real YOLO -> controller -> (latency) -> dynamics -> new pose."""
import csv
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .camera import FisheyeCamera
from .config import Config
from .controllers import Command, Observation
from .dynamics import Robot
from .render import Renderer
from .robot_code import controller as rc
from .world import Transect, build_floor, make_transect


@dataclass
class Scenario:
    """Everything that depends only on (path, seed): transect, floor texture, camera and renderer. Reusable across
    controllers / parameter sweeps (call renderer.reset between runs)."""
    cfg: Config
    transect: Transect
    camera: FisheyeCamera
    renderer: Renderer
    start: tuple  # (x, y, psi)


def build_scenario(cfg: Config) -> Scenario:
    sc = cfg.scenario
    rng = np.random.default_rng(sc.seed)
    transect = make_transect(sc.path, sc.length_m, rng, sc.n_gaps, sc.gap_length_m)
    floor = build_floor(transect, sc, rng)
    camera = FisheyeCamera(cfg.camera)
    renderer = Renderer(camera, floor, cfg.render, sc.view_range_m, seed=sc.seed)

    lat, hdg = sc.start_lateral_m, np.radians(sc.start_heading_deg)
    if sc.randomize_start:
        lat, hdg = rng.uniform(-lat, lat), rng.uniform(-hdg, hdg)
    return Scenario(cfg, transect, camera, renderer, (0.0, lat, hdg))


FRAME_FIELDS = ["t", "frame", "state", "detected", "score", "box_cx", "box_cy", "box_w", "box_h", "box_angle",
                "offset", "smoothed_offset", "surge", "yaw", "infer_ms", "x", "y", "psi_deg", "cte", "progress"]


@dataclass
class Result:
    metrics: dict
    frames: list = field(default_factory=list)       # one dict (FRAME_FIELDS) per perception frame
    traj: np.ndarray = None                           # (N, 5): t, x, y, psi, cte at every physics step

    def write_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
            w.writeheader()
            w.writerows(self.frames)


class Simulation:
    def __init__(self, scenario: Scenario, detector, controller, cfg: Config = None):
        self.sc = scenario
        self.cfg = cfg or scenario.cfg
        self.detector = detector
        self.controller = controller

    def run(self, on_frame=None):
        """Run until finished / lost / diverged / out of time. `on_frame(ctx)` is called after every perception
        frame (return False to abort); ctx is a dict with the frame, detection, pose and so on."""
        sim, sc = self.cfg.sim, self.sc
        tr, cam = sc.transect, sc.camera
        sc.renderer.reset(sim.seed)
        x0, y0, psi0 = sc.start
        robot = Robot(self.cfg.dynamics, x0, y0, psi0, seed=sim.seed)
        W, H = cam.cfg.width, cam.cfg.height

        t, next_frame, frame_no = 0.0, 0.0, 0
        cmd = Command()
        pending = deque()  # (time the command reaches the thrusters, Command)
        last_seen = 0.0
        frames, traj = [], []
        reason, max_s = "timeout", 0.0

        while t < sim.duration_s:
            if t >= next_frame - 1e-9:
                skip = on_frame is None and getattr(self.detector, "ignores_frame", False)
                frame = None if skip else sc.renderer.render(robot.x, robot.y, robot.psi, t)
                pose = (robot.x, robot.y, robot.psi) if getattr(self.detector, "needs_pose", False) else None
                dets, infer_ms = self.detector.detect(frame, pose=pose)
                det = rc_pick(dets)
                offset = rc.box_offset(det, W) if det is not None else None
                if offset is not None:
                    last_seen = t
                new_cmd, tel = self.controller.step(Observation(det, offset, W, H, t))
                pending.append((t + sim.command_latency_s, new_cmd))

                s, cte = tr.project(robot.x, robot.y)
                row = dict(t=round(t, 3), frame=frame_no, state=tel.state, detected=int(det is not None),
                           score=det[5] if det else "", box_cx=det[0] if det else "", box_cy=det[1] if det else "",
                           box_w=det[2] if det else "", box_h=det[3] if det else "", box_angle=det[4] if det else "",
                           offset=offset if offset is not None else "", smoothed_offset=tel.offset,
                           surge=new_cmd.surge, yaw=new_cmd.yaw, infer_ms=round(infer_ms, 1), x=robot.x, y=robot.y,
                           psi_deg=np.degrees(robot.psi), cte=cte, progress=s)
                frames.append(row)
                if on_frame is not None:
                    ctx = dict(frame=frame, dets=dets, det=det, tel=tel, cmd=new_cmd, applied=cmd, offset=offset, t=t, row=row,
                               robot=robot, traj=traj, scenario=sc, infer_ms=infer_ms)
                    if on_frame(ctx) is False:
                        reason = "aborted"
                        break
                next_frame += sim.perception_dt
                frame_no += 1
                if t - last_seen > sim.lost_timeout_s:
                    reason = "lost_timeout"
                    break

            while pending and pending[0][0] <= t:
                cmd = pending.popleft()[1]
            robot.step(cmd, sim.physics_dt)
            t += sim.physics_dt

            s, cte = tr.project(robot.x, robot.y)
            traj.append((t, robot.x, robot.y, robot.psi, cte))
            max_s = max(max_s, s)
            if max_s >= tr.length - sim.finish_margin_m:
                reason = "finished"
                break
            if abs(cte) > sim.diverge_cte_m:
                reason = "diverged"
                break

        traj = np.array(traj) if traj else np.zeros((0, 5))
        return Result(self._metrics(frames, traj, max_s, t, reason), frames, traj)

    def _metrics(self, frames, traj, max_s, t_end, reason):
        L = self.sc.transect.length
        cte = traj[:, 4] if len(traj) else np.zeros(1)
        states = [f["state"] for f in frames] or ["LOST"]
        n = len(states)
        return dict(
            end_reason=reason, finished=int(reason == "finished"), time_s=round(t_end, 2),
            progress_frac=round(min(max_s / L, 1.0), 3), mean_abs_cte=round(float(np.mean(np.abs(cte))), 4),
            rms_cte=round(float(np.sqrt(np.mean(cte ** 2))), 4), max_abs_cte=round(float(np.max(np.abs(cte))), 4),
            frac_track=round(states.count("TRACK") / n, 3), frac_coast=round(states.count("COAST") / n, 3),
            frac_lost=round(states.count("LOST") / n, 3),
            detect_rate=round(float(np.mean([f["detected"] for f in frames])) if frames else 0.0, 3),
            mean_infer_ms=round(float(np.mean([f["infer_ms"] for f in frames])) if frames else 0.0, 1),
            n_frames=len(frames))


def rc_pick(dets):
    return rc.pick_detection(dets)
