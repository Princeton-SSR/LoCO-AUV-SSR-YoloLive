"""Approximate planar vehicle dynamics: first-order lag from command to surge speed / yaw rate, plus a current."""
import numpy as np

from .config import DynamicsConfig


class Robot:
    """State: x, y (m, at the rotation centre), psi (rad, CCW+), forward speed u, yaw rate r (CCW+)."""

    def __init__(self, cfg: DynamicsConfig, x=0.0, y=0.0, psi=0.0, seed=0):
        self.cfg = cfg
        self.x, self.y, self.psi = x, y, psi
        self.u = 0.0
        self.r = 0.0
        self.rng = np.random.default_rng(seed + 99)
        self.drift = np.zeros(2)  # random velocity disturbance (OU process)
        ang = np.radians(cfg.current_dir_deg)
        self.current = cfg.current_speed_mps * np.array([np.cos(ang), np.sin(ang)])

    def step(self, cmd, dt):
        """Advance by dt seconds with the normalised command (surge, yaw in [-1, 1]; +yaw = clockwise)."""
        c = self.cfg
        surge = 0.0 if abs(cmd.surge) < c.cmd_deadzone else float(np.clip(cmd.surge, -1, 1))
        yaw = 0.0 if abs(cmd.yaw) < c.cmd_deadzone else float(np.clip(cmd.yaw, -1, 1))
        u_target = c.max_speed_mps * surge
        r_target = -np.radians(c.max_yaw_rate_dps) * yaw          # clockwise command -> psi decreases

        self.u += dt * (u_target - self.u) / c.tau_speed_s
        self.r += dt * (r_target - self.r) / c.tau_yaw_s
        self.drift += (-dt * self.drift / c.drift_tau_s
                       + c.drift_noise_mps * np.sqrt(2 * dt / c.drift_tau_s) * self.rng.standard_normal(2))

        self.psi += (self.r + np.radians(c.yaw_noise_dps) * self.rng.standard_normal()) * dt
        vx = self.u * np.cos(self.psi) + self.current[0] + self.drift[0]
        vy = self.u * np.sin(self.psi) + self.current[1] + self.drift[1]
        self.x += vx * dt
        self.y += vy * dt
