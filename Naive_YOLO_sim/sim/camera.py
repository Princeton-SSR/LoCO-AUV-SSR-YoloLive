"""Fisheye camera model: pixel <-> viewing ray, and intersection of the rays with the (flat) pool floor.

Pixel -> ray rays are computed once (at construction). Per frame only a rotation by the robot heading and the
ray/floor intersection are needed, which is cheap numpy.
"""
import numpy as np

from .config import CameraConfig


class FisheyeCamera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        W, H = cfg.width, cfg.height
        ss = max(1, int(cfg.supersample))
        self.ss = ss
        self.cx = (W - 1) / 2.0 + cfg.center_dx
        self.cy = (H - 1) / 2.0 + cfg.center_dy
        self.R = cfg.circle_radius_px or H / 2.0
        self.theta_max = np.radians(cfg.fov_deg / 2.0)
        self.f = self.R / self._poly(self.theta_max)  # focal length so the circle edge sits at theta_max
        self.pitch = np.radians(cfg.pitch_deg)

        # radius -> angle lookup table (monotonic), used to invert the lens polynomial
        th = np.linspace(0.0, self.theta_max * 1.6, 8192)
        self._th_grid, self._r_grid = th, self.f * self._poly(th)

        # output-resolution helpers (vignette / circle mask)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = np.hypot(xx - self.cx, yy - self.cy)
        self.radius_norm = (rr / self.R).astype(np.float32)
        self.circle_mask = rr <= self.R

        # supersampled rays, in the camera frame -> robot body frame (forward, right, down)
        u = (np.arange(W * ss, dtype=np.float32) + 0.5) / ss - 0.5
        v = (np.arange(H * ss, dtype=np.float32) + 0.5) / ss - 0.5
        dx = u[None, :] - self.cx
        dy = v[:, None] - self.cy
        r = np.hypot(dx, dy)
        theta = np.interp(r, self._r_grid, self._th_grid).astype(np.float32)
        inv_r = np.where(r > 1e-9, 1.0 / np.maximum(r, 1e-9), 0.0)
        xc = np.sin(theta) * dx * inv_r
        yc = np.sin(theta) * dy * inv_r
        zc = np.cos(theta)
        p = self.pitch
        self.fwd = (zc * np.cos(p) - yc * np.sin(p)).astype(np.float32)
        self.right = xc.astype(np.float32)
        self.down = (zc * np.sin(p) + yc * np.cos(p)).astype(np.float32)
        self.inside = r <= self.R

    def _poly(self, th):
        c = self.cfg
        return th + c.k1 * th ** 3 + c.k2 * th ** 5

    # ------------------------------------------------------------------ geometry
    def camera_position(self, x, y, psi):
        off = self.cfg.forward_offset_m
        return x + off * np.cos(psi), y + off * np.sin(psi)

    def floor_hits(self, x, y, psi, max_range):
        """Where each (supersampled) pixel's ray meets the floor.

        Returns X, Y (world metres), slant (distance along the ray) and a bool mask `valid` (False for pixels outside
        the image circle, above the horizon, or beyond `max_range`)."""
        c, s = np.cos(psi), np.sin(psi)
        cam_x, cam_y = self.camera_position(x, y, psi)
        dir_x = self.fwd * c + self.right * s
        dir_y = self.fwd * s - self.right * c
        valid = self.inside & (self.down > 0.02)
        slant = np.where(valid, self.cfg.height_m / np.maximum(self.down, 1e-6), 0.0).astype(np.float32)
        valid &= slant < max_range * 1.5
        slant = np.where(valid, slant, 0.0).astype(np.float32)
        X = (cam_x + slant * dir_x).astype(np.float32)
        Y = (cam_y + slant * dir_y).astype(np.float32)
        return X, Y, slant, valid

    def world_to_pixel(self, X, Y, x, y, psi):
        """Project floor points (world metres) into the image. Returns (u, v) in output pixels (NaN if behind)."""
        cam_x, cam_y = self.camera_position(x, y, psi)
        dx, dy = np.asarray(X, float) - cam_x, np.asarray(Y, float) - cam_y
        c, s = np.cos(psi), np.sin(psi)
        f_b = dx * c + dy * s
        r_b = dx * s - dy * c
        d_b = np.full_like(f_b, self.cfg.height_m)
        p = self.pitch
        zc = f_b * np.cos(p) + d_b * np.sin(p)
        yc = -f_b * np.sin(p) + d_b * np.cos(p)
        xc = r_b
        rho = np.hypot(xc, yc)
        theta = np.arctan2(rho, zc)
        rad = self.f * self._poly(theta)
        scale = np.where(rho > 1e-9, rad / np.maximum(rho, 1e-9), 0.0)
        return self.cx + xc * scale, self.cy + yc * scale
