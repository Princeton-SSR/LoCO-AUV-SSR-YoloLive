"""Turns (robot pose, time) into the RGB frame the real camera would deliver.

Pipeline: fisheye rays -> floor intersection -> texture lookup -> caustics -> underwater attenuation / veiling light
-> exposure -> (downsample) -> lens blur -> vignette + image-circle mask -> sensor noise -> uint8.
"""
from pathlib import Path

import cv2
import numpy as np

from .camera import FisheyeCamera
from .config import RenderConfig
from .world import Floor


class Renderer:
    def __init__(self, camera: FisheyeCamera, floor: Floor, cfg: RenderConfig, view_range_m: float, seed: int = 0):
        self.cam, self.floor, self.cfg, self.view_range = camera, floor, cfg, view_range_m
        self.atten = np.array(cfg.attenuation_per_m, np.float32)
        self.water = np.array(cfg.water_color, np.float32)
        self.water_up = np.array(cfg.water_color_up, np.float32)
        self.housing = self._load_housing(cfg.housing_image, camera.cfg.width, camera.cfg.height)
        self.reset(seed)

    @staticmethod
    def _load_housing(path, w, h):
        """RGB float image (0..1) of the static housing ring, or None."""
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        if not p.exists():
            return None
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (w, h)).astype(np.float32) / 255.0

    def reset(self, seed):
        """Re-seed the per-run randomness (exposure drift, sensor noise, caustic phase)."""
        self.rng = np.random.default_rng(seed + 1234)
        self._exp_base = 1.0 + self.cfg.exposure_jitter * self.rng.uniform(-1, 1)
        self._exp_phase = self.rng.uniform(0, 2 * np.pi)
        self._caus_phase = self.rng.uniform(0, 2 * np.pi, 4)

    def render(self, x, y, psi, t):
        cam, cfg = self.cam, self.cfg
        X, Y, slant, valid = cam.floor_hits(x, y, psi, self.view_range)

        mx, my = self.floor.maps(X, Y)
        img = cv2.remap(self.floor.texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        img = img.astype(np.float32) * (1.0 / 255.0)

        if cfg.caustics > 0:  # two drifting sine products ~ moving caustic light on the floor
            k = 2 * np.pi / cfg.caustic_scale_m
            p = self._caus_phase
            c = np.sin(k * X + 0.9 * t + p[0]) * np.sin(k * 0.8 * Y - 0.7 * t + p[1])
            c += np.sin(k * 1.7 * (X + Y) * 0.7 + 1.3 * t + p[2]) * np.sin(k * 1.3 * (X - Y) * 0.7 - 1.1 * t + p[3])
            img *= (1.0 + cfg.caustics * 0.5 * c)[..., None]

        # underwater: per-channel transmittance along the slant range; far / above-horizon pixels become pure veil
        fade = np.clip((self.view_range - slant) / (0.3 * self.view_range), 0.0, 1.0)
        trans = np.exp(-self.atten[None, None, :] * slant[..., None]) * fade[..., None]
        trans = np.where(valid[..., None], trans, 0.0)
        up = np.clip(-cam.down * 1.5, 0.0, 1.0)[..., None]  # looking up -> water colour shifts towards the surface tone
        veil = self.water[None, None, :] * (1.0 - up) + self.water_up[None, None, :] * up
        img = img * trans + veil * (1.0 - trans)

        exposure = cfg.exposure * self._exp_base * (1.0 + 0.5 * cfg.exposure_jitter * np.sin(0.15 * t + self._exp_phase))
        img *= exposure

        if cam.ss > 1:
            img = cv2.resize(img, (cam.cfg.width, cam.cfg.height), interpolation=cv2.INTER_AREA)
        if cfg.blur_sigma_px > 0:
            img = cv2.GaussianBlur(img, (0, 0), cfg.blur_sigma_px)

        img *= (1.0 - cfg.vignette * np.clip(cam.radius_norm, 0, 1.2) ** 2.5)[..., None]
        outside = self.housing if self.housing is not None else np.array(cfg.outside_circle_color, np.float32)[None, None, :]
        img = np.where(cam.circle_mask[..., None], img, outside)

        if cfg.noise_sigma > 0:
            img += cfg.noise_sigma * self.rng.standard_normal(img.shape, dtype=np.float32)
        img = np.clip(img, 0.0, 1.0)
        return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
