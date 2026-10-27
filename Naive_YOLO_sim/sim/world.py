"""The world: a transect polyline (yellow tape) and the pool-floor texture it is drawn on."""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ScenarioConfig

PATH_KINDS = ("straight", "gentle_s", "sine", "arc", "turn90", "random")
MAX_TEXTURE_PIXELS = 90e6


# ---------------------------------------------------------------------------------------------- transect
class Transect:
    """Densely sampled polyline starting at the origin heading +x, with optional gaps (no tape)."""

    def __init__(self, pts, gaps=()):
        self.pts = np.asarray(pts, dtype=np.float64)
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])
        self.gaps = list(gaps)  # [(s0, s1), ...] in metres of arc length
        self._p0 = self.pts[:-1]
        self._d = seg
        self._l2 = np.maximum(self.seg_len ** 2, 1e-12)

    def project(self, x, y):
        """Nearest point on the transect. Returns (arc length s, signed cross-track error; + = robot is left)."""
        w = ((x - self._p0[:, 0]) * self._d[:, 0] + (y - self._p0[:, 1]) * self._d[:, 1]) / self._l2
        w = np.clip(w, 0.0, 1.0)
        px = self._p0[:, 0] + w * self._d[:, 0]
        py = self._p0[:, 1] + w * self._d[:, 1]
        d2 = (x - px) ** 2 + (y - py) ** 2
        i = int(np.argmin(d2))
        cross = self._d[i, 0] * (y - py[i]) - self._d[i, 1] * (x - px[i])
        cte = float(np.sign(cross) * np.sqrt(d2[i])) if cross != 0 else float(np.sqrt(d2[i]))
        return float(self.s[i] + w[i] * self.seg_len[i]), cte

    def pose_at(self, s):
        """Position and heading (rad) of the transect at arc length s."""
        i = int(np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.seg_len) - 1))
        w = np.clip((s - self.s[i]) / max(self.seg_len[i], 1e-9), 0, 1)
        p = self._p0[i] + w * self._d[i]
        return float(p[0]), float(p[1]), float(np.arctan2(self._d[i, 1], self._d[i, 0]))

    def drawn_runs(self):
        """Index ranges of consecutive points that carry tape (gaps removed)."""
        mid = 0.5 * (self.s[:-1] + self.s[1:])
        keep = np.ones(len(mid), bool)
        for g0, g1 in self.gaps:
            keep &= ~((mid >= g0) & (mid <= g1))
        runs, start = [], None
        for i, k in enumerate(keep):
            if k and start is None:
                start = i
            if (not k) and start is not None:
                runs.append((start, i))  # points start .. i inclusive
                start = None
        if start is not None:
            runs.append((start, len(keep)))
        return runs


def make_transect(kind, length, rng, n_gaps=0, gap_len=(0.3, 0.8), ds=0.05):
    """Generate a transect of the given kind. Everything is built from a curvature profile kappa(s)."""
    if kind not in PATH_KINDS:
        raise ValueError(f"unknown path '{kind}', choose from {PATH_KINDS}")
    n = int(round(length / ds))
    s = np.arange(n) * ds
    sign = rng.choice([-1.0, 1.0])

    if kind == "straight":
        kappa = np.zeros(n)
    elif kind == "gentle_s":      # heading swings +-A once over the run
        A, w = np.radians(rng.uniform(15, 30)), 2 * np.pi / length
        kappa = sign * A * w * np.cos(w * s)
    elif kind == "sine":          # several, tighter swings
        A, w = np.radians(rng.uniform(20, 38)), 2 * np.pi / rng.uniform(4.5, 7.0)
        kappa = sign * A * w * np.cos(w * s)
    elif kind == "arc":           # constant-radius bend after a short straight
        R = rng.uniform(4.0, 8.0)
        kappa = np.where(s > 2.0, sign / R, 0.0)
    elif kind == "turn90":        # straight, 90 degree corner (smoothed, 1.5 m radius), straight
        R, s0 = 1.5, length * 0.35
        arc = np.pi / 2 * R
        kappa = np.where((s > s0) & (s < s0 + arc), sign / R, 0.0)
    else:                         # random: low-pass filtered curvature, limited to a 2 m minimum radius
        k = np.zeros(n)
        for i in range(1, n):
            k[i] = 0.97 * k[i - 1] + 0.06 * rng.standard_normal()
        kappa = np.clip(k, -0.5, 0.5)

    theta = np.cumsum(kappa) * ds
    x = np.cumsum(np.cos(theta)) * ds
    y = np.cumsum(np.sin(theta)) * ds
    pts = np.column_stack([np.concatenate([[0.0], x]), np.concatenate([[0.0], y])])

    t = Transect(pts)
    gaps = []
    for _ in range(n_gaps):
        g = rng.uniform(*gap_len)
        g0 = rng.uniform(2.0, max(2.1, t.length - g - 1.0))
        gaps.append((g0, g0 + g))
    t.gaps = gaps
    return t


# ---------------------------------------------------------------------------------------------- floor texture
@dataclass
class Floor:
    texture: np.ndarray  # (H, W, 3) uint8 RGB, row 0 = ymax
    xmin: float
    ymax: float
    res: float

    @property
    def bounds(self):
        h, w = self.texture.shape[:2]
        return self.xmin, self.xmin + w * self.res, self.ymax - h * self.res, self.ymax

    def maps(self, X, Y):
        """World coordinates -> cv2.remap source maps (pixel centres at integer coordinates)."""
        mx = ((X - self.xmin) / self.res - 0.5).astype(np.float32)
        my = ((self.ymax - Y) / self.res - 0.5).astype(np.float32)
        return mx, my


def _procedural_floor(h, w, res, color, tile_m, rng, contrast=1.0):
    """Speckled, blotchy single-colour floor (luminance noise at several scales, optional tile grid)."""
    lum = np.zeros((h, w), np.float32)
    for scale_m, amp in ((3.0, 0.10), (0.6, 0.07), (0.12, 0.04)):
        gh, gw = int(h * res / scale_m) + 3, int(w * res / scale_m) + 3
        lum += contrast * amp * cv2.resize(rng.standard_normal((gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    if tile_m > 0:
        pitch = tile_m / res
        thick = max(1, int(round(0.01 / res)))
        for i in range(int(w / pitch) + 1):
            cv2.line(lum, (int(i * pitch), 0), (int(i * pitch), h), -0.25, thick)
        for j in range(int(h / pitch) + 1):
            cv2.line(lum, (0, int(j * pitch)), (w, int(j * pitch)), -0.25, thick)
    base = np.array(color, np.float32)
    tex = np.empty((h, w, 3), np.uint8)
    step = 512
    for r0 in range(0, h, step):  # row chunks keep the float temporaries small
        l = 1.0 + lum[r0:r0 + step] + 0.03 * contrast * rng.standard_normal(lum[r0:r0 + step].shape).astype(np.float32)
        tex[r0:r0 + step] = np.clip(base[None, None, :] * l[..., None], 0, 255).astype(np.uint8)
    return tex


def _image_floor(h, w, res, path, width_m):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"cannot read floor image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tw = max(8, int(width_m / res))
    th = max(8, int(img.shape[0] * tw / img.shape[1]))
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return np.tile(img, (h // th + 1, w // tw + 1, 1))[:h, :w].copy()


def build_floor(transect: Transect, cfg: ScenarioConfig, rng) -> Floor:
    """Render the floor texture covering the transect plus `view_range_m` of margin, with the tape drawn on it."""
    m = cfg.view_range_m + 1.0
    xmin, xmax = transect.pts[:, 0].min() - m, transect.pts[:, 0].max() + m
    ymin, ymax = transect.pts[:, 1].min() - m, transect.pts[:, 1].max() + m
    res = cfg.floor_res_m
    w, h = int(np.ceil((xmax - xmin) / res)), int(np.ceil((ymax - ymin) / res))
    if w * h > MAX_TEXTURE_PIXELS:
        raise MemoryError(f"floor texture would be {w}x{h}px; raise floor_res_m or lower view_range_m")

    if cfg.floor_image:
        tex = _image_floor(h, w, res, cfg.floor_image, cfg.floor_image_width_m)
    else:
        tex = _procedural_floor(h, w, res, cfg.floor_color, cfg.tile_m, rng, cfg.floor_contrast)

    thick = max(1, int(round(cfg.tape_width_m / res)))
    for a, b in transect.drawn_runs():
        pts = transect.pts[a:b + 1]
        px = np.column_stack([(pts[:, 0] - xmin) / res, (ymax - pts[:, 1]) / res])
        cv2.polylines(tex, [np.round(px * 16).astype(np.int32)], False, tuple(int(c) for c in cfg.tape_color),
                      thick, cv2.LINE_AA, shift=4)
    tex = cv2.GaussianBlur(tex, (0, 0), 0.6)  # soften drawing aliasing
    return Floor(tex, xmin, ymax, res)
