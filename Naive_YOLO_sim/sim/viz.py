"""Visualisation: a side-by-side composite (top-down map | camera view with detection) and a pygame window for it.

compose() works on plain numpy RGB arrays so the same image feeds the live window, saved video and PNG dumps.
"""
import cv2
import numpy as np

PANEL_W, PANEL_H = 640, 480
CONTROL_H = 230  # height of the pygame control strip (command bars + sliders) under the two panels
GREEN, RED, WHITE, YELLOW, CYAN = (0, 255, 0), (255, 60, 60), (255, 255, 255), (255, 230, 0), (0, 220, 255)


class TopDown:
    """World -> panel pixel transform plus a pre-rendered background (the floor texture with its tape)."""

    def __init__(self, scenario):
        floor = scenario.renderer.floor
        x0, x1, y0, y1 = floor.bounds
        pts = scenario.transect.pts
        pad = 1.5  # show the transect and a little margin, not the whole 8 m view margin
        self.xmin, self.xmax = pts[:, 0].min() - pad, pts[:, 0].max() + pad
        self.ymin, self.ymax = pts[:, 1].min() - pad, pts[:, 1].max() + pad
        self.scale = min(PANEL_W / (self.xmax - self.xmin), PANEL_H / (self.ymax - self.ymin))
        wpx, hpx = int((self.xmax - self.xmin) * self.scale), int((self.ymax - self.ymin) * self.scale)
        self.size = (wpx, hpx)
        c0, c1 = int((self.xmin - x0) / floor.res), int((self.xmax - x0) / floor.res)
        r0, r1 = int((floor.ymax - self.ymax) / floor.res), int((floor.ymax - self.ymin) / floor.res)
        crop = floor.texture[max(r0, 0):r1, max(c0, 0):c1]
        self.bg = cv2.resize(crop, (wpx, hpx), interpolation=cv2.INTER_AREA)
        self.transect = scenario.transect

    def px(self, x, y):
        return (int((x - self.xmin) * self.scale), int((self.ymax - y) * self.scale))

    def draw(self, robot, traj, cam_cfg):
        img = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
        view = self.bg.copy()
        pts = np.array([self.px(*p) for p in self.transect.pts[::4]], np.int32)
        cv2.polylines(view, [pts], False, YELLOW, 1, cv2.LINE_AA)
        if len(traj) > 1:
            tp = np.array([self.px(p[1], p[2]) for p in traj[::5]], np.int32)
            cv2.polylines(view, [tp], False, RED, 2, cv2.LINE_AA)
        cx, cy = self.px(robot.x, robot.y)
        tip = self.px(robot.x + 0.5 * np.cos(robot.psi), robot.y + 0.5 * np.sin(robot.psi))
        left = self.px(robot.x - 0.2 * np.cos(robot.psi) + 0.18 * np.sin(robot.psi),
                       robot.y - 0.2 * np.sin(robot.psi) - 0.18 * np.cos(robot.psi))
        right = self.px(robot.x - 0.2 * np.cos(robot.psi) - 0.18 * np.sin(robot.psi),
                        robot.y - 0.2 * np.sin(robot.psi) + 0.18 * np.cos(robot.psi))
        cv2.fillConvexPoly(view, np.array([tip, left, right], np.int32), CYAN, cv2.LINE_AA)
        h, w = view.shape[:2]
        img[:h, :w] = view
        return img


def draw_camera_panel(frame, ctx, deadband=None, deadband_center=0.0):
    img = frame.copy()
    H, W = img.shape[:2]
    if deadband:
        bc = W / 2 + deadband_center * W / 2
        cv2.line(img, (int(bc), 0), (int(bc), H), (255, 255, 255), 1)
        for sgn in (-1, 1):
            x = int(bc + sgn * deadband * W / 2)
            cv2.line(img, (x, 0), (x, H), (150, 150, 150), 1)
    det = ctx["det"]
    for d in ctx["dets"]:
        cx, cy, w, h, r = d[:5]
        c, s = np.cos(r), np.sin(r)
        corners = [(cx + x * c - y * s, cy + x * s + y * c)
                   for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
        cv2.polylines(img, [np.array(corners, np.int32)], True, GREEN if d is det else (0, 120, 0), 2, cv2.LINE_AA)
    if det is not None and deadband is not None and ctx.get("tel") is not None:
        # box centre: black inside the deadband (no yaw), red outside (steering); uses the smoothed offset like the controller
        inside = abs(ctx["tel"].offset - deadband_center) < deadband
        pt = (int(det[0]), int(det[1]))
        cv2.circle(img, pt, 8, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 6, (0, 0, 0) if inside else (255, 0, 0), -1, cv2.LINE_AA)
    return img


def compose(ctx, topdown: TopDown, cam_cfg, deadband=None, deadband_center=0.0):
    """Top-down map | camera view with detection and deadband lines, as one RGB array (also used for video)."""
    robot, row, tel, cmd = ctx["robot"], ctx["row"], ctx["tel"], ctx["cmd"]
    left = topdown.draw(robot, ctx["traj"], cam_cfg)
    right = draw_camera_panel(ctx["frame"], ctx, deadband, deadband_center)
    out = np.concatenate([left, right], axis=1)
    lines = [f"t={ctx['t']:6.1f}s  {tel.state}  offset={tel.offset:+.2f}",
             f"cte={row['cte']:+.2f}m  progress={row['progress']:.1f}/{topdown.transect.length:.1f}m",
             f"surge={cmd.surge:+.0f} yaw={cmd.yaw:+.0f}  infer={ctx['infer_ms']:.0f}ms"]
    for i, text in enumerate(lines):
        cv2.putText(out, text, (PANEL_W + 8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return out


STATE_COLORS = {"TRACK": (27, 127, 59), "COAST": (183, 121, 31), "LOST": (185, 28, 28)}


class ControlPanel:
    """Live readout + sliders under the video, like the robot's web page, but values are NOT saved to params.json.

    Left: state, offset bar (with the deadband band), computed surge / yaw command bars, and the command that has
    actually reached the thrusters after latency. Right: one slider (or checkbox) per tunable parameter.
    Dragging a slider edits the running controller immediately. Press R to reset all sliders to their starting values.
    """

    def __init__(self, pg, controller, tunables, y0, width):
        self.pg, self.ctrl, self.tun, self.y0, self.w = pg, controller, tunables, y0, width
        self.initial = dict(controller.get_params())
        self.font = pg.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.big = pg.font.SysFont("menlo,dejavusansmono,monospace", 22, bold=True)
        self.drag = None
        # slider geometry: two columns on the right
        self.col_x, self.col_w, self.row_h = (520, 780), 215, 52
        self.items = []  # (spec, rect)
        for i, spec in enumerate(tunables):
            col, row = divmod(i, 4)
            x = self.col_x[col]
            self.items.append((spec, pg.Rect(x, y0 + 44 + row * self.row_h, self.col_w, 22)))

    # --- value <-> pixel
    def _value_at(self, spec, rect, mx):
        _, _, _, lo, hi, step = spec
        v = lo + (hi - lo) * min(1.0, max(0.0, (mx - rect.x) / rect.w))
        return round(v / step) * step

    def handle(self, e):
        pg = self.pg
        if e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
            for spec, rect in self.items:
                hit = rect.inflate(8, 14)
                if hit.collidepoint(e.pos):
                    if spec[2] == "float":
                        self.drag = (spec, rect)
                        self.ctrl.set_param(spec[0], self._value_at(spec, rect, e.pos[0]))
                    else:
                        cur = self.ctrl.get_params()[spec[0]]
                        on = cur < 0 if spec[2] == "sign" else bool(cur)
                        new = (-1.0 if not on else 1.0) if spec[2] == "sign" else (not on)
                        self.ctrl.set_param(spec[0], new)
        elif e.type == pg.MOUSEMOTION and self.drag:
            spec, rect = self.drag
            self.ctrl.set_param(spec[0], self._value_at(spec, rect, e.pos[0]))
        elif e.type == pg.MOUSEBUTTONUP:
            self.drag = None
        elif e.type == pg.KEYDOWN and e.key == pg.K_r:
            for k, v in self.initial.items():
                self.ctrl.set_param(k, v)

    # --- drawing
    def _bar(self, screen, x, y, w, label, value, color, lo=-1.0, hi=1.0, band=None):
        pg = self.pg
        value = float(value)
        if not np.isfinite(value):
            value = 0.0
        screen.blit(self.font.render(f"{label}", True, (170, 170, 170)), (x, y))
        txt = self.font.render(f"{value:+.2f}", True, WHITE)
        screen.blit(txt, (x + w - txt.get_width(), y))
        bar = pg.Rect(x, y + 17, w, 16)
        pg.draw.rect(screen, (42, 42, 42), bar, border_radius=3)
        if band:  # (centre, half-width) in value units
            c, hw = band
            bx0 = bar.x + (max(lo, c - hw) - lo) / (hi - lo) * w
            bx1 = bar.x + (min(hi, c + hw) - lo) / (hi - lo) * w
            pg.draw.rect(screen, (30, 90, 130), (int(bx0), bar.y, int(max(bx1 - bx0, 1)), bar.h))
        mid = bar.x + (0 - lo) / (hi - lo) * w
        pg.draw.line(screen, (110, 110, 110), (int(mid), bar.y), (int(mid), bar.bottom), 2)
        vx = bar.x + (min(hi, max(lo, value)) - lo) / (hi - lo) * w
        if band is None:
            pg.draw.rect(screen, color, (int(min(mid, vx)), bar.y, int(abs(vx - mid)), bar.h))
        else:
            pg.draw.rect(screen, (255, 212, 0), (int(vx) - 3, bar.y - 2, 6, bar.h + 4), border_radius=3)

    def draw(self, screen, ctx):
        pg = self.pg
        screen.fill((17, 17, 17), (0, self.y0, self.w, CONTROL_H))
        tel, cmd, applied = ctx["tel"], ctx["cmd"], ctx["applied"]
        params = self.ctrl.get_params()

        pg.draw.rect(screen, STATE_COLORS.get(tel.state, (68, 68, 68)), (16, self.y0 + 10, 96, 34), border_radius=6)
        screen.blit(self.big.render(tel.state, True, WHITE), (24, self.y0 + 14))
        info = self.font.render(f"t={ctx['t']:5.1f}s  cte={ctx['row']['cte']:+.2f}m", True, (170, 170, 170))
        screen.blit(info, (128, self.y0 + 10))
        # differential mix of what actually reaches the thrusters (+yaw = clockwise: left faster, right slower)
        l_th = float(np.clip(applied.surge + applied.yaw, -1000, 1000))
        r_th = float(np.clip(applied.surge - applied.yaw, -1000, 1000))
        screen.blit(self.font.render(f"Thrusters  L={l_th:+.0f}  R={r_th:+.0f}", True, (255, 212, 0)),
                    (128, self.y0 + 28))

        bw = 470
        self._bar(screen, 16, self.y0 + 56, bw, "Offset (smoothed)  [blue = deadband]", tel.offset, None,
                  band=(params["deadband_center"], params["deadband"]))
        self._bar(screen, 16, self.y0 + 100, bw, "Surge command (computed)", cmd.surge, (56, 189, 248),
                  lo=-1000, hi=1000)
        self._bar(screen, 16, self.y0 + 144, bw, "Yaw command (computed)  - left / + right", cmd.yaw, (244, 114, 182),
                  lo=-1000, hi=1000)
        grey = (130, 130, 130)
        screen.blit(self.font.render(
            f"at thrusters (after latency): surge={applied.surge:+.0f} yaw={applied.yaw:+.0f}", True, grey),
            (16, self.y0 + 178))
        screen.blit(self.font.render("Sliders change the live controller only;", True, (100, 100, 100)), (16, self.y0 + 198))
        screen.blit(self.font.render("params.json is NOT modified.  R = reset sliders", True, (100, 100, 100)),
                    (16, self.y0 + 212))

        screen.blit(self.font.render("CONTROLS   (Space = pause, Esc = quit)", True, (170, 170, 170)), (self.col_x[0], self.y0 + 6))
        for spec, rect in self.items:
            name, label, kind, lo, hi, _ = spec
            v = params[name]
            if kind == "float":
                screen.blit(self.font.render(label, True, (170, 170, 170)), (rect.x, rect.y - 17))
                shown = f"{v:.0f}" if (hi - lo) > 5 else f"{v:.2f}"
                val = self.font.render(shown, True, WHITE)
                screen.blit(val, (rect.right - val.get_width(), rect.y - 17))
                pg.draw.rect(screen, (60, 60, 60), (rect.x, rect.centery - 2, rect.w, 4), border_radius=2)
                frac = (v - lo) / (hi - lo)
                if name == "deadband_center":
                    pg.draw.line(screen, (110, 110, 110), (rect.centerx, rect.y + 2), (rect.centerx, rect.bottom - 2), 1)
                pg.draw.circle(screen, (56, 189, 248), (int(rect.x + float(frac) * rect.w), rect.centery), 8)
            else:
                on = v < 0 if kind == "sign" else bool(v)
                box = pg.Rect(rect.x, rect.y, 18, 18)
                pg.draw.rect(screen, (60, 60, 60), box, border_radius=3)
                if on:
                    pg.draw.rect(screen, (56, 189, 248), box.inflate(-6, -6), border_radius=2)
                screen.blit(self.font.render(label, True, (200, 200, 200)), (box.right + 8, box.y + 2))


class Window:
    """pygame window: composite view on top, control panel below. Space = pause, Esc / close = abort the run."""

    def __init__(self, size, controller=None, tunables=None, title="Naive_YOLO_sim"):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.font.init()
        self.size = size
        self.screen = pygame.display.set_mode((size[0], size[1] + (CONTROL_H if controller else 0)))
        pygame.display.set_caption(title)
        self.paused = False
        self.panel = ControlPanel(pygame, controller, tunables, size[1], size[0]) if controller else None

    def show(self, rgb, ctx=None):
        """Returns False if the user asked to quit. Sliders keep working while paused."""
        pg = self.pg
        self.screen.blit(pg.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1))), (0, 0))
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    return False
                if e.type == pg.KEYDOWN and e.key == pg.K_SPACE:
                    self.paused = not self.paused
                if self.panel:
                    self.panel.handle(e)
            if self.panel and ctx is not None:
                self.panel.draw(self.screen, ctx)
            pg.display.flip()
            if not self.paused:
                return True
            pg.time.wait(30)

    def close(self):
        self.pg.quit()
