#!/usr/bin/env python3
"""Live web view for follow_transect.py: annotated camera MJPEG stream + a live status panel.

    Streamer(port, deadband).update(frame, det, tel, cmd, raw_offset, infer_ms)   # call once per frame

Open http://<pi-ip>:PORT/ in a browser. Endpoints:
    /         web page: video, state, offset bar, surge/yaw bars, detection score, inference time
    /stream   raw MJPEG (annotated frames), usable in any <img> tag or VLC
    /status   latest status as JSON (poll it from other tools)

The server runs in background threads. update() is cheap: status is just a dict copy, and the
frame is only drawn + JPEG-encoded when a viewer is connected and the fps limit allows it.
"""
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw

from detection import box_corners


# --------------------------------------------------------------------------- drawing
def render(frame, det, tel, cmd, deadband, deadband_center=0.0):
    """Annotated PIL image: OBB (green), image centre and deadband (blue), and a status line."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    if det is not None:
        d.polygon(box_corners(*det[:5]), outline=(0, 255, 0), width=2)
    cx, half_db = w / 2, deadband * w / 2
    bc = cx + deadband_center * w / 2          # centre of the deadband (image centre shifted by deadband_center)
    d.line([bc, 0, bc, h], fill=(0, 160, 255), width=1)
    d.rectangle([bc - half_db, 0, bc + half_db, h], outline=(0, 160, 255))
    if det is not None:  # box centre: black inside the deadband (no yaw), red outside (steering). Uses the smoothed offset like the controller
        inside = abs(tel.offset - deadband_center) < deadband
        col = (0, 0, 0) if inside else (255, 0, 0)
        x, y = det[0], det[1]
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col, outline=(255, 255, 255))
    d.text((8, 8), f"{tel.state} offset={tel.offset:+.2f} surge={cmd.surge:+.2f} yaw={cmd.yaw:+.2f}",
           fill=(255, 255, 0))
    return img


def annotate(frame, det, tel, cmd, deadband, path="last_frame.jpg", deadband_center=0.0):
    """Save the annotated image to disk (atomically, so viewers never see a half-written file)."""
    render(frame, det, tel, cmd, deadband, deadband_center).save(path + ".tmp.jpg")
    os.replace(path + ".tmp.jpg", path)


# --------------------------------------------------------------------------- tunable parameters
# (NaiveConfig field, label, kind, min, max, step); kind "sign" is a checkbox that maps checked -> -1
TUNABLES = [
    ("surge", "Surge", "float", 0.0, 0.5, 0.01),
    ("yaw_cmd", "Yaw command", "float", 0.0, 1.0, 0.01),
    ("deadband", "Deadband", "float", 0.0, 0.5, 0.01),
    ("deadband_center", "Deadband centre (- left / + right)", "float", -0.5, 0.5, 0.01),
    ("smooth", "Smoothing (1 = none)", "float", 0.05, 1.0, 0.05),
    ("lost_hold", "Lost hold (s)", "float", 0.0, 5.0, 0.1),
    ("proportional", "Proportional yaw", "bool", 0, 1, 1),
    ("yaw_sign", "Invert yaw", "sign", -1, 1, 1),
]


# --------------------------------------------------------------------------- web page
PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Transect follower</title>
<style>
  body{margin:0;background:#111;color:#eee;font:14px/1.4 system-ui,sans-serif;display:flex;flex-wrap:wrap}
  #video{flex:2 1 480px;min-width:0}
  #video img{width:100%;display:block}
  #panel{flex:1 1 300px;padding:16px;max-width:460px}
  h2{margin:0 0 12px;font-size:16px;color:#aaa;font-weight:600}
  .state{font-size:28px;font-weight:700;padding:6px 14px;border-radius:6px;display:inline-block;margin-bottom:14px}
  .TRACK{background:#1b7f3b}.COAST{background:#b7791f}.LOST{background:#b91c1c}.WAIT{background:#444}
  .row{margin:14px 0}
  .lbl{display:flex;justify-content:space-between;color:#aaa;margin-bottom:4px}
  .lbl b{color:#fff;font-variant-numeric:tabular-nums}
  .bar{position:relative;height:18px;background:#2a2a2a;border-radius:4px;overflow:hidden}
  .bar .mid{position:absolute;left:50%;top:0;bottom:0;width:2px;background:#666}
  .bar .db{position:absolute;top:0;bottom:0;background:rgba(0,160,255,.25)}
  .bar .fill{position:absolute;top:0;bottom:0}
  .bar .dot{position:absolute;top:-2px;bottom:-2px;width:6px;margin-left:-3px;background:#ffd400;border-radius:3px}
  .ends{display:flex;justify-content:space-between;font-size:11px;color:#666;margin-top:2px}
  table{width:100%;border-collapse:collapse;margin-top:8px}
  td{padding:3px 0;color:#aaa}td+td{text-align:right;color:#fff;font-variant-numeric:tabular-nums}
  #stale{color:#f87171;display:none;margin-top:10px}
</style></head><body>
<div id="video"><img src="/stream"></div>
<div id="panel">
  <h2>TRANSECT FOLLOWER <span id="mode"></span></h2>
  <div id="state" class="state WAIT">--</div>

  <div class="row">
    <div class="lbl"><span>Offset (smoothed)</span><b id="offset">--</b></div>
    <div class="bar"><div class="db" id="db"></div><div class="mid"></div><div class="dot" id="odot"></div></div>
    <div class="ends"><span>line left</span><span>centre</span><span>line right</span></div>
  </div>

  <div class="row">
    <div class="lbl"><span>Surge command</span><b id="surge">--</b></div>
    <div class="bar"><div class="mid"></div><div class="fill" id="sfill" style="background:#38bdf8"></div></div>
    <div class="ends"><span>-1 back</span><span>0</span><span>+1 fwd</span></div>
  </div>

  <div class="row">
    <div class="lbl"><span>Yaw command</span><b id="yaw">--</b></div>
    <div class="bar"><div class="mid"></div><div class="fill" id="yfill" style="background:#f472b6"></div></div>
    <div class="ends"><span>-1 left</span><span>0</span><span>+1 right</span></div>
  </div>

  <table>
    <tr><td>Raw offset</td><td id="raw">--</td></tr>
    <tr><td>Detection score</td><td id="score">--</td></tr>
    <tr><td>Inference</td><td id="infer">--</td></tr>
    <tr><td>Loop rate</td><td id="fps">--</td></tr>
    <tr><td>Run time</td><td id="t">--</td></tr>
  </table>
  <h2 style="margin-top:22px">CONTROLS <span id="saved" style="color:#4ade80;font-weight:400"></span></h2>
  <div id="ctl"></div>
  <div id="stale">No status updates (script stopped?)</div>
</div>
<script>
const $ = id => document.getElementById(id);
let saveTimer = null;
function post(name, value) {
  fetch("/params", {method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({[name]: value})})
    .then(() => { $("saved").textContent = "saved"; clearTimeout(saveTimer);
                  saveTimer = setTimeout(() => $("saved").textContent = "", 1200); });
}
async function buildControls() {
  const p = await (await fetch("/params", {cache: "no-store"})).json();
  p.spec.forEach(s => {
    const row = document.createElement("div"); row.className = "row";
    const v = p.values[s.name];
    if (s.kind === "float") {
      row.innerHTML = '<div class="lbl"><span>' + s.label + '</span><input type="number" min="' + s.min +
        '" max="' + s.max + '" step="' + s.step + '" value="' + v + '" style="width:70px"></div>' +
        '<input type="range" min="' + s.min + '" max="' + s.max + '" step="' + s.step + '" value="' + v +
        '" style="width:100%">';
      const num = row.querySelector("input[type=number]"), rng = row.querySelector("input[type=range]");
      let t = null;
      rng.oninput = () => { num.value = rng.value; clearTimeout(t); t = setTimeout(() => post(s.name, +rng.value), 150); };
      num.onchange = () => { rng.value = num.value; post(s.name, +num.value); };
    } else {                                   // "bool" or "sign" (checked = -1)
      const on = s.kind === "sign" ? v < 0 : !!v;
      row.innerHTML = '<label><input type="checkbox"' + (on ? " checked" : "") + '> ' + s.label + '</label>';
      row.querySelector("input").onchange = e =>
        post(s.name, s.kind === "sign" ? (e.target.checked ? -1 : 1) : e.target.checked);
    }
    $("ctl").appendChild(row);
  });
}
buildControls();
const f = (v, n=2, sign=true) => v == null ? "--" : (sign && v >= 0 ? "+" : "") + v.toFixed(n);
function centredFill(el, v) {          // bar growing from the middle, v in -1..1
  v = Math.max(-1, Math.min(1, v));
  el.style.left = (v < 0 ? 50 + v * 50 : 50) + "%";
  el.style.width = Math.abs(v) * 50 + "%";
}
let dbSet = false;
async function tick() {
  try {
    const s = await (await fetch("/status", {cache: "no-store"})).json();
    $("stale").style.display = (Date.now() / 1000 - s.server_time > 3 || s.age > 3) ? "block" : "none";
    $("mode").textContent = s.dry_run ? "(dry run)" : "";
    $("state").textContent = s.state; $("state").className = "state " + s.state;
    $("offset").textContent = f(s.offset);
    $("odot").style.left = (50 + Math.max(-1, Math.min(1, s.offset)) * 50) + "%";
    $("db").style.left = (50 + (s.deadband_center - s.deadband) * 50) + "%"; $("db").style.width = s.deadband * 100 + "%";
    $("surge").textContent = f(s.surge); centredFill($("sfill"), s.surge);
    $("yaw").textContent = f(s.yaw);     centredFill($("yfill"), s.yaw);
    $("raw").textContent = s.raw_offset == null ? "no detection" : f(s.raw_offset);
    $("score").textContent = s.score == null ? "--" : s.score.toFixed(2);
    $("infer").textContent = s.infer_ms == null ? "--" : s.infer_ms.toFixed(0) + " ms";
    $("fps").textContent = s.fps == null ? "--" : s.fps.toFixed(1) + " Hz";
    $("t").textContent = s.elapsed.toFixed(1) + " s";
  } catch (e) { $("stale").style.display = "block"; }
}
setInterval(tick, 150); tick();
</script></body></html>
"""


# --------------------------------------------------------------------------- server
class Streamer:
    """Background HTTP server; feed it one update() per control-loop iteration."""

    def __init__(self, port, deadband, fps=10.0, quality=70, dry_run=False, cfg=None, params_path=None):
        self._deadband, self.min_dt, self.quality, self.dry_run = deadband, 1.0 / fps, quality, dry_run
        self.cfg, self.params_path = cfg, params_path     # cfg: live NaiveConfig edited from the web page
        self._plock = threading.Lock()
        self._load_params()
        self._jpeg = None
        self._cond = threading.Condition()
        self._status = {"state": "WAIT", "offset": 0.0, "raw_offset": None, "surge": 0.0, "yaw": 0.0,
                        "score": None, "infer_ms": None, "fps": None, "elapsed": 0.0}
        self._status_t = time.monotonic()
        self._t0 = self._last_push = self._last_update = time.monotonic()
        self._clients = 0                 # MJPEG viewers currently connected
        self._lock = threading.Lock()
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                try:
                    if self.path in ("/", "/index.html"):
                        self._send(PAGE, "text/html; charset=utf-8")
                    elif self.path == "/status":
                        self._send(streamer._status_json(), "application/json")
                    elif self.path == "/params":
                        self._send(streamer._params_json(), "application/json")
                    elif self.path.startswith("/stream"):
                        self._stream()
                    else:
                        self.send_error(404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                try:
                    if self.path != "/params":
                        self.send_error(404)
                        return
                    n = int(self.headers.get("Content-Length", 0))
                    streamer.set_params(json.loads(self.rfile.read(n) or b"{}"))
                    self._send(streamer._params_json(), "application/json")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except (ValueError, TypeError):
                    self.send_error(400)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with streamer._lock:
                    streamer._clients += 1
                try:
                    last = None
                    while True:
                        with streamer._cond:
                            streamer._cond.wait(timeout=5)
                            jpeg = streamer._jpeg
                        if jpeg is None or jpeg is last:
                            continue
                        last = jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with streamer._lock:
                        streamer._clients -= 1

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def update(self, frame, det, tel, cmd, raw_offset=None, infer_ms=None):
        """Call once per loop. Always refreshes the status; renders/encodes the frame only if someone is watching."""
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now
        with self._lock:
            old_fps = self._status["fps"]
            inst = 1.0 / dt if dt > 0 else None
            fps = inst if old_fps is None or inst is None else old_fps + 0.2 * (inst - old_fps)  # smoothed
            self._status = {
                "state": tel.state, "offset": float(tel.offset),
                "raw_offset": float(raw_offset) if raw_offset is not None else None,
                "surge": float(cmd.surge), "yaw": float(cmd.yaw),
                "score": float(det[5]) if det is not None else None,
                "infer_ms": float(infer_ms) if infer_ms is not None else None,
                "fps": fps, "elapsed": now - self._t0,
            }
            self._status_t = now
            watching = self._clients > 0
        if watching and now - self._last_push >= self.min_dt:
            self._last_push = now
            buf = io.BytesIO()
            render(frame, det, tel, cmd, self.deadband, self.deadband_center).save(buf, format="JPEG", quality=self.quality)
            with self._cond:
                self._jpeg = buf.getvalue()
                self._cond.notify_all()

    @property
    def deadband(self):
        """Current deadband (live value from the config if there is one, so the overlay follows the slider)."""
        return self.cfg.deadband if self.cfg is not None else self._deadband

    @property
    def deadband_center(self):
        """Current deadband centre offset (live from the config), 0 = image centre."""
        return getattr(self.cfg, "deadband_center", 0.0) if self.cfg is not None else 0.0

    # ---- tunable parameters (edited from the web page, saved to params_path)
    def _load_params(self):
        """Apply values saved in params_path (if it exists) to the config at startup."""
        if self.cfg is None or not self.params_path or not os.path.exists(self.params_path):
            return
        try:
            with open(self.params_path) as f:
                self._apply(json.load(f))
            print(f"Loaded parameters from {self.params_path}")
        except (OSError, ValueError) as e:
            print(f"Could not load {self.params_path}: {e}")

    def _apply(self, d):
        """Validate/clamp each known value and set it on the config."""
        for name, _label, kind, lo, hi, _step in TUNABLES:
            if name not in d:
                continue
            v = d[name]
            if kind == "float":
                v = max(lo, min(hi, float(v)))
            elif kind == "bool":
                v = bool(v)
            else:  # sign
                v = -1.0 if float(v) < 0 else 1.0
            setattr(self.cfg, name, v)

    def set_params(self, d):
        """Apply new values to the live controller config and write all current values to params_path."""
        if self.cfg is None:
            return
        with self._plock:
            self._apply(d)
            if self.params_path:
                values = {t[0]: getattr(self.cfg, t[0]) for t in TUNABLES}
                with open(self.params_path + ".tmp", "w") as f:
                    json.dump(values, f, indent=2)
                os.replace(self.params_path + ".tmp", self.params_path)

    def _params_json(self):
        spec = [dict(name=n, label=l, kind=k, min=lo, max=hi, step=st) for n, l, k, lo, hi, st in TUNABLES]
        values = {t[0]: getattr(self.cfg, t[0]) for t in TUNABLES} if self.cfg is not None else {}
        return json.dumps({"spec": spec, "values": values}, default=float).encode()

    def _status_json(self):
        with self._lock:
            s = dict(self._status, deadband=self.deadband, deadband_center=self.deadband_center, dry_run=self.dry_run,
                     age=time.monotonic() - self._status_t, server_time=time.time())
        return json.dumps(s, default=float).encode()  # default=float catches any stray numpy scalars

    def close(self):
        self._server.shutdown()
        self._server.server_close()
