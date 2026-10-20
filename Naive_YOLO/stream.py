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

from yolo_live import box_corners


# --------------------------------------------------------------------------- drawing
def render(frame, det, tel, cmd, deadband):
    """Annotated PIL image: OBB (green), image centre and deadband (blue), and a status line."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    if det is not None:
        d.polygon(box_corners(*det[:5]), outline=(0, 255, 0), width=2)
    cx, half_db = w / 2, deadband * w / 2
    d.line([cx, 0, cx, h], fill=(0, 160, 255), width=1)
    d.rectangle([cx - half_db, 0, cx + half_db, h], outline=(0, 160, 255))
    d.text((8, 8), f"{tel.state} offset={tel.offset:+.2f} surge={cmd.surge:+.2f} yaw={cmd.yaw:+.2f}",
           fill=(255, 255, 0))
    return img


def annotate(frame, det, tel, cmd, deadband, path="last_frame.jpg"):
    """Save the annotated image to disk (atomically, so viewers never see a half-written file)."""
    render(frame, det, tel, cmd, deadband).save(path + ".tmp.jpg")
    os.replace(path + ".tmp.jpg", path)


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
  <div id="stale">No status updates (script stopped?)</div>
</div>
<script>
const $ = id => document.getElementById(id);
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
    if (!dbSet) { $("db").style.left = (50 - s.deadband * 50) + "%"; $("db").style.width = s.deadband * 100 + "%"; dbSet = true; }
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

    def __init__(self, port, deadband, fps=10.0, quality=70, dry_run=False):
        self.deadband, self.min_dt, self.quality, self.dry_run = deadband, 1.0 / fps, quality, dry_run
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
                    elif self.path.startswith("/stream"):
                        self._stream()
                    else:
                        self.send_error(404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

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
                "state": tel.state, "offset": float(tel.offset), "raw_offset": raw_offset,
                "surge": float(cmd.surge), "yaw": float(cmd.yaw),
                "score": float(det[5]) if det is not None else None,
                "infer_ms": infer_ms, "fps": fps, "elapsed": now - self._t0,
            }
            self._status_t = now
            watching = self._clients > 0
        if watching and now - self._last_push >= self.min_dt:
            self._last_push = now
            buf = io.BytesIO()
            render(frame, det, tel, cmd, self.deadband).save(buf, format="JPEG", quality=self.quality)
            with self._cond:
                self._jpeg = buf.getvalue()
                self._cond.notify_all()

    def _status_json(self):
        with self._lock:
            s = dict(self._status, deadband=self.deadband, dry_run=self.dry_run,
                     age=time.monotonic() - self._status_t, server_time=time.time())
        return json.dumps(s).encode()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
