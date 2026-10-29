#!/usr/bin/env python3
"""Live web view for follow_transect.py: annotated camera MJPEG stream + a live status panel.

    Streamer(port, deadband).update(frame, det, tel, cmd, raw_offset, infer_ms, nav)   # call once per frame

Open http://<pi-ip>:PORT/ in a browser. Endpoints:
    /         web page: video, state, offset bar, surge/yaw bars, detection score, inference time.
              "start recording" keeps the JPEGs in the browser and downloads one zip on "stop recording".
    /stream   raw MJPEG (annotated frames), usable in any <img> tag or VLC
    /record   same MJPEG, plus an X-Frame-Meta header on each part (the command for that frame)
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
    d.text((8, 8), f"{tel.state} offset={tel.offset:+.2f} surge={cmd.surge:+.0f} yaw={cmd.yaw:+.0f}",
           fill=(255, 255, 0))
    return img


def annotate(frame, det, tel, cmd, deadband, path="last_frame.jpg", deadband_center=0.0):
    """Save the annotated image to disk (atomically, so viewers never see a half-written file)."""
    render(frame, det, tel, cmd, deadband, deadband_center).save(path + ".tmp.jpg")
    os.replace(path + ".tmp.jpg", path)


# --------------------------------------------------------------------------- tunable parameters
# (NaiveConfig field, label, kind, min, max, step); kind "sign" is a checkbox that maps checked -> -1
TUNABLES = [
    ("surge", "Surge (-1000..1000)", "float", -1000, 1000, 10),
    ("yaw", "Yaw command (0..1000)", "float", 0, 1000, 10),
    ("deadband", "Deadband", "float", 0.0, 0.5, 0.01),
    ("deadband_center", "Deadband centre (- left / + right)", "float", -0.5, 0.5, 0.01),
    ("right_bias", "Right-turn bias (extra yaw, 0..1000)", "float", 0, 1000, 5),
    ("smooth", "Smoothing (1 = none)", "float", 0.05, 1.0, 0.05),
    ("lost_hold", "Lost hold (s)", "float", 0.0, 5.0, 0.1),
    ("lost_timeout", "Lost timeout (s, 0 = never stop)", "float", 0.0, 120.0, 1.0),
    ("lost_spin", "Lost spin yaw (0 = off, 0..1000)", "float", 0, 1000, 5),
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
  #statusrow{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:14px}
  #recbar{display:flex;flex-direction:column;align-items:flex-end;gap:6px}
  #rec{font:600 14px system-ui,sans-serif;padding:8px 16px;border:0;border-radius:6px;background:#334155;color:#fff;cursor:pointer}
  #rec.recording{background:#b91c1c}
  #rec:disabled{opacity:.6;cursor:default}
  #reccount{color:#aaa;font-variant-numeric:tabular-nums}
  .state{font-size:28px;font-weight:700;padding:6px 14px;border-radius:6px;display:inline-block}
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
  <div id="statusrow">
    <div id="state" class="state WAIT">--</div>
    <div id="recbar">
      <button id="rec" type="button">start recording</button>
      <span id="reccount"></span>
    </div>
  </div>

  <div class="row">
    <div class="lbl"><span>Offset (smoothed)</span><b id="offset">--</b></div>
    <div class="bar"><div class="db" id="db"></div><div class="mid"></div><div class="dot" id="odot"></div></div>
    <div class="ends"><span>line left</span><span>centre</span><span>line right</span></div>
  </div>

  <div class="row">
    <div class="lbl"><span>Surge command</span><b id="surge">--</b></div>
    <div class="bar"><div class="mid"></div><div class="fill" id="sfill" style="background:#38bdf8"></div></div>
    <div class="ends"><span>-1000 back</span><span>0</span><span>+1000 fwd</span></div>
  </div>

  <div class="row">
    <div class="lbl"><span>Yaw command</span><b id="yaw">--</b></div>
    <div class="bar"><div class="mid"></div><div class="fill" id="yfill" style="background:#f472b6"></div></div>
    <div class="ends"><span>-1000 left</span><span>0</span><span>+1000 right</span></div>
  </div>

  <table>
    <tr><td>Raw offset</td><td id="raw">--</td></tr>
    <tr><td>Detection score</td><td id="score">--</td></tr>
    <tr><td>Inference</td><td id="infer">--</td></tr>
    <tr><td>Loop rate</td><td id="fps">--</td></tr>
    <tr><td>Run time</td><td id="t">--</td></tr>
    <tr><td>Depth</td><td id="depth">--</td></tr>
    <tr><td>Heading</td><td id="heading">--</td></tr>
    <tr><td>Roll / pitch</td><td id="rp">--</td></tr>
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
  p.spec.forEach(s => {   // every tunable is also a CSV column (deadband / deadband_center already have their own)
    if (s.name !== "deadband" && s.name !== "deadband_center") CSV_COLS.push("param_" + s.name);
  });
  p.spec.forEach(s => {
    const row = document.createElement("div"); row.className = "row";
    const v = p.values[s.name];
    if (s.kind === "float") {
      row.innerHTML = '<div class="lbl"><span>' + s.label + '</span><input type="number" min="' + s.min +
        '" max="' + s.max + '" step="' + s.step + '" value="' + v + '" style="width:84px"></div>' +
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
function centredFill(el, v) {          // bar growing from the middle, v in -1000..1000
  v = Math.max(-1000, Math.min(1000, v)) / 1000;
  el.style.left = (v < 0 ? 50 + v * 50 : 50) + "%";
  el.style.width = Math.abs(v) * 50 + "%";
}
let dbSet = false;
// The <img> does not reopen a motion-JPEG connection after the script exits. Once /status
// has failed for a few polls and then answers again, point the image at /stream once more.
let streamMisses = 0;
let streamDown = false;
function reconnectStream() {
  document.querySelector("#video img").src = "/stream?t=" + Date.now();
}
async function tick() {
  try {
    const s = await (await fetch("/status", {cache: "no-store"})).json();
    if (streamDown) { streamDown = false; reconnectStream(); }
    streamMisses = 0;
    $("stale").style.display = (Date.now() / 1000 - s.server_time > 3 || s.age > 3) ? "block" : "none";
    $("mode").textContent = s.dry_run ? "(dry run)" : "";
    $("state").textContent = s.state; $("state").className = "state " + s.state;
    $("offset").textContent = f(s.offset);
    $("odot").style.left = (50 + Math.max(-1, Math.min(1, s.offset)) * 50) + "%";
    $("db").style.left = (50 + (s.deadband_center - s.deadband) * 50) + "%"; $("db").style.width = s.deadband * 100 + "%";
    $("surge").textContent = f(s.surge, 0); centredFill($("sfill"), s.surge);
    $("yaw").textContent = f(s.yaw, 0);     centredFill($("yfill"), s.yaw);
    $("raw").textContent = s.raw_offset == null ? "no detection" : f(s.raw_offset);
    $("score").textContent = s.score == null ? "--" : s.score.toFixed(2);
    $("infer").textContent = s.infer_ms == null ? "--" : s.infer_ms.toFixed(0) + " ms";
    $("fps").textContent = s.fps == null ? "--" : s.fps.toFixed(1) + " Hz";
    $("t").textContent = s.elapsed.toFixed(1) + " s";
    $("depth").textContent = s.depth_m == null ? "--" : s.depth_m.toFixed(2) + " m";
    $("heading").textContent = s.heading_deg == null ? "--" : s.heading_deg.toFixed(0) + " deg";
    $("rp").textContent = s.roll_deg == null ? "--" : f(s.roll_deg, 1) + " / " + f(s.pitch_deg, 1) + " deg";
  } catch (e) {
    streamMisses += 1;
    if (streamMisses >= 3) streamDown = true;
    $("stale").style.display = "block";
  }
}
setInterval(tick, 150); tick();

// Recording stays in the browser. Stop builds one zip (commands.csv + frames/frame_#####.jpg) and downloads it.
const CSV_COLS = ["frame","wall_time","elapsed","state","score","box_cx","box_cy","box_w","box_h","box_angle","raw_offset","offset","surge","yaw","infer_ms","fps","deadband","deadband_center",
  "roll_deg","pitch_deg","heading_deg","roll_rate_dps","pitch_rate_dps","yaw_rate_dps","acc_x_ms2","acc_y_ms2","acc_z_ms2","depth_m","pressure_mbar","water_temp_c"];
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(data) {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function csvCell(v) {
  if (v == null || v !== v) return "";
  const s = String(v);
  const lf = String.fromCharCode(10);
  if (s.indexOf(",") < 0 && s.indexOf('"') < 0 && s.indexOf(lf) < 0) return s;
  return '"' + s.split('"').join('""') + '"';
}
function fileStamp(d) {
  const p = n => (n < 10 ? "0" : "") + n;
  return "transect_" + d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate())
    + "_" + p(d.getHours()) + "-" + p(d.getMinutes()) + "-" + p(d.getSeconds());
}
function frameName(n) {
  let s = String(n);
  while (s.length < 5) s = "0" + s;
  return "frames/frame_" + s + ".jpg";
}
function zipParts(frames) {
  const enc = new TextEncoder();
  const lf = String.fromCharCode(10);
  let csv = CSV_COLS.join(",") + lf;
  const files = [];
  for (let i = 0; i < frames.length; i++) {
    const meta = frames[i].meta || {};
    const n = i + 1;
    files.push({name: frameName(n), data: frames[i].jpeg});
    const row = [n];
    for (let c = 1; c < CSV_COLS.length; c++) row.push(meta[CSV_COLS[c]]);
    csv += row.map(csvCell).join(",") + lf;
  }
  files.unshift({name: "commands.csv", data: enc.encode(csv)});
  const parts = [], central = [];
  let offset = 0;
  const now = new Date();
  const dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const dosDate = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  files.forEach(f => {
    const name = enc.encode(f.name);
    const crc = crc32(f.data);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(8, 0, true);
    local.setUint16(10, dosTime, true);
    local.setUint16(12, dosDate, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, f.data.length, true);
    local.setUint32(22, f.data.length, true);
    local.setUint16(26, name.length, true);
    parts.push(new Uint8Array(local.buffer), name, f.data);
    const cen = new DataView(new ArrayBuffer(46));
    cen.setUint32(0, 0x02014b50, true);
    cen.setUint16(4, 20, true);
    cen.setUint16(6, 20, true);
    cen.setUint16(10, 0, true);
    cen.setUint16(12, dosTime, true);
    cen.setUint16(14, dosDate, true);
    cen.setUint32(16, crc, true);
    cen.setUint32(20, f.data.length, true);
    cen.setUint32(24, f.data.length, true);
    cen.setUint16(28, name.length, true);
    cen.setUint32(42, offset, true);
    central.push(new Uint8Array(cen.buffer), name);
    offset += 30 + name.length + f.data.length;
  });
  let centralSize = 0;
  central.forEach(p => { centralSize += p.length; });
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);
  central.forEach(p => parts.push(p));
  parts.push(new Uint8Array(end.buffer));
  return parts;
}
function concatBytes(a, b) {
  const n = new Uint8Array(a.length + b.length);
  n.set(a, 0); n.set(b, a.length);
  return n;
}
function findHeaderEnd(buf) {
  for (let i = 0; i + 3 < buf.length; i++) {
    if (buf[i] === 13 && buf[i + 1] === 10 && buf[i + 2] === 13 && buf[i + 3] === 10) return i;
  }
  return -1;
}
function headerValue(header, name) {
  const lf = String.fromCharCode(10);
  const want = name.toLowerCase();
  const lines = header.split(lf);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const c = line.indexOf(":");
    if (c > 0 && line.slice(0, c).trim().toLowerCase() === want) return line.slice(c + 1).trim();
  }
  return "";
}
async function readFrames(signal, onFrame) {
  let res;
  try { res = await fetch("/record", {cache: "no-store", signal}); }
  catch (e) { if (e.name === "AbortError") return; throw e; }
  if (!res.ok || !res.body) throw new Error("record stream failed");
  const reader = res.body.getReader();
  let buf = new Uint8Array(0);
  const dec = new TextDecoder();
  try {
    while (true) {
      const got = await reader.read();
      if (got.done) break;
      buf = concatBytes(buf, got.value);
      while (true) {
        const headerEnd = findHeaderEnd(buf);
        if (headerEnd < 0) break;
        const header = dec.decode(buf.subarray(0, headerEnd));
        const len = parseInt(headerValue(header, "Content-Length"), 10);
        if (!(len >= 0)) { buf = buf.slice(headerEnd + 4); continue; }
        const start = headerEnd + 4;
        if (buf.length < start + len) break;
        const jpeg = buf.slice(start, start + len);
        let meta = {};
        const raw = headerValue(header, "X-Frame-Meta");
        if (raw) { try { meta = JSON.parse(raw); } catch (e) { meta = {}; } }
        onFrame(jpeg, meta);
        let next = start + len;
        if (buf[next] === 13 && buf[next + 1] === 10) next += 2;
        buf = buf.slice(next);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") throw e;
  } finally {
    try { reader.cancel(); } catch (e) {}
  }
}
let session = null;
$("rec").onclick = async () => {
  const btn = $("rec");
  if (session === null) {
    const frames = [];
    const ctrl = new AbortController();
    btn.classList.add("recording");
    btn.textContent = "stop recording";
    $("reccount").textContent = "0 frames";
    const done = readFrames(ctrl.signal, (jpeg, meta) => {
      frames.push({jpeg: jpeg, meta: meta});
      const n = frames.length;
      $("reccount").textContent = n + (n === 1 ? " frame" : " frames");
    });
    session = {frames: frames, ctrl: ctrl, started: new Date(), done: done};
    return;
  }
  if (btn.disabled) return;
  btn.disabled = true;
  const cur = session;
  session = null;
  cur.ctrl.abort();
  try { await cur.done; } catch (e) {}
  if (!cur.frames.length) {
    $("reccount").textContent = "no frames received";
  } else {
    $("reccount").textContent = "preparing download...";
    await new Promise(r => setTimeout(r, 30));
    const blob = new Blob(zipParts(cur.frames), {type: "application/zip"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = fileStamp(cur.started) + ".zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 20000);
    const n = cur.frames.length;
    $("reccount").textContent = n + (n === 1 ? " frame saved" : " frames saved");
  }
  btn.classList.remove("recording");
  btn.textContent = "start recording";
  btn.disabled = false;
};
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
        self._meta = b"{}"          # JSON for the JPEG currently in _jpeg (sent only on /record)
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
                        self._stream(with_meta=False)
                    elif self.path.startswith("/record"):
                        self._stream(with_meta=True)
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

            def _stream(self, with_meta):
                self.send_response(200)
                # /record is read with fetch(), which streams a plain body. The <img> live view stays
                # multipart/x-mixed-replace; browsers do not reliably expose that type to fetch().
                ctype = "application/octet-stream" if with_meta else "multipart/x-mixed-replace; boundary=frame"
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with streamer._lock:
                    streamer._clients += 1
                try:
                    last = None
                    while True:
                        with streamer._cond:
                            streamer._cond.wait(timeout=5)
                            jpeg, meta = streamer._jpeg, streamer._meta
                        if jpeg is None or jpeg is last:
                            continue
                        last = jpeg
                        head = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                        if with_meta:
                            head += b"\r\nX-Frame-Meta: " + meta
                        self.wfile.write(head + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with streamer._lock:
                        streamer._clients -= 1

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def update(self, frame, det, tel, cmd, raw_offset=None, infer_ms=None, nav=None):
        """Call once per loop. Always refreshes the status; renders/encodes the frame only if someone is watching.

        nav: vehicle.telemetry() dict (attitude / IMU / depth), added to the status and the recorded CSV.
        """
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
                **{k: None if v is None else float(v) for k, v in (nav or {}).items()},
            }
            self._status_t = now
            snap = dict(self._status)
            watching = self._clients > 0
        if watching and now - self._last_push >= self.min_dt:
            self._last_push = now
            snap["wall_time"] = time.time()
            snap["deadband"] = float(self.deadband)
            snap["deadband_center"] = float(self.deadband_center)
            if self.cfg is not None:
                for name, *_ in TUNABLES:
                    v = getattr(self.cfg, name)
                    snap["param_" + name] = int(v) if isinstance(v, bool) else float(v)
            if det is not None:
                snap["box_cx"], snap["box_cy"], snap["box_w"], snap["box_h"], snap["box_angle"] = (float(v) for v in det[:5])
            else:
                snap["box_cx"] = snap["box_cy"] = snap["box_w"] = snap["box_h"] = snap["box_angle"] = None
            buf = io.BytesIO()
            render(frame, det, tel, cmd, self.deadband, self.deadband_center).save(buf, format="JPEG", quality=self.quality)
            with self._cond:
                self._jpeg = buf.getvalue()
                self._meta = json.dumps(snap, separators=(",", ":")).encode()
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
