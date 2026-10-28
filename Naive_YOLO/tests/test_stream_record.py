"""Recording stream: each /record JPEG carries the command for that frame. Run from Naive_YOLO/:
    python3 tests/test_stream_record.py
"""
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from controller import Command, Telemetry
from stream import PAGE, Streamer


def _recv_until(sock, buf, marker):
    while marker not in buf:
        chunk = sock.recv(65536)
        assert chunk, "connection closed early"
        buf += chunk
    return buf


def _open(port, path):
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.settimeout(5)
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode())
    return sock


def _wait_clients(streamer, n):
    for _ in range(100):
        if streamer._clients >= n:
            return
        time.sleep(0.01)
    raise AssertionError(f"expected {n} viewers, have {streamer._clients}")


def _next_part(sock, buf):
    buf = _recv_until(sock, buf, b"\r\n\r\n")
    head, buf = buf.split(b"\r\n\r\n", 1)
    length = None
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    assert length and length > 0
    while len(buf) < length:
        buf += sock.recv(65536)
    return head, buf[:length], buf[length:]


def test_page_has_record_button():
    page = PAGE.decode()
    assert ">start recording</button>" in page
    assert 'btn.textContent = "stop recording"' in page
    assert "#rec.recording{background:#b91c1c}" in page
    assert 'fetch("/record"' in page
    print("ok  record button")


def test_record_part_carries_command():
    streamer = Streamer(0, deadband=0.1, fps=100)
    port = streamer._server.server_address[1]
    frame = np.zeros((32, 48, 3), np.uint8)
    frame[:, :, 0] = 220
    det = (24.0, 16.0, 8.0, 3.0, 0.2, 0.87, 0)
    rec = live = None
    try:
        rec = _open(port, "/record")
        _wait_clients(streamer, 1)
        streamer._last_push = 0.0
        streamer.update(frame, det, Telemetry("TRACK", 0.15), Command(0.25, -0.1), raw_offset=0.2, infer_ms=33.0)
        buf = _recv_until(rec, b"", b"\r\n\r\n")
        http, buf = buf.split(b"\r\n\r\n", 1)
        assert b"application/octet-stream" in http.lower()
        head, jpeg, _rest = _next_part(rec, buf)
        assert jpeg[:2] == b"\xff\xd8"
        meta_line = next(line for line in head.split(b"\r\n") if line.lower().startswith(b"x-frame-meta:"))
        meta = json.loads(meta_line.split(b":", 1)[1])
        assert meta["state"] == "TRACK"
        assert meta["surge"] == 0.25 and meta["yaw"] == -0.1
        assert meta["offset"] == 0.15 and abs(meta["raw_offset"] - 0.2) < 1e-9
        assert abs(meta["score"] - 0.87) < 1e-9 and meta["infer_ms"] == 33.0
        assert meta["box_cx"] == 24.0 and meta["deadband"] == 0.1 and meta["deadband_center"] == 0.0
        assert b"\n" not in meta_line

        live = _open(port, "/stream")
        _wait_clients(streamer, 2)
        streamer._last_push = 0.0
        streamer.update(frame, None, Telemetry("COAST", 0.0), Command(0.1, 0.0), raw_offset=None, infer_ms=10.0)
        buf = _recv_until(live, b"", b"\r\n\r\n")
        http, buf = buf.split(b"\r\n\r\n", 1)
        assert b"multipart/x-mixed-replace" in http.lower()
        live_head, live_jpeg, _rest = _next_part(live, buf)
        assert live_jpeg[:2] == b"\xff\xd8"
        assert not any(line.lower().startswith(b"x-frame-meta:") for line in live_head.split(b"\r\n"))
    finally:
        for sock in (rec, live):
            if sock is not None:
                sock.close()
        streamer.close()
    print("ok  record part carries the command")


if __name__ == "__main__":
    test_page_has_record_button()
    test_record_part_carries_command()
    print("all passed")
