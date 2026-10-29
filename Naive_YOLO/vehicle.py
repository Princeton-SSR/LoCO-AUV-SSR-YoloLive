"""Vehicle interfaces: MAVLink (same link/convention as LatestControlScript.py) and a dry-run stand-in.

MANUAL_CONTROL mapping used by LatestControlScript.send_manual_control(x, y, z, r):
    x = surge (+ forward, -1000..1000), y = 0 (robot has no sway), z = throttle (0..1000, 500 = neutral),
    r = yaw (+ clockwise, -1000..1000). Commands are already in these units; they are only clipped here.
A background thread re-sends the latest command at 20 Hz (so the autopilot's pilot-input failsafe never trips) and
falls back to neutral if no fresh command arrives within `cmd_timeout` (e.g. if the vision loop hangs).

A receiver thread is the only reader of the link: it caches the latest attitude / IMU / depth messages
(see telemetry()) and prints autopilot STATUSTEXT messages. arm()/disarm() poll its state instead of reading.

NOTE: only one process can own /dev/ttyAMA0, so don't run this at the same time as LatestControlScript.py
(or mavproxy.service).
"""
import math
import threading
import time

from controller import Command

# Navigation data logged with every frame (blank when unavailable, e.g. dry run or a stale link).
# Depth assumes ArduSub, where VFR_HUD.alt is the altitude relative to the surface (negative underwater).
NAV_FIELDS = ["roll_deg", "pitch_deg", "heading_deg", "roll_rate_dps", "pitch_rate_dps", "yaw_rate_dps",
              "acc_x_ms2", "acc_y_ms2", "acc_z_ms2", "depth_m", "pressure_mbar", "water_temp_c"]
NAV_MSGS = ("ATTITUDE", "RAW_IMU", "VFR_HUD", "SCALED_PRESSURE2")
NAV_MAX_AGE = 2.0   # seconds; older messages are treated as missing


def _clip(v):
    """Round to int and clamp to the MANUAL_CONTROL range [-1000, 1000]."""
    return max(-1000, min(1000, int(round(v))))


class DryRunVehicle:
    """Same interface as MavVehicle but never touches the autopilot; prints commands (about 4x/s) instead."""

    def __init__(self, **_):
        self._last = Command()
        self._t = 0.0

    def connect(self): print("[dry-run] no MAVLink connection; commands are only printed")
    def arm(self): print("[dry-run] arm")
    def disarm(self): print("[dry-run] disarm")
    def close(self): pass
    def telemetry(self): return dict.fromkeys(NAV_FIELDS)

    def set_command(self, cmd: Command):
        """Print the command as it would be sent over MAVLink (rate-limited)."""
        self._last = cmd
        now = time.monotonic()
        if now - self._t > 0.25:
            self._t = now
            print(f"\n[dry-run] x(surge)={_clip(cmd.surge):5d}  y(sway)=    0  "
                  f"r(yaw)={_clip(cmd.yaw):5d}", end="")


class MavVehicle:
    """Real vehicle over MAVLink: sends surge/yaw as MANUAL_CONTROL and handles heartbeat, mode, arm/disarm."""

    def __init__(self, port="/dev/ttyAMA0", baud=921600, mode="MANUAL", send_hz=20.0, cmd_timeout=0.5,
                 source_system=255, nav_hz=10.0):
        # source_system must equal the autopilot's SYSID_MYGCS (default 255), or MANUAL_CONTROL / arm are ignored
        self.port, self.baud, self.mode, self.source_system = port, baud, mode, source_system
        self.send_period = 1.0 / send_hz
        self.cmd_timeout = cmd_timeout
        self._cmd, self._cmd_t = Command(), 0.0
        self._cmd_lock = threading.Lock()
        self._tx_lock = threading.Lock()   # pymavlink isn't thread-safe for concurrent writes
        self._stop = threading.Event()
        self._armed_by_us = False
        self.nav_hz = nav_hz
        self._nav = {}                     # message type -> (monotonic receive time, message)
        self._nav_lock = threading.Lock()
        self.loco = None

    def connect(self):
        """Open the serial link, wait for the autopilot heartbeat, set the flight mode, start the background threads."""
        from pymavlink import mavutil
        self._mavutil = mavutil
        self.loco = mavutil.mavlink_connection(self.port, baud=self.baud, source_system=self.source_system)
        self.loco.wait_heartbeat()
        self.loco.target_system = 1
        self.loco.target_component = 1
        print("Connected to Pixhawk")
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        mode_id = self.loco.mode_mapping()[self.mode]
        with self._tx_lock:
            self.loco.mav.set_mode_send(self.loco.target_system,
                                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
        print(f"Mode set to {self.mode}")
        self._request_nav_streams()
        threading.Thread(target=self._send_loop, daemon=True).start()

    def _request_nav_streams(self):
        """Ask the autopilot to send the NAV_MSGS at nav_hz (without mavproxy nothing else requests them)."""
        m = self._mavutil.mavlink
        interval_us = 1e6 / self.nav_hz
        with self._tx_lock:
            for name in NAV_MSGS:
                self.loco.mav.command_long_send(
                    self.loco.target_system, self.loco.target_component, m.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                    getattr(m, "MAVLINK_MSG_ID_" + name), interval_us, 0, 0, 0, 0, 0)

    def _recv_loop(self):
        """Thread: sole reader of the link. Caches nav messages, prints autopilot text; armed state updates inside pymavlink."""
        while not self._stop.is_set():
            try:
                msg = self.loco.recv_match(blocking=True, timeout=0.5)
            except Exception as e:  # e.g. SerialException if another process grabs the port
                print(f"\n[mavlink] receive error: {e}")
                time.sleep(0.5)
                continue
            if msg is None:
                continue
            kind = msg.get_type()
            if kind in NAV_MSGS:
                with self._nav_lock:
                    self._nav[kind] = (time.monotonic(), msg)
            elif kind == "STATUSTEXT":
                print(f"\n[autopilot] {msg.text}")
            elif kind == "COMMAND_ACK" and msg.command == 400:
                print(f"\n[autopilot] arm/disarm command ack result={msg.result} (0 = accepted)")

    def telemetry(self):
        """Latest attitude / IMU / depth as {NAV_FIELDS name: value}; None for anything missing or older than NAV_MAX_AGE."""
        now = time.monotonic()
        with self._nav_lock:
            fresh = {k: m for k, (t, m) in self._nav.items() if now - t < NAV_MAX_AGE}
        d = dict.fromkeys(NAV_FIELDS)
        att, imu, hud, prs = (fresh.get(k) for k in NAV_MSGS)
        if att:
            d.update(roll_deg=math.degrees(att.roll), pitch_deg=math.degrees(att.pitch),
                     heading_deg=math.degrees(att.yaw) % 360, roll_rate_dps=math.degrees(att.rollspeed),
                     pitch_rate_dps=math.degrees(att.pitchspeed), yaw_rate_dps=math.degrees(att.yawspeed))
        if imu:  # ArduPilot fills RAW_IMU acceleration in milli-g
            g = 9.80665e-3
            d.update(acc_x_ms2=imu.xacc * g, acc_y_ms2=imu.yacc * g, acc_z_ms2=imu.zacc * g)
        if hud:
            d["depth_m"] = -hud.alt
        if prs:  # external (Bar30) pressure sensor on ArduSub
            d.update(pressure_mbar=prs.press_abs, water_temp_c=prs.temperature / 100.0)
        return d

    def _heartbeat_loop(self):
        """Thread: send a GCS heartbeat once per second so the autopilot knows a controller is attached."""
        m = self._mavutil.mavlink
        while not self._stop.is_set():
            with self._tx_lock:
                self.loco.mav.heartbeat_send(m.MAV_TYPE_GCS, m.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            time.sleep(1)

    def _send_loop(self):
        """Thread: stream the latest command at `send_hz`; send neutral if it is older than `cmd_timeout`."""
        while not self._stop.is_set():
            with self._cmd_lock:
                cmd = self._cmd if time.monotonic() - self._cmd_t < self.cmd_timeout else Command()
            self._send(cmd)
            time.sleep(self.send_period)

    def _send(self, cmd):
        """Send one MANUAL_CONTROL message (x=surge, y=0, z=500 neutral throttle, r=yaw)."""
        with self._tx_lock:
            self.loco.mav.manual_control_send(
                self.loco.target_system,
                _clip(cmd.surge), 0, 500, _clip(cmd.yaw), 0)

    def set_command(self, cmd: Command):
        """Store the newest command (thread-safe); the send loop streams it to the autopilot."""
        with self._cmd_lock:
            self._cmd, self._cmd_t = cmd, time.monotonic()

    def arm(self, timeout=8.0):
        """Arm the motors; resend every 2 s and give up after `timeout` s (the receiver prints the autopilot's messages).

        Raises RuntimeError if arming isn't confirmed (e.g. a pre-arm check failed or the command was ignored).
        """
        self._armed_by_us = True   # so close() disarms even if the arm confirmation arrives late
        end, last_send = time.monotonic() + timeout, 0.0
        while time.monotonic() < end:
            if time.monotonic() - last_send > 2.0:
                with self._tx_lock:
                    self.loco.arducopter_arm()
                last_send = time.monotonic()
            if self.loco.motors_armed():
                print("Armed")
                return
            time.sleep(0.1)
        raise RuntimeError(f"Arming not confirmed after {timeout:.0f} s (check pre-arm messages above, "
                           "SYSID_MYGCS, or arm from QGC to see the reason)")

    def disarm(self, timeout=5.0):
        """Command neutral for a second, then disarm (resent every second) until the autopilot confirms or `timeout` s."""
        self.set_command(Command())
        time.sleep(1.0)  # let a second of neutral go out first
        end, last_send = time.monotonic() + timeout, 0.0
        while time.monotonic() < end:
            if time.monotonic() - last_send > 1.0:
                with self._tx_lock:
                    self.loco.arducopter_disarm()
                last_send = time.monotonic()
            if not self.loco.motors_armed():
                self._armed_by_us = False
                print("Disarmed")
                return
            time.sleep(0.1)
        print(f"WARNING: disarm not confirmed after {timeout:.0f} s - check the vehicle / disarm from QGC")

    def close(self):
        """Stop the robot (neutral), disarm if this object armed it, and stop the background threads."""
        if self.loco is None:
            return
        self.set_command(Command())
        time.sleep(0.3)
        if self._armed_by_us:
            self.disarm()
        self._stop.set()
