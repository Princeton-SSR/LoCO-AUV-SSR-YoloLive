"""Vehicle interfaces: MAVLink (same link/convention as LatestControlScript.py) and a dry-run stand-in.

MANUAL_CONTROL mapping used by LatestControlScript.send_manual_control(x, y, z, r):
    x = surge (+ forward), y = 0 (robot has no sway), z = throttle (0..1000, 500 = neutral), r = yaw (+ clockwise)
A background thread re-sends the latest command at 20 Hz (so the autopilot's pilot-input failsafe never trips) and
falls back to neutral if no fresh command arrives within `cmd_timeout` (e.g. if the vision loop hangs).

NOTE: only one process can own /dev/ttyAMA0, so don't run this at the same time as LatestControlScript.py.
"""
import threading
import time

from controller import Command


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

    def set_command(self, cmd: Command):
        """Print the command as it would be sent over MAVLink (rate-limited)."""
        self._last = cmd
        now = time.monotonic()
        if now - self._t > 0.25:
            self._t = now
            print(f"\n[dry-run] x(surge)={_clip(cmd.surge*1000):5d}  y(sway)=    0  "
                  f"r(yaw)={_clip(cmd.yaw*1000):5d}", end="")


class MavVehicle:
    """Real vehicle over MAVLink: sends surge/yaw as MANUAL_CONTROL and handles heartbeat, mode, arm/disarm."""

    def __init__(self, port="/dev/ttyAMA0", baud=921600, mode="STABILIZE", send_hz=20.0, cmd_timeout=0.5):
        self.port, self.baud, self.mode = port, baud, mode
        self.send_period = 1.0 / send_hz
        self.cmd_timeout = cmd_timeout
        self._cmd, self._cmd_t = Command(), 0.0
        self._cmd_lock = threading.Lock()
        self._tx_lock = threading.Lock()   # pymavlink isn't thread-safe for concurrent writes
        self._stop = threading.Event()
        self._armed_by_us = False
        self.loco = None

    def connect(self):
        """Open the serial link, wait for the autopilot heartbeat, set the flight mode, start the background threads."""
        from pymavlink import mavutil
        self._mavutil = mavutil
        self.loco = mavutil.mavlink_connection(self.port, baud=self.baud, source_system=1)
        self.loco.wait_heartbeat()
        self.loco.target_system = 1
        self.loco.target_component = 1
        print("Connected to Pixhawk")
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        mode_id = self.loco.mode_mapping()[self.mode]
        with self._tx_lock:
            self.loco.mav.set_mode_send(self.loco.target_system,
                                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
        print(f"Mode set to {self.mode}")
        threading.Thread(target=self._send_loop, daemon=True).start()

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
                _clip(cmd.surge * 1000), 0, 500, _clip(cmd.yaw * 1000), 0)

    def set_command(self, cmd: Command):
        """Store the newest command (thread-safe); the send loop streams it to the autopilot."""
        with self._cmd_lock:
            self._cmd, self._cmd_t = cmd, time.monotonic()

    def arm(self):
        """Arm the motors and block until the autopilot confirms."""
        self.loco.arducopter_arm()
        self.loco.motors_armed_wait()
        self._armed_by_us = True
        print("Armed")

    def disarm(self):
        """Command neutral for a second, then disarm and block until the autopilot confirms."""
        self.set_command(Command())
        time.sleep(1.0)  # let a second of neutral go out first
        self.loco.arducopter_disarm()
        self.loco.motors_disarmed_wait()
        self._armed_by_us = False
        print("Disarmed")

    def close(self):
        """Stop the robot (neutral), disarm if this object armed it, and stop the background threads."""
        if self.loco is None:
            return
        self.set_command(Command())
        time.sleep(0.3)
        if self._armed_by_us:
            self.disarm()
        self._stop.set()
