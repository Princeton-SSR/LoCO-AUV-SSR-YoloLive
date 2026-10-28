"""Hard-coded test run (no camera / model): straight, then right, then left.

Run from Naive_YOLO/:
    python3 tests/test_vehicle.py --dry-run          # just print the commands
    python3 tests/test_vehicle.py --arm              # real vehicle: arm, run the sequence, disarm

Ctrl+C always sends neutral (and disarms if this script armed).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `vehicle` / `controller` are importable

from controller import Command
from vehicle import DryRunVehicle, MavVehicle

# ---- test parameters (MANUAL_CONTROL units, -1000..1000) ----
SURGE = 50          # forward command used in every segment
YAW = 20            # yaw magnitude: + = right (clockwise), - = left
SEGMENT_SECONDS = 2.0
START_DELAY = 5.0   # seconds before the sequence starts (after arming)
SEND_HZ = 10        # how often the command is refreshed (MavVehicle falls back to neutral after 0.5 s of silence)

# (label, surge, yaw)
SEQUENCE = [
    ("straight", SURGE, 0),
    ("right", SURGE, +YAW),
    ("left", SURGE, -YAW),
]


def run_segment(vehicle, label, surge, yaw, seconds):
    """Hold one command for `seconds`, refreshing it so the vehicle's command timeout never trips."""
    print(f"\n{label}: surge={surge} yaw={yaw:+} for {seconds:.1f} s")
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        vehicle.set_command(Command(surge=surge, yaw=yaw))
        time.sleep(1.0 / SEND_HZ)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="don't touch MAVLink, just print commands")
    p.add_argument("--arm", action="store_true", help="arm before the sequence (and disarm at the end)")
    p.add_argument("--port", default="/dev/ttyAMA0")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--source-system", type=int, default=255)
    p.add_argument("--mode", default="STABILIZE")
    p.add_argument("--start-delay", type=float, default=START_DELAY)
    args = p.parse_args()

    vehicle = DryRunVehicle() if args.dry_run else MavVehicle(
        args.port, args.baud, args.mode, source_system=args.source_system)
    vehicle.connect()

    try:
        print(f"Starting in {args.start_delay:.0f} s (Ctrl+C aborts)...")
        time.sleep(args.start_delay)
        if args.arm:
            vehicle.arm()
        for label, surge, yaw in SEQUENCE:
            run_segment(vehicle, label, surge, yaw, SEGMENT_SECONDS)
        vehicle.set_command(Command())  # neutral at the end
        print("\nSequence finished")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        vehicle.close()  # sends neutral, disarms if we armed
        print("Done.")


if __name__ == "__main__":
    main()
