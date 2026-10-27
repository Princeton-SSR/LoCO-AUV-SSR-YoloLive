# MainControl_1.py
# This script combines thruster control and camera capture on a Raspberry Pi connected to a Pixhawk. 
# It moves forward while taking photos at regular intervals, then pauses, and moves forward again. Photos are saved with timestamps in a specified directory.

from pymavlink import mavutil
from picamera2 import Picamera2
import time
import threading
import os

# --- Setup ---
PHOTO_DIR = '/home/{}/Pictures/ExpPhotos'.format(os.getlogin()) # or: PHOTO_DIR = '/home/ssr-loco/Pictures/ExpPhotos'
PHOTO_INTERVAL = 0.5  # seconds between photos
FORWARD_SPEED = 300   # forward speed (-1000 to 1000), 300 is slow
SWIMDUR = 10    # seconds spent swimming

os.makedirs(PHOTO_DIR, exist_ok=True)
print(f"Photos will be saved to: {PHOTO_DIR}")

# --- Connect to Pixhawk ---
master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=921600, source_system=1)
master.wait_heartbeat()
master.target_system = 1
master.target_component = 1
print("Connected to Pixhawk")

# --- Heartbeat thread ---
def send_heartbeat():
    while True:
        master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )
        time.sleep(1)

threading.Thread(target=send_heartbeat, daemon=True).start()

# --- Camera setup ---
camera = Picamera2()
camera.configure(camera.create_still_configuration())
camera.start()
print("Camera ready")

# --- Photo capture thread ---
photo_active = False
photo_count = 0

def capture_photos():
    global photo_count
    while True:
        if photo_active:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            path = f"{PHOTO_DIR}/photo_{timestamp}_{photo_count:04d}.jpg"
            camera.capture_file(path)
            print(f"Captured: {path}")
            photo_count += 1
            time.sleep(PHOTO_INTERVAL)
        else:
            time.sleep(0.1)

threading.Thread(target=capture_photos, daemon=True).start()

# --- Control functions ---
def send_manual_control(x=0, y=0, z=500, r=0):
    """
    x: forward/back (-1000 to 1000)
    y: left/right (-1000 to 1000)
    z: throttle (0 to 1000, 500 = neutral)
    r: yaw (-1000 to 1000)
    """
    master.mav.manual_control_send(
        master.target_system,
        x, y, z, r, 0
    )

def move_forward(duration, speed=FORWARD_SPEED):
    global photo_active
    print(f"Moving forward at speed {speed} for {duration}s...")
    photo_active = True
    start = time.time()
    while time.time() - start < duration:
        send_manual_control(x=speed)
        time.sleep(0.05)
    photo_active = False
    send_manual_control(x=0)
    print("Stopped")

def pause(duration):
    print(f"Pausing for {duration}s...")
    send_manual_control(x=0)
    time.sleep(duration)

# --- Main procedure ---
try:
    # 5 second delay before starting
    print("Starting in 5 seconds...")
    time.sleep(2)
    master.arducopter_arm()
    master.motors_armed_wait()
    print("Armed")
    time.sleep(3)

    # Run the procedure
    move_forward(duration=SWIMDUR)
    #pause(duration=2)
    #move_forward(duration=5)

    # Stop and disarm
    send_manual_control(x=0)
    time.sleep(0.5)
    master.arducopter_disarm()
    master.motors_disarmed_wait()
    print("Disarmed")

except KeyboardInterrupt:
    print("Interrupted — stopping and disarming...")
    send_manual_control(x=0)
    photo_active = False
    time.sleep(0.5)
    master.arducopter_disarm()
    master.motors_disarmed_wait()

finally:
    camera.stop()
    print(f"Done. {photo_count} photos saved to {PHOTO_DIR}")