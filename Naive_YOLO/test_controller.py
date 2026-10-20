"""Sanity tests for NaiveController (no hardware needed).  Run:  python3 test_controller.py"""
from naive_controller import NaiveController, NaiveConfig, box_offset


def test_directions():
    ctrl = NaiveController()
    cmd, tel = ctrl.update(0.6, 0.0)       # box right of centre
    assert tel.state == "TRACK" and cmd.yaw > 0 and cmd.surge > 0
    ctrl = NaiveController()
    cmd, _ = ctrl.update(-0.6, 0.0)        # box left of centre
    assert cmd.yaw < 0
    ctrl = NaiveController()
    cmd, _ = ctrl.update(0.03, 0.0)        # inside deadband
    assert cmd.yaw == 0 and cmd.surge > 0
    print("ok  directions + deadband")


def test_proportional_and_sign():
    ctrl = NaiveController(NaiveConfig(proportional=True, smooth=1.0))
    small, _ = ctrl.update(0.2, 0.0)
    big, _ = NaiveController(NaiveConfig(proportional=True, smooth=1.0)).update(0.8, 0.0)
    assert 0 < small.yaw < big.yaw
    flipped, _ = NaiveController(NaiveConfig(yaw_sign=-1.0)).update(0.6, 0.0)
    assert flipped.yaw < 0
    print("ok  proportional + yaw_sign")


def test_closed_loop():
    """Toy robot: the box offset drifts with heading; the controller should keep it near the centre."""
    ctrl = NaiveController()
    off, dt = 0.7, 0.1
    for k in range(300):
        cmd, _ = ctrl.update(off, k * dt)
        off -= cmd.yaw * 1.0 * dt          # yawing right moves the line toward the left of the image
    assert abs(off) < 0.2, off
    print(f"ok  closed loop (final offset {off:+.2f})")


def test_offset_and_lost():
    assert abs(box_offset((320, 0), 640)) < 1e-9 and box_offset((640, 0), 640) == 1.0
    ctrl = NaiveController()
    ctrl.update(0.5, 0.0)
    cmd, tel = ctrl.update(None, 0.5)
    assert tel.state == "COAST" and cmd.surge > 0 and cmd.yaw == 0
    cmd, tel = ctrl.update(None, 2.0)
    assert tel.state == "LOST" and cmd.surge == 0
    print("ok  offset + lost behaviour")


if __name__ == "__main__":
    test_offset_and_lost()
    test_directions()
    test_proportional_and_sign()
    test_closed_loop()
    print("all passed")
