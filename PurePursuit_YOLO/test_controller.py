"""Closed-loop sanity test of TransectController with a crude kinematic robot (no hardware needed).

Run:  python3 test_controller.py
"""
import math

import numpy as np

from transect_controller import Line, TransectController, ControllerConfig, GroundMap, line_from_obb


def simulate(start_y, start_heading, cfg=None, seconds=40.0, hz=10.0, v_max=0.5, seg=2.0):
    """World: line is the x-axis (y=0). +y is 'right' when heading=0. Returns final (y, heading) and max |y|."""
    ctrl = TransectController(cfg)
    x, y, th = 0.0, start_y, start_heading
    surge = yaw = 0.0
    dt, max_y = 1.0 / hz, abs(start_y)
    for k in range(int(seconds * hz)):
        fwd = np.array([math.cos(th), math.sin(th)])
        rgt = np.array([-math.sin(th), math.cos(th)])

        def to_robot(p):
            d = np.asarray(p) - np.array([x, y])
            return np.array([d @ fwd, d @ rgt])

        # the camera sees a finite segment of the line ahead of / around the robot
        line = Line(to_robot((x - 0.3, 0.0)), to_robot((x + seg, 0.0)))
        if line.p_near[0] > line.p_far[0]:
            line = Line(line.p_far, line.p_near)
        cmd, _ = ctrl.update(line, k * dt)

        # first-order actuator lag
        surge += 0.4 * (cmd.surge - surge)
        yaw += 0.4 * (cmd.yaw - yaw)
        vel = surge * v_max * fwd
        x, y = x + vel[0] * dt, y + vel[1] * dt
        th += yaw * 1.0 * dt
        max_y = max(max_y, abs(y))
    return y, th, max_y


def test_converges():
    for sy, sh in [(-0.5, 0.3), (0.5, -0.3), (0.0, 0.8), (0.3, 0.0), (-0.6, -0.5)]:
        y, th, _ = simulate(sy, sh)
        wrapped = (th + math.pi) % (2 * math.pi) - math.pi
        assert abs(y) < 0.05 and abs(wrapped) < 0.08, f"start y={sy} th={sh} -> end y={y:.3f} th={wrapped:.3f}"
        print(f"ok  start y={sy:+.1f} th={sh:+.1f} -> end y={y:+.3f} th={wrapped:+.3f}")


def test_obb_geometry():
    gm = GroundMap(640, 480, ref_frac=(0.5, 0.85))
    # vertical thin box straight ahead of the reference pixel: w < h, angle 0 -> long axis is image-y
    line = line_from_obb((320, 240, 20, 300, 0.0), gm)
    assert line is not None and abs(line.p_near[1]) < 1e-6 and abs(line.p_far[1]) < 1e-6
    assert line.p_far[0] > line.p_near[0]
    # box shifted right of the reference pixel -> positive (right) cross-track
    line = line_from_obb((420, 240, 20, 300, 0.0), gm)
    ctrl = TransectController()
    cmd, tel = ctrl.update(line, 0.0)
    assert tel.e > 0 and cmd.yaw > 0  # line on the right -> yaw right
    # stubby box is rejected
    assert line_from_obb((320, 240, 40, 50, 0.0), gm) is None
    print("ok  obb geometry")


def test_lost_behaviour():
    ctrl = TransectController()
    gm = GroundMap(640, 480)
    ctrl.update(line_from_obb((320, 240, 20, 300, 0.0), gm), 0.0)
    cmd, tel = ctrl.update(None, 0.5)
    assert tel.state == "COAST" and cmd.surge > 0 and cmd.yaw == 0
    cmd, tel = ctrl.update(None, 2.0)
    assert tel.state == "LOST" and cmd.surge == 0
    print("ok  lost behaviour")


if __name__ == "__main__":
    test_obb_geometry()
    test_lost_behaviour()
    test_converges()
    print("all passed")
