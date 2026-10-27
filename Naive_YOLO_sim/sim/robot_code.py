"""Loads the REAL robot modules (Naive_YOLO/controller.py and detection.py) without modifying or shadowing them.

They are loaded by file path under unique module names, so the other files in Naive_YOLO (log.py, vehicle.py, ...)
can never clash with names used here. Whatever you change in those two files is what the simulator runs.
"""
import importlib.util
import sys
from pathlib import Path

ROBOT_DIR = Path(__file__).resolve().parent.parent.parent / "Naive_YOLO"


def _load(name, filename):
    path = ROBOT_DIR / filename
    spec = importlib.util.spec_from_file_location(f"naive_yolo_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # needed so @dataclass can resolve the module
    spec.loader.exec_module(mod)
    return mod


controller = _load("controller", "controller.py")
detection = _load("detection", "detection.py")  # picamera2 is only imported inside main(), so this is safe off the Pi
