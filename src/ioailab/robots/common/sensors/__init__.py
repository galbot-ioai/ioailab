"""Robot-generic IsaacLab sensor config helpers."""

from ioailab.robots.common.sensors.core import BaseSensors
from ioailab.robots.common.sensors.camera import (
    CameraMountSpec,
    add_camera_cfg,
    camera_prim_path,
    make_camera_cfg,
)

__all__ = [
    "BaseSensors",
    "CameraMountSpec",
    "add_camera_cfg",
    "camera_prim_path",
    "make_camera_cfg",
]
