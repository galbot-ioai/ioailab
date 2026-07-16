"""Shared robot capability and runtime interface contracts."""

from __future__ import annotations

from ioailab.robots.common.actions.core import BaseActions
from ioailab.robots.common.articulation import BaseArticulation
from ioailab.robots.common.assemble import BaseRobot
from ioailab.robots.common.interfaces import (
    CameraSensor,
    RobotHandle,
    RuntimeScene,
    TaskActionBuilder,
)
from ioailab.robots.common.sensors.core import BaseSensors

__all__ = [
    "BaseActions",
    "BaseArticulation",
    "BaseRobot",
    "BaseSensors",
    "CameraSensor",
    "RobotHandle",
    "RuntimeScene",
    "TaskActionBuilder",
]
