"""Robot profile abstraction for multi-robot agent support.

A RobotProfile encapsulates all robot-specific facts that agents need to
produce action tensors: joint names, DOF groups, kinematics, packing
functions, and planner specs. Agents accept a RobotProfile at construction
time, making them robot-agnostic in their interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RobotProfile:
    """Robot-specific configuration consumed by all agent types.

    Each supported robot defines a single canonical profile instance. Agents
    receive the profile at construction time and use it to resolve joint names,
    action packing, and planner setup without importing robot-specific modules
    directly.
    """

    name: str

    base_velocity_packer: Callable[..., Any]
    base_body_name: str
    base_wheel_dof_names: tuple[str, ...]

    arm_dof_names: dict[str, tuple[str, ...]]
    gripper_dof_names: dict[str, tuple[str, ...]]

    default_arm: str = "left"
    default_max_nav_speed: float = 0.45
    default_nav_success_radius: float = 0.15

    planner_spec_factory: Callable[..., Any] | None = None

    extra: dict[str, Any] = field(default_factory=dict)
