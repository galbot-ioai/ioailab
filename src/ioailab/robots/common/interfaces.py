"""Runtime shape protocols for robot control and sensors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TaskActionBuilder(Protocol):
    """Build one task action accepted by a live ioailab env step."""

    def joint_position(self, targets: Mapping[str, float]) -> "TaskActionBuilder":
        """Select joint-position targets for the next action."""

    def build(self) -> Any:
        """Return the full task action payload for ``env.step(action)``."""


@runtime_checkable
class CameraSensor(Protocol):
    """Camera sensor view exposed by a runtime robot handle."""

    def read_rgb(self) -> Any:
        """Read an RGB frame batch from the live runtime sensor."""

    def read_rgb_depth(self) -> tuple[Any, Any]:
        """Read RGB and depth frames from the live runtime sensor."""


@runtime_checkable
class RobotHandle(Protocol):
    """Robot control surface exposed from a live ioailab scene."""

    def action_builder(self) -> TaskActionBuilder:
        """Return an action builder for the current task action layout."""

    def get_sensor(self, name: str) -> CameraSensor:
        """Return a named robot-mounted sensor."""


@runtime_checkable
class RuntimeScene(Protocol):
    """Scene view exposed by a live direct env."""

    @property
    def robot(self) -> RobotHandle:
        """Return the primary robot handle for this scene."""
