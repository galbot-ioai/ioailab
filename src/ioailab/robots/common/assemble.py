"""Base robot assembly object."""

from __future__ import annotations

from dataclasses import dataclass

from ioailab.robots.common.actions.core import BaseActions
from ioailab.robots.common.articulation import BaseArticulation
from ioailab.robots.common.sensors.core import BaseSensors


@dataclass(frozen=True, slots=True)
class BaseRobot:
    """Reviewable robot capability map.

    Attributes:
        name: Stable robot identifier.
        articulation: Articulation cfg and asset facts.
        actions: Static action cfg factories and runtime tensor helpers.
        sensors: Robot-mounted sensor cfg factories.
        display_name: Optional human-readable robot name.
    """

    name: str
    articulation: BaseArticulation
    actions: BaseActions
    sensors: BaseSensors
    display_name: str | None = None

    @property
    def capability_names(self) -> tuple[str, ...]:
        """Return the component names exposed by this robot definition."""

        return ("articulation", "actions", "sensors")
