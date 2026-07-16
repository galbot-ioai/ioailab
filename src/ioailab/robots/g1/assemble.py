"""Assemble the canonical Galbot G1 robot object."""

from __future__ import annotations

from ioailab.robots.common import BaseRobot
from ioailab.robots.g1.actions.core import G1Actions
from ioailab.robots.g1.articulation import DISPLAY_NAME, ROBOT_NAME, G1Articulation
from ioailab.robots.g1.sensors.core import G1Sensors


class G1Robot(BaseRobot):
    """Galbot G1 robot capability map."""

    def __init__(self) -> None:
        """Initialize the reviewable G1 component map."""

        super().__init__(
            name=ROBOT_NAME,
            display_name=DISPLAY_NAME,
            articulation=G1Articulation(),
            actions=G1Actions(),
            sensors=G1Sensors(),
        )


g1 = G1Robot()
"""Canonical Galbot G1 robot capability object."""

# Public uppercase robot object alias.
G1 = g1
