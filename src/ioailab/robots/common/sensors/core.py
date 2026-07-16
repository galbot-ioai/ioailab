"""Base robot sensor capability object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSensors(ABC):
    """Base sensor capability for robot-mounted sensor cfgs."""

    @property
    @abstractmethod
    def mount_names(self) -> tuple[str, ...]:
        """Return public robot-mounted sensor names."""

    @abstractmethod
    def camera(self, mount: str) -> Any:
        """Return an IsaacLab camera cfg for one named robot-mounted camera."""
