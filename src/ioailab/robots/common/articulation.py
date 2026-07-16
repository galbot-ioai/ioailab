"""Base robot articulation capability object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseArticulation(ABC):
    """Base articulation capability for a robot asset."""

    @abstractmethod
    def cfg(self, **kwargs: Any) -> Any:
        """Return a default IsaacLab articulation cfg for this robot."""
