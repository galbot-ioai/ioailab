"""Base robot action capability object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseActions(ABC):
    """Base action capability for robot action cfgs and tensor helpers."""

    @abstractmethod
    def action_cfg(self, *parts: Any, **kwargs: Any) -> Any:
        """Return an IsaacLab action cfg object for selected robot parts."""
