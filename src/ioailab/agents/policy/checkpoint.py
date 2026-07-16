"""Policy checkpoint references."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicyCheckpoint:
    """Reference to a trained policy checkpoint."""

    path: Path | str
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize checkpoint fields."""

        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "metadata", dict(self.metadata))
