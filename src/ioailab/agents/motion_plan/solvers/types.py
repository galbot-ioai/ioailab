"""Solver-neutral motion-planning request and result contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SolverRequest:
    """Backend-ready request plus optional solver context."""

    native_request: Any
    context: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize metadata to a plain dict."""

        object.__setattr__(self, "metadata", dict(self.metadata))


WaypointPlan = Any


@runtime_checkable
class MotionSolver(Protocol):
    """Protocol implemented by motion-planning solver backends."""

    def solve(self, request: SolverRequest) -> WaypointPlan:
        """Return a waypoint plan for ``request`` without touching IsaacLab envs."""


__all__ = ["MotionSolver", "SolverRequest", "WaypointPlan"]
