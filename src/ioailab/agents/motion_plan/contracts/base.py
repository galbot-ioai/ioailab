"""Robot contract protocol for motion-planning agents."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ioailab.agents.motion_plan.solvers.types import SolverRequest, WaypointPlan


@runtime_checkable
class MotionPlanningContract(Protocol):
    """Bridge robot/task facts to solver requests and IsaacLab action tensors."""

    robot_name: str

    def build_solver_request(
        self,
        env: Any,
        commands: tuple[Any, ...],
        *,
        layout: Any,
        action_tensor: Any,
    ) -> SolverRequest:
        """Return a backend-ready solver request for recorded task commands."""

    def frames_from_waypoint_plan(
        self,
        request: SolverRequest,
        plan: WaypointPlan,
        *,
        layout: Any,
        action_tensor: Any,
        commands: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Convert solver waypoints into executable action frames."""

    def write_frame_action(
        self, env: Any, *, layout: Any, action_tensor: Any, frame: Any
    ) -> None:
        """Write one executable frame into a full IsaacLab action tensor."""


__all__ = ["MotionPlanningContract"]
