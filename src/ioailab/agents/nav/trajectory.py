"""Trajectory-planning navigation agent for mobile-base tasks."""

from __future__ import annotations

import math
from typing import Any

import torch

from ioailab.agents.nav import _chassis
from ioailab.agents.nav.goal import GoalNavAgent
from ioailab.agents.robot_profile import RobotProfile


class TrajectoryNavAgent(GoalNavAgent):
    """Plan a straight-line XY trajectory and follow it waypoint by waypoint.

    The navigation algorithm lives in :meth:`plan_target_xy`: on first sight of
    a row it plans evenly spaced waypoints from the current pose to the goal,
    then returns the next unreached waypoint each step. The shared base loop
    turns that target into base twist commands and aligns the final yaw.
    """

    @classmethod
    def from_task(cls, task_id: str, **overrides: Any) -> "TrajectoryNavAgent":
        """Create a trajectory nav agent from task-local metadata."""

        from ioailab import tasks

        if "subtask" in overrides:
            raise TypeError(
                "TrajectoryNavAgent.from_task(...) no longer accepts subtask=; "
                "use the phase task ID directly."
            )
        return tasks.nav_agent_for_task(
            str(task_id),
            agent_cls=cls,
            **overrides,
        )

    def __init__(
        self,
        *,
        robot: RobotProfile,
        goal_xy: tuple[float, float],
        goal_yaw: float | None = None,
        success_radius: float | None = None,
        yaw_tolerance: float = 0.15,
        rotate_before_translate: bool = False,
        waypoint_spacing: float = 0.25,
        waypoint_tolerance: float | None = None,
    ) -> None:
        super().__init__(
            robot=robot,
            goal_xy=goal_xy,
            goal_yaw=goal_yaw,
            success_radius=success_radius,
            yaw_tolerance=yaw_tolerance,
            rotate_before_translate=rotate_before_translate,
        )
        if waypoint_spacing <= 0.0:
            raise ValueError("waypoint_spacing must be greater than zero.")
        self._waypoint_spacing = float(waypoint_spacing)
        self._waypoint_tolerance = (
            float(waypoint_tolerance)
            if waypoint_tolerance is not None
            else min(float(self._success_radius), float(waypoint_spacing))
        )
        self._trajectories: list[torch.Tensor | None] = []
        self._waypoint_indices: list[int] = []

    def reset(self, env, env_ids=None) -> None:
        """Reset completion state and discard stale row trajectories."""

        super().reset(env, env_ids)
        self._ensure_trajectory_state(env)
        ids = (
            range(len(self._trajectories))
            if env_ids is None
            else (int(env_id) for env_id in env_ids)
        )
        for env_id in ids:
            self._trajectories[env_id] = None
            self._waypoint_indices[env_id] = 0

    def plan_target_xy(
        self, current_xy: torch.Tensor, env_ids: tuple[int, ...]
    ) -> torch.Tensor:
        """Plan/advance the driven rows, then return the active target per row.

        Only ``env_ids`` are planned and advanced so row-scoped task flows do
        not disturb rows that are not currently navigating; other rows still get a
        target (their current waypoint or the goal), which is sliced out upstream.
        """

        self._ensure_trajectory_state_for_rows(current_xy.shape[0])
        for env_id in env_ids:
            if self._trajectories[env_id] is None:
                self._trajectories[env_id] = self._plan_xy_trajectory(
                    current_xy[env_id]
                )
                self._waypoint_indices[env_id] = 0
            self._advance_reached_waypoints(env_id, current_xy[env_id])
        targets = [
            self._active_waypoint(env_id, current_xy)
            for env_id in range(current_xy.shape[0])
        ]
        return torch.stack(targets, dim=0)

    def _plan_xy_trajectory(self, start_xy: torch.Tensor) -> torch.Tensor:
        """Return evenly spaced waypoints from ``start_xy`` to ``goal_xy``."""

        goal_xy = start_xy.new_tensor(self._goal_xy)
        delta = goal_xy - start_xy
        distance = float(torch.linalg.vector_norm(delta).item())
        if distance <= self._waypoint_tolerance:
            return goal_xy.reshape(1, 2)
        segment_count = max(1, int(math.ceil(distance / self._waypoint_spacing)))
        fractions = torch.linspace(
            1.0 / segment_count,
            1.0,
            segment_count,
            device=start_xy.device,
            dtype=start_xy.dtype,
        ).unsqueeze(1)
        return start_xy.reshape(1, 2) + fractions * delta.reshape(1, 2)

    def _active_waypoint(self, env_id: int, current_xy: torch.Tensor) -> torch.Tensor:
        """Return the active waypoint for one row, falling back to the goal."""

        trajectory = self._trajectories[env_id]
        if trajectory is None or trajectory.numel() == 0:
            return current_xy.new_tensor(self._goal_xy)
        index = min(self._waypoint_indices[env_id], trajectory.shape[0] - 1)
        return trajectory[index]

    def _advance_reached_waypoints(
        self, env_id: int, current_xy_row: torch.Tensor
    ) -> None:
        """Advance one row's waypoint index while waypoints are reached."""

        trajectory = self._trajectories[env_id]
        if trajectory is None:
            return
        index = self._waypoint_indices[env_id]
        while index < trajectory.shape[0] - 1:
            distance = torch.linalg.vector_norm(trajectory[index] - current_xy_row)
            if float(distance.item()) > self._waypoint_tolerance:
                break
            index += 1
        self._waypoint_indices[env_id] = index

    def _ensure_trajectory_state(self, env) -> None:
        self._ensure_trajectory_state_for_rows(
            int(getattr(_chassis.unwrapped(env), "num_envs"))
        )

    def _ensure_trajectory_state_for_rows(self, num_envs: int) -> None:
        if len(self._trajectories) == num_envs:
            return
        self._trajectories = [None for _ in range(num_envs)]
        self._waypoint_indices = [0 for _ in range(num_envs)]
