"""Goal-seeking navigation policy shared by the concrete nav algorithms."""

from __future__ import annotations

from abc import abstractmethod

import torch

from ioailab.agents.nav import _chassis
from ioailab.agents.nav.base import BaseNavAgent
from ioailab.agents.robot_profile import RobotProfile

_MAX_YAW_SPEED = 1.0
"""Fixed in-place rotation rate (rad/s); not a per-agent knob."""


class GoalNavAgent(BaseNavAgent):
    """Drive to a goal pose: follow a planned target, then align the goal yaw.

    Holds the goal and arrival/approach tuning and implements the shared follow
    law (steer toward the planned target while sizing the speed off distance to
    the final goal, clamped to the robot's max nav speed), the yaw alignment, and
    arrival detection. Subclasses implement :meth:`plan_target_xy` -- the
    navigation algorithm that decides where to head each step.
    """

    def __init__(
        self,
        *,
        robot: RobotProfile,
        goal_xy: tuple[float, float],
        goal_yaw: float | None = None,
        success_radius: float | None = None,
        yaw_tolerance: float = 0.15,
        rotate_before_translate: bool = False,
    ) -> None:
        super().__init__(robot=robot)
        self._goal_xy = goal_xy
        self._goal_yaw = goal_yaw
        self._max_speed = robot.default_max_nav_speed
        self._success_radius = (
            success_radius
            if success_radius is not None
            else robot.default_nav_success_radius
        )
        self._yaw_tolerance = yaw_tolerance
        self._rotate_before_translate = bool(rotate_before_translate)

    @property
    def goal_xy(self) -> tuple[float, float]:
        return self._goal_xy

    @property
    def rotate_before_translate(self) -> bool:
        """Whether yaw alignment is completed before XY translation."""

        return self._rotate_before_translate

    @abstractmethod
    def plan_target_xy(
        self, current_xy: torch.Tensor, env_ids: tuple[int, ...]
    ) -> torch.Tensor:
        """Return the env-local XY point each row should drive toward this step.

        This is the navigation algorithm. ``current_xy`` is ``(num_envs, 2)`` in
        the env-local frame and the return value must match its shape. ``env_ids``
        are the rows being driven this step; planners with per-row state should
        plan and advance only those rows (other rows are sliced out downstream).
        """

    def _navigate(
        self,
        current_xy: torch.Tensor,
        current_yaw: torch.Tensor,
        env_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_xy = self.plan_target_xy(current_xy, env_ids)
        goal_distance = self._goal_distance(current_xy)
        vx, vy = self._velocity_toward(
            target_xy - current_xy, current_yaw, goal_distance
        )

        xy_reached = goal_distance <= self._success_radius
        vx[xy_reached] = 0.0
        vy[xy_reached] = 0.0
        wz = torch.zeros_like(current_yaw)
        if self._goal_yaw is None:
            return vx, vy, wz, xy_reached

        yaw_error = _chassis.wrap_angle(self._goal_yaw - current_yaw)
        yaw_done = yaw_error.abs() <= self._yaw_tolerance
        wz = torch.clamp(yaw_error * 2.0, min=-_MAX_YAW_SPEED, max=_MAX_YAW_SPEED)
        if self._rotate_before_translate:
            # Hold position until the goal heading is reached.
            vx[~yaw_done] = 0.0
            vy[~yaw_done] = 0.0
        else:
            # Rotate in place only after arriving at the XY goal.
            wz[~xy_reached] = 0.0
        wz[yaw_done] = 0.0
        return vx, vy, wz, torch.logical_and(xy_reached, yaw_done)

    def _velocity_toward(
        self,
        delta_xy_w: torch.Tensor,
        current_yaw: torch.Tensor,
        goal_distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steer toward the target, sizing the speed off distance-to-goal.

        The heading comes from the planned target (e.g. the next waypoint) while
        the speed magnitude is governed by the distance to the *final* goal. This
        lets the base cruise at ``max_speed`` between dense intermediate
        waypoints -- which are collinear with the goal and so only steer, never
        throttle -- and decelerate just once inside ``max_speed`` metres of the
        goal, instead of stopping and re-accelerating at every waypoint.
        """

        delta_xy_b = _chassis.world_to_base_frame(delta_xy_w, current_yaw)
        norm = torch.linalg.vector_norm(delta_xy_b, dim=1).clamp_min(1.0e-6)
        direction = delta_xy_b / norm.unsqueeze(1)
        speed = goal_distance.clamp(max=self._max_speed)
        command_xy = direction * speed.unsqueeze(1)
        return command_xy[:, 0], command_xy[:, 1]

    def _goal_distance(self, current_xy: torch.Tensor) -> torch.Tensor:
        """Return each env row's Euclidean distance to the goal XY."""

        goal_xy = current_xy.new_tensor(self._goal_xy).reshape(1, 2)
        return torch.linalg.vector_norm(goal_xy - current_xy, dim=1)

    def _goal_reached(self, current_xy: torch.Tensor) -> torch.Tensor:
        """Return whether each env row is within ``success_radius`` of the goal."""

        return self._goal_distance(current_xy) <= self._success_radius
