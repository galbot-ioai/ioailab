"""Proportional navigation agent: drive straight at the goal."""

from __future__ import annotations

import torch

from ioailab.agents.nav.goal import GoalNavAgent


class ProportionalNavAgent(GoalNavAgent):
    """Head directly toward the goal each step.

    The simplest navigation algorithm: the planned target is always the final
    goal, so the base drives straight at it (clamped to the robot's max nav
    speed by the shared follow law) and aligns ``goal_yaw`` on arrival.
    """

    def plan_target_xy(
        self, current_xy: torch.Tensor, env_ids: tuple[int, ...]
    ) -> torch.Tensor:
        del env_ids  # Stateless: every row always heads at the final goal.
        return current_xy.new_tensor(self._goal_xy).reshape(1, 2).expand_as(current_xy)
