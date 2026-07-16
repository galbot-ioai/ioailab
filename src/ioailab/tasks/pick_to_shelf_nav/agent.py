"""Navigation-agent config and factory for the PickToShelf nav phase task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ioailab.agents import TrajectoryNavAgent
from ioailab.agents.robot_profile import RobotProfile
from ioailab.robots.g1.profile import G1_PROFILE
from ioailab.tasks.pick_to_shelf_nav.mdp.goals import SHELF_NAV_XY, SHELF_NAV_YAW


@dataclass
class PickToShelfNavAgentCfg:
    """Trajectory-nav config for the nav phase task."""

    task_id: str = "GalbotG1-PickToShelf-Nav-v0"
    robot: RobotProfile = G1_PROFILE
    goal_xy: tuple[float, float] = SHELF_NAV_XY
    goal_yaw: float = float(SHELF_NAV_YAW)
    success_radius: float = 0.02
    yaw_tolerance: float = 0.03
    rotate_before_translate: bool = True
    waypoint_spacing: float = 10.0


def nav_agent(
    config: Any | None = None,
    *,
    agent_cls: type[TrajectoryNavAgent] = TrajectoryNavAgent,
    **overrides: Any,
) -> TrajectoryNavAgent:
    """Return the nav phase trajectory agent with its bundled config."""

    cfg = config if config is not None else PickToShelfNavAgentCfg()
    kwargs = {
        "robot": cfg.robot,
        "goal_xy": cfg.goal_xy,
        "goal_yaw": cfg.goal_yaw,
        "success_radius": cfg.success_radius,
        "yaw_tolerance": cfg.yaw_tolerance,
        "rotate_before_translate": cfg.rotate_before_translate,
        "waypoint_spacing": cfg.waypoint_spacing,
    }
    kwargs.update(overrides)
    return agent_cls(**kwargs)


__all__ = ["PickToShelfNavAgentCfg", "nav_agent"]
