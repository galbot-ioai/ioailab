"""Motion-planning config and plan factory for the PickToShelf pick task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ioailab.agents.motion_plan.yaml_motion_plan import YamlMotionPlan


@dataclass
class PickToShelfPickMotionPlanningCfg:
    """cuRobo motion-planning config for the pick phase task."""

    task_id: str = "GalbotG1-PickToShelf-Pick-v0"
    planner: str = "curobov2"
    robot_asset_name: str = "robot"
    cube_asset_name: str = "cube"
    shelf_deck_asset_name: str = "shelf_deck"
    ready_hold_frames: int = 60
    target_settle_steps: int = 12
    post_plan_hold_seconds: float = 10.0
    max_joint_step: float = 0.04
    position_tolerance: float = 0.035
    orientation_tolerance: float = 0.15


def pick_motion_plan(config: Any | None = None) -> YamlMotionPlan:
    """Return the pick phase task's YAML motion plan."""

    return YamlMotionPlan.from_package(
        __package__,
        "motion_plan.yaml",
        config=config if config is not None else PickToShelfPickMotionPlanningCfg(),
    )


__all__ = ["PickToShelfPickMotionPlanningCfg", "pick_motion_plan"]
