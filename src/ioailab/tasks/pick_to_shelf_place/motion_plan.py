"""Motion-planning config and plan factory for the PickToShelf place task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ioailab.agents.motion_plan.yaml_motion_plan import YamlMotionPlan
from ioailab.tasks.pick_to_shelf_pick.motion_plan import (
    PickToShelfPickMotionPlanningCfg,
)


@dataclass
class PickToShelfPlaceMotionPlanningCfg(PickToShelfPickMotionPlanningCfg):
    """cuRobo motion-planning config for the place phase task."""

    task_id: str = "GalbotG1-PickToShelf-Place-v0"


def place_motion_plan(config: Any | None = None) -> YamlMotionPlan:
    """Return the place phase task's YAML motion plan."""

    return YamlMotionPlan.from_package(
        __package__,
        "motion_plan.yaml",
        config=config if config is not None else PickToShelfPlaceMotionPlanningCfg(),
    )


__all__ = ["PickToShelfPlaceMotionPlanningCfg", "place_motion_plan"]
