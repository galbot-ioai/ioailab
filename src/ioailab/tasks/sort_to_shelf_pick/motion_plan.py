"""Motion-planning config and plan factory for the SortToShelf pick phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ioailab.agents.motion_plan.yaml_motion_plan import YamlMotionPlan
from ioailab.tasks.sort_to_shelf.scene import sorting_object_name
from ioailab.tasks.sort_to_shelf_pick import GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID
from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
    G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
)

_SORTING_PICK_QUAT_XYZW = (1.0, 0.0, 0.0, 0.0)
_SORTING_PICK_APPROACH_OFFSET = (-0.12, 0.0, 0.025)
_SORTING_PICK_GRASP_OFFSET = (-0.01, 0.0, 0.025)
_SORTING_PICK_LIFT_OFFSET = (0.0, 0.0, 0.22)


@dataclass
class SortToShelfMotionPlanningCfg:
    """cuRobo motion-planning config shared by SortToShelf phases."""

    task_id: str = GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID
    planner: str = "curobov2"
    robot_asset_name: str = "robot"
    object_asset_name: str = "red_cube"
    shelf_deck_asset_name: str = "shelf_deck"
    ready_hold_frames: int = 60
    target_settle_steps: int = 6
    post_plan_hold_seconds: float = 10.0
    max_joint_step: float = 0.04
    position_tolerance: float = 0.035
    orientation_tolerance: float = 0.15
    sorting_object: str = "red_cube"

    @property
    def selected_sorting_object(self) -> str:
        """Return the selected SortToShelf object asset name."""

        return sorting_object_name(self.sorting_object)

    @property
    def left_arm_ready_joint_positions(self) -> dict[str, float]:
        """Return the ready carry posture for YAML motion plans."""

        return dict(G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS)

    @property
    def pick_approach_step_name(self) -> str:
        """Return the selected-object approach step name."""

        return f"approach_{self.selected_sorting_object}"

    @property
    def pick_descend_step_name(self) -> str:
        """Return the selected-object descend step name."""

        return f"descend_to_{self.selected_sorting_object}"

    @property
    def pick_lift_step_name(self) -> str:
        """Return the selected-object lift step name."""

        return f"lift_{self.selected_sorting_object}"

    @property
    def pick_carry_step_name(self) -> str:
        """Return the selected-object carry step name."""

        return f"carry_{self.selected_sorting_object}"

    def apply_task_options(self, task_options: dict[str, object]) -> None:
        """Apply the selected sorting object from planner task options."""

        options = dict(task_options)
        unknown = tuple(sorted(set(options) - {"sorting_object"}))
        if unknown:
            raise ValueError(
                f"Unknown sorting motion option(s): {unknown}. "
                "Allowed options: ('sorting_object',)."
            )
        object_name = sorting_object_name(
            options.get("sorting_object", self.sorting_object)
        )
        self.sorting_object = object_name
        self.object_asset_name = object_name


@dataclass
class SortToShelfPickMotionPlanningCfg(SortToShelfMotionPlanningCfg):
    """cuRobo motion-planning config for the SortToShelf pick phase."""


def pick_motion_plan(config: Any | None = None) -> YamlMotionPlan:
    """Return the sort-to-shelf pick YAML motion plan."""

    return YamlMotionPlan.from_package(
        __package__,
        "motion_plan.yaml",
        config=config if config is not None else SortToShelfPickMotionPlanningCfg(),
    )


__all__ = [
    "SortToShelfMotionPlanningCfg",
    "SortToShelfPickMotionPlanningCfg",
    "pick_motion_plan",
]
