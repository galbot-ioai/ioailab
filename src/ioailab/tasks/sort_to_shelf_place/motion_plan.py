"""Motion-planning config and plan factory for the SortToShelf place phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ioailab.agents.motion_plan.yaml_motion_plan import YamlMotionPlan
from ioailab.tasks.sort_to_shelf.scene import (
    SORTING_SHELF_A_CELLS,
    sorting_object_name,
    sorting_place_board_asset_name_for_object,
    sorting_place_target_offset_from_board_for_object,
    sorting_target_cell_for_object,
)
from ioailab.tasks.sort_to_shelf_pick.motion_plan import (
    SortToShelfMotionPlanningCfg,
)
from ioailab.tasks.sort_to_shelf_place import GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID

_SORTING_PLACE_QUAT_XYZW = (0.70710678, -0.70710678, 0.0, 0.0)
_SORTING_PLACE_APPROACH_OFFSET = (0.0, 0.24, 0.07)
_SORTING_PLACE_INSERT_OFFSET = (0.0, 0.0, 0.07)
_SORTING_PLACE_OFFSET = (0.0, 0.0, 0.0)
_SORTING_LOWER_ROW_PLACE_Z_LIFT = 0.06
_SORTING_UPPER_ROW_PLACE_Z_LOWERING = 0.18


@dataclass
class SortToShelfPlaceMotionPlanningCfg(SortToShelfMotionPlanningCfg):
    """cuRobo motion-planning config for the SortToShelf place phase."""

    task_id: str = GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID

    @property
    def place_object_name(self) -> str:
        """Return the selected SortToShelf object asset name."""

        return sorting_object_name(self.sorting_object)

    @property
    def place_target_cell(self) -> str:
        """Return the shelf cell assigned to the selected object."""

        return sorting_target_cell_for_object(self.place_object_name)

    @property
    def place_board_asset_name(self) -> str:
        """Return the selected object's place-board asset name."""

        return sorting_place_board_asset_name_for_object(self.place_object_name)

    @property
    def place_center_offset(self) -> tuple[float, float, float]:
        """Return the object-center offset from the selected place board."""

        board_to_center = sorting_place_target_offset_from_board_for_object(
            self.place_object_name
        )
        if self.place_target_cell in SORTING_SHELF_A_CELLS:
            z_adjust = -_SORTING_UPPER_ROW_PLACE_Z_LOWERING
        else:
            z_adjust = _SORTING_LOWER_ROW_PLACE_Z_LIFT
        return (
            board_to_center[0],
            board_to_center[1],
            board_to_center[2] + z_adjust,
        )

    @property
    def place_approach_offset(self) -> tuple[float, float, float]:
        """Return the pre-approach and retreat offset for the selected cell."""

        return _summed_offset(self.place_center_offset, _SORTING_PLACE_APPROACH_OFFSET)

    @property
    def place_insert_offset(self) -> tuple[float, float, float]:
        """Return the insertion offset for the selected cell."""

        return _summed_offset(self.place_center_offset, _SORTING_PLACE_INSERT_OFFSET)

    @property
    def place_release_offset(self) -> tuple[float, float, float]:
        """Return the final place/release offset for the selected cell."""

        return _summed_offset(self.place_center_offset, _SORTING_PLACE_OFFSET)

    @property
    def place_approach_step_name(self) -> str:
        """Return the selected-cell approach step name."""

        return f"approach_{self.place_target_cell}"

    @property
    def place_insert_step_name(self) -> str:
        """Return the selected-cell insert step name."""

        return f"insert_to_{self.place_target_cell}"

    @property
    def place_descend_step_name(self) -> str:
        """Return the selected-cell descend step name."""

        return f"descend_to_{self.place_target_cell}"

    @property
    def place_release_step_name(self) -> str:
        """Return the selected-cell release step name."""

        return f"release_on_{self.place_target_cell}"

    @property
    def place_retreat_step_name(self) -> str:
        """Return the selected-cell retreat step name."""

        return f"retreat_from_{self.place_target_cell}"

    @property
    def place_retract_step_name(self) -> str:
        """Return the selected-cell retract step name."""

        return f"retract_left_arm_from_{self.place_target_cell}"


def _summed_offset(
    base: tuple[float, float, float], extra: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (base[0] + extra[0], base[1] + extra[1], base[2] + extra[2])


def place_motion_plan(config: Any | None = None) -> YamlMotionPlan:
    """Return the sort-to-shelf place YAML motion plan."""

    return YamlMotionPlan.from_package(
        __package__,
        "motion_plan.yaml",
        config=config if config is not None else SortToShelfPlaceMotionPlanningCfg(),
    )


__all__ = [
    "SortToShelfPlaceMotionPlanningCfg",
    "place_motion_plan",
]
