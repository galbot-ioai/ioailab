"""Galbot G1 sort-to-shelf place phase task registry."""

from __future__ import annotations

from typing import Final

from ioailab.tasks import TaskSpec

GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID: Final = "GalbotG1-SortToShelf-Place-v0"
GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_IDS: Final = (GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID,)

GALBOT_G1_SORT_TO_SHELF_PLACE_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.sort_to_shelf_place.config.g1.env_cfg:GalbotG1SortToShelfPlaceEnvCfg",
    },
    motion_plan_entry_point="ioailab.tasks.sort_to_shelf_place.motion_plan:place_motion_plan",
    requires_cameras=True,
)

GALBOT_G1_SORT_TO_SHELF_PLACE_TASKS: Final = (GALBOT_G1_SORT_TO_SHELF_PLACE_TASK,)

__all__ = [
    "GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID",
    "GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_IDS",
    "GALBOT_G1_SORT_TO_SHELF_PLACE_TASK",
    "GALBOT_G1_SORT_TO_SHELF_PLACE_TASKS",
]
