"""Galbot G1 sort-to-shelf pick phase task registry."""

from __future__ import annotations

from typing import Final

from ioailab.tasks import TaskSpec

GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID: Final = "GalbotG1-SortToShelf-Pick-v0"
GALBOT_G1_SORT_TO_SHELF_PICK_TASK_IDS: Final = (GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID,)

GALBOT_G1_SORT_TO_SHELF_PICK_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.sort_to_shelf_pick.config.g1.env_cfg:GalbotG1SortToShelfPickEnvCfg",
    },
    motion_plan_entry_point="ioailab.tasks.sort_to_shelf_pick.motion_plan:pick_motion_plan",
    requires_cameras=True,
    reset_randomization_events=("randomize_pick_and_place_positions",),
)

GALBOT_G1_SORT_TO_SHELF_PICK_TASKS: Final = (GALBOT_G1_SORT_TO_SHELF_PICK_TASK,)

__all__ = [
    "GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID",
    "GALBOT_G1_SORT_TO_SHELF_PICK_TASK_IDS",
    "GALBOT_G1_SORT_TO_SHELF_PICK_TASK",
    "GALBOT_G1_SORT_TO_SHELF_PICK_TASKS",
]
