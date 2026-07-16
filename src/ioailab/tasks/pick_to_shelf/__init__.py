"""Galbot G1 coherent pick-to-shelf task registry."""

from __future__ import annotations

from typing import Final

from ioailab.tasks import TaskSpec
from ioailab.tasks.pick_to_shelf_nav import GALBOT_G1_PICK_TO_SHELF_NAV_TASK_ID
from ioailab.tasks.pick_to_shelf_pick import GALBOT_G1_PICK_TO_SHELF_PICK_TASK_ID
from ioailab.tasks.pick_to_shelf_place import GALBOT_G1_PICK_TO_SHELF_PLACE_TASK_ID

GALBOT_G1_PICK_TO_SHELF_TASK_ID: Final = "GalbotG1-PickToShelf-v0"
GALBOT_G1_PICK_TO_SHELF_TASK_IDS: Final = (
    GALBOT_G1_PICK_TO_SHELF_TASK_ID,
    GALBOT_G1_PICK_TO_SHELF_PICK_TASK_ID,
    GALBOT_G1_PICK_TO_SHELF_NAV_TASK_ID,
    GALBOT_G1_PICK_TO_SHELF_PLACE_TASK_ID,
)


GALBOT_G1_PICK_TO_SHELF_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_PICK_TO_SHELF_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.pick_to_shelf.config.g1.env_cfg:GalbotG1PickToShelfEnvCfg",
    },
    requires_cameras=True,
    reset_randomization_events=("randomize_pick_and_place_positions",),
    task_flow_entry_point=(
        "ioailab.tasks.pick_to_shelf.config.g1.env_cfg:"
        "GalbotG1PickToShelfEnvCfg.task_flow"
    ),
)

GALBOT_G1_PICK_TO_SHELF_TASKS: Final = (GALBOT_G1_PICK_TO_SHELF_TASK,)


__all__ = [
    "GALBOT_G1_PICK_TO_SHELF_TASK_ID",
    "GALBOT_G1_PICK_TO_SHELF_TASK_IDS",
    "GALBOT_G1_PICK_TO_SHELF_TASK",
    "GALBOT_G1_PICK_TO_SHELF_TASKS",
]
