"""Galbot G1 sort-to-shelf navigation phase task registry."""

from __future__ import annotations

from typing import Final

from ioailab.tasks import TaskSpec

GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID: Final = "GalbotG1-SortToShelf-Nav-v0"
GALBOT_G1_SORT_TO_SHELF_NAV_TASK_IDS: Final = (GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID,)

GALBOT_G1_SORT_TO_SHELF_NAV_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.sort_to_shelf_nav.config.g1.env_cfg:GalbotG1SortToShelfNavEnvCfg",
    },
    nav_agent_entry_point="ioailab.tasks.sort_to_shelf_nav.agent:nav_agent",
    requires_cameras=True,
)

GALBOT_G1_SORT_TO_SHELF_NAV_TASKS: Final = (GALBOT_G1_SORT_TO_SHELF_NAV_TASK,)

__all__ = [
    "GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID",
    "GALBOT_G1_SORT_TO_SHELF_NAV_TASK_IDS",
    "GALBOT_G1_SORT_TO_SHELF_NAV_TASK",
    "GALBOT_G1_SORT_TO_SHELF_NAV_TASKS",
]
