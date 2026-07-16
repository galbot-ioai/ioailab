"""Galbot G1 base navigation task registry."""

from __future__ import annotations

from typing import Final

from ioailab.tasks import TaskSpec

GALBOT_G1_BASE_NAV_TASK_ID: Final = "GalbotG1-BaseNav-v0"
GALBOT_G1_BASE_NAV_TASK_IDS: Final = (GALBOT_G1_BASE_NAV_TASK_ID,)

GALBOT_G1_BASE_NAV_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_BASE_NAV_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.base_nav.config.g1.env_cfg:GalbotG1BaseNavEnvCfg",
    },
)

GALBOT_G1_BASE_NAV_TASKS: Final = (GALBOT_G1_BASE_NAV_TASK,)


__all__ = [
    "GALBOT_G1_BASE_NAV_TASK_ID",
    "GALBOT_G1_BASE_NAV_TASK_IDS",
    "GALBOT_G1_BASE_NAV_TASK",
    "GALBOT_G1_BASE_NAV_TASKS",
]
