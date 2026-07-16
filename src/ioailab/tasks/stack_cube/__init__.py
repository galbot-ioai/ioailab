"""Galbot G1 stack-cube IsaacLab task registration."""

from __future__ import annotations

from typing import Final

from ioailab.tasks import TaskSpec

GALBOT_G1_STACK_CUBE_TASK_ID: Final = "GalbotG1-StackCube-v0"

GALBOT_G1_STACK_CUBE_TASK_IDS: Final = (GALBOT_G1_STACK_CUBE_TASK_ID,)


GALBOT_G1_STACK_CUBE_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_STACK_CUBE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.stack_cube.config.g1.env_cfg:GalbotG1StackCubeEnvCfg",
    },
    motion_plan_entry_point="ioailab.tasks.stack_cube.config.g1.agent_cfg.motion_plan:StackCubeMotionPlan",
    reset_randomization_events=(
        "randomize_cube_positions",
        "randomize_ground_material",
        "randomize_table_material",
        "randomize_hdri_texture",
    ),
)

GALBOT_G1_STACK_CUBE_TASKS: Final = (GALBOT_G1_STACK_CUBE_TASK,)


__all__ = [
    "GALBOT_G1_STACK_CUBE_TASK_ID",
    "GALBOT_G1_STACK_CUBE_TASK_IDS",
    "GALBOT_G1_STACK_CUBE_TASK",
    "GALBOT_G1_STACK_CUBE_TASKS",
]
