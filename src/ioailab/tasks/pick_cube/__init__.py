"""Galbot G1 pick-cube task registry."""

from __future__ import annotations

from typing import Final

from ioailab.tasks import TaskSpec

GALBOT_G1_PICK_CUBE_TASK_ID: Final = "GalbotG1-PickCube-v0"
GALBOT_G1_PICK_CUBE_TELEOP_TASK_ID: Final = "GalbotG1-PickCube-Teleop-v0"
GALBOT_G1_PICK_CUBE_MIMIC_TASK_ID: Final = "GalbotG1-PickCube-Mimic-v0"

GALBOT_G1_PICK_CUBE_TASK_IDS: Final = (
    GALBOT_G1_PICK_CUBE_TASK_ID,
    GALBOT_G1_PICK_CUBE_TELEOP_TASK_ID,
    GALBOT_G1_PICK_CUBE_MIMIC_TASK_ID,
)


GALBOT_G1_PICK_CUBE_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_PICK_CUBE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.pick_cube.config.g1.env_cfg:GalbotG1PickCubeEnvCfg",
    },
    motion_plan_entry_point="ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan:PickCubeMotionPlan",
    requires_cameras=True,
    reset_randomization_events=(
        "randomize_pick_and_place_positions",
        "randomize_ground_material",
        "randomize_table_material",
        "randomize_hdri_texture",
    ),
)

GALBOT_G1_PICK_CUBE_TELEOP_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_PICK_CUBE_TELEOP_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.pick_cube.config.g1.env_cfg:GalbotG1PickCubeTeleopEnvCfg",
    },
    requires_cameras=True,
    reset_randomization_events=(
        "randomize_pick_and_place_positions",
        "randomize_ground_material",
        "randomize_table_material",
        "randomize_hdri_texture",
    ),
)

GALBOT_G1_PICK_CUBE_MIMIC_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_PICK_CUBE_MIMIC_TASK_ID,
    entry_point="ioailab.datasets.mimic.env:ioailabMimicEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.pick_cube.config.g1.env_cfg:GalbotG1PickCubeMimicEnvCfg",
    },
)

GALBOT_G1_PICK_CUBE_TASKS: Final = (
    GALBOT_G1_PICK_CUBE_TASK,
    GALBOT_G1_PICK_CUBE_TELEOP_TASK,
    GALBOT_G1_PICK_CUBE_MIMIC_TASK,
)


__all__ = [
    "GALBOT_G1_PICK_CUBE_TASK_ID",
    "GALBOT_G1_PICK_CUBE_MIMIC_TASK_ID",
    "GALBOT_G1_PICK_CUBE_TELEOP_TASK_ID",
    "GALBOT_G1_PICK_CUBE_TASK_IDS",
    "GALBOT_G1_PICK_CUBE_MIMIC_TASK",
    "GALBOT_G1_PICK_CUBE_TASK",
    "GALBOT_G1_PICK_CUBE_TELEOP_TASK",
    "GALBOT_G1_PICK_CUBE_TASKS",
]
