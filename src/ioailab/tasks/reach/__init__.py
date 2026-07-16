"""Galbot G1 reach task registry."""

from __future__ import annotations

from typing import Final

from ioailab.tasks import TaskSpec

GALBOT_G1_REACH_TASK_ID: Final = "GalbotG1-Reach-v0"

GALBOT_G1_REACH_TASK_IDS: Final = (GALBOT_G1_REACH_TASK_ID,)


GALBOT_G1_REACH_TASK: Final = TaskSpec(
    task_id=GALBOT_G1_REACH_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": "ioailab.tasks.reach.config.g1.env_cfg:GalbotG1ReachEnvCfg",
        "rsl_rl_cfg_entry_point": "ioailab.tasks.reach.config.g1.agent_cfg.rsl_rl_ppo_cfg:GalbotG1ReachPPORunnerCfg",
    },
    motion_plan_entry_point="ioailab.tasks.reach.config.g1.agent_cfg.motion_plan:ReachMotionPlan",
)

GALBOT_G1_REACH_TASKS: Final = (GALBOT_G1_REACH_TASK,)


__all__ = [
    "GALBOT_G1_REACH_TASK_ID",
    "GALBOT_G1_REACH_TASK_IDS",
    "GALBOT_G1_REACH_TASK",
    "GALBOT_G1_REACH_TASKS",
]
