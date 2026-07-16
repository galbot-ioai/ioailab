"""Robot-agnostic MDP terms for the stack-cube task.

The action group and the assembled ``StackCubeMdpCfg`` (which composes G1
actions) live in the G1 binding at ``stack_cube/config/g1/mdp_cfg.py``. The
observation/reward/termination/event groups and predicate functions here are
robot-agnostic -- they read the gripper by name via ``ioailab.tasks.common.mdp``.
"""

from ioailab.tasks.stack_cube.mdp.events import StackCubeEventCfg
from ioailab.tasks.stack_cube.mdp.observations import StackCubeObservationsCfg
from ioailab.tasks.stack_cube.mdp.observations import single_gripper_pos
from ioailab.tasks.stack_cube.mdp.rewards import (
    StackCubeRewardsCfg,
    action_l2_penalty,
    cube_to_stack_alignment_reward,
    cubes_stacked_on_base_cube,
    objects_stacked_on_base,
    stack_success_reward,
)
from ioailab.tasks.stack_cube.mdp.terminations import StackCubeMimicSuccessCfg
from ioailab.tasks.stack_cube.mdp.terminations import StackCubeTerminationsCfg
from ioailab.tasks.stack_cube.mdp.terminations import cube_2_grasped
from ioailab.tasks.stack_cube.mdp.terminations import cube_2_on_cube_1
from ioailab.tasks.stack_cube.mdp.terminations import cube_3_grasped
from ioailab.tasks.stack_cube.mdp.terminations import cube_3_on_cube_2
from ioailab.tasks.stack_cube.mdp.terminations import object_grasped_by_single_gripper
from ioailab.tasks.stack_cube.mdp.terminations import object_stacked_single_gripper
from ioailab.tasks.stack_cube.mdp.terminations import stack_cube_success

__all__ = [
    "StackCubeEventCfg",
    "StackCubeObservationsCfg",
    "StackCubeRewardsCfg",
    "StackCubeMimicSuccessCfg",
    "StackCubeTerminationsCfg",
    "action_l2_penalty",
    "cube_2_grasped",
    "cube_2_on_cube_1",
    "cube_3_grasped",
    "cube_3_on_cube_2",
    "cube_to_stack_alignment_reward",
    "cubes_stacked_on_base_cube",
    "object_grasped_by_single_gripper",
    "object_stacked_single_gripper",
    "objects_stacked_on_base",
    "single_gripper_pos",
    "stack_cube_success",
    "stack_success_reward",
]
