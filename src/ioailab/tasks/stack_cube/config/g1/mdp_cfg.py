"""Galbot G1 binding for the stack-cube MDP.

The action group names G1 joints, so it lives here with the G1 config; the
assembled ``StackCubeMdpCfg`` composes it with the robot-agnostic observation,
reward, termination, and event groups imported up from ``stack_cube/mdp/``.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.actions import g1_action_cfg
from ioailab.tasks.stack_cube.mdp.events import StackCubeEventCfg
from ioailab.tasks.stack_cube.mdp.observations import StackCubeObservationsCfg
from ioailab.tasks.stack_cube.mdp.rewards import StackCubeRewardsCfg
from ioailab.tasks.stack_cube.mdp.terminations import StackCubeTerminationsCfg


@configclass
class StackCubeActionsCfg:
    """Action specifications for the canonical G1 stack-cube MDP."""

    arm_action = g1_action_cfg("left_arm", "absolute")
    gripper_action = g1_action_cfg("left_gripper", "absolute")


@configclass
class StackCubeMdpCfg:
    """Top-level IsaacLab manager configs for the stack-cube MDP."""

    observations: StackCubeObservationsCfg = StackCubeObservationsCfg()
    actions: StackCubeActionsCfg = StackCubeActionsCfg()
    rewards: StackCubeRewardsCfg = StackCubeRewardsCfg()
    terminations: StackCubeTerminationsCfg = StackCubeTerminationsCfg()
    events: StackCubeEventCfg = StackCubeEventCfg()
    commands = None
    curriculum = None


__all__ = [
    "StackCubeActionsCfg",
    "StackCubeMdpCfg",
]
