"""Galbot G1 binding for the pick-cube MDP.

The action and observation groups name G1 entities (left arm/gripper joints, the
front-head camera), so they live here in the G1 config rather than the
robot-agnostic ``pick_cube/mdp/`` package. Robot-agnostic event and termination
terms are imported up from ``pick_cube/mdp/``.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.actions import g1_action_cfg
from ioailab.tasks.common.mdp import rgb_image_obs_term
from ioailab.tasks.pick_cube.mdp.events import PickCubeEventCfg
from ioailab.tasks.pick_cube.mdp.terminations import (
    PickCubeMimicSuccessCfg,
    PickCubeTerminationsCfg,
)


@configclass
class PickCubeActionsCfg:
    """Absolute joint actions for motion-planning data collection."""

    arm_action = g1_action_cfg("left_arm", "absolute")
    gripper_action = g1_action_cfg("left_gripper", "absolute")
    leg_action = None


@configclass
class PickCubePolicyObs(ObsGroup):
    """Policy observations with state values useful for planning/debugging."""

    actions = ObsTerm(func=base_mdp.last_action)
    robot_joint_pos = ObsTerm(
        func=base_mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    front_head_rgb = rgb_image_obs_term("front_head_rgb_camera")

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class PickCubeTeleopPolicyObs(PickCubePolicyObs):
    """Low-dimensional pick-cube observations plus raw camera RGB images."""

    left_wrist_rgb = rgb_image_obs_term("left_wrist_rgb_camera")
    front_head_rgb = rgb_image_obs_term("front_head_rgb_camera")


@configclass
class PickCubeObservationsCfg:
    """Observations for motion-planning data collection."""

    policy: PickCubePolicyObs = PickCubePolicyObs()


@configclass
class PickCubeTeleopObservationsCfg:
    """Observations for teleop collection with task-owned RGB cameras."""

    policy: PickCubeTeleopPolicyObs = PickCubeTeleopPolicyObs()


@configclass
class PickCubeMdpCfg:
    """MDP config for motion-planning data collection (no rewards)."""

    observations: PickCubeObservationsCfg = PickCubeObservationsCfg()
    actions: PickCubeActionsCfg = PickCubeActionsCfg()
    terminations: PickCubeTerminationsCfg = PickCubeTerminationsCfg()
    events: PickCubeEventCfg = PickCubeEventCfg()
    commands = None
    rewards = None
    curriculum = None


@configclass
class PickCubeMimicMdpCfg(PickCubeMdpCfg):
    """MDP config for Mimic annotation/generation."""

    terminations: PickCubeMimicSuccessCfg = PickCubeMimicSuccessCfg()


__all__ = [
    "PickCubeActionsCfg",
    "PickCubeMdpCfg",
    "PickCubeMimicMdpCfg",
    "PickCubeObservationsCfg",
    "PickCubePolicyObs",
    "PickCubeTeleopObservationsCfg",
    "PickCubeTeleopPolicyObs",
]
