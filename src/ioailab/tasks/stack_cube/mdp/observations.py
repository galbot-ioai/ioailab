"""Observation terms and policy group for Galbot G1 stack-cube variants."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

from ioailab.tasks.common.mdp import single_dof_gripper_pos

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def single_gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_names: Sequence[str] | str | None = None,
) -> torch.Tensor:
    """Return the G1 single-joint gripper position as an observation term."""

    return single_dof_gripper_pos(env, robot_cfg, gripper_joint_names).unsqueeze(1)


@configclass
class StackCubePolicyObs(ObsGroup):
    """Policy observations with robot state and object poses."""

    actions = ObsTerm(func=stack_mdp.last_action)
    joint_pos = ObsTerm(func=stack_mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=stack_mdp.joint_vel_rel)
    object = ObsTerm(
        func=stack_mdp.object_abs_obs_in_base_frame,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    cube_positions = ObsTerm(
        func=stack_mdp.cube_poses_in_base_frame,
        params={"robot_cfg": SceneEntityCfg("robot"), "return_key": "pos"},
    )
    cube_orientations = ObsTerm(
        func=stack_mdp.cube_poses_in_base_frame,
        params={"robot_cfg": SceneEntityCfg("robot"), "return_key": "quat"},
    )
    eef_pos = ObsTerm(
        func=stack_mdp.ee_frame_pose_in_base_frame,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "return_key": "pos",
        },
    )
    eef_quat = ObsTerm(
        func=stack_mdp.ee_frame_pose_in_base_frame,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "return_key": "quat",
        },
    )
    gripper_pos = ObsTerm(func=single_gripper_pos)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StackCubeObservationsCfg:
    """State observations for G1 stack-cube training."""

    policy: StackCubePolicyObs = StackCubePolicyObs()
