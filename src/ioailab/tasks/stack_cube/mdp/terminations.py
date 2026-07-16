"""Termination terms and stage-completion predicates for G1 stack-cube.

The boolean predicates here report completion of stack-cube Mimic stages and
regular task completion signals. They live with termination semantics rather
than in observation config.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

from ioailab.tasks.common.mdp import (
    resolve_gripper_open_value,
    resolve_gripper_threshold,
    single_dof_gripper_pos,
)
from ioailab.utils.scene_state import asset_root_pos_w
from ioailab.utils.tensors import as_torch_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class StackCubeTerminationsCfg:
    """Termination terms for the G1 stack-cube task."""

    time_out = DoneTerm(func=stack_mdp.time_out, time_out=True)
    cube_1_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")},
    )
    cube_2_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")},
    )
    cube_3_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")},
    )
    # Keep success non-terminal so the motion-plan expert can release and lift away.
    success = None


def object_grasped_by_single_gripper(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    gripper_joint_names: Sequence[str] | str | None = None,
    gripper_open_val: float | None = None,
    gripper_threshold: float | None = None,
) -> torch.Tensor:
    """Return whether a closed G1 single-joint gripper is near an object."""

    ee_frame = env.scene[ee_frame_cfg.name]
    object_pos = asset_root_pos_w(env, object_cfg.name)
    end_effector_pos = as_torch_tensor(ee_frame.data.target_pos_w)[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    gripper_pos = single_dof_gripper_pos(env, robot_cfg, gripper_joint_names)
    open_target = gripper_pos.new_tensor(
        resolve_gripper_open_value(env, gripper_open_val)
    )
    close_threshold = gripper_pos.new_tensor(
        resolve_gripper_threshold(env, gripper_threshold)
    )
    gripper_closed = torch.abs(gripper_pos - open_target) > close_threshold
    return torch.logical_and(pose_diff < diff_threshold, gripper_closed)


def object_stacked_single_gripper(
    env: ManagerBasedRLEnv,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.006,
    height_diff: float = 0.05,
    gripper_joint_names: Sequence[str] | str | None = None,
    gripper_open_val: float | None = None,
    gripper_open_atol: float = 1e-3,
    gripper_open_rtol: float = 1e-3,
) -> torch.Tensor:
    """Return whether one object is stacked on another and the gripper is open."""

    upper_object_pos = asset_root_pos_w(env, upper_object_cfg.name)
    lower_object_pos = asset_root_pos_w(env, lower_object_cfg.name)
    pos_diff = upper_object_pos - lower_object_pos
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_error = torch.abs(pos_diff[:, 2] - height_diff)

    gripper_pos = single_dof_gripper_pos(env, robot_cfg, gripper_joint_names)
    gripper_open = torch.isclose(
        gripper_pos,
        gripper_pos.new_tensor(resolve_gripper_open_value(env, gripper_open_val)),
        atol=gripper_open_atol,
        rtol=gripper_open_rtol,
    )
    stacked = torch.logical_and(xy_dist < xy_threshold, z_error < height_threshold)
    return torch.logical_and(stacked, gripper_open)


def cube_2_grasped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether cube_2 has been grasped by the left gripper."""

    return object_grasped_by_single_gripper(
        env,
        SceneEntityCfg("robot"),
        SceneEntityCfg("tcp_frame"),
        SceneEntityCfg("cube_2"),
    )


def cube_2_on_cube_1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether cube_2 is released on top of cube_1."""

    return object_stacked_single_gripper(
        env,
        upper_object_cfg=SceneEntityCfg("cube_2"),
        lower_object_cfg=SceneEntityCfg("cube_1"),
    )


def cube_3_grasped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether cube_3 has been grasped by the left gripper."""

    return object_grasped_by_single_gripper(
        env,
        SceneEntityCfg("robot"),
        SceneEntityCfg("tcp_frame"),
        SceneEntityCfg("cube_3"),
    )


def cube_3_on_cube_2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether cube_3 is released on top of cube_2."""

    return object_stacked_single_gripper(
        env,
        upper_object_cfg=SceneEntityCfg("cube_3"),
        lower_object_cfg=SceneEntityCfg("cube_2"),
    )


def stack_cube_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether the full three-cube stack has been completed."""

    return cube_2_on_cube_1(env) & cube_3_on_cube_2(env)


@configclass
class StackCubeMimicSuccessCfg:
    """Mimic-only success term required by IsaacLab Mimic."""

    # IsaacLab Mimic reads final task completion from the fixed path
    # ``env_cfg.terminations.success``. Keep this separate from the normal
    # StackCube termination cfg, where full-stack success stays non-terminal so
    # the expert can finish release/lift motions.
    success = DoneTerm(func=stack_cube_success)
