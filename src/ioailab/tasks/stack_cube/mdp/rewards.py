"""Reward terms and config for Galbot G1 stack-cube task variants."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ioailab.utils.scene_state import asset_root_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cubes_stacked_on_base_cube(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.04,
    cube_height: float = 0.05,
    height_threshold: float = 0.006,
) -> torch.Tensor:
    """Return whether the three cubes form a vertical stack on the base cube."""

    return objects_stacked_on_base(
        env,
        object_cfgs=(cube_1_cfg, cube_2_cfg, cube_3_cfg),
        xy_threshold=xy_threshold,
        object_height=cube_height,
        height_threshold=height_threshold,
    )


def objects_stacked_on_base(
    env: ManagerBasedRLEnv,
    object_cfgs: Sequence[SceneEntityCfg],
    xy_threshold: float = 0.04,
    object_height: float = 0.05,
    height_threshold: float = 0.006,
) -> torch.Tensor:
    """Return whether ordered objects form a vertical stack on the first object."""

    object_cfgs = tuple(object_cfgs)
    if len(object_cfgs) < 2:
        raise ValueError("Expected at least two object configs.")

    object_positions = tuple(
        asset_root_pos_w(env, object_cfg.name) for object_cfg in object_cfgs
    )
    stacked = torch.ones(
        object_positions[0].shape[0],
        device=object_positions[0].device,
        dtype=torch.bool,
    )

    for lower_object_pos, upper_object_pos in zip(
        object_positions, object_positions[1:]
    ):
        pos_diff = upper_object_pos - lower_object_pos
        stacked = torch.logical_and(
            stacked,
            torch.linalg.vector_norm(pos_diff[:, :2], dim=1) < xy_threshold,
        )
        stacked = torch.logical_and(
            stacked,
            torch.abs(pos_diff[:, 2] - object_height) < height_threshold,
        )
    return stacked


def stack_success_reward(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.04,
    cube_height: float = 0.05,
    height_threshold: float = 0.006,
) -> torch.Tensor:
    """Reward completed stacks."""

    return cubes_stacked_on_base_cube(
        env,
        cube_1_cfg=cube_1_cfg,
        cube_2_cfg=cube_2_cfg,
        cube_3_cfg=cube_3_cfg,
        xy_threshold=xy_threshold,
        cube_height=cube_height,
        height_threshold=height_threshold,
    ).to(torch.float32)


def cube_to_stack_alignment_reward(
    env: ManagerBasedRLEnv,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    height_diff: float = 0.05,
    distance_scale: float = 8.0,
) -> torch.Tensor:
    """Reward an object for moving toward its target stack pose."""

    pos_error = asset_root_pos_w(env, upper_object_cfg.name) - asset_root_pos_w(
        env, lower_object_cfg.name
    )
    target_error = torch.cat((pos_error[:, :2], pos_error[:, 2:3] - height_diff), dim=1)
    return 1.0 - torch.tanh(
        distance_scale * torch.linalg.vector_norm(target_error, dim=1)
    )


def action_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large policy actions."""

    return torch.sum(torch.square(env.action_manager.action), dim=1)


@configclass
class StackCubeRewardsCfg:
    """Dense rewards for first-stage PPO stack-cube experiments."""

    cube_2_on_cube_1 = RewTerm(
        func=cube_to_stack_alignment_reward,
        weight=2.0,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_1"),
            "height_diff": 0.05,
        },
    )
    cube_3_on_cube_2 = RewTerm(
        func=cube_to_stack_alignment_reward,
        weight=2.0,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_3"),
            "lower_object_cfg": SceneEntityCfg("cube_2"),
            "height_diff": 0.05,
        },
    )
    stack_success = RewTerm(
        func=stack_success_reward,
        weight=20.0,
        params={
            "cube_height": 0.05,
            "xy_threshold": 0.04,
            "height_threshold": 0.006,
        },
    )
    action_rate = RewTerm(func=action_l2_penalty, weight=-1.0e-4)
