"""Shared termination helpers for SortToShelf phase tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
import torch

from ioailab.tasks.base_nav.mdp.terminations import goal_reached
from ioailab.tasks.sort_to_shelf.scene import (
    SORTING_SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT,
)

SHELF_PLACE_XY_THRESHOLD = 0.10
SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT = SORTING_SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT


def goal_reached_and_joints_at_targets(
    env,
    *,
    robot_cfg: SceneEntityCfg,
    target_joint_names: Sequence[str],
    target_joint_pos_by_name: Mapping[str, float],
    max_joint_abs_error: float,
) -> torch.Tensor:
    """Return whether base navigation and named joint targets are both done."""

    success = goal_reached(env)
    return torch.logical_and(
        success,
        joints_at_named_targets(
            env,
            robot_cfg=robot_cfg,
            target_joint_names=target_joint_names,
            target_joint_pos_by_name=target_joint_pos_by_name,
            max_joint_abs_error=max_joint_abs_error,
            device=success.device,
        ),
    )


def joints_at_named_targets(
    env,
    *,
    robot_cfg: SceneEntityCfg,
    target_joint_names: Sequence[str],
    target_joint_pos_by_name: Mapping[str, float],
    max_joint_abs_error: float,
    device: torch.device,
) -> torch.Tensor:
    """Return whether named joints are within tolerance of target positions."""

    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene[robot_cfg.name]
    joint_ids, resolved_joint_names = robot.find_joints(tuple(target_joint_names))

    joint_pos = torch.as_tensor(
        robot.data.joint_pos, device=device, dtype=torch.float32
    )
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(1, -1)
    actual = joint_pos[:, joint_ids]
    target = torch.tensor(
        [target_joint_pos_by_name[joint_name] for joint_name in resolved_joint_names],
        device=device,
        dtype=torch.float32,
    )
    return torch.all(torch.abs(actual - target) <= float(max_joint_abs_error), dim=1)


@configclass
class SortToShelfTimeOutTerminationsCfg:
    """Base timeout-only termination group for task-specific phase cfgs."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


__all__ = [
    "SortToShelfTimeOutTerminationsCfg",
    "goal_reached_and_joints_at_targets",
    "joints_at_named_targets",
]
