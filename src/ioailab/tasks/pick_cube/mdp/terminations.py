"""Termination and evaluation terms for Galbot G1 pick-cube tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.mdp import (
    resolve_gripper_open_value,
    resolve_gripper_threshold,
    single_dof_gripper_pos,
)
from ioailab.utils.scene_state import asset_root_pos_w
from ioailab.utils.tensors import as_torch_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

PICK_CUBE_CUBE_SIZE = 0.05
PICK_CUBE_BLUE_BLOCK_SIZE = (0.15, 0.15, 0.02)
PICK_CUBE_GRIPPER_OPEN_THRESHOLD = 0.05


def grasped_cube(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tcp_frame_cfg: SceneEntityCfg = SceneEntityCfg("tcp_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.06,
    gripper_joint_names: Sequence[str] | str | None = None,
    gripper_open_val: float | None = None,
    gripper_threshold: float | None = None,
) -> torch.Tensor:
    """Return whether the cube is close to the TCP and the gripper is closed."""

    cube_pos = asset_root_pos_w(env, cube_cfg.name)
    tcp_frame = env.scene[tcp_frame_cfg.name]
    tcp_pos = as_torch_tensor(tcp_frame.data.target_pos_w)[:, 0, :]
    tcp_to_cube = torch.linalg.vector_norm(cube_pos - tcp_pos, dim=1)

    gripper_pos = single_dof_gripper_pos(env, robot_cfg, gripper_joint_names)
    open_val = gripper_pos.new_tensor(resolve_gripper_open_value(env, gripper_open_val))
    threshold = gripper_pos.new_tensor(
        resolve_gripper_threshold(env, gripper_threshold)
    )
    gripper_closed = torch.abs(gripper_pos - open_val) > threshold
    return torch.logical_and(tcp_to_cube < diff_threshold, gripper_closed)


def cube_released_on_blue_block(
    env: ManagerBasedEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    blue_block_cfg: SceneEntityCfg = SceneEntityCfg("blue_block"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.08,
    platform_top_to_cube_center: float = 0.035,
    height_threshold: float = 0.03,
    gripper_joint_names: Sequence[str] | str | None = None,
    gripper_open_val: float | None = None,
    gripper_open_threshold: float = PICK_CUBE_GRIPPER_OPEN_THRESHOLD,
) -> torch.Tensor:
    """Return whether the cube is on the blue block and released."""

    cube_pos = asset_root_pos_w(env, cube_cfg.name)
    blue_block_pos = asset_root_pos_w(env, blue_block_cfg.name)
    xy_aligned = (
        torch.linalg.vector_norm(cube_pos[:, :2] - blue_block_pos[:, :2], dim=1)
        < xy_threshold
    )
    z_aligned = (
        torch.abs(cube_pos[:, 2] - blue_block_pos[:, 2] - platform_top_to_cube_center)
        < height_threshold
    )
    gripper_pos = single_dof_gripper_pos(env, robot_cfg, gripper_joint_names)
    open_val = gripper_pos.new_tensor(resolve_gripper_open_value(env, gripper_open_val))
    open_threshold = gripper_pos.new_tensor(float(gripper_open_threshold))
    gripper_open = torch.abs(gripper_pos - open_val) <= open_threshold
    return xy_aligned & z_aligned & gripper_open


def make_pick_cube_release_termination_term() -> DoneTerm:
    """Return the PickCube task-complete termination term config."""

    return DoneTerm(
        func=cube_released_on_blue_block,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "blue_block_cfg": SceneEntityCfg("blue_block"),
            "platform_top_to_cube_center": PICK_CUBE_BLUE_BLOCK_SIZE[2] / 2.0
            + PICK_CUBE_CUBE_SIZE / 2.0,
            "gripper_open_threshold": PICK_CUBE_GRIPPER_OPEN_THRESHOLD,
        },
    )


def make_pick_cube_evaluation_success_term() -> DoneTerm:
    """Return the PickCube evaluation-success term config."""

    return make_pick_cube_release_termination_term()


@configclass
class PickCubeTerminationsCfg:
    """Termination terms for motion-planning and teleop data collection."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    released_on_blue_block = make_pick_cube_release_termination_term()


@configclass
class PickCubeMimicSuccessCfg:
    """Mimic-only success term required by IsaacLab Mimic."""

    # IsaacLab Mimic reads final task completion from the fixed path
    # ``env_cfg.terminations.success``. Keep this separate from the normal
    # PickCube termination cfg so data collection keeps its explicit
    # ``released_on_blue_block`` termination name.
    success = make_pick_cube_release_termination_term()


__all__ = [
    "PICK_CUBE_GRIPPER_OPEN_THRESHOLD",
    "PickCubeMimicSuccessCfg",
    "PickCubeTerminationsCfg",
    "cube_released_on_blue_block",
    "grasped_cube",
    "make_pick_cube_evaluation_success_term",
    "make_pick_cube_release_termination_term",
]
