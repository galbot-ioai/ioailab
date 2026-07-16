"""Termination terms for the PickToShelf place phase."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
import torch

from ioailab.tasks.common.mdp import (
    condition_held_for_min_steps,
    resolve_gripper_open_value,
    single_dof_gripper_pos,
)
from ioailab.tasks.pick_to_shelf.scene import CUBE_SIZE, SHELF_DECK_SIZE
from ioailab.utils.pose import quat_xyzw_local_z_dot_world_z
from ioailab.utils.scene_state import asset_root_pos_w, asset_root_quat_xyzw

SHELF_TOP_TO_CUBE_CENTER = SHELF_DECK_SIZE[2] / 2.0 + CUBE_SIZE[2] / 2.0
SHELF_PLACE_XY_THRESHOLD = 0.10
SHELF_PLACE_Z_THRESHOLD = 0.04
SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT = 0.90
SHELF_PLACE_GRIPPER_OPEN_THRESHOLD = 0.05
SHELF_PLACE_MIN_SUCCESS_STEPS = 20


def cube_placed_on_shelf(
    env,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    shelf_deck_cfg: SceneEntityCfg = SceneEntityCfg("shelf_deck"),
    platform_top_to_cube_center: float = SHELF_TOP_TO_CUBE_CENTER,
    xy_threshold: float = SHELF_PLACE_XY_THRESHOLD,
    z_threshold: float = SHELF_PLACE_Z_THRESHOLD,
    upright_z_axis_min_dot: float = SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_names: Sequence[str] | str | None = None,
    gripper_open_val: float | None = None,
    gripper_open_threshold: float = SHELF_PLACE_GRIPPER_OPEN_THRESHOLD,
    min_success_steps: int = SHELF_PLACE_MIN_SUCCESS_STEPS,
) -> torch.Tensor:
    """Return whether the released cube remained correctly placed on the shelf."""

    cube_pos_w = asset_root_pos_w(env, cube_cfg.name)
    shelf_pos_w = asset_root_pos_w(env, shelf_deck_cfg.name)
    xy_distance = torch.linalg.vector_norm(
        cube_pos_w[:, :2] - shelf_pos_w[:, :2], dim=1
    )
    expected_cube_z = shelf_pos_w[:, 2] + platform_top_to_cube_center
    z_distance = torch.abs(cube_pos_w[:, 2] - expected_cube_z)
    cube_quat_w = asset_root_quat_xyzw(env, cube_cfg.name).to(
        device=cube_pos_w.device,
        dtype=cube_pos_w.dtype,
    )
    upright_dot = torch.abs(quat_xyzw_local_z_dot_world_z(cube_quat_w))
    position_ok = torch.logical_and(
        xy_distance <= xy_threshold, z_distance <= z_threshold
    )
    upright_ok = upright_dot >= float(upright_z_axis_min_dot)
    gripper_pos = single_dof_gripper_pos(env, robot_cfg, gripper_joint_names)
    open_val = gripper_pos.new_tensor(resolve_gripper_open_value(env, gripper_open_val))
    open_threshold = gripper_pos.new_tensor(float(gripper_open_threshold))
    gripper_open = torch.abs(gripper_pos - open_val) <= open_threshold
    placed_and_released = position_ok & upright_ok & gripper_open
    return condition_held_for_min_steps(
        env,
        condition=placed_and_released,
        min_steps=min_success_steps,
        state_key="pick_to_shelf_place_success",
    )


def make_shelf_place_success_term() -> DoneTerm:
    """Return the place phase shelf-placement success term."""

    return DoneTerm(
        func=cube_placed_on_shelf,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "shelf_deck_cfg": SceneEntityCfg("shelf_deck"),
            "platform_top_to_cube_center": SHELF_TOP_TO_CUBE_CENTER,
            "xy_threshold": SHELF_PLACE_XY_THRESHOLD,
            "z_threshold": SHELF_PLACE_Z_THRESHOLD,
            "upright_z_axis_min_dot": SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT,
            "robot_cfg": SceneEntityCfg("robot"),
            "gripper_open_threshold": SHELF_PLACE_GRIPPER_OPEN_THRESHOLD,
            "min_success_steps": SHELF_PLACE_MIN_SUCCESS_STEPS,
        },
    )


@configclass
class PickToShelfPlaceTerminationsCfg:
    """Place phase terminations: time out plus shelf-placement success."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cube_on_shelf = make_shelf_place_success_term()


__all__ = [
    "PickToShelfPlaceTerminationsCfg",
    "SHELF_PLACE_GRIPPER_OPEN_THRESHOLD",
    "SHELF_PLACE_MIN_SUCCESS_STEPS",
    "SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT",
    "SHELF_PLACE_XY_THRESHOLD",
    "SHELF_PLACE_Z_THRESHOLD",
    "SHELF_TOP_TO_CUBE_CENTER",
    "cube_placed_on_shelf",
    "make_shelf_place_success_term",
]
