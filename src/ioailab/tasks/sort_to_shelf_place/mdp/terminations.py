"""Termination terms for the SortToShelf place phase."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
import torch

from ioailab.tasks.common.mdp import (
    resolve_gripper_open_value,
    single_dof_gripper_pos,
)
from ioailab.tasks.sort_to_shelf.mdp.terminations import (
    SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT,
    SHELF_PLACE_XY_THRESHOLD,
    joints_at_named_targets,
)
from ioailab.tasks.sort_to_shelf.scene import (
    SORTING_SHELF_PLACE_Z_THRESHOLD,
    sorting_object_name,
    sorting_place_board_asset_name_for_object,
    sorting_place_target_offset_from_board_for_object,
    sorting_place_upright_z_axis_min_dot_for_object,
)
from ioailab.utils.pose import quat_xyzw_local_z_dot_world_z
from ioailab.utils.scene_state import asset_root_pos_w, asset_root_quat_xyzw


def object_placed_at_target_position(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    target_pos_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_asset_cfg: SceneEntityCfg | None = None,
    target_offset_xyz: tuple[float, float, float] | None = None,
    xy_threshold: float = SHELF_PLACE_XY_THRESHOLD,
    z_threshold: float = SORTING_SHELF_PLACE_Z_THRESHOLD,
    upright_z_axis_min_dot: float = SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_names: Sequence[str] | str | None = None,
    gripper_open_val: float | None = None,
    gripper_open_threshold: float | None = None,
    target_joint_pos_by_name: Mapping[str, float] | None = None,
    required_joint_names: Sequence[str] = (),
    max_joint_abs_error: float = 0.12,
) -> torch.Tensor:
    """Return whether an object is placed, released, and optionally retracted."""

    object_pos_w = asset_root_pos_w(env, object_cfg.name)
    if target_asset_cfg is None:
        target_pos = object_pos_w.new_tensor(target_pos_xyz).reshape(1, 3)
    else:
        target_pos = asset_root_pos_w(env, target_asset_cfg.name)
        offset = object_pos_w.new_tensor(target_offset_xyz or (0.0, 0.0, 0.0))
        target_pos = target_pos + offset.reshape(1, 3)
    xy_distance = torch.linalg.vector_norm(
        object_pos_w[:, :2] - target_pos[:, :2], dim=1
    )
    z_distance = torch.abs(object_pos_w[:, 2] - target_pos[:, 2])
    object_quat_w = asset_root_quat_xyzw(env, object_cfg.name).to(
        device=object_pos_w.device,
        dtype=object_pos_w.dtype,
    )
    upright_dot = torch.abs(quat_xyzw_local_z_dot_world_z(object_quat_w))
    success = torch.logical_and(
        torch.logical_and(xy_distance <= xy_threshold, z_distance <= z_threshold),
        upright_dot >= float(upright_z_axis_min_dot),
    )

    if gripper_open_threshold is not None:
        gripper_pos = single_dof_gripper_pos(env, robot_cfg, gripper_joint_names)
        open_val = gripper_pos.new_tensor(
            resolve_gripper_open_value(env, gripper_open_val)
        )
        open_threshold = gripper_pos.new_tensor(float(gripper_open_threshold))
        gripper_open = torch.abs(gripper_pos - open_val) <= open_threshold
        success = torch.logical_and(success, gripper_open)

    if target_joint_pos_by_name is not None:
        requested_joint_names = tuple(required_joint_names or target_joint_pos_by_name)
        target_joint_names = tuple(
            str(joint_name)
            for joint_name in requested_joint_names
            if joint_name in target_joint_pos_by_name
        )
        if len(target_joint_names) != len(requested_joint_names):
            missing = sorted(set(requested_joint_names) - set(target_joint_names))
            raise ValueError(f"Place posture is missing joint(s): {missing}")
        success = torch.logical_and(
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

    return success


def make_place_success_term(
    object_name: str | None = "red_cube",
    *,
    gripper_open_threshold: float,
    target_joint_pos_by_name: Mapping[str, float],
    required_joint_names: Sequence[str],
    max_joint_abs_error: float = 0.12,
) -> DoneTerm:
    """Return the selected-object shelf-cell placement success termination term."""

    resolved = sorting_object_name(object_name)
    return DoneTerm(
        func=object_placed_at_target_position,
        params={
            "object_cfg": SceneEntityCfg(resolved),
            "target_asset_cfg": SceneEntityCfg(
                sorting_place_board_asset_name_for_object(resolved)
            ),
            "target_offset_xyz": sorting_place_target_offset_from_board_for_object(
                resolved
            ),
            "xy_threshold": SHELF_PLACE_XY_THRESHOLD,
            "z_threshold": SORTING_SHELF_PLACE_Z_THRESHOLD,
            "upright_z_axis_min_dot": sorting_place_upright_z_axis_min_dot_for_object(
                resolved
            ),
            "robot_cfg": SceneEntityCfg("robot"),
            "gripper_open_threshold": float(gripper_open_threshold),
            "target_joint_pos_by_name": dict(target_joint_pos_by_name),
            "required_joint_names": tuple(required_joint_names),
            "max_joint_abs_error": float(max_joint_abs_error),
        },
    )


def make_place_terminations_cfg(
    *,
    gripper_open_threshold: float,
    target_joint_pos_by_name: Mapping[str, float],
    required_joint_names: Sequence[str],
    max_joint_abs_error: float = 0.12,
) -> type:
    """Return the place phase termination cfg for a robot binding."""

    @configclass
    class SortToShelfPlaceTerminationsCfg:
        """Place phase terminations: time out plus shelf-cell placement success."""

        time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
        placed = make_place_success_term(
            gripper_open_threshold=gripper_open_threshold,
            target_joint_pos_by_name=target_joint_pos_by_name,
            required_joint_names=required_joint_names,
            max_joint_abs_error=max_joint_abs_error,
        )

    return SortToShelfPlaceTerminationsCfg


__all__ = [
    "make_place_success_term",
    "make_place_terminations_cfg",
    "object_placed_at_target_position",
]
