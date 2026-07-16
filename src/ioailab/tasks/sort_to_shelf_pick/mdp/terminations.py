"""Termination terms for the SortToShelf pick phase."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
import torch

from ioailab.tasks.sort_to_shelf.mdp.terminations import joints_at_named_targets
from ioailab.tasks.sort_to_shelf.scene import (
    sorting_object_name,
    sorting_object_pick_lift_min_z,
)
from ioailab.utils.scene_state import asset_root_pos_w


def object_lifted_and_left_arm_at_carry(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_object_center_z: float | None = None,
    target_joint_pos_by_name: Mapping[str, float] | None = None,
    required_joint_names: Sequence[str] = (),
    max_joint_abs_error: float = 0.12,
) -> torch.Tensor:
    """Return whether an object is lifted and configured arm joints are at carry."""

    if target_joint_pos_by_name is None:
        raise ValueError("target_joint_pos_by_name is required.")
    if not required_joint_names:
        required_joint_names = tuple(str(name) for name in target_joint_pos_by_name)
    target_joint_names = tuple(
        joint_name
        for joint_name in required_joint_names
        if joint_name in target_joint_pos_by_name
    )
    if len(target_joint_names) != len(tuple(required_joint_names)):
        missing = sorted(set(required_joint_names) - set(target_joint_names))
        raise ValueError(f"Carry posture is missing joint(s): {missing}")

    object_pos_w = asset_root_pos_w(env, object_cfg.name)
    if min_object_center_z is None:
        min_object_center_z = sorting_object_pick_lift_min_z(object_cfg.name)
    lifted = object_pos_w[:, 2] >= float(min_object_center_z)
    return torch.logical_and(
        lifted,
        joints_at_named_targets(
            env,
            robot_cfg=robot_cfg,
            target_joint_names=target_joint_names,
            target_joint_pos_by_name=target_joint_pos_by_name,
            max_joint_abs_error=max_joint_abs_error,
            device=lifted.device,
        ),
    )


def make_pick_success_term(
    object_name: str | None = "red_cube",
    *,
    target_joint_pos_by_name: Mapping[str, float],
    required_joint_names: Sequence[str],
    max_joint_abs_error: float = 0.12,
) -> DoneTerm:
    """Return the selected-object pick success termination term."""

    resolved = sorting_object_name(object_name)
    return DoneTerm(
        func=object_lifted_and_left_arm_at_carry,
        params={
            "object_cfg": SceneEntityCfg(resolved),
            "min_object_center_z": sorting_object_pick_lift_min_z(resolved),
            "target_joint_pos_by_name": dict(target_joint_pos_by_name),
            "required_joint_names": tuple(required_joint_names),
            "max_joint_abs_error": float(max_joint_abs_error),
        },
    )


def make_pick_terminations_cfg(
    *,
    target_joint_pos_by_name: Mapping[str, float],
    required_joint_names: Sequence[str],
    max_joint_abs_error: float = 0.12,
) -> type:
    """Return the pick phase termination cfg for a robot binding."""

    @configclass
    class SortToShelfPickTerminationsCfg:
        """Pick phase terminations: time out plus carry-posture success."""

        time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
        at_carry = make_pick_success_term(
            target_joint_pos_by_name=target_joint_pos_by_name,
            required_joint_names=required_joint_names,
            max_joint_abs_error=max_joint_abs_error,
        )

    return SortToShelfPickTerminationsCfg


__all__ = [
    "make_pick_success_term",
    "make_pick_terminations_cfg",
    "object_lifted_and_left_arm_at_carry",
]
