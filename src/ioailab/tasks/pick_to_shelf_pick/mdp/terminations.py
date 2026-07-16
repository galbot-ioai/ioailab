"""Termination terms for the PickToShelf pick phase."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
import torch

from ioailab.tasks.pick_to_shelf.scene import CUBE_SIZE, TABLE_TOP_Z
from ioailab.utils.scene_state import asset_root_pos_w

CUBE_PICK_LIFT_MIN_Z = TABLE_TOP_Z + CUBE_SIZE[2] / 2.0 + 0.06


def cube_lifted_and_left_arm_at_carry(
    env,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_cube_center_z: float = CUBE_PICK_LIFT_MIN_Z,
    target_joint_pos_by_name: Mapping[str, float] | None = None,
    required_joint_names: Sequence[str] = (),
    max_joint_abs_error: float = 0.12,
) -> torch.Tensor:
    """Return whether the cube is lifted and configured arm joints are at carry."""

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

    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene[robot_cfg.name]
    joint_ids, resolved_joint_names = robot.find_joints(target_joint_names)
    if tuple(resolved_joint_names) != target_joint_names:
        raise ValueError(
            "Robot joint resolution changed carry order: "
            f"{tuple(resolved_joint_names)} != {target_joint_names}."
        )

    lifted = cube_lifted_from_table(
        env,
        cube_cfg=cube_cfg,
        min_cube_center_z=min_cube_center_z,
    )
    joint_pos = torch.as_tensor(
        robot.data.joint_pos,
        device=lifted.device,
        dtype=torch.float32,
    )
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(1, -1)
    actual = joint_pos[:, joint_ids]
    target = torch.tensor(
        [target_joint_pos_by_name[joint_name] for joint_name in target_joint_names],
        device=lifted.device,
        dtype=torch.float32,
    )
    at_carry = torch.all(
        torch.abs(actual - target) <= float(max_joint_abs_error), dim=1
    )
    return torch.logical_and(lifted, at_carry)


def cube_lifted_from_table(
    env,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    min_cube_center_z: float = CUBE_PICK_LIFT_MIN_Z,
) -> torch.Tensor:
    """Return whether the cube has been lifted above the pick threshold."""

    cube_pos_w = asset_root_pos_w(env, cube_cfg.name)
    return cube_pos_w[:, 2] >= min_cube_center_z


def make_pick_carry_success_term(
    *,
    target_joint_pos_by_name: Mapping[str, float],
    required_joint_names: Sequence[str],
    max_joint_abs_error: float = 0.12,
) -> DoneTerm:
    """Return the pick phase success termination term."""

    return DoneTerm(
        func=cube_lifted_and_left_arm_at_carry,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "min_cube_center_z": CUBE_PICK_LIFT_MIN_Z,
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
    class PickToShelfPickTerminationsCfg:
        """Pick phase terminations: time out plus carry-posture success."""

        time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
        at_carry = make_pick_carry_success_term(
            target_joint_pos_by_name=target_joint_pos_by_name,
            required_joint_names=required_joint_names,
            max_joint_abs_error=max_joint_abs_error,
        )

    return PickToShelfPickTerminationsCfg


__all__ = [
    "CUBE_PICK_LIFT_MIN_Z",
    "cube_lifted_and_left_arm_at_carry",
    "cube_lifted_from_table",
    "make_pick_carry_success_term",
    "make_pick_terminations_cfg",
]
