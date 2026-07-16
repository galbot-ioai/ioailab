"""Observation terms for the SortToShelf pick phase."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
import torch

from ioailab.tasks.common.mdp import rgb_image_obs_term


def canonical_robot_joint_pos(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: Sequence[str] = (),
    neutral_joint_names: Sequence[str] = (),
) -> torch.Tensor:
    """Return robot joint positions in a caller-specified canonical order."""

    if not joint_names:
        raise ValueError("canonical_robot_joint_pos requires joint_names.")

    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene[asset_cfg.name]
    available_names = tuple(str(name) for name in getattr(robot, "joint_names", ()))
    missing = tuple(
        joint_name for joint_name in joint_names if joint_name not in available_names
    )
    if missing:
        raise ValueError(f"SortToShelf robot_joint_pos is missing joint(s): {missing}")

    joint_ids = [available_names.index(joint_name) for joint_name in joint_names]
    neutral_names = frozenset(str(joint_name) for joint_name in neutral_joint_names)
    neutral_columns = tuple(
        index
        for index, joint_name in enumerate(joint_names)
        if joint_name in neutral_names
    )
    joint_pos = torch.as_tensor(
        robot.data.joint_pos,
        device=getattr(unwrapped, "device", None),
        dtype=torch.float32,
    )
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(1, -1)
    obs = joint_pos[:, joint_ids].clone()
    if neutral_columns:
        obs[:, neutral_columns] = 0.0
    return obs


def make_sorting_observations_cfg(
    *,
    joint_names: Sequence[str],
    neutral_joint_names: Sequence[str] = (),
    camera_name: str,
    robot_entity_name: str = "robot",
) -> type:
    """Return the SortToShelf policy observation cfg for a robot binding."""

    canonical_joint_names = tuple(str(joint_name) for joint_name in joint_names)
    if not canonical_joint_names:
        raise ValueError("make_sorting_observations_cfg requires joint_names.")
    neutral_joints = tuple(str(joint_name) for joint_name in neutral_joint_names)
    camera_entity_name = str(camera_name)
    robot_cfg = SceneEntityCfg(str(robot_entity_name))

    @configclass
    class SortToShelfPolicyObs(ObsGroup):
        """Last action, canonical robot joints, and one task camera."""

        actions = ObsTerm(func=base_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=canonical_robot_joint_pos,
            params={
                "asset_cfg": robot_cfg,
                "joint_names": canonical_joint_names,
                "neutral_joint_names": neutral_joints,
            },
        )
        front_head_rgb = rgb_image_obs_term(camera_entity_name)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SortToShelfObservationsCfg:
        """Policy observations for vision-based collection and evaluation."""

        policy: SortToShelfPolicyObs = SortToShelfPolicyObs()

    return SortToShelfObservationsCfg


__all__ = ["canonical_robot_joint_pos", "make_sorting_observations_cfg"]
