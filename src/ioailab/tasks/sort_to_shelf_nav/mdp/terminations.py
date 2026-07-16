"""Termination terms for the SortToShelf navigation phase."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
import torch

from ioailab.tasks.base_nav.mdp.observations import vector_to_goal_xy
from ioailab.tasks.common.mdp import condition_held_for_min_steps
from ioailab.tasks.sort_to_shelf.mdp.terminations import (
    joints_at_named_targets,
)


def nav_place_start_reached(
    env,
    *,
    robot_cfg: SceneEntityCfg,
    target_joint_names: Sequence[str],
    target_joint_pos_by_name: Mapping[str, float],
    max_joint_abs_error: float,
    base_success_radius: float,
    min_ready_steps: int,
) -> torch.Tensor:
    """Return whether nav reached the shelf and the place-start posture."""

    distance_to_goal = torch.linalg.vector_norm(vector_to_goal_xy(env), dim=1)
    base_at_place_start = distance_to_goal <= float(base_success_radius)
    joints_at_place_start = joints_at_named_targets(
        env,
        robot_cfg=robot_cfg,
        target_joint_names=target_joint_names,
        target_joint_pos_by_name=target_joint_pos_by_name,
        max_joint_abs_error=max_joint_abs_error,
        device=base_at_place_start.device,
    )
    ready = torch.logical_and(base_at_place_start, joints_at_place_start)
    return ready_held_for_min_steps(
        env, ready=ready, min_ready_steps=int(min_ready_steps)
    )


def ready_held_for_min_steps(
    env,
    *,
    ready: torch.Tensor,
    min_ready_steps: int,
) -> torch.Tensor:
    """Return rows whose ready mask has been true for consecutive steps."""

    return condition_held_for_min_steps(
        env,
        condition=ready,
        min_steps=min_ready_steps,
        state_key="sort_to_shelf_nav_ready",
    )


def make_nav_success_term(
    *,
    target_joint_names: Sequence[str],
    target_joint_pos_by_name: Mapping[str, float],
    max_joint_abs_error: float,
    base_success_radius: float,
    min_ready_steps: int,
    robot_name: str = "robot",
) -> DoneTerm:
    """Return nav success after base arrival and place-start posture settling."""

    return DoneTerm(
        func=nav_place_start_reached,
        params={
            "robot_cfg": SceneEntityCfg(robot_name),
            "target_joint_names": tuple(target_joint_names),
            "target_joint_pos_by_name": dict(target_joint_pos_by_name),
            "max_joint_abs_error": float(max_joint_abs_error),
            "base_success_radius": float(base_success_radius),
            "min_ready_steps": int(min_ready_steps),
        },
    )


def make_nav_terminations_cfg(
    *,
    target_joint_names: Sequence[str],
    target_joint_pos_by_name: Mapping[str, float],
    max_joint_abs_error: float,
    base_success_radius: float,
    min_ready_steps: int,
) -> type:
    """Return the nav termination cfg for a robot binding."""

    @configclass
    class SortToShelfNavTerminationsCfg:
        """End nav only after the base and place-start posture are both ready."""

        time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
        at_place_start = make_nav_success_term(
            target_joint_names=target_joint_names,
            target_joint_pos_by_name=target_joint_pos_by_name,
            max_joint_abs_error=max_joint_abs_error,
            base_success_radius=base_success_radius,
            min_ready_steps=min_ready_steps,
        )

    return SortToShelfNavTerminationsCfg


__all__ = [
    "make_nav_success_term",
    "make_nav_terminations_cfg",
    "nav_place_start_reached",
    "ready_held_for_min_steps",
]
