"""Reward terms and config for the G1 base navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from ioailab.tasks.base_nav.mdp.observations import vector_to_goal_xy
from ioailab.tasks.base_nav.mdp.terminations import goal_reached

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_to_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return Euclidean XY distance from base to goal."""

    return torch.linalg.vector_norm(vector_to_goal_xy(env), dim=1)


def negative_distance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward smaller goal distance."""

    return -distance_to_goal(env)


def goal_reached_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward successful arrival at the goal."""

    return goal_reached(env).to(torch.float32)


def action_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize high wheel velocity commands."""

    return torch.sum(torch.square(env.action_manager.action), dim=1)


@configclass
class BaseNavRewardsCfg:
    """Dense distance reward with a success bonus."""

    distance_to_goal = RewTerm(func=negative_distance_reward, weight=2.0)
    goal_reached = RewTerm(func=goal_reached_reward, weight=10.0)
    action_l2 = RewTerm(func=action_l2_penalty, weight=-0.02)
