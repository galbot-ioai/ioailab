"""Termination terms and config for the G1 base navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ioailab.tasks.base_nav.mdp.observations import vector_to_goal_xy

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether the base is within the configured goal radius."""

    distance_to_goal = torch.linalg.vector_norm(vector_to_goal_xy(env), dim=1)
    return distance_to_goal <= float(env.cfg.success_radius)


@configclass
class BaseNavTerminationsCfg:
    """End the episode on timeout or goal arrival."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=goal_reached)
