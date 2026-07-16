"""Galbot G1 binding for the base-navigation MDP.

The wheel velocity action group names the G1 base wheels, so it lives here with
the G1 config; the assembled ``BaseNavMdpCfg`` composes it with the
robot-agnostic observation, reward, termination, and event groups imported up
from ``base_nav/mdp/``.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.actions import (
    DEFAULT_BASE_WHEEL_RADIUS,
    G1_BASE_WHEEL_DOF_ORDER,
    g1_action_cfg,
)
from ioailab.tasks.base_nav.mdp.events import BaseNavEventCfg
from ioailab.tasks.base_nav.mdp.observations import BaseNavObservationsCfg
from ioailab.tasks.base_nav.mdp.rewards import BaseNavRewardsCfg
from ioailab.tasks.base_nav.mdp.terminations import BaseNavTerminationsCfg

# Wheel speed cap matching GalbotG1BaseNavEnvCfg.max_command_speed (m/s).
_MAX_WHEEL_VELOCITY = 0.45 / DEFAULT_BASE_WHEEL_RADIUS


@configclass
class BaseNavActionsCfg:
    """Wheel velocity action for the G1 mobile base."""

    base_action = g1_action_cfg(
        "base",
        "velocity",
        clip={
            joint_name: (-_MAX_WHEEL_VELOCITY, _MAX_WHEEL_VELOCITY)
            for joint_name in G1_BASE_WHEEL_DOF_ORDER
        },
    )


@configclass
class BaseNavMdpCfg:
    """Top-level manager configs for base navigation."""

    observations: BaseNavObservationsCfg = BaseNavObservationsCfg()
    actions: BaseNavActionsCfg = BaseNavActionsCfg()
    rewards: BaseNavRewardsCfg = BaseNavRewardsCfg()
    terminations: BaseNavTerminationsCfg = BaseNavTerminationsCfg()
    events: BaseNavEventCfg = BaseNavEventCfg()
    commands = None
    curriculum = None


__all__ = [
    "BaseNavActionsCfg",
    "BaseNavMdpCfg",
]
