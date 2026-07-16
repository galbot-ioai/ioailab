"""Galbot G1 binding for the PickToShelf navigation phase MDP."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.actions import (
    DEFAULT_BASE_WHEEL_RADIUS,
    G1_BASE_WHEEL_DOF_ORDER,
    g1_action_cfg,
)
from ioailab.tasks.base_nav.mdp.observations import BaseNavObservationsCfg
from ioailab.tasks.base_nav.mdp.rewards import BaseNavRewardsCfg
from ioailab.tasks.base_nav.mdp.terminations import BaseNavTerminationsCfg
from ioailab.tasks.common.scenario import load_scenario
from ioailab.tasks.pick_to_shelf_nav.mdp.events import make_nav_events_cfg

NAV_SCENARIO = load_scenario(Path(__file__).with_name("scenarios") / "nav_default.yaml")
PickToShelfNavEventsCfg = make_nav_events_cfg(NAV_SCENARIO)


def _base_wheel_velocity_clip() -> dict[str, tuple[float, float]]:
    max_wheel_velocity = 0.45 / DEFAULT_BASE_WHEEL_RADIUS
    return {
        joint_name: (-max_wheel_velocity, max_wheel_velocity)
        for joint_name in G1_BASE_WHEEL_DOF_ORDER
    }


@configclass
class PickToShelfBaseVelocityActionsCfg:
    """Mobile-base velocity action for navigation-only policy data."""

    base_action = g1_action_cfg("base", "velocity", clip=_base_wheel_velocity_clip())


@configclass
class PickToShelfNavMdpCfg:
    """Navigation MDP with base velocity commands toward the shelf."""

    observations: BaseNavObservationsCfg = BaseNavObservationsCfg()
    actions: PickToShelfBaseVelocityActionsCfg = PickToShelfBaseVelocityActionsCfg()
    rewards: BaseNavRewardsCfg = BaseNavRewardsCfg()
    terminations: BaseNavTerminationsCfg = BaseNavTerminationsCfg()
    events: PickToShelfNavEventsCfg = PickToShelfNavEventsCfg()
    commands = None
    curriculum = None


__all__ = [
    "PickToShelfBaseVelocityActionsCfg",
    "PickToShelfNavMdpCfg",
]
