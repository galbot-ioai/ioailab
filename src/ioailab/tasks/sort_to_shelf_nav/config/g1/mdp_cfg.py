"""Galbot G1 binding for the SortToShelf navigation phase MDP."""

from __future__ import annotations

from pathlib import Path

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.actions import (
    DEFAULT_BASE_WHEEL_RADIUS,
    G1_BASE_WHEEL_DOF_ORDER,
    G1_LEG_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER,
    g1_action_cfg,
)
from ioailab.tasks.base_nav.mdp.observations import BaseNavObservationsCfg
from ioailab.tasks.base_nav.mdp.rewards import BaseNavRewardsCfg
from ioailab.tasks.common.scenario import load_scenario
from ioailab.tasks.sort_to_shelf.scene import (
    sorting_object_name,
    sorting_object_requires_leg_lift,
)
from ioailab.tasks.sort_to_shelf_nav.mdp.events import make_nav_events_cfg
from ioailab.tasks.sort_to_shelf_nav.mdp.terminations import (
    make_nav_success_term as _make_nav_success_term,
    make_nav_terminations_cfg,
)
from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
    SORTING_A_CELL_LEG_LIFT_JOINT_POS,
    SORTING_DEFAULT_LEG_JOINT_POS,
    sorting_place_approach_left_arm_joint_pos_for_object,
)


def _nav_scenario(object_name: str | None):
    scenario_path = (
        Path(__file__).resolve().parent
        / "scenarios"
        / f"{sorting_object_name(object_name)}.yaml"
    )
    return load_scenario(scenario_path)


SortToShelfNavEventsCfg = make_nav_events_cfg(_nav_scenario("red_cube"))


def _base_wheel_velocity_clip() -> dict[str, tuple[float, float]]:
    max_wheel_velocity = 0.45 / DEFAULT_BASE_WHEEL_RADIUS
    return {
        joint_name: (-max_wheel_velocity, max_wheel_velocity)
        for joint_name in G1_BASE_WHEEL_DOF_ORDER
    }


@configclass
class SortToShelfNavActionsCfg:
    """Base drive plus place-start posture actions for navigation data."""

    base_action = g1_action_cfg("base", "velocity", clip=_base_wheel_velocity_clip())
    leg_action = g1_action_cfg("legs", "absolute")
    arm_action = g1_action_cfg("left_arm", "absolute")


SORT_TO_SHELF_NAV_PLACE_START_JOINT_NAMES = (
    *G1_LEG_DOF_ORDER,
    *G1_LEFT_ARM_DOF_ORDER,
)
SORT_TO_SHELF_NAV_POSTURE_MAX_JOINT_ABS_ERROR = 0.12
SORT_TO_SHELF_NAV_PLACE_START_BASE_SUCCESS_RADIUS = 0.05
SORT_TO_SHELF_NAV_PLACE_START_MIN_READY_STEPS = 20


def sorting_place_start_joint_pos_for_object(
    object_name: str | None,
) -> dict[str, float]:
    """Return leg and left-arm targets for the nav phase place-start posture."""

    resolved = sorting_object_name(object_name)
    leg_targets = (
        SORTING_A_CELL_LEG_LIFT_JOINT_POS
        if sorting_object_requires_leg_lift(resolved)
        else SORTING_DEFAULT_LEG_JOINT_POS
    )
    return {
        **dict(leg_targets),
        **sorting_place_approach_left_arm_joint_pos_for_object(resolved),
    }


def make_nav_success_term(object_name: str | None = "red_cube") -> DoneTerm:
    """Return nav success after base arrival and place-start posture settling."""

    return _make_nav_success_term(
        target_joint_names=SORT_TO_SHELF_NAV_PLACE_START_JOINT_NAMES,
        target_joint_pos_by_name=sorting_place_start_joint_pos_for_object(object_name),
        max_joint_abs_error=SORT_TO_SHELF_NAV_POSTURE_MAX_JOINT_ABS_ERROR,
        base_success_radius=SORT_TO_SHELF_NAV_PLACE_START_BASE_SUCCESS_RADIUS,
        min_ready_steps=SORT_TO_SHELF_NAV_PLACE_START_MIN_READY_STEPS,
    )


SortToShelfNavTerminationsCfg = make_nav_terminations_cfg(
    target_joint_names=SORT_TO_SHELF_NAV_PLACE_START_JOINT_NAMES,
    target_joint_pos_by_name=sorting_place_start_joint_pos_for_object("red_cube"),
    max_joint_abs_error=SORT_TO_SHELF_NAV_POSTURE_MAX_JOINT_ABS_ERROR,
    base_success_radius=SORT_TO_SHELF_NAV_PLACE_START_BASE_SUCCESS_RADIUS,
    min_ready_steps=SORT_TO_SHELF_NAV_PLACE_START_MIN_READY_STEPS,
)


@configclass
class SortToShelfNavMdpCfg:
    """Navigation MDP with base velocity commands toward the sorting shelf."""

    observations: BaseNavObservationsCfg = BaseNavObservationsCfg()
    actions: SortToShelfNavActionsCfg = SortToShelfNavActionsCfg()
    rewards: BaseNavRewardsCfg = BaseNavRewardsCfg()
    terminations: SortToShelfNavTerminationsCfg = SortToShelfNavTerminationsCfg()
    events: SortToShelfNavEventsCfg = SortToShelfNavEventsCfg()
    commands = None
    curriculum = None


__all__ = [
    "SORT_TO_SHELF_NAV_PLACE_START_JOINT_NAMES",
    "SORT_TO_SHELF_NAV_PLACE_START_BASE_SUCCESS_RADIUS",
    "SORT_TO_SHELF_NAV_POSTURE_MAX_JOINT_ABS_ERROR",
    "SORT_TO_SHELF_NAV_PLACE_START_MIN_READY_STEPS",
    "SortToShelfNavActionsCfg",
    "SortToShelfNavMdpCfg",
    "SortToShelfNavTerminationsCfg",
    "make_nav_success_term",
    "sorting_place_start_joint_pos_for_object",
]
