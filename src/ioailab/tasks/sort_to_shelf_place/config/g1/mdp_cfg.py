"""Galbot G1 binding for the SortToShelf place phase MDP."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.scenario import load_scenario
from ioailab.tasks.sort_to_shelf.scene import sorting_object_name
from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
    SortToShelfManipulationActionsCfg,
    SortToShelfObservationsCfg,
    SortToShelfPlaceTerminationsCfg,
)
from ioailab.tasks.sort_to_shelf_place.mdp.events import make_place_events_cfg


def _place_scenario(object_name: str | None):
    scenario_path = (
        Path(__file__).resolve().parent
        / "scenarios"
        / f"{sorting_object_name(object_name)}.yaml"
    )
    return load_scenario(scenario_path)


SortToShelfPlaceEventsCfg = make_place_events_cfg(_place_scenario("red_cube"))


@configclass
class SortToShelfPlaceMdpCfg:
    """MDP config for the standalone SortToShelf place phase."""

    observations: SortToShelfObservationsCfg = SortToShelfObservationsCfg()
    actions: SortToShelfManipulationActionsCfg = SortToShelfManipulationActionsCfg()
    rewards = None
    terminations: SortToShelfPlaceTerminationsCfg = SortToShelfPlaceTerminationsCfg()
    events: SortToShelfPlaceEventsCfg = SortToShelfPlaceEventsCfg()
    commands = None
    curriculum = None


__all__ = ["SortToShelfPlaceMdpCfg"]
