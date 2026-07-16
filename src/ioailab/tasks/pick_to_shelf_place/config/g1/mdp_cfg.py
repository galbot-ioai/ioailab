"""Galbot G1 binding for the PickToShelf place phase MDP."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.scenario import load_scenario
from ioailab.tasks.pick_to_shelf_pick.config.g1.mdp_cfg import (
    PickToShelfManipulationActionsCfg,
    PickToShelfObservationsCfg,
)
from ioailab.tasks.pick_to_shelf_pick.mdp.rewards import PickToShelfRewardsCfg
from ioailab.tasks.pick_to_shelf_place.mdp.events import make_place_events_cfg
from ioailab.tasks.pick_to_shelf_place.mdp.terminations import (
    PickToShelfPlaceTerminationsCfg,
)

PLACE_SCENARIO = load_scenario(
    Path(__file__).with_name("scenarios") / "place_default.yaml"
)
PickToShelfPlaceEventsCfg = make_place_events_cfg(PLACE_SCENARIO)


@configclass
class PickToShelfPlaceMdpCfg:
    """Manipulation MDP for placing the cube on the shelf."""

    observations: PickToShelfObservationsCfg = PickToShelfObservationsCfg()
    actions: PickToShelfManipulationActionsCfg = PickToShelfManipulationActionsCfg()
    rewards: PickToShelfRewardsCfg = PickToShelfRewardsCfg()
    terminations: PickToShelfPlaceTerminationsCfg = PickToShelfPlaceTerminationsCfg()
    events: PickToShelfPlaceEventsCfg = PickToShelfPlaceEventsCfg()
    commands = None
    curriculum = None


__all__ = ["PickToShelfPlaceMdpCfg"]
