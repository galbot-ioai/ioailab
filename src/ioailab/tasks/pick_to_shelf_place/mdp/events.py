"""Reset event helpers for the PickToShelf place phase."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.scenario import Scenario, scenario_reset_event


def make_place_events_cfg(scenario: Scenario | str | Path) -> type:
    """Return an events cfg class that resets from ``scenario``."""

    @configclass
    class PickToShelfPlaceEventsCfg:
        reset_all = scenario_reset_event(scenario)

    return PickToShelfPlaceEventsCfg


__all__ = ["make_place_events_cfg"]
