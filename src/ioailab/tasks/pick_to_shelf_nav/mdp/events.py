"""Reset event helpers for the PickToShelf nav phase."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.scenario import Scenario, scenario_reset_event


def make_nav_events_cfg(scenario: Scenario | str | Path) -> type:
    """Return an events cfg class that resets from ``scenario``."""

    @configclass
    class PickToShelfNavEventsCfg:
        reset_all = scenario_reset_event(scenario)

    return PickToShelfNavEventsCfg


__all__ = ["make_nav_events_cfg"]
