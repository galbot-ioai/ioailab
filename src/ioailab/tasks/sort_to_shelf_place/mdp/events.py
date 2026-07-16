"""Reset event helpers for the SortToShelf place phase."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.scenario import Scenario, scenario_reset_event


def make_place_events_cfg(scenario: Scenario | str | Path) -> type:
    """Return place reset events that restore the selected object scenario."""

    @configclass
    class SortToShelfPlacePolicyEventCfg:
        """Place reset events for the selected shelf-facing start."""

        reset_all = scenario_reset_event(scenario)

    return SortToShelfPlacePolicyEventCfg


__all__ = ["make_place_events_cfg"]
