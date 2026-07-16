"""Reset event helpers for the SortToShelf navigation phase."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.scenario import Scenario, scenario_reset_event


def make_nav_events_cfg(scenario: Scenario | str | Path) -> type:
    """Return nav reset events that restore the selected object scenario."""

    @configclass
    class SortToShelfNavEventsCfg:
        """Navigation reset events for the carry phase start."""

        reset_all = scenario_reset_event(scenario)

    return SortToShelfNavEventsCfg


__all__ = ["make_nav_events_cfg"]
