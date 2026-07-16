"""Shared robot-agnostic MDP terms for SortToShelf phase tasks."""

from ioailab.tasks.sort_to_shelf.mdp.terminations import (
    SortToShelfTimeOutTerminationsCfg,
    joints_at_named_targets,
)

__all__ = [
    "SortToShelfTimeOutTerminationsCfg",
    "joints_at_named_targets",
]
