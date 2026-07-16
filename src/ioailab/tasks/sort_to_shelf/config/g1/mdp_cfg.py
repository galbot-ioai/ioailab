"""Generated MDP class exports for the coherent SortToShelf task."""

from __future__ import annotations

from ioailab.tasks.common.composition import combined_task_definition
from ioailab.tasks.sort_to_shelf.config.g1.env_cfg import GalbotG1SortToShelfEnvCfg

_DEFINITION = combined_task_definition(GalbotG1SortToShelfEnvCfg)

SortToShelfActionsCfg = _DEFINITION.actions_cfg_cls
SortToShelfFlowTerminationsCfg = _DEFINITION.terminations_cfg_cls
SortToShelfMdpCfg = _DEFINITION.mdp_cfg_cls


__all__ = [
    "SortToShelfActionsCfg",
    "SortToShelfFlowTerminationsCfg",
    "SortToShelfMdpCfg",
]
