"""Generated MDP class exports for the coherent PickToShelf task."""

from __future__ import annotations

from ioailab.tasks.common.composition import combined_task_definition
from ioailab.tasks.pick_to_shelf.config.g1.env_cfg import GalbotG1PickToShelfEnvCfg

_DEFINITION = combined_task_definition(GalbotG1PickToShelfEnvCfg)

PickToShelfActionsCfg = _DEFINITION.actions_cfg_cls
PickToShelfFlowTerminationsCfg = _DEFINITION.terminations_cfg_cls
PickToShelfMdpCfg = _DEFINITION.mdp_cfg_cls


__all__ = [
    "PickToShelfActionsCfg",
    "PickToShelfFlowTerminationsCfg",
    "PickToShelfMdpCfg",
]
