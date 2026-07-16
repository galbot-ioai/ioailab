"""Robot-agnostic MDP terms for the pick-cube task.

Action and observation groups (which name G1 entities) live in the G1 binding at
``pick_cube/config/g1/mdp_cfg.py``; this package holds only robot-agnostic event
and termination terms.
"""

from ioailab.tasks.pick_cube.mdp.events import PickCubeEventCfg
from ioailab.tasks.pick_cube.mdp.terminations import (
    PickCubeMimicSuccessCfg,
    PickCubeTerminationsCfg,
    make_pick_cube_evaluation_success_term,
    make_pick_cube_release_termination_term,
)

__all__ = [
    "PickCubeEventCfg",
    "PickCubeMimicSuccessCfg",
    "PickCubeTerminationsCfg",
    "make_pick_cube_evaluation_success_term",
    "make_pick_cube_release_termination_term",
]
