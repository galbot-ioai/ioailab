"""Robot-agnostic MDP terms for base navigation.

The wheel action group and the assembled ``BaseNavMdpCfg`` (which composes G1
actions) live in the G1 binding at ``base_nav/config/g1/mdp_cfg.py``. The
observation/reward/termination/event groups and terms here are robot-agnostic --
the mobile-base body is resolved from ``env.cfg.base_body_name``.
"""

from ioailab.tasks.base_nav.mdp.events import BaseNavEventCfg
from ioailab.tasks.base_nav.mdp.observations import (
    BaseNavObservationsCfg,
    base_position_xy,
    goal_position_xy,
    vector_to_goal_xy,
)
from ioailab.tasks.base_nav.mdp.rewards import (
    BaseNavRewardsCfg,
    action_l2_penalty,
    distance_to_goal,
    goal_reached_reward,
    negative_distance_reward,
)
from ioailab.tasks.base_nav.mdp.terminations import (
    BaseNavTerminationsCfg,
    goal_reached,
)

__all__ = [
    "BaseNavEventCfg",
    "BaseNavObservationsCfg",
    "BaseNavRewardsCfg",
    "BaseNavTerminationsCfg",
    "action_l2_penalty",
    "base_position_xy",
    "distance_to_goal",
    "goal_position_xy",
    "goal_reached",
    "goal_reached_reward",
    "negative_distance_reward",
    "vector_to_goal_xy",
]
