"""Reset events for the G1 base navigation task."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.configclass import configclass


@configclass
class BaseNavEventCfg:
    """Reset scene state at episode start."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
