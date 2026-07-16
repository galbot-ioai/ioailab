"""Reward config for the PickToShelf pick phase."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass


@configclass
class PickToShelfRewardsCfg:
    """No reward terms for motion-planning execution."""


__all__ = ["PickToShelfRewardsCfg"]
