"""Reset events for the SortToShelf pick phase."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ioailab.randomizers import ObjectSlotAssignmentRandomizer
from ioailab.tasks.sort_to_shelf.scene import (
    SORTING_OBJECT_NAMES,
    SORTING_OBJECT_POSITIONS,
)


def make_sort_to_shelf_object_randomization_event() -> EventTerm:
    """Return reset-time slot permutation for the four-object sorting scene."""

    return EventTerm(
        func=ObjectSlotAssignmentRandomizer.apply,
        mode="reset",
        params={
            "asset_cfgs": [
                SceneEntityCfg(object_name) for object_name in SORTING_OBJECT_NAMES
            ],
            "slot_positions": SORTING_OBJECT_POSITIONS,
        },
    )


@configclass
class SortToShelfPickPolicyEventCfg:
    """Sorting pick-policy reset events for the four tabletop objects."""

    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    randomize_pick_and_place_positions = make_sort_to_shelf_object_randomization_event()


__all__ = [
    "SortToShelfPickPolicyEventCfg",
    "make_sort_to_shelf_object_randomization_event",
]
