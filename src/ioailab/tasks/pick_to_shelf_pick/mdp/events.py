"""Reset events for the PickToShelf pick phase."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ioailab.randomizers import ObjectPoseRandomizer
from ioailab.tasks.pick_to_shelf.scene import CUBE_POSITION


@configclass
class PickToShelfPickPolicyEventCfg:
    """Pick reset events with tabletop cube position randomization."""

    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    randomize_pick_and_place_positions = EventTerm(
        func=ObjectPoseRandomizer.apply,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("cube")],
            "asset_pose_ranges": {
                "cube": {
                    "x": (-0.58, -0.42),
                    "y": (0.10, 0.30),
                    "z": (CUBE_POSITION[2], CUBE_POSITION[2]),
                    "yaw": (0.0, 0.0),
                },
            },
        },
    )


__all__ = ["PickToShelfPickPolicyEventCfg"]
