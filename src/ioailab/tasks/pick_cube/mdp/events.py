"""Event terms and config groups for Galbot G1 pick-cube tasks."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ioailab.randomizers import (
    DomeLightTextureRandomizer,
    ObjectPoseRandomizer,
    VisualMaterialRandomizer,
)
from ioailab.utils.asset_utils import list_hdri_paths, list_visual_material_paths


@configclass
class PickCubeEventCfg:
    """Reset events for Galbot G1 pick-cube tasks."""

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
                    "x": (-0.38, -0.22),
                    "y": (0.12, 0.26),
                    "z": (0.125, 0.125),
                    "yaw": (0.0, 0.0),
                },
            },
        },
    )
    randomize_ground_material = EventTerm(
        func=VisualMaterialRandomizer.apply,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("plane"),
            "material_paths": tuple(
                str(path)
                for path in list_visual_material_paths(
                    categories=("Concrete", "Ground", "Stone", "Ceramic")
                )
            ),
        },
    )
    randomize_table_material = EventTerm(
        func=VisualMaterialRandomizer.apply,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "material_paths": tuple(
                str(path)
                for path in list_visual_material_paths(
                    categories=("Wood", "Metal", "Plastic", "Composite")
                )
            ),
        },
    )
    randomize_hdri_texture = EventTerm(
        func=DomeLightTextureRandomizer.apply,
        mode="reset",
        params={
            "light_prim_path": "/World/ioailabLight",
            "texture_paths": tuple(str(path) for path in list_hdri_paths()),
        },
    )
