"""Reset events for Galbot G1 stack-cube task variants."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

from ioailab.randomizers import (
    DomeLightTextureRandomizer,
    ObjectPoseRandomizer,
    VisualMaterialRandomizer,
)
from ioailab.utils.asset_utils import list_hdri_paths, list_visual_material_paths


@configclass
class StackCubeEventCfg:
    """Reset events for the G1 stack-cube scene."""

    reset_all = EventTerm(
        func=stack_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    randomize_cube_positions = EventTerm(
        func=ObjectPoseRandomizer.apply,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.35, -0.25),
                "y": (0.14, 0.23),
                "z": (0.125, 0.125),
                "yaw": (-1.0, 1.0),
            },
            "min_separation": 0.09,
            "asset_cfgs": [
                SceneEntityCfg("cube_1"),
                SceneEntityCfg("cube_2"),
                SceneEntityCfg("cube_3"),
            ],
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
