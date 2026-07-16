"""Env cfg for the PickToShelf pick phase task."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.articulation import make_galbot_g1_mobile_base_articulation_cfg
from ioailab.robots.g1.sensors.camera import make_g1_camera_cfg
from ioailab.tasks.common.defaults import DefaultEnvCfg
from ioailab.tasks.pick_to_shelf.scene import PickToShelfSceneCfg
from ioailab.tasks.pick_to_shelf_pick.config.g1.mdp_cfg import (
    PickToShelfPickMdpCfg,
    PickToShelfPickTerminationsCfg,
)


@configclass
class G1PickToShelfSceneCfg(PickToShelfSceneCfg):
    """Pick-to-shelf world with mobile-base G1 and front-head camera."""

    robot = make_galbot_g1_mobile_base_articulation_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        required_asset=False,
        base_position=(-1.2, 0.0, 0.0),
        base_orientation=(0.0, 0.0, 0.0, 1.0),
        use_usd_controller_graphs=False,
    )
    robot.init_state.joint_pos.update(
        {
            "left_arm_joint1": 1.910009444500404,
            "left_arm_joint2": -1.460010959112611,
            "left_arm_joint3": -0.4741512242415168,
            "left_arm_joint4": -2.467893642457805,
            "left_arm_joint5": -0.0016785070526536992,
            "left_arm_joint6": -0.1221698763763522,
            "left_arm_joint7": -0.09424494344765931,
            "right_arm_joint1": -1.91,
            "right_arm_joint2": 1.46,
            "right_arm_joint3": 0.57,
            "right_arm_joint4": 2.10,
            "right_arm_joint5": 0.0,
            "right_arm_joint6": -0.71,
            "right_arm_joint7": -0.03,
            "head_joint1": 0.0,
            "head_joint2": 0.25,
        }
    )
    front_head_rgb_camera = make_g1_camera_cfg(
        mount="front_head",
        data="rgb",
        width=298,
        height=224,
    )


@configclass
class GalbotG1PickToShelfPickEnvCfg(PickToShelfPickMdpCfg, DefaultEnvCfg):
    """Standalone pick phase env for policy collection, training, and eval."""

    gripper_joint_names = ["left_gripper_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 1.2

    scene: G1PickToShelfSceneCfg = G1PickToShelfSceneCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    evaluation_success = PickToShelfPickTerminationsCfg().at_carry


__all__ = ["G1PickToShelfSceneCfg", "GalbotG1PickToShelfPickEnvCfg"]
