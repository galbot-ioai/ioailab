"""Galbot G1 binding for the PickToShelf pick phase MDP."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.actions import (
    G1_LEFT_ARM_DOF_ORDER,
    g1_action_cfg,
)
from ioailab.tasks.pick_to_shelf_pick.mdp.events import PickToShelfPickPolicyEventCfg
from ioailab.tasks.pick_to_shelf_pick.mdp.observations import (
    make_pick_observations_cfg,
)
from ioailab.tasks.pick_to_shelf_pick.mdp.rewards import PickToShelfRewardsCfg
from ioailab.tasks.pick_to_shelf_pick.mdp.terminations import (
    make_pick_terminations_cfg,
)

PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER = (
    "leg_joint1",
    "wheel1_joint",
    "wheel2_joint",
    "wheel3_joint",
    "wheel4_joint",
    "leg_joint2",
    "wheel_1_passive_0_joint",
    "wheel_1_passive_1_joint",
    "wheel_1_passive_2_joint",
    "wheel_1_passive_3_joint",
    "wheel_1_passive_4_joint",
    "wheel_1_passive_5_joint",
    "wheel_1_passive_6_joint",
    "wheel_1_passive_7_joint",
    "wheel_1_passive_8_joint",
    "wheel_1_passive_9_joint",
    "wheel_2_passive_0_joint",
    "wheel_2_passive_1_joint",
    "wheel_2_passive_2_joint",
    "wheel_2_passive_3_joint",
    "wheel_2_passive_4_joint",
    "wheel_2_passive_5_joint",
    "wheel_2_passive_6_joint",
    "wheel_2_passive_7_joint",
    "wheel_2_passive_8_joint",
    "wheel_2_passive_9_joint",
    "wheel_3_passive_0_joint",
    "wheel_3_passive_1_joint",
    "wheel_3_passive_2_joint",
    "wheel_3_passive_3_joint",
    "wheel_3_passive_4_joint",
    "wheel_3_passive_5_joint",
    "wheel_3_passive_6_joint",
    "wheel_3_passive_7_joint",
    "wheel_3_passive_8_joint",
    "wheel_3_passive_9_joint",
    "wheel_4_passive_0_joint",
    "wheel_4_passive_1_joint",
    "wheel_4_passive_2_joint",
    "wheel_4_passive_3_joint",
    "wheel_4_passive_4_joint",
    "wheel_4_passive_5_joint",
    "wheel_4_passive_6_joint",
    "wheel_4_passive_7_joint",
    "wheel_4_passive_8_joint",
    "wheel_4_passive_9_joint",
    "leg_joint3",
    "leg_joint4",
    "leg_joint5",
    "head_joint1",
    "left_arm_joint1",
    "right_arm_joint1",
    "head_joint2",
    "left_arm_joint2",
    "right_arm_joint2",
    "left_arm_joint3",
    "right_arm_joint3",
    "left_arm_joint4",
    "right_arm_joint4",
    "left_arm_joint5",
    "right_arm_joint5",
    "left_arm_joint6",
    "right_arm_joint6",
    "left_arm_joint7",
    "right_arm_joint7",
    "left_gripper_joint",
    "left_gripper_l_inner_knuckle_joint",
    "left_gripper_l_knuckle_joint",
    "left_gripper_r_inner_knuckle_joint",
    "right_gripper_joint",
    "right_gripper_l_inner_knuckle_joint",
    "right_gripper_l_knuckle_joint",
    "right_gripper_r_inner_knuckle_joint",
    "left_gripper_l_finger_joint",
    "left_gripper_r_finger_joint",
    "right_gripper_l_finger_joint",
    "right_gripper_r_finger_joint",
)

PICK_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS = tuple(
    joint_name
    for joint_name in PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER
    if joint_name.startswith("wheel")
)

PICK_TO_SHELF_CARRY_JOINT_POS_BY_NAME = {
    "left_arm_joint1": 1.910009444500404,
    "left_arm_joint2": -1.460010959112611,
    "left_arm_joint3": -0.4741512242415168,
    "left_arm_joint4": -2.467893642457805,
    "left_arm_joint5": -0.0016785070526536992,
    "left_arm_joint6": -0.1221698763763522,
    "left_arm_joint7": -0.09424494344765931,
}


@configclass
class PickToShelfManipulationActionsCfg:
    """Absolute left-arm and left-gripper actions for phase policy data."""

    arm_action = g1_action_cfg("left_arm", "absolute")
    gripper_action = g1_action_cfg("left_gripper", "absolute")
    leg_action = None


PickToShelfObservationsCfg = make_pick_observations_cfg(
    joint_names=PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER,
    neutral_joint_names=PICK_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS,
    camera_name="front_head_rgb_camera",
)
PickToShelfPickTerminationsCfg = make_pick_terminations_cfg(
    target_joint_pos_by_name=PICK_TO_SHELF_CARRY_JOINT_POS_BY_NAME,
    required_joint_names=G1_LEFT_ARM_DOF_ORDER,
)


@configclass
class PickToShelfPickMdpCfg:
    """Manipulation MDP for lifting the cube to the carry posture."""

    observations: PickToShelfObservationsCfg = PickToShelfObservationsCfg()
    actions: PickToShelfManipulationActionsCfg = PickToShelfManipulationActionsCfg()
    rewards: PickToShelfRewardsCfg = PickToShelfRewardsCfg()
    terminations: PickToShelfPickTerminationsCfg = PickToShelfPickTerminationsCfg()
    events: PickToShelfPickPolicyEventCfg = PickToShelfPickPolicyEventCfg()
    commands = None
    curriculum = None


__all__ = [
    "PICK_TO_SHELF_CARRY_JOINT_POS_BY_NAME",
    "PICK_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS",
    "PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER",
    "PickToShelfManipulationActionsCfg",
    "PickToShelfObservationsCfg",
    "PickToShelfPickMdpCfg",
    "PickToShelfPickTerminationsCfg",
]
