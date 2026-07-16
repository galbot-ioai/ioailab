"""Galbot G1 robot profile for agent construction."""

from __future__ import annotations

from ioailab.agents.robot_profile import RobotProfile
from ioailab.robots.g1.actions import pack_g1_base_velocity_command
from ioailab.robots.g1.spec import (
    G1_BASE_WHEEL_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER,
    G1_LEFT_GRIPPER_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER,
    G1_RIGHT_GRIPPER_DOF_ORDER,
)
from ioailab.robots.g1.articulation import G1_MOBILE_BASE_BODY_NAME

G1_PROFILE = RobotProfile(
    name="galbot_g1",
    base_velocity_packer=pack_g1_base_velocity_command,
    base_body_name=G1_MOBILE_BASE_BODY_NAME,
    base_wheel_dof_names=G1_BASE_WHEEL_DOF_ORDER,
    arm_dof_names={
        "left": G1_LEFT_ARM_DOF_ORDER,
        "right": G1_RIGHT_ARM_DOF_ORDER,
    },
    gripper_dof_names={
        "left": G1_LEFT_GRIPPER_DOF_ORDER,
        "right": G1_RIGHT_GRIPPER_DOF_ORDER,
    },
    default_arm="left",
    default_max_nav_speed=0.45,
    default_nav_success_radius=0.15,
)
