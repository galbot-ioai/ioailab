"""G1 IsaacLab action config factories and tensor packers.

Two stages:

1. Static config: use ``g1_action_cfg(group, type)`` to build action term cfgs
   and assign them to ``env_cfg.actions`` before env construction.
2. Runtime packing: call ``pack_g1_*_command()`` to build per-term tensors,
   then concatenate them in env_cfg.actions order before ``env.step(action)``.
"""

from ioailab.robots.g1.actions.core import (
    G1Actions,
    absolute,
    baselink,
    binary,
    left_arm,
    left_gripper,
    legs,
    relative,
    right_arm,
    right_gripper,
    velocity,
)
from ioailab.robots.g1.spec import (
    DEFAULT_BASE_WHEEL_RADIUS,
    DEFAULT_BASE_WHEEL_X,
    DEFAULT_BASE_WHEEL_Y,
    DEFAULT_GRIPPER_CLOSED_POSITION,
    DEFAULT_GRIPPER_OPEN_POSITION,
    G1_GRIPPER_VELOCITY_LIMIT_SIM,
)
from ioailab.robots.g1.actions.dispatch import g1_action
from ioailab.robots.g1.actions.pack import (
    G1_BASE_WHEEL_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER,
    G1_LEFT_GRIPPER_DOF_ORDER,
    G1_LEG_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER,
    G1_RIGHT_GRIPPER_DOF_ORDER,
    pack_g1_base_velocity_command,
    pack_g1_left_arm_absolute_joint_command,
    pack_g1_left_arm_relative_joint_command,
    pack_g1_left_gripper_binary_command,
    pack_g1_legs_absolute_joint_command,
    pack_g1_legs_relative_joint_command,
    pack_g1_right_arm_absolute_joint_command,
    pack_g1_right_arm_relative_joint_command,
    pack_g1_right_gripper_binary_command,
)

_CFG_EXPORTS = {"G1_ACTION_GROUPS", "g1_action_cfg"}


def __getattr__(name: str):
    """Load IsaacLab action config factories only when requested."""

    if name in _CFG_EXPORTS:
        from ioailab.robots.g1.actions import cfg

        value = getattr(cfg, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "G1Actions",
    "absolute",
    "baselink",
    "binary",
    "left_arm",
    "left_gripper",
    "legs",
    "relative",
    "right_arm",
    "right_gripper",
    "velocity",
    "DEFAULT_BASE_WHEEL_RADIUS",
    "DEFAULT_BASE_WHEEL_X",
    "DEFAULT_BASE_WHEEL_Y",
    "DEFAULT_GRIPPER_CLOSED_POSITION",
    "DEFAULT_GRIPPER_OPEN_POSITION",
    "G1_ACTION_GROUPS",
    "G1_BASE_WHEEL_DOF_ORDER",
    "G1_GRIPPER_VELOCITY_LIMIT_SIM",
    "G1_LEFT_ARM_DOF_ORDER",
    "G1_LEFT_GRIPPER_DOF_ORDER",
    "G1_LEG_DOF_ORDER",
    "G1_RIGHT_ARM_DOF_ORDER",
    "G1_RIGHT_GRIPPER_DOF_ORDER",
    "g1_action",
    "g1_action_cfg",
    "pack_g1_base_velocity_command",
    "pack_g1_left_arm_absolute_joint_command",
    "pack_g1_left_arm_relative_joint_command",
    "pack_g1_left_gripper_binary_command",
    "pack_g1_legs_absolute_joint_command",
    "pack_g1_legs_relative_joint_command",
    "pack_g1_right_arm_absolute_joint_command",
    "pack_g1_right_arm_relative_joint_command",
    "pack_g1_right_gripper_binary_command",
]
