"""Robot-generic IsaacLab action config and tensor helpers."""

from ioailab.robots.common.actions.core import BaseActions
from ioailab.robots.common.actions.pack import (
    JointValueInput,
    current_joint_positions_from_env,
    normalize_env_indices,
    pack_absolute_joint_command,
    pack_joint_value_command,
    pack_relative_joint_command,
    resolve_tensor_context,
)

_CFG_EXPORTS = {
    "make_absolute_joint_position_action_cfg",
    "make_joint_velocity_action_cfg",
    "make_relative_joint_position_action_cfg",
}


def __getattr__(name: str):
    """Load IsaacLab action config factories only when requested."""

    if name in _CFG_EXPORTS:
        from ioailab.robots.common.actions import cfg

        value = getattr(cfg, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseActions",
    "JointValueInput",
    "current_joint_positions_from_env",
    "normalize_env_indices",
    "make_absolute_joint_position_action_cfg",
    "make_joint_velocity_action_cfg",
    "make_relative_joint_position_action_cfg",
    "pack_absolute_joint_command",
    "pack_joint_value_command",
    "pack_relative_joint_command",
    "resolve_tensor_context",
]
