"""Robot-agnostic IsaacLab joint action config helpers."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs.mdp.actions import (
    JointPositionActionCfg,
    JointVelocityActionCfg,
    RelativeJointPositionActionCfg,
)


def make_relative_joint_position_action_cfg(
    *,
    asset_name: str,
    joint_names: Sequence[str],
    scale: float,
) -> RelativeJointPositionActionCfg:
    """Build a relative position cfg while preserving the given joint order."""

    return RelativeJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=list(joint_names),
        scale=scale,
        preserve_order=True,
        use_zero_offset=True,
    )


def make_absolute_joint_position_action_cfg(
    *,
    asset_name: str,
    joint_names: Sequence[str],
    scale: float,
) -> JointPositionActionCfg:
    """Build an absolute position cfg while preserving the given joint order."""

    return JointPositionActionCfg(
        asset_name=asset_name,
        joint_names=list(joint_names),
        scale=scale,
        preserve_order=True,
        use_default_offset=False,
    )


def make_joint_velocity_action_cfg(
    *,
    asset_name: str,
    joint_names: Sequence[str],
    scale: float,
    clip: dict[str, tuple[float, float]] | None = None,
) -> JointVelocityActionCfg:
    """Build a velocity cfg while preserving the given joint order."""

    return JointVelocityActionCfg(
        asset_name=asset_name,
        joint_names=list(joint_names),
        scale=scale,
        clip=clip,
        preserve_order=True,
        use_default_offset=False,
    )
