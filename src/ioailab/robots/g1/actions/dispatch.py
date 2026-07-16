"""Unified G1 runtime action dispatch.

Provides ``g1_action()`` — a single function for all G1 action tensor
packing at runtime, mirroring the config-time ``g1_action_cfg()`` interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ioailab.robots.g1.actions.pack import (
    JointValueInput,
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

_ABSOLUTE_PACKERS: dict[str, Any] = {
    "legs": pack_g1_legs_absolute_joint_command,
    "left_arm": pack_g1_left_arm_absolute_joint_command,
    "right_arm": pack_g1_right_arm_absolute_joint_command,
}

_RELATIVE_PACKERS: dict[str, Any] = {
    "legs": pack_g1_legs_relative_joint_command,
    "left_arm": pack_g1_left_arm_relative_joint_command,
    "right_arm": pack_g1_right_arm_relative_joint_command,
}

_BINARY_PACKERS: dict[str, Any] = {
    "left_gripper": pack_g1_left_gripper_binary_command,
    "right_gripper": pack_g1_right_gripper_binary_command,
}

_ALL_GROUPS = frozenset(
    ("base", "legs", "left_arm", "right_arm", "left_gripper", "right_gripper")
)


def g1_action(
    group: str,
    action_type: str,
    *,
    joint_names: str | Sequence[str] | None = None,
    values: JointValueInput | None = None,
    is_open: bool | None = None,
    vx: JointValueInput = 0.0,
    vy: JointValueInput = 0.0,
    wz: JointValueInput = 0.0,
    env: Any | None = None,
    asset_name: str = "robot",
    baseline: JointValueInput | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch a G1 action command to the appropriate tensor packer.

    Args:
        group: Body group — "base", "legs", "left_arm", "right_arm",
            "left_gripper", or "right_gripper".
        action_type: Action mode — "absolute", "relative", "velocity",
            or "binary".
        joint_names: Joint names to command (for absolute/relative on
            legs, left_arm, right_arm).
        values: Target values for the specified joints.
        is_open: Gripper state for binary gripper commands.
        vx: Forward velocity for base velocity commands.
        vy: Lateral velocity for base velocity commands.
        wz: Yaw velocity for base velocity commands.
        env: Optional IsaacLab environment for tensor context.
        asset_name: Robot asset name.
        baseline: Baseline joint positions for absolute commands.
        env_indices: Environment rows to command.
        num_envs: Number of environments when env is not provided.
        device: Tensor device when env is not provided.
        dtype: Output tensor dtype.
        **kwargs: Additional keyword arguments forwarded to the packer.

    Returns:
        Action tensor for one IsaacLab action term.
    """

    if group not in _ALL_GROUPS:
        raise ValueError(
            f"Unknown G1 action group {group!r}. Available: {sorted(_ALL_GROUPS)}"
        )

    if group == "base" and action_type == "velocity":
        return pack_g1_base_velocity_command(
            vx=vx,
            vy=vy,
            wz=wz,
            env=env,
            env_indices=env_indices,
            num_envs=num_envs,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    if action_type == "binary":
        packer = _BINARY_PACKERS.get(group)
        if packer is None:
            raise ValueError(
                f"Action type 'binary' is only valid for gripper groups, got {group!r}."
            )
        if is_open is None:
            raise TypeError("is_open is required for binary gripper commands.")
        return packer(
            is_open,
            env=env,
            asset_name=asset_name,
            baseline=baseline,
            env_indices=env_indices,
            num_envs=num_envs,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    if action_type == "absolute":
        packer = _ABSOLUTE_PACKERS.get(group)
        if packer is None:
            raise ValueError(
                f"Action type 'absolute' is not supported for {group!r}. "
                f"Use 'binary' for gripper groups or 'velocity' for base."
            )
        if joint_names is None or values is None:
            raise TypeError(
                "joint_names and values are required for absolute joint commands."
            )
        return packer(
            joint_names,
            values,
            env=env,
            asset_name=asset_name,
            baseline=baseline,
            env_indices=env_indices,
            num_envs=num_envs,
            device=device,
            dtype=dtype,
        )

    if action_type == "relative":
        packer = _RELATIVE_PACKERS.get(group)
        if packer is None:
            raise ValueError(
                f"Action type 'relative' is not supported for {group!r}. "
                f"Use 'binary' for gripper groups or 'velocity' for base."
            )
        if joint_names is None or values is None:
            raise TypeError(
                "joint_names and values are required for relative joint commands."
            )
        return packer(
            joint_names,
            values,
            env=env,
            env_indices=env_indices,
            num_envs=num_envs,
            device=device,
            dtype=dtype,
        )

    valid_types = _valid_action_types(group)
    raise ValueError(
        f"Unknown action type {action_type!r} for group {group!r}. Valid: {valid_types}"
    )


def _valid_action_types(group: str) -> list[str]:
    types = []
    if group == "base":
        types.append("velocity")
    if group in _ABSOLUTE_PACKERS:
        types.extend(["absolute", "relative"])
    if group in _BINARY_PACKERS:
        types.append("binary")
    return types
