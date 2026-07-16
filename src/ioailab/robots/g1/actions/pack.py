"""Dynamic G1 action tensor packing helpers for IsaacLab ``env.step`` calls.

Packers are the runtime half of the ioailab action API. User code gives a
plain command, such as joint names plus target values, and receives one
``torch.Tensor`` slice for one configured IsaacLab action term.

Usage rules:
    * Configure the matching action term once with
      :mod:`ioailab.robots.g1.actions.cfg` before creating the environment.
    * Call packers inside the control loop to build dynamic command tensors.
    * Concatenate packed tensors in the same order as ``env_cfg.actions``.
    * Pass the concatenated tensor directly to ``env.step(action_tensor)``.

Relative leg/arm packers output deltas and leave unspecified joints at zero.
Absolute leg/arm packers output final joint targets; sparse absolute commands
need ``env`` or ``baseline`` so unspecified joints can hold their current value.
Gripper bool packers control only the master gripper joint; the USD/PhysX asset
owns the detailed finger motion.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Any

import torch

from ioailab.robots.common.actions.pack import (
    JointValueInput,
    pack_absolute_joint_command,
    pack_joint_value_command,
    pack_relative_joint_command,
    resolve_tensor_context,
    normalize_env_indices,
)
from ioailab.robots.g1.spec import (
    DEFAULT_BASE_WHEEL_RADIUS,
    DEFAULT_BASE_WHEEL_X,
    DEFAULT_BASE_WHEEL_Y,
    DEFAULT_GRIPPER_CLOSED_POSITION,
    DEFAULT_GRIPPER_OPEN_POSITION,
    DEFAULT_ROBOT_ASSET_NAME,
)

from ioailab.robots.g1.spec import (
    G1_BASE_WHEEL_DOF_ORDER,
    G1_LEG_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER,
    G1_LEFT_GRIPPER_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER,
    G1_RIGHT_GRIPPER_DOF_ORDER,
)


DEFAULT_BASE_LINEAR_VELOCITY_SCALE = 1.0
DEFAULT_BASE_ANGULAR_VELOCITY_SCALE = 1.0


def pack_g1_base_velocity_command(
    *,
    vx: JointValueInput = 0.0,
    vy: JointValueInput = 0.0,
    wz: JointValueInput = 0.0,
    env: Any | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    wheel_radius: float = DEFAULT_BASE_WHEEL_RADIUS,
    wheel_x: float = DEFAULT_BASE_WHEEL_X,
    wheel_y: float = DEFAULT_BASE_WHEEL_Y,
    linear_velocity_scale: float = DEFAULT_BASE_LINEAR_VELOCITY_SCALE,
    angular_velocity_scale: float = DEFAULT_BASE_ANGULAR_VELOCITY_SCALE,
) -> torch.Tensor:
    """Pack a G1 base-level velocity command into wheel velocity actions.

    Args:
        vx: Desired base forward velocity in meters per second.
        vy: Desired base lateral velocity in meters per second.
        wz: Desired base yaw velocity in radians per second.
        env: Optional IsaacLab environment used to infer ``num_envs`` and
            ``device``.
        env_indices: Optional environment rows to command. Non-selected rows
            receive zero wheel velocity.
        num_envs: Number of environments when ``env`` is not provided.
        device: Tensor device when ``env`` is not provided.
        dtype: Output tensor dtype.
        wheel_radius: Main wheel radius in meters from the G1 USD metadata.
        wheel_x: Absolute x offset of each wheel from the chassis center.
        wheel_y: Absolute y offset of each wheel from the chassis center.
        linear_velocity_scale: Debug/tuning scale for translational velocity.
        angular_velocity_scale: Debug/tuning scale for yaw velocity.

    Returns:
        Wheel velocity tensor with shape ``(num_envs, 4)`` in
        ``G1_BASE_WHEEL_DOF_ORDER``.
    """

    resolved_num_envs, resolved_device = resolve_tensor_context(
        env=env, num_envs=num_envs, device=device
    )
    selected_env_ids = normalize_env_indices(
        env_indices, num_envs=resolved_num_envs, device=resolved_device
    )
    selected_count = int(selected_env_ids.numel())

    base_twist = _normalize_g1_base_twist(
        vx=vx,
        vy=vy,
        wz=wz,
        selected_env_count=selected_count,
        device=resolved_device,
        dtype=dtype,
    )
    wheel_velocities = _g1_base_twist_to_wheel_velocities(
        base_twist,
        wheel_radius=wheel_radius,
        wheel_x=wheel_x,
        wheel_y=wheel_y,
        linear_velocity_scale=linear_velocity_scale,
        angular_velocity_scale=angular_velocity_scale,
    )

    action = torch.zeros(
        (resolved_num_envs, len(G1_BASE_WHEEL_DOF_ORDER)),
        device=resolved_device,
        dtype=dtype,
    )
    action[selected_env_ids, :] = wheel_velocities
    return action


def pack_g1_legs_relative_joint_command(
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack sparse named G1 leg relative deltas for one action term.

    Unspecified joints receive zero delta. ``env_indices=None`` applies the
    same command to all environments; otherwise only selected rows are changed.
    """

    return _pack_g1_relative_joint_command(
        G1_LEG_DOF_ORDER,
        joint_names,
        values,
        env=env,
        env_indices=env_indices,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
    )


def pack_g1_legs_absolute_joint_command(
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None = None,
    asset_name: str = DEFAULT_ROBOT_ASSET_NAME,
    baseline: JointValueInput | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack sparse named G1 leg absolute targets for one action term.

    Unspecified joints hold ``baseline`` or the current positions read from
    ``env``. ``env_indices=None`` applies the command to all environments.
    """

    return _pack_g1_absolute_joint_command(
        G1_LEG_DOF_ORDER,
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


def pack_g1_left_arm_relative_joint_command(
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack sparse named G1 left-arm relative deltas for one action term.

    Unspecified joints receive zero delta. ``env_indices=None`` applies the
    same command to all environments; otherwise only selected rows are changed.
    """

    return _pack_g1_relative_joint_command(
        G1_LEFT_ARM_DOF_ORDER,
        joint_names,
        values,
        env=env,
        env_indices=env_indices,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
    )


def pack_g1_left_arm_absolute_joint_command(
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None = None,
    asset_name: str = DEFAULT_ROBOT_ASSET_NAME,
    baseline: JointValueInput | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack sparse named G1 left-arm absolute targets for one action term.

    Unspecified joints hold ``baseline`` or the current positions read from
    ``env``. ``env_indices=None`` applies the command to all environments.
    """

    return _pack_g1_absolute_joint_command(
        G1_LEFT_ARM_DOF_ORDER,
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


def pack_g1_right_arm_relative_joint_command(
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack sparse named G1 right-arm relative deltas for one action term.

    Unspecified joints receive zero delta. ``env_indices=None`` applies the
    same command to all environments; otherwise only selected rows are changed.
    """

    return _pack_g1_relative_joint_command(
        G1_RIGHT_ARM_DOF_ORDER,
        joint_names,
        values,
        env=env,
        env_indices=env_indices,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
    )


def pack_g1_right_arm_absolute_joint_command(
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None = None,
    asset_name: str = DEFAULT_ROBOT_ASSET_NAME,
    baseline: JointValueInput | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack sparse named G1 right-arm absolute targets for one action term.

    Unspecified joints hold ``baseline`` or the current positions read from
    ``env``. ``env_indices=None`` applies the command to all environments.
    """

    return _pack_g1_absolute_joint_command(
        G1_RIGHT_ARM_DOF_ORDER,
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


def pack_g1_left_gripper_binary_command(
    is_open: bool,
    *,
    env: Any | None = None,
    asset_name: str = DEFAULT_ROBOT_ASSET_NAME,
    baseline: JointValueInput | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    open_position: float = DEFAULT_GRIPPER_OPEN_POSITION,
    closed_position: float = DEFAULT_GRIPPER_CLOSED_POSITION,
) -> torch.Tensor:
    """Pack a bool open/close command for the G1 left gripper master joint.

    The tensor has one column because ioailab commands only the master
    gripper joint; detailed finger motion stays inside the robot asset.
    """

    return _pack_g1_gripper_command(
        is_open,
        G1_LEFT_GRIPPER_DOF_ORDER,
        env=env,
        asset_name=asset_name,
        baseline=baseline,
        env_indices=env_indices,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        open_position=open_position,
        closed_position=closed_position,
    )


def pack_g1_right_gripper_binary_command(
    is_open: bool,
    *,
    env: Any | None = None,
    asset_name: str = DEFAULT_ROBOT_ASSET_NAME,
    baseline: JointValueInput | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    open_position: float = DEFAULT_GRIPPER_OPEN_POSITION,
    closed_position: float = DEFAULT_GRIPPER_CLOSED_POSITION,
) -> torch.Tensor:
    """Pack a bool open/close command for the G1 right gripper master joint.

    The tensor has one column because ioailab commands only the master
    gripper joint; detailed finger motion stays inside the robot asset.
    """

    return _pack_g1_gripper_command(
        is_open,
        G1_RIGHT_GRIPPER_DOF_ORDER,
        env=env,
        asset_name=asset_name,
        baseline=baseline,
        env_indices=env_indices,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        open_position=open_position,
        closed_position=closed_position,
    )


def _normalize_g1_base_twist(
    *,
    vx: JointValueInput,
    vy: JointValueInput,
    wz: JointValueInput,
    selected_env_count: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return base twist rows as ``(selected_env_count, 3)``."""

    components = [
        _normalize_g1_base_velocity_component(
            vx, selected_env_count=selected_env_count, device=device, dtype=dtype
        ),
        _normalize_g1_base_velocity_component(
            vy, selected_env_count=selected_env_count, device=device, dtype=dtype
        ),
        _normalize_g1_base_velocity_component(
            wz, selected_env_count=selected_env_count, device=device, dtype=dtype
        ),
    ]
    return torch.stack(components, dim=1)


def _normalize_g1_base_velocity_component(
    values: JointValueInput,
    *,
    selected_env_count: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return one base velocity component for selected env rows."""

    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if tensor.ndim == 0 or tensor.numel() == 1:
        return tensor.reshape(1).repeat(selected_env_count)
    if tensor.shape == (selected_env_count,):
        return tensor.clone()
    raise ValueError(
        "base velocity components must be scalar or have shape "
        f"({selected_env_count},), got {tuple(tensor.shape)}."
    )


def _g1_base_twist_to_wheel_velocities(
    base_twist: torch.Tensor,
    *,
    wheel_radius: float,
    wheel_x: float,
    wheel_y: float,
    linear_velocity_scale: float,
    angular_velocity_scale: float,
) -> torch.Tensor:
    """Convert ``vx, vy, wz`` rows to G1 main wheel angular velocities.

    The output order follows ``G1_BASE_WHEEL_DOF_ORDER`` and the same
    base-axis convention as the USD holonomic-controller graph: ``vx`` maps to
    the front/back wheel split, ``vy`` maps to the left/right diagonal split,
    and ``wz`` maps to all wheels turning together.
    """

    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive.")

    linear_factor = linear_velocity_scale / (sqrt(2.0) * wheel_radius)
    yaw_factor = angular_velocity_scale / (
        2.0 * sqrt(2.0) * wheel_radius * (abs(wheel_x) + abs(wheel_y))
    )

    vx = base_twist[:, 0] * linear_factor
    vy = base_twist[:, 1] * linear_factor
    yaw = base_twist[:, 2] * yaw_factor

    return torch.stack(
        (
            -vx + vy + yaw,
            -vx - vy + yaw,
            vx - vy + yaw,
            vx + vy + yaw,
        ),
        dim=1,
    )


def _pack_g1_gripper_command(
    is_open: bool,
    dof_order: Sequence[str],
    *,
    env: Any | None,
    asset_name: str,
    baseline: JointValueInput | None,
    env_indices: int | Sequence[int] | torch.Tensor | None,
    num_envs: int | None,
    device: torch.device | str | None,
    dtype: torch.dtype,
    open_position: float,
    closed_position: float,
) -> torch.Tensor:
    """Pack a bool gripper command as an absolute one-joint target tensor."""

    target = open_position if is_open else closed_position
    return pack_joint_value_command(
        dof_order,
        dof_order,
        target,
        env=env,
        # Gripper bool commands use the IsaacLab env only for tensor context.
        # The one-column action is fully specified by ``target``; unspecified
        # rows fall back to the G1 open-position default unless a caller
        # provides an explicit baseline.
        asset_name=None,
        baseline=baseline,
        env_indices=env_indices,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        default_value=open_position,
        joint_label="G1 joint",
    )


def _pack_g1_relative_joint_command(
    dof_order: Sequence[str],
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None,
    env_indices: int | Sequence[int] | torch.Tensor | None,
    num_envs: int | None,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Bind a G1 DOF order to the generic relative joint command packer."""

    return pack_relative_joint_command(
        dof_order,
        joint_names,
        values,
        env=env,
        env_indices=env_indices,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        joint_label="G1 joint",
    )


def _pack_g1_absolute_joint_command(
    dof_order: Sequence[str],
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None,
    asset_name: str,
    baseline: JointValueInput | None,
    env_indices: int | Sequence[int] | torch.Tensor | None,
    num_envs: int | None,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Bind a G1 DOF order to the generic absolute joint command packer."""

    return pack_absolute_joint_command(
        dof_order,
        joint_names,
        values,
        env=env,
        asset_name=asset_name,
        baseline=baseline,
        env_indices=env_indices,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        joint_label="G1 joint",
    )
