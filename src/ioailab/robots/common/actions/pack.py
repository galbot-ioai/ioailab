"""Robot-agnostic action tensor packing helpers for IsaacLab env steps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ioailab.utils.tensors import as_torch_tensor

JointValueInput = float | Sequence[float] | torch.Tensor


def pack_relative_joint_command(
    dof_order: Sequence[str],
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    joint_label: str = "joint",
) -> torch.Tensor:
    """Pack sparse joint deltas into the order expected by one relative action term."""

    resolved_num_envs, resolved_device = resolve_tensor_context(
        env=env, num_envs=num_envs, device=device
    )
    selected_env_ids = normalize_env_indices(
        env_indices, num_envs=resolved_num_envs, device=resolved_device
    )
    names = normalize_joint_names(joint_names, joint_label=joint_label)
    columns = [joint_column(dof_order, name, joint_label=joint_label) for name in names]
    deltas = normalize_selected_values(
        values,
        joint_count=len(names),
        selected_env_count=int(selected_env_ids.numel()),
        device=resolved_device,
        dtype=dtype,
    )
    action = torch.zeros(
        (resolved_num_envs, len(dof_order)), device=resolved_device, dtype=dtype
    )
    for value_index, column in enumerate(columns):
        action[selected_env_ids, column] = deltas[:, value_index]
    return action


def pack_absolute_joint_command(
    dof_order: Sequence[str],
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None = None,
    asset_name: str,
    baseline: JointValueInput | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    joint_label: str = "joint",
) -> torch.Tensor:
    """Pack sparse joint targets into the order expected by one absolute action term."""

    resolved_num_envs, resolved_device = resolve_tensor_context(
        env=env, num_envs=num_envs, device=device
    )
    selected_env_ids = normalize_env_indices(
        env_indices, num_envs=resolved_num_envs, device=resolved_device
    )
    names = normalize_joint_names(joint_names, joint_label=joint_label)
    columns = [joint_column(dof_order, name, joint_label=joint_label) for name in names]
    targets = normalize_selected_values(
        values,
        joint_count=len(names),
        selected_env_count=int(selected_env_ids.numel()),
        device=resolved_device,
        dtype=dtype,
    )
    action = make_absolute_baseline(
        dof_order,
        names,
        env=env,
        asset_name=asset_name,
        baseline=baseline,
        num_envs=resolved_num_envs,
        device=resolved_device,
        dtype=dtype,
        require_initialized_baseline=env_indices is not None
        and int(selected_env_ids.numel()) < resolved_num_envs,
        joint_label=joint_label,
    )
    for value_index, column in enumerate(columns):
        action[selected_env_ids, column] = targets[:, value_index]
    return action


def pack_joint_value_command(
    dof_order: Sequence[str],
    joint_names: str | Sequence[str],
    values: JointValueInput,
    *,
    env: Any | None = None,
    asset_name: str | None = None,
    baseline: JointValueInput | None = None,
    env_indices: int | Sequence[int] | torch.Tensor | None = None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    default_value: float | None = None,
    joint_label: str = "joint",
) -> torch.Tensor:
    """Pack explicit joint values while preserving unspecified columns from a baseline."""

    resolved_num_envs, resolved_device = resolve_tensor_context(
        env=env, num_envs=num_envs, device=device
    )
    selected_env_ids = normalize_env_indices(
        env_indices, num_envs=resolved_num_envs, device=resolved_device
    )
    names = normalize_joint_names(joint_names, joint_label=joint_label)
    columns = [joint_column(dof_order, name, joint_label=joint_label) for name in names]
    targets = normalize_selected_values(
        values,
        joint_count=len(names),
        selected_env_count=int(selected_env_ids.numel()),
        device=resolved_device,
        dtype=dtype,
    )
    action = make_value_baseline(
        dof_order,
        env=env,
        asset_name=asset_name,
        baseline=baseline,
        default_value=default_value,
        num_envs=resolved_num_envs,
        device=resolved_device,
        dtype=dtype,
    )
    for value_index, column in enumerate(columns):
        action[selected_env_ids, column] = targets[:, value_index]
    return action


def make_absolute_baseline(
    dof_order: Sequence[str],
    names: Sequence[str],
    *,
    env: Any | None,
    asset_name: str,
    baseline: JointValueInput | None,
    num_envs: int,
    device: torch.device | str,
    dtype: torch.dtype,
    require_initialized_baseline: bool,
    joint_label: str = "joint",
) -> torch.Tensor:
    """Return baseline targets for sparse absolute commands."""

    if baseline is not None:
        return normalize_baseline(
            baseline,
            action_width=len(dof_order),
            num_envs=num_envs,
            device=device,
            dtype=dtype,
        )
    if env is not None:
        return current_joint_positions_from_env(
            env,
            asset_name=asset_name,
            dof_order=dof_order,
            device=device,
            dtype=dtype,
        )
    if require_initialized_baseline:
        raise ValueError(
            f"baseline or env is required for sparse absolute {joint_label} commands."
        )
    if set(names) == set(dof_order):
        return torch.empty((num_envs, len(dof_order)), device=device, dtype=dtype)
    raise ValueError(
        f"baseline or env is required for sparse absolute {joint_label} commands."
    )


def make_value_baseline(
    dof_order: Sequence[str],
    *,
    env: Any | None,
    asset_name: str | None,
    baseline: JointValueInput | None,
    default_value: float | None,
    num_envs: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return baseline targets for selected-row value commands."""

    if baseline is not None:
        return normalize_baseline(
            baseline,
            action_width=len(dof_order),
            num_envs=num_envs,
            device=device,
            dtype=dtype,
        )
    if env is not None and asset_name is not None:
        return current_joint_positions_from_env(
            env,
            asset_name=asset_name,
            dof_order=dof_order,
            device=device,
            dtype=dtype,
        )
    if default_value is not None:
        return torch.full(
            (num_envs, len(dof_order)), float(default_value), device=device, dtype=dtype
        )
    return torch.zeros((num_envs, len(dof_order)), device=device, dtype=dtype)


def current_joint_positions_from_env(
    env: Any,
    *,
    asset_name: str,
    dof_order: Sequence[str],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read current joint positions from an IsaacLab env for sparse commands."""

    unwrapped = getattr(env, "unwrapped", env)
    asset = unwrapped.scene[asset_name]
    joint_names = list(asset.joint_names)
    joint_pos = as_torch_tensor(asset.data.joint_pos, device=device, dtype=dtype)
    columns = [joint_names.index(joint_name) for joint_name in dof_order]
    return joint_pos[:, columns].clone()


def resolve_tensor_context(
    *,
    env: Any | None,
    num_envs: int | None,
    device: torch.device | str | None,
) -> tuple[int, torch.device | str]:
    """Resolve action tensor shape/device from an env or explicit values."""

    if env is not None:
        unwrapped = getattr(env, "unwrapped", env)
        if num_envs is None:
            num_envs = int(unwrapped.num_envs)
        if device is None:
            device = unwrapped.device
    if num_envs is None or device is None:
        raise ValueError(
            "Provide env or both num_envs and device to build an IsaacLab action tensor."
        )
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1.")
    return num_envs, device


def normalize_env_indices(
    env_indices: int | Sequence[int] | torch.Tensor | None,
    *,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return selected env rows as a one-dimensional integer tensor."""

    if env_indices is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)
    if isinstance(env_indices, int) and not isinstance(env_indices, bool):
        index_tensor = torch.tensor([env_indices], dtype=torch.long)
    elif isinstance(env_indices, torch.Tensor):
        if (
            env_indices.dtype == torch.bool
            or torch.is_floating_point(env_indices)
            or torch.is_complex(env_indices)
        ):
            raise ValueError("env_indices must contain integers.")
        index_tensor = env_indices.detach().cpu().reshape(-1).to(dtype=torch.long)
    elif isinstance(env_indices, Sequence) and not isinstance(env_indices, str):
        index_values = tuple(env_indices)
        if not index_values:
            raise ValueError("env_indices must not be empty.")
        if not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in index_values
        ):
            raise ValueError("env_indices must contain integers.")
        index_tensor = torch.tensor(index_values, dtype=torch.long)
    else:
        raise ValueError("env_indices must contain integers.")

    if index_tensor.numel() == 0:
        raise ValueError("env_indices must not be empty.")
    if torch.any(index_tensor < 0) or torch.any(index_tensor >= num_envs):
        raise ValueError(f"env_indices out of range for num_envs={num_envs}.")
    if torch.unique(index_tensor).numel() != index_tensor.numel():
        raise ValueError("env_indices must be unique.")
    return index_tensor.to(device=device)


def normalize_joint_names(
    joint_names: str | Sequence[str], *, joint_label: str = "joint"
) -> tuple[str, ...]:
    """Return one or more joint names as a tuple."""

    names = (joint_names,) if isinstance(joint_names, str) else tuple(joint_names)
    if not names:
        raise ValueError(f"joint_names must contain at least one {joint_label} name.")
    if len(set(names)) != len(names):
        raise ValueError("joint_names must not contain duplicates.")
    return names


def normalize_selected_values(
    values: JointValueInput,
    *,
    joint_count: int,
    selected_env_count: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return values as ``(selected_env_count, joint_count)``."""

    value_tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if value_tensor.ndim == 0 or value_tensor.numel() == 1:
        if joint_count != 1:
            raise ValueError(
                f"values must contain {joint_count} item(s), got {value_tensor.numel()}."
            )
        return value_tensor.reshape(1, 1).repeat(selected_env_count, joint_count)
    if value_tensor.ndim == 1 and value_tensor.shape[0] == joint_count:
        return value_tensor.reshape(1, joint_count).repeat(selected_env_count, 1)
    if (
        value_tensor.ndim == 1
        and joint_count == 1
        and value_tensor.shape[0] == selected_env_count
    ):
        return value_tensor.reshape(selected_env_count, 1)
    if value_tensor.shape == (selected_env_count, joint_count):
        return value_tensor.clone()
    raise ValueError(
        "values must be a scalar, have shape "
        f"({joint_count},), ({selected_env_count},) for one joint, or "
        f"({selected_env_count}, {joint_count}); got {tuple(value_tensor.shape)}."
    )


def normalize_baseline(
    baseline: JointValueInput,
    *,
    action_width: int,
    num_envs: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a baseline action tensor with one row per environment."""

    baseline_tensor = torch.as_tensor(baseline, device=device, dtype=dtype)
    if baseline_tensor.ndim == 0 and action_width == 1:
        return baseline_tensor.reshape(1, 1).repeat(num_envs, 1)
    if baseline_tensor.shape == (action_width,):
        return baseline_tensor.reshape(1, action_width).repeat(num_envs, 1)
    if baseline_tensor.shape == (num_envs, action_width):
        return baseline_tensor.clone()
    raise ValueError(
        "baseline must have shape "
        f"({action_width},) or ({num_envs}, {action_width}), got {tuple(baseline_tensor.shape)}."
    )


def joint_column(
    dof_order: Sequence[str], joint_name: str, *, joint_label: str = "joint"
) -> int:
    """Return the action column for a joint name in one action term."""

    try:
        return tuple(dof_order).index(joint_name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown {joint_label} for this action term: {joint_name}"
        ) from exc
