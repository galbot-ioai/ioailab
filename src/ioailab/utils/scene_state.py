"""IsaacLab scene-state accessors.

These helpers preserve IsaacLab's root-pose convention. Scene/root asset poses
are returned as ``xyz + xyzw`` because IsaacLab root quaternions use
``(x, y, z, w)`` order. Planner-specific conversions belong at planner
boundaries, not here.
"""

from __future__ import annotations

from typing import Any

import torch

from ioailab.utils.tensors import as_torch_tensor, batched_tensor


def scene_asset(env: Any, name: str) -> Any:
    """Return a named IsaacLab scene asset from an env or unwrapped env."""

    unwrapped = getattr(env, "unwrapped", env)
    return unwrapped.scene[str(name)]


def asset_root_pos_w(env: Any, name: str) -> torch.Tensor:
    """Return an IsaacLab scene asset root position in world frame.

    Args:
        env: IsaacLab environment or unwrapped environment.
        name: Scene asset name, such as ``"cube"``.

    Returns:
        A float32 tensor with shape ``(num_envs, 3)``.
    """

    asset = scene_asset(env, name)
    device = _env_device(env)
    data = asset.data
    if not hasattr(data, "root_pos_w"):
        raise AttributeError(f"Scene asset {name!r} must expose root_pos_w.")
    return batched_tensor(
        as_torch_tensor(data.root_pos_w, device=device),
        width=3,
        field_name=f"{name}.root_pos_w",
    )


def asset_root_quat_xyzw(env: Any, name: str) -> torch.Tensor:
    """Return an IsaacLab scene asset root quaternion in ``xyzw`` order."""

    asset = scene_asset(env, name)
    device = _env_device(env)
    data = asset.data
    if not hasattr(data, "root_quat_w"):
        raise AttributeError(f"Scene asset {name!r} must expose root_quat_w.")
    return batched_tensor(
        as_torch_tensor(data.root_quat_w, device=device),
        width=4,
        field_name=f"{name}.root_quat_w",
    )


def asset_root_pose_xyz_xyzw(env: Any, name: str) -> torch.Tensor:
    """Return an IsaacLab scene asset root pose in ``xyz + xyzw`` order."""

    asset = scene_asset(env, name)
    device = _env_device(env)
    data = asset.data
    if not hasattr(data, "root_pos_w") or not hasattr(data, "root_quat_w"):
        raise AttributeError(
            f"Scene asset {name!r} must expose root_pos_w and root_quat_w."
        )
    pos_w = batched_tensor(
        as_torch_tensor(data.root_pos_w, device=device),
        width=3,
        field_name=f"{name}.root_pos_w",
    )
    quat_xyzw = batched_tensor(
        as_torch_tensor(data.root_quat_w, device=device),
        width=4,
        field_name=f"{name}.root_quat_w",
    )
    if pos_w.shape[0] != quat_xyzw.shape[0]:
        raise ValueError(
            f"{name!r} root position and quaternion batches must match: "
            f"{pos_w.shape[0]} != {quat_xyzw.shape[0]}."
        )
    return torch.cat((pos_w, quat_xyzw), dim=1)


def _env_device(env: Any) -> torch.device | None:
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "device"):
        return torch.device(unwrapped.device)
    return None
