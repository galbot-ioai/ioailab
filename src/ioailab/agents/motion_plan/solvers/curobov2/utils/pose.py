"""Pose and quaternion helpers for cuRobo v2 planner data."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from ioailab.agents.motion_plan.solvers.curobov2.waypoint_plan import TargetPose


def normalize_pose_xyz_wxyz(
    value: Any, *, field_name: str = "pose_xyz_wxyz"
) -> np.ndarray:
    """Return batched ``xyz + wxyz`` poses with unit quaternions.

    Args:
        value: Pose array with shape ``(7,)`` or ``(num_envs, 7)``.
        field_name: Human-readable field name for validation errors.

    Returns:
        A ``float32`` array with shape ``(num_envs, 7)``.

    Raises:
        ValueError: If the value cannot be interpreted as one or more poses.
    """

    pose = np.asarray(value, dtype=np.float32)
    if pose.ndim == 1:
        pose = pose[None, :]
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(
            f"{field_name} must have shape (7,) or (num_envs, 7), got {pose.shape}."
        )

    normalized = pose.astype(np.float32, copy=True)
    quat = normalized[:, 3:7]
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-8):
        raise ValueError(f"{field_name} contains a zero-length quaternion.")
    normalized[:, 3:7] = quat / norms
    return normalized


def quat_xyzw_to_wxyz(value: Any, *, field_name: str = "quat_xyzw") -> Any:
    """Convert quaternion values from IsaacLab ``xyzw`` order to ``wxyz``.

    Args:
        value: Quaternion array with shape ``(4,)`` or ``(..., 4)``.
        field_name: Human-readable field name for validation errors.

    Returns:
        A tensor/array with the same backend as ``value`` and reordered
        quaternion components.
    """

    return _reorder_quat(value, indices=(3, 0, 1, 2), field_name=field_name)


def quat_wxyz_to_xyzw(value: Any, *, field_name: str = "quat_wxyz") -> Any:
    """Convert quaternion values from cuRobo ``wxyz`` order to ``xyzw``."""

    return _reorder_quat(value, indices=(1, 2, 3, 0), field_name=field_name)


def pose_xyz_xyzw_to_xyz_wxyz(value: Any, *, field_name: str = "pose_xyz_xyzw") -> Any:
    """Convert pose values from IsaacLab ``xyz + xyzw`` to ``xyz + wxyz``."""

    return _convert_pose_quat_order(
        value,
        quat_converter=quat_xyzw_to_wxyz,
        field_name=field_name,
        quat_field_name=f"{field_name}.quat_xyzw",
    )


def pose_xyz_wxyz_to_xyz_xyzw(value: Any, *, field_name: str = "pose_xyz_wxyz") -> Any:
    """Convert pose values from cuRobo ``xyz + wxyz`` to ``xyz + xyzw``."""

    return _convert_pose_quat_order(
        value,
        quat_converter=quat_wxyz_to_xyzw,
        field_name=field_name,
        quat_field_name=f"{field_name}.quat_wxyz",
    )


def _validate_base_pose_by_env(base_pose_by_env: Any, *, num_envs: int) -> np.ndarray:
    """Validate a world-frame robot base pose array.

    Args:
        base_pose_by_env: ``xyz + wxyz`` base pose with shape ``(num_envs, 7)``.
        num_envs: Expected number of environments.

    Returns:
        Normalized ``float32`` base poses.
    """

    base_pose = normalize_pose_xyz_wxyz(base_pose_by_env, field_name="base_pose_by_env")
    if base_pose.shape[0] != int(num_envs):
        raise ValueError(
            "base_pose_by_env must have one pose per environment: "
            f"expected {int(num_envs)}, got {base_pose.shape[0]}."
        )
    return base_pose


def _reorder_quat(
    value: Any, *, indices: tuple[int, int, int, int], field_name: str
) -> Any:
    if isinstance(value, torch.Tensor):
        if value.shape[-1] != 4:
            raise ValueError(
                f"{field_name} must have shape (..., 4), got {tuple(value.shape)}."
            )
        return value[..., list(indices)]

    quat = np.asarray(value, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"{field_name} must have shape (..., 4), got {quat.shape}.")
    return quat[..., list(indices)].astype(np.float32, copy=False)


def _convert_pose_quat_order(
    value: Any,
    *,
    quat_converter: Callable[..., Any],
    field_name: str,
    quat_field_name: str,
) -> Any:
    if isinstance(value, torch.Tensor):
        if value.shape[-1] != 7:
            raise ValueError(
                f"{field_name} must have shape (..., 7), got {tuple(value.shape)}."
            )
        return torch.cat(
            (
                value[..., :3],
                quat_converter(value[..., 3:7], field_name=quat_field_name),
            ),
            dim=-1,
        )

    pose = np.asarray(value, dtype=np.float32)
    if pose.shape[-1] != 7:
        raise ValueError(f"{field_name} must have shape (..., 7), got {pose.shape}.")
    return np.concatenate(
        (pose[..., :3], quat_converter(pose[..., 3:7], field_name=quat_field_name)),
        axis=-1,
    ).astype(np.float32, copy=False)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-8):
        raise ValueError("Quaternion array contains a zero-length quaternion.")
    return (quat / norms).astype(np.float32, copy=False)


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float32).copy()
    out[..., 1:4] *= -1.0
    return out


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(lhs, dtype=np.float32), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(rhs, dtype=np.float32), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    zeros = np.zeros((*np.asarray(vector).shape[:-1], 1), dtype=np.float32)
    vector_quat = np.concatenate([zeros, np.asarray(vector, dtype=np.float32)], axis=-1)
    rotated = _quat_mul_wxyz(
        _quat_mul_wxyz(quat, vector_quat), _quat_conjugate_wxyz(quat)
    )
    return rotated[..., 1:4].astype(np.float32, copy=False)


def resolve_target_pose_xyz_wxyz(
    target_pose: "TargetPose",
    *,
    num_envs: int,
    base_pose_by_env: Any | None = None,
) -> np.ndarray:
    """Resolve a target pose into the robot-base frame expected by cuRobo.

    ``frame='base'`` poses are already in cuRobo's planning frame. ``frame='world'``
    poses require the per-env robot base pose so this utility can transform the
    target into the base frame.
    """

    pose = normalize_pose_xyz_wxyz(
        target_pose.pose_xyz_wxyz, field_name=f"{target_pose.group_name}.pose_xyz_wxyz"
    )
    if pose.shape[0] == 1 and int(num_envs) > 1:
        pose = np.repeat(pose, int(num_envs), axis=0)
    if pose.shape[0] != int(num_envs):
        raise ValueError(
            f"Target pose for {target_pose.group_name!r} must have one pose per environment: "
            f"expected {int(num_envs)}, got {pose.shape[0]}."
        )

    frame = str(target_pose.frame).lower()
    if frame == "base":
        return pose
    if frame != "world":
        raise ValueError(
            f"Unsupported target frame {target_pose.frame!r}; expected 'base' or 'world'."
        )
    if base_pose_by_env is None:
        raise ValueError(
            "base_pose_by_env is required when a TargetPose uses frame='world'."
        )

    base_pose = _validate_base_pose_by_env(base_pose_by_env, num_envs=int(num_envs))
    base_pos = base_pose[:, :3]
    base_quat = base_pose[:, 3:7]
    target_pos = pose[:, :3]
    target_quat = pose[:, 3:7]

    inv_base_quat = _quat_conjugate_wxyz(base_quat)
    relative_pos = _quat_rotate_wxyz(inv_base_quat, target_pos - base_pos)
    relative_quat = _normalize_quat_wxyz(_quat_mul_wxyz(inv_base_quat, target_quat))
    return np.concatenate([relative_pos, relative_quat], axis=1).astype(
        np.float32, copy=False
    )


def curobo_pose_from_robot_base_position(
    *,
    target_base_pos: torch.Tensor,
    target_robot_wxyz: np.ndarray,
) -> np.ndarray:
    """Build cuRobo ``xyz+wxyz`` poses from base-frame positions."""

    target_base_wxyz = torch.as_tensor(
        target_robot_wxyz, device=target_base_pos.device, dtype=torch.float32
    )
    if target_base_wxyz.shape != (target_base_pos.shape[0], 4):
        raise ValueError(
            "target_robot_wxyz must have shape "
            f"({target_base_pos.shape[0]}, 4), got {tuple(target_base_wxyz.shape)}"
        )
    return (
        torch.cat([target_base_pos, target_base_wxyz], dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
