"""Stateless mobile-base chassis mechanics shared by navigation agents.

These helpers are the low-level plumbing :class:`~ioailab.agents.nav.base.
BaseNavAgent` uses to control a chassis: resolving env rows, rotating world
deltas into the base frame, reading yaw from a quaternion, and packing a base
twist into the full task action while preserving non-base action columns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from ioailab.agents.io import EnvIds


def unwrapped(env: Any) -> Any:
    """Peel workflow/Gymnasium wrappers down to the live IsaacLab env."""

    inner = getattr(env, "unwrapped", getattr(env, "raw_env", env))
    return getattr(inner, "unwrapped", inner)


def resolve_env_ids(env: Any, env_ids: EnvIds = None) -> tuple[int, ...]:
    """Return selected env row ids, defaulting to all rows."""

    if env_ids is None:
        return tuple(range(int(getattr(env, "num_envs"))))
    ids = tuple(int(env_id) for env_id in env_ids)
    if not ids:
        raise ValueError("env_ids must not be empty.")
    if len(set(ids)) != len(ids):
        raise ValueError("env_ids must be unique.")
    num_envs = int(getattr(env, "num_envs"))
    if any(env_id < 0 or env_id >= num_envs for env_id in ids):
        raise ValueError(f"env_ids out of range for num_envs={num_envs}.")
    return ids


def world_to_base_frame(
    delta_xy_w: torch.Tensor, base_yaw_w: torch.Tensor
) -> torch.Tensor:
    """Rotate env/world XY deltas into the robot base frame."""

    cos_yaw = torch.cos(base_yaw_w)
    sin_yaw = torch.sin(base_yaw_w)
    delta_x_w = delta_xy_w[:, 0]
    delta_y_w = delta_xy_w[:, 1]
    delta_x_b = cos_yaw * delta_x_w + sin_yaw * delta_y_w
    delta_y_b = -sin_yaw * delta_x_w + cos_yaw * delta_y_w
    return torch.stack((delta_x_b, delta_y_b), dim=1)


def quat_to_yaw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from an xyzw quaternion."""

    x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: torch.Tensor | float) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""

    if isinstance(angle, float):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return torch.tensor(angle)
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compose_full_action(
    env: Any, base_action: torch.Tensor, base_wheel_dof_names: Sequence[str]
) -> torch.Tensor:
    """Write the base action slice while preserving existing non-base columns."""

    base_width = int(base_action.shape[1])
    base_slice = _base_action_slice(env, base_width, base_wheel_dof_names)
    action_manager = getattr(env, "action_manager", None)
    prior = getattr(action_manager, "action", None)
    prior_width = (
        int(prior.shape[1])
        if isinstance(prior, torch.Tensor) and prior.ndim == 2
        else 0
    )
    total_action_dim = int(getattr(action_manager, "total_action_dim", 0) or 0)
    action_dim = max(total_action_dim, int(base_slice.stop), base_width, prior_width)
    if isinstance(prior, torch.Tensor) and prior.shape == (
        base_action.shape[0],
        action_dim,
    ):
        full_action = prior.to(
            device=base_action.device, dtype=base_action.dtype
        ).clone()
    else:
        full_action = torch.zeros(
            base_action.shape[0],
            action_dim,
            device=base_action.device,
            dtype=base_action.dtype,
        )
    full_action[:, base_slice] = base_action[:, : base_slice.stop - base_slice.start]
    return full_action


def _base_action_slice(
    env: Any, base_width: int, base_wheel_dof_names: Sequence[str]
) -> slice:
    """Resolve the base-wheel action slice from action-manager term order."""

    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return slice(0, int(base_width))
    term_names = _action_term_names(action_manager)
    if not term_names:
        return slice(0, int(base_width))
    terms_by_name = getattr(action_manager, "_terms", None)
    if not isinstance(terms_by_name, dict):
        terms_by_name = {}
    base_wheels = tuple(base_wheel_dof_names)
    cursor = 0
    for term_name in term_names:
        term = terms_by_name.get(term_name)
        cfg_term = _env_cfg_action_term(env, term_name)
        joint_names = _joint_names(term, cfg_term)
        if joint_names is None:
            width = int(
                getattr(term, "action_dim", 0)
                or getattr(cfg_term, "action_dim", 0)
                or 0
            )
        else:
            width = len(joint_names)
            if joint_names == base_wheels:
                return slice(cursor, cursor + width)
        cursor += width
    total_action_dim = int(getattr(action_manager, "total_action_dim", 0) or 0)
    if (
        len(term_names) > 1
        or cursor > int(base_width)
        or total_action_dim > int(base_width)
    ):
        raise ValueError(
            "Cannot resolve base action slice from action-manager metadata; "
            "multi-term navigation actions require the base action term to expose "
            "G1 base wheel joint_names."
        )
    return slice(0, int(base_width))


def _action_term_names(action_manager: Any) -> tuple[str, ...]:
    for attr_name in ("active_terms", "term_names"):
        names = getattr(action_manager, attr_name, None)
        if isinstance(names, (list, tuple)):
            return tuple(str(name) for name in names)
    terms = getattr(action_manager, "_terms", None)
    if isinstance(terms, dict):
        return tuple(str(name) for name in terms)
    return ()


def _env_cfg_action_term(env: Any, term_name: str) -> Any | None:
    env_cfg = getattr(env, "cfg", None)
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        return None
    return getattr(actions_cfg, term_name, None)


def _joint_names(term: Any | None, cfg_term: Any | None) -> tuple[str, ...] | None:
    for source in (
        term,
        getattr(term, "cfg", None) if term is not None else None,
        getattr(term, "_cfg", None) if term is not None else None,
        cfg_term,
    ):
        if source is not None and hasattr(source, "joint_names"):
            return tuple(str(name) for name in getattr(source, "joint_names"))
    return None
