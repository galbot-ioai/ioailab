"""Unified task-space target vocabulary for motion plans.

Both YAML and Python motion plans describe arm targets with the same two types:
:class:`WorldTarget` for absolute (or computed) poses and
:class:`AssetRelativeTarget` for poses expressed relative to a live scene asset.
Each target owns its own resolution against the env, so the asset-lookup logic
lives in exactly one place and reads identically from either authoring path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ioailab.agents.motion_plan.solvers.curobov2.utils import quat_xyzw_to_wxyz
from ioailab.utils.scene_state import asset_root_pose_xyz_xyzw

_VALID_FRAMES = frozenset(("world", "base"))


def _validate_frame(frame: Any) -> str:
    """Return a normalized target frame or raise for an unknown frame."""

    normalized = str(frame).lower()
    if normalized not in _VALID_FRAMES:
        raise ValueError(
            f"motion target frame must be 'world' or 'base', got {frame!r}."
        )
    return normalized


def _normalize_pos_xyz(pos_xyz: Any) -> torch.Tensor:
    """Return a float position tensor of shape ``(3,)`` or ``(num_envs, 3)``."""

    pos = torch.as_tensor(pos_xyz, dtype=torch.float32)
    if pos.shape == (3,) or (pos.ndim == 2 and pos.shape[1] == 3):
        return pos
    raise ValueError(
        "motion target position must have shape (3,) or (num_envs, 3), "
        f"got {tuple(pos.shape)}."
    )


def _quat_xyzw_to_wxyz(quat_xyzw: Any) -> torch.Tensor:
    """Return a ``wxyz`` quaternion tensor from an ``xyzw`` declaration."""

    quat = torch.as_tensor(quat_xyzw, dtype=torch.float32)
    return quat_xyzw_to_wxyz(
        quat,
        field_name="motion target quat_xyzw",
    )


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """A concrete arm target resolved against live env state.

    Attributes:
        pos_xyz: Position tensor of shape ``(3,)`` or ``(num_envs, 3)``.
        quat_wxyz: Orientation tensor in ``wxyz`` order, or ``None`` when the
            target did not declare an orientation (the planner then applies its
            robot-specific default TCP orientation).
        frame: ``"world"`` or ``"base"``.
    """

    pos_xyz: torch.Tensor
    quat_wxyz: torch.Tensor | None
    frame: str


@dataclass(frozen=True, slots=True)
class WorldTarget:
    """Absolute world/base-frame target.

    ``pos_xyz`` accepts a literal ``(3,)`` sequence shared by all envs, or a
    ``(num_envs, 3)`` tensor of per-env targets (e.g. computed from live poses).

    Attributes:
        pos_xyz: Target position in the declared ``frame``.
        quat_xyzw: Optional orientation in IsaacLab ``xyzw`` order.
        frame: ``"world"`` (default) or ``"base"`` (robot-base frame).
    """

    pos_xyz: Any
    quat_xyzw: Any | None = None
    frame: str = "world"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _validate_frame(self.frame))

    def resolve(self, env: Any) -> ResolvedTarget:
        """Return the resolved target; ``env`` is unused for world targets."""

        del env
        quat = None if self.quat_xyzw is None else _quat_xyzw_to_wxyz(self.quat_xyzw)
        return ResolvedTarget(
            pos_xyz=_normalize_pos_xyz(self.pos_xyz),
            quat_wxyz=quat,
            frame=self.frame,
        )


@dataclass(frozen=True, slots=True)
class AssetRelativeTarget:
    """Target expressed relative to a live scene asset's root pose.

    Resolved against the env at normalization time: the asset's world position
    plus ``offset``. The result is vectorized over ``num_envs``.

    Attributes:
        asset: Scene asset name passed to :func:`asset_root_pose_xyz_xyzw`.
        offset: ``(x, y, z)`` offset added to the asset position.
        quat_xyzw: Optional orientation in IsaacLab ``xyzw`` order.
        frame: ``"world"`` (default) or ``"base"``.
    """

    asset: str
    offset: Sequence[float] = (0.0, 0.0, 0.0)
    quat_xyzw: Any | None = None
    frame: str = "world"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _validate_frame(self.frame))

    def resolve(self, env: Any) -> ResolvedTarget:
        """Return the asset position plus offset as a vectorized target."""

        pos_xyz = asset_root_pose_xyz_xyzw(env, str(self.asset))[:, :3]
        pos_xyz = pos_xyz + pos_xyz.new_tensor(tuple(float(v) for v in self.offset))
        quat = None if self.quat_xyzw is None else _quat_xyzw_to_wxyz(self.quat_xyzw)
        return ResolvedTarget(pos_xyz=pos_xyz, quat_wxyz=quat, frame=self.frame)


Target = WorldTarget | AssetRelativeTarget


__all__ = [
    "AssetRelativeTarget",
    "ResolvedTarget",
    "Target",
    "WorldTarget",
]
