"""Normalize task-authored motion-plan steps into G1 commands."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

from ioailab.agents.motion_plan.motion_plan import (
    MotionStep,
    TaskMotionPlan,
)
from ioailab.agents.motion_plan.targets import ResolvedTarget
from ioailab.robots.g1.actions import (
    G1_LEFT_ARM_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER,
)
from ioailab.robots.g1.articulation import G1_TOP_DOWN_TCP_WXYZ

_ARM_ALIASES = {
    "left": "left_arm",
    "left_arm": "left_arm",
    "right": "right_arm",
    "right_arm": "right_arm",
}
_ARM_TO_GRIPPER = {
    "left_arm": "left_gripper",
    "right_arm": "right_gripper",
}
_ARM_GROUPS = frozenset(_ARM_TO_GRIPPER)


_JOINT_POSITION_GROUP_DOF_ORDER = {
    "left_arm": G1_LEFT_ARM_DOF_ORDER,
    "right_arm": G1_RIGHT_ARM_DOF_ORDER,
}


@dataclass(frozen=True, slots=True)
class G1MotionCommand:
    """One normalized G1 terminal command derived from a public MotionStep."""

    name: str
    tcp_targets_w: Mapping[str, Any] = field(default_factory=dict)
    tcp_wxyz_by_group: Mapping[str, Any] = field(default_factory=dict)
    tcp_frame_by_group: Mapping[str, str] = field(default_factory=dict)
    joint_targets_by_group: Mapping[str, Any] = field(default_factory=dict)
    gripper_open_by_group: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize command keys after construction."""

        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "tcp_targets_w", _string_key_dict(self.tcp_targets_w))
        object.__setattr__(
            self, "tcp_wxyz_by_group", _string_key_dict(self.tcp_wxyz_by_group)
        )
        object.__setattr__(
            self,
            "tcp_frame_by_group",
            {
                str(group_name): _normalize_target_frame(frame)
                for group_name, frame in self.tcp_frame_by_group.items()
            },
        )
        object.__setattr__(
            self,
            "joint_targets_by_group",
            _string_key_dict(self.joint_targets_by_group),
        )
        object.__setattr__(
            self, "gripper_open_by_group", _string_key_dict(self.gripper_open_by_group)
        )


@dataclass(slots=True)
class G1MotionCommandContext:
    """Active command buffer for one executable G1 motion plan."""

    env: Any
    motion_cfg: Any
    available_joint_groups: tuple[str, ...]
    available_binary_groups: tuple[str, ...]
    commands: list[G1MotionCommand] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize group names after construction."""

        self.available_joint_groups = tuple(
            str(group) for group in self.available_joint_groups
        )
        self.available_binary_groups = tuple(
            str(group) for group in self.available_binary_groups
        )


_ACTIVE_CONTEXT: ContextVar[G1MotionCommandContext | None] = ContextVar(
    "ioailab_g1_motion_command_context",
    default=None,
)


@contextmanager
def g1_motion_command_context(
    *,
    env: Any,
    motion_cfg: Any,
    available_joint_groups: Iterable[str],
    available_binary_groups: Iterable[str],
):
    """Record normalized commands for one G1 task motion plan."""

    context = G1MotionCommandContext(
        env=env,
        motion_cfg=motion_cfg,
        available_joint_groups=tuple(available_joint_groups),
        available_binary_groups=tuple(available_binary_groups),
    )
    token = _ACTIVE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACTIVE_CONTEXT.reset(token)


def execute_motion_plan(motion_plan: TaskMotionPlan, *, env: Any) -> None:
    """Record the steps returned by a task-local motion-plan instance."""

    if not isinstance(motion_plan, TaskMotionPlan):
        raise TypeError("motion_plan must be a TaskMotionPlan instance.")
    for step in motion_plan.build(env):
        record_motion_step(step)


def record_motion_step(step: MotionStep) -> None:
    """Normalize and record one public MotionStep into the active context."""

    context = _require_context()
    repeats = int(step.hold_steps)
    if repeats < 1:
        raise ValueError(f"hold_steps must be positive, got {step.hold_steps!r}.")
    target = step.target
    if target is None and step.joint_positions is None and step.gripper_open is None:
        raise ValueError(
            "MotionStep requires a target, joint_positions, gripper_open, or a combination."
        )

    has_joint_target = step.joint_positions is not None
    arm_group = _resolve_arm_group(
        step.arm,
        context,
        needs_target=target is not None or has_joint_target,
    )
    tcp_targets_w: dict[str, Any] = {}
    tcp_wxyz_by_group: dict[str, Any] = {}
    tcp_frame_by_group: dict[str, str] = {}
    if target is not None:
        resolved: ResolvedTarget = target.resolve(context.env)
        tcp_targets_w[arm_group] = resolved.pos_xyz
        tcp_wxyz_by_group[arm_group] = _normalize_quat_for_pos(
            resolved.quat_wxyz, resolved.pos_xyz
        )
        tcp_frame_by_group[arm_group] = resolved.frame

    joint_targets_by_group: dict[str, Any] = {}
    if step.joint_positions is not None:
        joint_targets_by_group[arm_group] = _joint_positions_to_group_targets(
            step.joint_positions,
            group_name=arm_group,
        )

    gripper_open_by_group: dict[str, Any] = {}
    if step.gripper_open is not None:
        gripper_group = _resolve_gripper_group(arm_group, context)
        gripper_open_by_group[gripper_group] = bool(step.gripper_open)

    command_name = (
        str(step.name)
        if step.name
        else _default_command_name(context, arm_group, target, step.gripper_open)
    )
    command = G1MotionCommand(
        name=command_name,
        tcp_targets_w=tcp_targets_w,
        tcp_wxyz_by_group=tcp_wxyz_by_group,
        tcp_frame_by_group=tcp_frame_by_group,
        joint_targets_by_group=joint_targets_by_group,
        gripper_open_by_group=gripper_open_by_group,
    )
    context.commands.extend(command for _ in range(repeats))


def _require_context() -> G1MotionCommandContext:
    context = _ACTIVE_CONTEXT.get()
    if context is None:
        raise RuntimeError(
            "G1 motion-plan helpers are available only while ioailab executes a "
            "task motion-plan. Run the task through the motion-plan executor."
        )
    return context


def _resolve_arm_group(
    arm: str | None, context: G1MotionCommandContext, *, needs_target: bool
) -> str:
    if arm is not None:
        group = _ARM_ALIASES.get(str(arm))
        if group is None:
            raise ValueError(
                f"Unsupported G1 arm selector {arm!r}; expected 'left' or 'right'."
            )
        if needs_target and group not in context.available_joint_groups:
            raise ValueError(
                f"G1 arm group {group!r} is not writable by this task action layout: "
                f"{context.available_joint_groups!r}."
            )
        return group

    available = [
        group for group in context.available_joint_groups if group in _ARM_GROUPS
    ]
    if needs_target and len(available) == 1:
        return available[0]
    raise ValueError(
        "MotionStep could not infer an arm. Pass arm='left' or arm='right'. "
        f"Writable arm groups: {tuple(available)!r}."
    )


def _resolve_gripper_group(arm_group: str, context: G1MotionCommandContext) -> str:
    gripper_group = _ARM_TO_GRIPPER[arm_group]
    if gripper_group not in context.available_binary_groups:
        raise ValueError(
            f"G1 gripper group {gripper_group!r} is not writable by this task "
            f"action layout: {context.available_binary_groups!r}."
        )
    return gripper_group


def _joint_positions_to_group_targets(
    joint_positions: Mapping[str, float],
    *,
    group_name: str,
) -> torch.Tensor:
    """Return ordered joint targets for one G1 arm group."""

    if group_name not in _JOINT_POSITION_GROUP_DOF_ORDER:
        raise ValueError(
            f"G1 group {group_name!r} does not support direct joint_positions steps."
        )
    dof_order = _JOINT_POSITION_GROUP_DOF_ORDER[group_name]
    positions = {str(name): float(value) for name, value in joint_positions.items()}
    missing = tuple(
        joint_name for joint_name in dof_order if joint_name not in positions
    )
    unexpected = tuple(
        joint_name for joint_name in positions if joint_name not in dof_order
    )
    if missing or unexpected:
        raise ValueError(
            f"G1 group {group_name!r} joint_positions must exactly match {dof_order!r}; "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    return torch.tensor(
        [positions[joint_name] for joint_name in dof_order], dtype=torch.float32
    )


def _normalize_quat_for_pos(quat_wxyz: Any | None, pos_xyz: Any) -> Any:
    """Return a ``wxyz`` quaternion broadcast to match the position batch.

    Applies the robot default TCP orientation when the target declared none.
    """

    batch = 1 if torch.as_tensor(pos_xyz).ndim == 1 else int(pos_xyz.shape[0])
    if quat_wxyz is None:
        base = torch.tensor(G1_TOP_DOWN_TCP_WXYZ, dtype=torch.float32)
    else:
        base = torch.as_tensor(quat_wxyz, dtype=torch.float32)
    if base.ndim == 1:
        if batch == 1:
            return base
        return base.unsqueeze(0).repeat(batch, 1)
    if base.shape[0] != batch:
        raise ValueError(
            f"Target quaternion batch {tuple(base.shape)} does not match "
            f"position batch {batch}."
        )
    return base


def _default_command_name(
    context: G1MotionCommandContext,
    arm_group: str,
    target: Any | None,
    gripper_open: bool | None,
) -> str:
    del context
    if target is not None and gripper_open is not None:
        suffix = "open" if gripper_open else "close"
        return f"{arm_group}_{suffix}_target"
    if target is not None:
        return f"{arm_group}_target"
    suffix = "open" if gripper_open else "close"
    return f"{arm_group}_{suffix}_gripper"


def _string_key_dict(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in mapping.items()}


def _normalize_target_frame(frame: Any) -> str:
    """Return a normalized motion target frame."""

    normalized = str(frame).lower()
    if normalized not in ("world", "base"):
        raise ValueError(
            f"motion target frame must be 'world' or 'base', got {frame!r}."
        )
    return normalized


__all__ = [
    "G1MotionCommand",
    "G1MotionCommandContext",
    "execute_motion_plan",
    "g1_motion_command_context",
    "record_motion_step",
]
