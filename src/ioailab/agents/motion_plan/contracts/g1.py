"""G1 bridge between motion-plan commands and IsaacLab action tensors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch

from ioailab.agents.motion_plan.commands import G1MotionCommand
from ioailab.robots.g1.actions import (
    DEFAULT_GRIPPER_CLOSED_POSITION,
    DEFAULT_GRIPPER_OPEN_POSITION,
    G1_BASE_WHEEL_DOF_ORDER,
    G1_LEG_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER,
    G1_LEFT_GRIPPER_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER,
    G1_RIGHT_GRIPPER_DOF_ORDER,
    pack_g1_legs_absolute_joint_command,
    pack_g1_left_arm_absolute_joint_command,
    pack_g1_right_arm_absolute_joint_command,
)

G1_ACTION_GROUP_DOF_ORDER: dict[str, tuple[str, ...]] = {
    "base": G1_BASE_WHEEL_DOF_ORDER,
    "legs": G1_LEG_DOF_ORDER,
    "left_arm": G1_LEFT_ARM_DOF_ORDER,
    "right_arm": G1_RIGHT_ARM_DOF_ORDER,
    "left_gripper": G1_LEFT_GRIPPER_DOF_ORDER,
    "right_gripper": G1_RIGHT_GRIPPER_DOF_ORDER,
}
"""Supported G1 action groups and their tensor widths.

``base`` is a pass-through layout group so motion-plan action tensors can
coexist with navigation actions; it is intentionally not a cuRobo joint group.
"""

G1_JOINT_GROUP_NAMES = frozenset(("legs", "left_arm", "right_arm"))
G1_BINARY_GROUP_NAMES = frozenset(("left_gripper", "right_gripper"))
_G1_ACTION_GROUP_BY_DOF_ORDER = {
    joint_names: group_name
    for group_name, joint_names in G1_ACTION_GROUP_DOF_ORDER.items()
}
_G1_JOINT_GROUP_PACKERS: dict[
    str, tuple[tuple[str, ...], Callable[..., torch.Tensor]]
] = {
    "legs": (G1_LEG_DOF_ORDER, pack_g1_legs_absolute_joint_command),
    "left_arm": (G1_LEFT_ARM_DOF_ORDER, pack_g1_left_arm_absolute_joint_command),
    "right_arm": (G1_RIGHT_ARM_DOF_ORDER, pack_g1_right_arm_absolute_joint_command),
}


@dataclass(frozen=True, slots=True)
class G1ActionTerm:
    """One resolved IsaacLab action term that the motion-plan source may fill."""

    term_name: str
    group_name: str
    action_slice: slice
    joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate explicit action tensor bounds."""

        if self.action_slice.start is None or self.action_slice.stop is None:
            raise ValueError(
                "G1 action tensor term slices must have explicit start and stop indices."
            )
        if (
            self.action_slice.start < 0
            or self.action_slice.stop <= self.action_slice.start
        ):
            raise ValueError(
                f"Invalid action slice for term {self.term_name!r}: {self.action_slice}."
            )

    @property
    def width(self) -> int:
        """Return the tensor width of this action term."""

        return int(self.action_slice.stop - self.action_slice.start)


@dataclass(frozen=True, slots=True)
class G1ActionLayout:
    """Resolved task action tensor layout for G1 motion-plan action sourcing."""

    terms: tuple[G1ActionTerm, ...]

    def __post_init__(self) -> None:
        """Validate term and group uniqueness plus tensor slice overlap."""

        if not self.terms:
            raise ValueError("G1 action tensor layout must contain at least one term.")
        term_names = [term.term_name for term in self.terms]
        if len(set(term_names)) != len(term_names):
            raise ValueError(
                f"G1 action tensor layout has duplicate term names: {term_names!r}."
            )
        group_names = [term.group_name for term in self.terms]
        if len(set(group_names)) != len(group_names):
            raise ValueError(
                f"G1 action tensor layout has duplicate group names: {group_names!r}."
            )
        _validate_non_overlapping_slices(self.terms)

    @property
    def action_dim(self) -> int:
        """Return the minimum full action tensor width required by the layout."""

        return max(int(term.action_slice.stop) for term in self.terms)

    @property
    def joint_group_names(self) -> tuple[str, ...]:
        """Return declared joint-position groups."""

        return tuple(
            term.group_name
            for term in self.terms
            if term.group_name in G1_JOINT_GROUP_NAMES
        )

    @property
    def binary_group_names(self) -> tuple[str, ...]:
        """Return declared binary gripper groups."""

        return tuple(
            term.group_name
            for term in self.terms
            if term.group_name in G1_BINARY_GROUP_NAMES
        )

    def term_for_group(self, group_name: str) -> G1ActionTerm:
        """Return the declared tensor term for a G1 action group."""

        group = str(group_name)
        for term in self.terms:
            if term.group_name == group:
                return term
        raise ValueError(f"G1 action layout does not declare writable group {group!r}.")

    def slice_for_group(self, group_name: str) -> slice:
        """Return the full action-tensor slice for a G1 action group."""

        return self.term_for_group(group_name).action_slice


@dataclass(frozen=True, slots=True)
class G1ActionFrame:
    """One executable action frame resolved from a waypoint plan."""

    name: str
    joint_targets_by_group: Mapping[str, torch.Tensor]
    binary_values_by_group: Mapping[str, torch.Tensor]


class _ActionRowSelection(NamedTuple):
    """Mapping from selected env ids to rows in the provided action tensor."""

    action_rows: torch.Tensor
    env_ids: torch.Tensor | None
    env_row_count: int | None


def make_g1_action_layout_from_env(env: Any) -> G1ActionLayout:
    """Build a G1 action tensor layout from IsaacLab action joint names."""

    unwrapped = getattr(env, "unwrapped", env)
    action_manager = getattr(unwrapped, "action_manager", None)
    if action_manager is None:
        raise ValueError(
            "Cannot resolve IsaacLab action term order: env has no action_manager."
        )

    term_names = _action_term_names(action_manager)
    terms_by_name = getattr(action_manager, "_terms", None)
    if not isinstance(terms_by_name, dict):
        terms_by_name = {}

    layout_terms: list[G1ActionTerm] = []
    cursor = 0
    for term_name in term_names:
        term = terms_by_name.get(term_name)
        cfg_term = _env_cfg_action_term(unwrapped, term_name)
        sources = _action_cfg_sources(term, cfg_term)
        joint_names = _resolve_joint_names(term_name, term, sources)
        group_name = _resolve_g1_action_group(term_name, joint_names)
        width = len(joint_names)
        layout_terms.append(
            G1ActionTerm(
                term_name=term_name,
                group_name=group_name,
                action_slice=slice(cursor, cursor + width),
                joint_names=joint_names,
            )
        )
        cursor += width

    return G1ActionLayout(tuple(layout_terms))


def make_g1_action_tensor(env: Any, *, action_dim: int) -> torch.Tensor:
    """Create a full action tensor on the unwrapped env device."""

    unwrapped = getattr(env, "unwrapped", env)
    action_manager = getattr(unwrapped, "action_manager", None)
    total_action_dim = int(getattr(action_manager, "total_action_dim", action_dim))
    return torch.zeros(
        (int(unwrapped.num_envs), max(int(action_dim), total_action_dim)),
        device=unwrapped.device,
        dtype=torch.float32,
    )


def write_g1_initial_action(
    env: Any,
    *,
    layout: G1ActionLayout,
    action_tensor: torch.Tensor,
    robot_asset_name: str,
    initial_joint_targets_by_group: Mapping[str, Any],
    initial_gripper_open_by_group: Mapping[str, Any],
    open_position: float = DEFAULT_GRIPPER_OPEN_POSITION,
    closed_position: float = DEFAULT_GRIPPER_CLOSED_POSITION,
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Write the initial hold action into a full or compact action tensor."""

    for group_name in layout.joint_group_names:
        targets = initial_joint_targets_by_group.get(group_name)
        if targets is None:
            targets = current_g1_group_joint_positions(
                env, robot_asset_name=robot_asset_name, group_name=group_name
            )
        write_g1_joint_targets(
            env,
            layout=layout,
            action_tensor=action_tensor,
            group_name=group_name,
            targets=targets,
            env_ids=env_ids,
        )
    for group_name in layout.binary_group_names:
        value = initial_gripper_open_by_group.get(group_name, True)
        write_g1_binary_values(
            env=env,
            layout=layout,
            action_tensor=action_tensor,
            group_name=group_name,
            values=value,
            open_position=open_position,
            closed_position=closed_position,
            env_ids=env_ids,
        )


def write_g1_frame_action(
    env: Any,
    *,
    layout: G1ActionLayout,
    action_tensor: torch.Tensor,
    frame: G1ActionFrame,
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Write one executable frame into a full or compact G1 action tensor."""

    for group_name, targets in frame.joint_targets_by_group.items():
        write_g1_joint_targets(
            env,
            layout=layout,
            action_tensor=action_tensor,
            group_name=group_name,
            targets=targets,
            env_ids=env_ids,
        )
    for group_name, values in frame.binary_values_by_group.items():
        write_g1_binary_values(
            env=env,
            layout=layout,
            action_tensor=action_tensor,
            group_name=group_name,
            values=values,
            env_ids=env_ids,
        )


def write_g1_joint_targets(
    env: Any,
    *,
    layout: G1ActionLayout,
    action_tensor: torch.Tensor,
    group_name: str,
    targets: Any,
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Pack and write joint-position targets for one G1 action group."""

    group = str(group_name)
    term = layout.term_for_group(group)
    selection = _action_row_selection(env, action_tensor, env_ids)
    selected_count = int(selection.action_rows.numel())
    target_tensor = torch.as_tensor(
        targets, device=action_tensor.device, dtype=action_tensor.dtype
    )
    if target_tensor.ndim == 1:
        target_tensor = target_tensor.reshape(1, -1).repeat(selected_count, 1)
    else:
        target_tensor = _select_action_values(
            target_tensor,
            action_tensor=action_tensor,
            selection=selection,
            value_width=len(term.joint_names),
            value_label=f"G1 group {group!r} targets",
        )
    expected_shape = (selected_count, len(term.joint_names))
    if target_tensor.shape != expected_shape:
        raise ValueError(
            f"G1 group {group!r} targets must have shape {expected_shape}, got {tuple(target_tensor.shape)}."
        )

    if group not in _G1_JOINT_GROUP_PACKERS:
        raise ValueError(f"G1 group {group!r} is not a joint-position planner group.")
    dof_order, packer = _G1_JOINT_GROUP_PACKERS[group]
    action_slice = packer(
        dof_order,
        target_tensor,
        env=None,
        baseline=action_tensor[:, term.action_slice],
        env_indices=selection.action_rows,
        num_envs=int(action_tensor.shape[0]),
        device=action_tensor.device,
        dtype=action_tensor.dtype,
    )
    action_tensor[selection.action_rows, term.action_slice] = action_slice[
        selection.action_rows
    ]


def write_g1_binary_values(
    *,
    env: Any | None = None,
    layout: G1ActionLayout,
    action_tensor: torch.Tensor,
    group_name: str,
    values: Any,
    open_position: float = DEFAULT_GRIPPER_OPEN_POSITION,
    closed_position: float = DEFAULT_GRIPPER_CLOSED_POSITION,
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Write binary gripper open/close values for one G1 gripper group."""

    group = str(group_name)
    if group not in G1_BINARY_GROUP_NAMES:
        raise ValueError(f"G1 group {group!r} is not a binary gripper group.")
    term = layout.term_for_group(group)
    if term.width != 1:
        raise ValueError(
            f"G1 binary group {group!r} expects a one-column tensor slice."
        )
    selection = _action_row_selection(env, action_tensor, env_ids)
    value_tensor = normalize_g1_bool_tensor(
        _selected_bool_values(
            values,
            action_tensor=action_tensor,
            selection=selection,
        ),
        num_envs=int(selection.action_rows.numel()),
        device=action_tensor.device,
    )
    target = torch.where(
        value_tensor,
        torch.as_tensor(
            open_position, device=action_tensor.device, dtype=action_tensor.dtype
        ),
        torch.as_tensor(
            closed_position, device=action_tensor.device, dtype=action_tensor.dtype
        ),
    )
    action_tensor[selection.action_rows, term.action_slice] = target.reshape((-1, 1))


def current_g1_group_joint_positions(
    env: Any, *, robot_asset_name: str, group_name: str
) -> torch.Tensor:
    """Return current IsaacLab joint positions for one G1 action group."""

    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene[robot_asset_name]
    robot_joint_names = tuple(str(name) for name in getattr(robot, "joint_names", ()))
    expected_joint_names = G1_ACTION_GROUP_DOF_ORDER[group_name]
    missing = tuple(
        name for name in expected_joint_names if name not in robot_joint_names
    )
    if missing:
        raise ValueError(
            f"Robot asset {robot_asset_name!r} is missing G1 joints for group {group_name!r}: {missing}."
        )
    joint_indices = [robot_joint_names.index(name) for name in expected_joint_names]
    joint_pos = torch.as_tensor(
        robot.data.joint_pos,
        device=torch.device(getattr(unwrapped, "device", "cpu")),
        dtype=torch.float32,
    )
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(1, -1)
    return joint_pos[:, joint_indices]


def validate_g1_motion_command_groups(command: G1MotionCommand) -> None:
    """Validate that one command requests only supported G1 action groups."""

    for group_name in command.tcp_targets_w:
        if group_name not in G1_JOINT_GROUP_NAMES:
            raise ValueError(
                f"G1 motion command {command.name!r} requests unsupported TCP group {group_name!r}."
            )
    for group_name in command.tcp_wxyz_by_group:
        if group_name not in command.tcp_targets_w:
            raise ValueError(
                f"G1 motion command {command.name!r} declares orientation for {group_name!r} "
                "without a matching TCP position target."
            )
        if group_name not in G1_JOINT_GROUP_NAMES:
            raise ValueError(
                f"G1 motion command {command.name!r} requests unsupported TCP group {group_name!r}."
            )
    for group_name, frame in command.tcp_frame_by_group.items():
        if group_name not in command.tcp_targets_w:
            raise ValueError(
                f"G1 motion command {command.name!r} declares target frame for {group_name!r} "
                "without a matching TCP position target."
            )
        if group_name not in G1_JOINT_GROUP_NAMES:
            raise ValueError(
                f"G1 motion command {command.name!r} requests unsupported TCP group {group_name!r}."
            )
        if frame not in ("world", "base"):
            raise ValueError(
                f"G1 motion command {command.name!r} uses unsupported target frame {frame!r}."
            )
    for group_name in command.joint_targets_by_group:
        if group_name not in G1_JOINT_GROUP_NAMES:
            raise ValueError(
                f"G1 motion command {command.name!r} requests unsupported joint-target group {group_name!r}."
            )
    for group_name in command.gripper_open_by_group:
        if group_name not in G1_BINARY_GROUP_NAMES:
            raise ValueError(
                f"G1 motion command {command.name!r} requests unsupported gripper group {group_name!r}."
            )


def with_target_settle_frames(
    frames: tuple[G1ActionFrame, ...], *, settle_steps: int
) -> tuple[G1ActionFrame, ...]:
    """Hold each named target, including the final target, for settle frames."""

    if not frames or settle_steps <= 0:
        return frames
    settled: list[G1ActionFrame] = []
    for index, frame in enumerate(frames):
        settled.append(frame)
        next_name = frames[index + 1].name if index + 1 < len(frames) else None
        if next_name != frame.name:
            settled.extend(frame for _ in range(int(settle_steps)))
    return tuple(settled)


def normalize_g1_bool_array(value: Any, *, num_envs: int) -> np.ndarray:
    """Return scalar or vector boolean values as a per-env numpy array."""

    values = np.asarray(value, dtype=bool)
    if values.ndim == 0:
        values = np.full((int(num_envs),), bool(values), dtype=bool)
    if values.shape == (1,) and int(num_envs) > 1:
        values = np.repeat(values, int(num_envs), axis=0)
    if values.shape != (int(num_envs),):
        raise ValueError(
            f"Binary G1 group values must be scalar or have shape ({int(num_envs)},), got {values.shape}."
        )
    return values.astype(bool, copy=False)


def normalize_g1_bool_tensor(
    value: Any, *, num_envs: int, device: torch.device | str
) -> torch.Tensor:
    """Return scalar or vector boolean values as a per-env torch tensor."""

    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.bool)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1).repeat(int(num_envs))
        if tensor.shape == (1,) and int(num_envs) > 1:
            tensor = tensor.repeat(int(num_envs))
        if tensor.shape != (int(num_envs),):
            raise ValueError(
                "Binary G1 group values must be scalar or have shape "
                f"({int(num_envs)},), got {tuple(tensor.shape)}."
            )
        return tensor
    return torch.as_tensor(
        normalize_g1_bool_array(value, num_envs=num_envs),
        device=device,
        dtype=torch.bool,
    )


def _action_row_selection(
    env: Any | None,
    action_tensor: torch.Tensor,
    env_ids: Sequence[int] | torch.Tensor | None,
) -> _ActionRowSelection:
    """Resolve requested env ids to local rows in ``action_tensor``.

    Sequence and task-flow agents often emit compact action tensors containing only the rows
    they own, while planner agents emit full vector-env tensors.  G1 writers
    operate on the tensor they receive, so local action-row indices must be kept
    separate from optional global env-row ids.
    """

    action_row_count = int(action_tensor.shape[0])
    env_row_count = _env_row_count(env)
    if env_ids is None:
        return _ActionRowSelection(
            action_rows=torch.arange(
                action_row_count, device=action_tensor.device, dtype=torch.long
            ),
            env_ids=None,
            env_row_count=env_row_count,
        )

    ids = _env_id_tensor(env_ids, device=action_tensor.device)
    if env_row_count is not None and torch.any(ids >= int(env_row_count)):
        raise ValueError(f"env_ids out of range for num_envs={int(env_row_count)}.")

    if env_row_count is not None and action_row_count == int(env_row_count):
        action_rows = ids
    elif action_row_count == int(ids.numel()):
        action_rows = torch.arange(
            action_row_count, device=action_tensor.device, dtype=torch.long
        )
    elif torch.all(ids < action_row_count):
        action_rows = ids
    else:
        raise ValueError(
            "env_ids must index a full action tensor or match compact action rows; "
            f"got action rows={action_row_count}, env_ids={tuple(int(i) for i in ids.tolist())}."
        )
    return _ActionRowSelection(
        action_rows=action_rows,
        env_ids=ids,
        env_row_count=env_row_count,
    )


def _env_id_tensor(
    env_ids: Sequence[int] | torch.Tensor, *, device: torch.device | str
) -> torch.Tensor:
    """Return validated global env-row ids on ``device``."""

    if isinstance(env_ids, torch.Tensor):
        if (
            env_ids.dtype == torch.bool
            or torch.is_floating_point(env_ids)
            or torch.is_complex(env_ids)
        ):
            raise ValueError("env_ids must contain integers.")
        ids = env_ids.detach().reshape(-1).to(device=device, dtype=torch.long)
    else:
        if isinstance(env_ids, str):
            raise ValueError("env_ids must contain integers.")
        values = tuple(env_ids)
        if not all(
            isinstance(env_id, int) and not isinstance(env_id, bool)
            for env_id in values
        ):
            raise ValueError("env_ids must contain integers.")
        ids = torch.tensor(values, device=device, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("env_ids must be a non-empty one-dimensional sequence.")
    if torch.any(ids < 0):
        raise ValueError("env_ids must be non-negative.")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("env_ids must be unique.")
    return ids


def _env_row_count(env: Any | None) -> int | None:
    """Return ``env.num_envs`` when available."""

    if env is None:
        return None
    unwrapped = getattr(env, "unwrapped", env)
    num_envs = getattr(unwrapped, "num_envs", None)
    if num_envs is None:
        return None
    return int(num_envs)


def _select_action_values(
    value_tensor: torch.Tensor,
    *,
    action_tensor: torch.Tensor,
    selection: _ActionRowSelection,
    value_width: int,
    value_label: str,
) -> torch.Tensor:
    """Select compact values for the requested action rows."""

    selected_count = int(selection.action_rows.numel())
    if value_tensor.shape == (int(action_tensor.shape[0]), int(value_width)):
        return value_tensor[selection.action_rows]
    if (
        selection.env_ids is not None
        and selection.env_row_count is not None
        and value_tensor.shape == (int(selection.env_row_count), int(value_width))
    ):
        return value_tensor[selection.env_ids]
    if value_tensor.shape == (selected_count, int(value_width)):
        return value_tensor
    raise ValueError(
        f"{value_label} must have shape ({value_width},), "
        f"({selected_count}, {value_width}), or match the action/env row count; "
        f"got {tuple(value_tensor.shape)}."
    )


def _selected_bool_values(
    values: Any, *, action_tensor: torch.Tensor, selection: _ActionRowSelection
) -> Any:
    """Return bool values scoped to selected action rows."""

    tensor = torch.as_tensor(values, device=action_tensor.device)
    selected_count = int(selection.action_rows.numel())
    if tensor.ndim == 0 or tensor.numel() == 1:
        return tensor
    tensor = tensor.reshape(-1)
    if tensor.shape == (int(action_tensor.shape[0]),):
        return tensor[selection.action_rows]
    if (
        selection.env_ids is not None
        and selection.env_row_count is not None
        and tensor.shape == (int(selection.env_row_count),)
    ):
        return tensor[selection.env_ids]
    if tensor.shape == (selected_count,):
        return tensor
    raise ValueError(
        "Binary G1 group values must be scalar, full-env shaped, or selected-env shaped; "
        f"got {tuple(tensor.shape)}."
    )


def _action_term_names(action_manager: Any) -> tuple[str, ...]:
    for attr_name in ("active_terms", "term_names"):
        names = getattr(action_manager, attr_name, None)
        if isinstance(names, (list, tuple)):
            return tuple(str(name) for name in names)
    terms = getattr(action_manager, "_terms", None)
    if isinstance(terms, dict):
        return tuple(str(name) for name in terms)
    raise ValueError(
        "Cannot resolve IsaacLab action term order: expected action_manager.active_terms, "
        "action_manager.term_names, or ordered action_manager._terms."
    )


def _env_cfg_action_term(unwrapped_env: Any, term_name: str) -> Any | None:
    env_cfg = getattr(unwrapped_env, "cfg", None)
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        return None
    return getattr(actions_cfg, term_name, None)


def _action_cfg_sources(term: Any | None, cfg_term: Any | None) -> tuple[Any, ...]:
    sources: list[Any] = []
    for source in (
        term,
        getattr(term, "cfg", None) if term is not None else None,
        getattr(term, "_cfg", None) if term is not None else None,
        cfg_term,
    ):
        if source is not None:
            sources.append(source)
    return tuple(sources)


def _resolve_g1_action_group(term_name: str, joint_names: Sequence[str]) -> str:
    group_name = _G1_ACTION_GROUP_BY_DOF_ORDER.get(tuple(joint_names))
    if group_name is not None:
        return group_name
    raise ValueError(
        f"Action term {term_name!r} uses joint names that do not match a "
        f"planner-supported G1 action group: {tuple(joint_names)}. "
        "Use G1 base, legs, left/right arm, or left/right gripper action cfgs. "
        "Base is supported as a pass-through action group and is not planned by cuRobo."
    )


def _resolve_joint_names(
    term_name: str, term: Any | None, sources: Sequence[Any]
) -> tuple[str, ...]:
    for source, attr_name in ((term, "joint_names"), (term, "_joint_names")):
        if source is not None and hasattr(source, attr_name):
            return tuple(str(name) for name in getattr(source, attr_name))
    for source in sources:
        if hasattr(source, "joint_names"):
            return tuple(str(name) for name in getattr(source, "joint_names"))
    raise ValueError(
        f"Action term {term_name!r} is missing joint-name validation data."
    )


def _validate_non_overlapping_slices(terms: Sequence[G1ActionTerm]) -> None:
    occupied: set[int] = set()
    for term in terms:
        for column in range(int(term.action_slice.start), int(term.action_slice.stop)):
            if column in occupied:
                raise ValueError(
                    f"G1 action tensor term {term.term_name!r} overlaps another term at column {column}."
                )
            occupied.add(column)


__all__ = [
    "G1_ACTION_GROUP_DOF_ORDER",
    "G1_BINARY_GROUP_NAMES",
    "G1_JOINT_GROUP_NAMES",
    "G1ActionFrame",
    "G1ActionLayout",
    "G1ActionTerm",
    "current_g1_group_joint_positions",
    "make_g1_action_layout_from_env",
    "make_g1_action_tensor",
    "validate_g1_motion_command_groups",
    "with_target_settle_frames",
    "write_g1_frame_action",
    "write_g1_initial_action",
]
