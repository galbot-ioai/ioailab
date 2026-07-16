"""Runtime request/result data for grouped cuRobo v2 waypoint plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ioailab.agents.motion_plan.solvers.curobov2.utils.adapter import (
    Curobo2WBIKRequest,
    Curobo2WBIKResult,
)
from ioailab.agents.motion_plan.solvers.curobov2.robot_spec import (
    RobotPlanningSpec,
    make_curobo_parallel_wbik,
)
from ioailab.agents.motion_plan.solvers.curobov2.utils.pose import (
    normalize_pose_xyz_wxyz,
    resolve_target_pose_xyz_wxyz,
)
from ioailab.agents.motion_plan.solvers.curobov2.utils.tensors import (
    expand_binary_values,
    map_q_to_group,
    validate_sample_target_indices,
    validate_step_success,
)


@dataclass(frozen=True, slots=True)
class TargetPose:
    """One group TCP target for a planning target step.

    Args:
        group_name: Motion group name in the robot planning spec.
        pose_xyz_wxyz: Pose array in ``xyz + wxyz`` order.
        frame: ``"base"`` for robot-base frame or ``"world"`` with request
            ``base_pose_by_env`` supplied.
    """

    group_name: str
    pose_xyz_wxyz: np.ndarray
    frame: str = "base"

    def __init__(
        self, group_name: str, pose_xyz_wxyz: Any, frame: str = "base"
    ) -> None:
        object.__setattr__(self, "group_name", str(group_name))
        object.__setattr__(
            self, "pose_xyz_wxyz", normalize_pose_xyz_wxyz(pose_xyz_wxyz)
        )
        object.__setattr__(self, "frame", _normalize_target_frame(frame))


@dataclass(frozen=True, slots=True)
class TargetStep:
    """Named target phase for a grouped cuRobo waypoint plan."""

    name: str
    target_poses_by_group: Mapping[str, TargetPose] = field(default_factory=dict)
    joint_targets_by_group: Mapping[str, Any] = field(default_factory=dict)
    active_groups: tuple[str, ...] | None = None
    binary_values_by_group: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        target_poses_by_group: Mapping[str, TargetPose] | None = None,
        joint_targets_by_group: Mapping[str, Any] | None = None,
        active_groups: Sequence[str] | None = None,
        binary_values_by_group: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "name", str(name))
        object.__setattr__(
            self,
            "target_poses_by_group",
            {
                str(group_name): pose
                for group_name, pose in (target_poses_by_group or {}).items()
            },
        )
        object.__setattr__(
            self,
            "joint_targets_by_group",
            {
                str(group_name): value
                for group_name, value in (joint_targets_by_group or {}).items()
            },
        )
        object.__setattr__(
            self,
            "active_groups",
            None
            if active_groups is None
            else tuple(str(group_name) for group_name in active_groups),
        )
        object.__setattr__(
            self,
            "binary_values_by_group",
            {
                str(group_name): value
                for group_name, value in (binary_values_by_group or {}).items()
            },
        )
        object.__setattr__(self, "metadata", dict(metadata or {}))


@dataclass(frozen=True, slots=True)
class CuroboPlanningRequest:
    """Runtime inputs for a grouped cuRobo waypoint solve."""

    spec: RobotPlanningSpec
    start_q: np.ndarray
    target_steps: tuple[TargetStep, ...]
    active_groups: tuple[str, ...]
    locked_groups: tuple[str, ...] = ()
    base_pose_by_env: np.ndarray | None = None
    nullspace_q: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        spec: RobotPlanningSpec,
        start_q: Any,
        target_steps: Sequence[TargetStep],
        active_groups: Sequence[str],
        locked_groups: Sequence[str] = (),
        base_pose_by_env: Any | None = None,
        nullspace_q: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        start = np.asarray(start_q, dtype=np.float32)
        if start.ndim == 1:
            start = start[None, :]
        if start.ndim != 2 or start.shape[1] != len(spec.whole_body_joint_names):
            raise ValueError(
                "start_q must have shape (num_envs, whole_body_joints), got "
                f"{start.shape} for {len(spec.whole_body_joint_names)} joints."
            )
        if not target_steps:
            raise ValueError("target_steps must contain at least one TargetStep.")
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "start_q", start)
        object.__setattr__(self, "target_steps", tuple(target_steps))
        object.__setattr__(
            self,
            "active_groups",
            tuple(str(group_name) for group_name in active_groups),
        )
        object.__setattr__(
            self,
            "locked_groups",
            tuple(str(group_name) for group_name in locked_groups),
        )
        object.__setattr__(
            self,
            "base_pose_by_env",
            None
            if base_pose_by_env is None
            else np.asarray(base_pose_by_env, dtype=np.float32),
        )
        nullspace = (
            None if nullspace_q is None else np.asarray(nullspace_q, dtype=np.float32)
        )
        if nullspace is not None:
            if nullspace.ndim == 1:
                nullspace = nullspace[None, :]
            if nullspace.shape != start.shape:
                raise ValueError(
                    f"nullspace_q must match start_q shape {start.shape}, got {nullspace.shape}."
                )
        object.__setattr__(self, "nullspace_q", nullspace)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        _validate_request_groups(self)


@dataclass(frozen=True, slots=True)
class JointGroupTrajectory:
    """Executable samples for one named joint group."""

    group_name: str
    joint_names: tuple[str, ...]
    positions: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float32)
        if positions.ndim != 3:
            raise ValueError(
                f"JointGroupTrajectory.positions must have shape (K, N, J), got {positions.shape}."
            )
        if positions.shape[2] != len(self.joint_names):
            raise ValueError(
                f"JointGroupTrajectory {self.group_name!r} has width {positions.shape[2]}, "
                f"expected {len(self.joint_names)}."
            )
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True, slots=True)
class BinaryGroupTrajectory:
    """Executable binary samples for one named group."""

    group_name: str
    value_names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=bool)
        if values.ndim != 2:
            raise ValueError(
                f"BinaryGroupTrajectory.values must have shape (K, N), got {values.shape}."
            )
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PoseGroupTrajectory:
    """Achieved pose samples for one named target frame, if provided."""

    group_name: str
    frame_name: str
    poses_xyz_wxyz: np.ndarray

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses_xyz_wxyz, dtype=np.float32)
        if poses.ndim == 2 and poses.shape[1] == 7:
            poses = poses[None, :, :]
        if poses.ndim != 3 or poses.shape[2] != 7:
            raise ValueError(
                f"PoseGroupTrajectory.poses_xyz_wxyz must have shape (K, N, 7), got {poses.shape}."
            )
        normalized = normalize_pose_xyz_wxyz(
            poses.reshape((-1, 7)), field_name="poses_xyz_wxyz"
        ).reshape(poses.shape)
        object.__setattr__(self, "poses_xyz_wxyz", normalized)


@dataclass(frozen=True, slots=True)
class GroupedWaypointPlan:
    """Planner output: named target steps plus executable grouped samples."""

    target_step_names: tuple[str, ...]
    step_success_by_env: np.ndarray
    sample_names: tuple[str, ...]
    sample_target_indices: np.ndarray
    joint_groups: Mapping[str, JointGroupTrajectory]
    binary_groups: Mapping[str, BinaryGroupTrajectory] = field(default_factory=dict)
    pose_groups: Mapping[str, PoseGroupTrajectory] = field(default_factory=dict)
    summaries_by_step: tuple[tuple[dict[str, Any], ...], ...] = ()
    raw_results: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_success = validate_step_success(
            self.step_success_by_env,
            step_count=len(self.target_step_names),
            num_envs=_infer_num_envs(
                self.joint_groups, self.binary_groups, self.step_success_by_env
            ),
        )
        sample_indices = validate_sample_target_indices(
            self.sample_target_indices,
            sample_count=len(self.sample_names),
            step_count=len(self.target_step_names),
        )
        object.__setattr__(self, "step_success_by_env", step_success)
        object.__setattr__(self, "sample_target_indices", sample_indices)
        object.__setattr__(self, "joint_groups", dict(self.joint_groups))
        object.__setattr__(self, "binary_groups", dict(self.binary_groups))
        object.__setattr__(self, "pose_groups", dict(self.pose_groups))
        object.__setattr__(self, "metadata", dict(self.metadata))


def compute_curobo_grouped_waypoints(
    request: CuroboPlanningRequest,
    *,
    context: Any | None = None,
) -> GroupedWaypointPlan:
    """Compute grouped waypoints without executing IsaacLab actions.

    The returned plan is plain data. User/example code remains responsible for
    converting each group into ioailab action tensors and advancing IsaacLab.
    """

    solver_context = context
    current_q = request.start_q.copy()
    step_q: list[np.ndarray] = []
    step_success: list[np.ndarray] = []
    summaries_by_step: list[tuple[dict[str, Any], ...]] = []
    raw_results: list[Any] = []

    for step in request.target_steps:
        target_poses = _target_poses_by_frame(request, step)
        if target_poses:
            if solver_context is None:
                solver_context = _make_default_context(request)
            result = solver_context.solver.solve(
                Curobo2WBIKRequest(
                    start_q=current_q,
                    target_poses_xyz_wxyz_by_frame=target_poses,
                    nullspace_q=request.nullspace_q,
                )
            )
            current_q = np.asarray(result.q, dtype=np.float32).copy()
            success = _result_success_vector(result, num_envs=request.start_q.shape[0])
            summaries = tuple(dict(summary) for summary in result.summaries)
            raw_result = result
        else:
            success = np.ones((request.start_q.shape[0],), dtype=bool)
            backend = "joint_target" if step.joint_targets_by_group else "hold"
            summaries = tuple(
                {"success": True, "backend": backend}
                for _ in range(request.start_q.shape[0])
            )
            raw_result = None
        if step.joint_targets_by_group:
            current_q = _apply_joint_targets(request, current_q, step)
        step_q.append(current_q.copy())
        step_success.append(success)
        summaries_by_step.append(summaries)
        raw_results.append(raw_result)

    if solver_context is not None:
        solver_context.current_q = current_q.copy()
    q_samples = np.stack(step_q, axis=0).astype(np.float32, copy=False)
    sample_target_indices = np.arange(len(request.target_steps), dtype=np.int64)
    sample_names = tuple(step.name for step in request.target_steps)
    joint_groups = _joint_group_trajectories(request, q_samples)
    binary_groups = _binary_group_trajectories(
        request, sample_target_indices=sample_target_indices
    )
    return GroupedWaypointPlan(
        target_step_names=tuple(step.name for step in request.target_steps),
        step_success_by_env=np.stack(step_success, axis=0).astype(bool, copy=False),
        sample_names=sample_names,
        sample_target_indices=sample_target_indices,
        joint_groups=joint_groups,
        binary_groups=binary_groups,
        summaries_by_step=tuple(summaries_by_step),
        raw_results=tuple(raw_results),
        metadata=request.metadata,
    )


def _target_poses_by_frame(
    request: CuroboPlanningRequest, step: TargetStep
) -> dict[str, np.ndarray]:
    target_poses: dict[str, np.ndarray] = {}
    for group_name, target_pose in step.target_poses_by_group.items():
        if group_name not in request.spec.motion_groups:
            raise ValueError(
                f"TargetStep {step.name!r} references unknown motion group {group_name!r}."
            )
        group = request.spec.motion_groups[group_name]
        if group.target_frame_name is None:
            raise ValueError(
                f"Motion group {group_name!r} does not define a target_frame_name."
            )
        target_poses[group.target_frame_name] = resolve_target_pose_xyz_wxyz(
            target_pose,
            num_envs=request.start_q.shape[0],
            base_pose_by_env=request.base_pose_by_env,
        )
    return target_poses


def _apply_joint_targets(
    request: CuroboPlanningRequest,
    current_q: np.ndarray,
    step: TargetStep,
) -> np.ndarray:
    """Apply direct joint targets to the current whole-body configuration."""

    q = np.asarray(current_q, dtype=np.float32).copy()
    num_envs = int(request.start_q.shape[0])
    for group_name, targets in step.joint_targets_by_group.items():
        if group_name not in request.spec.motion_groups:
            raise ValueError(
                f"TargetStep {step.name!r} references unknown joint-target group {group_name!r}."
            )
        group = request.spec.motion_groups[group_name]
        if not group.joint_names:
            raise ValueError(
                f"TargetStep {step.name!r} joint-target group {group_name!r} has no joints."
            )
        values = np.asarray(targets, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape == (1, len(group.joint_names)) and num_envs > 1:
            values = np.repeat(values, num_envs, axis=0)
        expected_shape = (num_envs, len(group.joint_names))
        if values.shape != expected_shape:
            raise ValueError(
                f"TargetStep {step.name!r} joint-target group {group_name!r} must have shape "
                f"{expected_shape}, got {values.shape}."
            )
        for column, joint_name in enumerate(group.joint_names):
            q[:, request.spec.whole_body_joint_names.index(joint_name)] = values[
                :, column
            ]
    return q


def _normalize_target_frame(frame: Any) -> str:
    """Return a normalized target frame name accepted by cuRobo planning."""

    normalized = str(frame).lower()
    if normalized not in ("base", "world"):
        raise ValueError(f"TargetPose.frame must be 'base' or 'world', got {frame!r}.")
    return normalized


def _joint_group_trajectories(
    request: CuroboPlanningRequest,
    q_samples: np.ndarray,
) -> dict[str, JointGroupTrajectory]:
    trajectories: dict[str, JointGroupTrajectory] = {}
    for group_name in request.active_groups:
        if group_name in request.locked_groups:
            continue
        group = request.spec.motion_groups[group_name]
        if not group.joint_names:
            continue
        trajectories[group_name] = JointGroupTrajectory(
            group_name=group_name,
            joint_names=group.joint_names,
            positions=map_q_to_group(
                q_samples, request.spec.whole_body_joint_names, group.joint_names
            ),
        )
    return trajectories


def _binary_group_trajectories(
    request: CuroboPlanningRequest,
    *,
    sample_target_indices: np.ndarray,
) -> dict[str, BinaryGroupTrajectory]:
    group_names = tuple(
        dict.fromkeys(
            group_name
            for step in request.target_steps
            for group_name in step.binary_values_by_group
            if group_name in request.spec.binary_groups
        )
    )
    trajectories: dict[str, BinaryGroupTrajectory] = {}
    for group_name in group_names:
        binary_spec = request.spec.binary_groups[group_name]
        per_step_values = []
        for step in request.target_steps:
            if group_name not in step.binary_values_by_group:
                raise ValueError(
                    f"Binary group {group_name!r} is missing from TargetStep {step.name!r}; "
                    "define the group on every step or omit it entirely."
                )
            per_step_values.append(step.binary_values_by_group[group_name])
        trajectories[group_name] = BinaryGroupTrajectory(
            group_name=group_name,
            value_names=binary_spec.value_names,
            values=expand_binary_values(
                per_step_values,
                num_envs=request.start_q.shape[0],
                sample_target_indices=sample_target_indices,
            ),
        )
    return trajectories


def _result_success_vector(result: Curobo2WBIKResult, *, num_envs: int) -> np.ndarray:
    success = np.asarray(result.success, dtype=bool)
    if success.ndim == 0:
        success = np.full((int(num_envs),), bool(success), dtype=bool)
    if success.shape != (int(num_envs),):
        raise ValueError(
            f"cuRobo result success must have shape ({int(num_envs)},), got {success.shape}."
        )
    return success


def _make_default_context(request: CuroboPlanningRequest) -> Any:
    solver = make_curobo_parallel_wbik(
        request.spec,
        active_joint_names=_active_joint_names_for_request(request),
        tool_frame_names=_tool_frame_names_for_request(request),
    )
    return _PlanningContext(solver=solver)


def _active_joint_names_for_request(request: CuroboPlanningRequest) -> tuple[str, ...]:
    names: list[str] = []
    for group_name in request.active_groups:
        group = request.spec.motion_groups[group_name]
        names.extend(group.joint_names)
    return tuple(dict.fromkeys(names)) or request.spec.whole_body_joint_names


def _tool_frame_names_for_request(request: CuroboPlanningRequest) -> tuple[str, ...]:
    frames: list[str] = []
    for step in request.target_steps:
        for group_name in step.target_poses_by_group:
            group = request.spec.motion_groups[group_name]
            if group.target_frame_name is not None:
                frames.append(group.target_frame_name)
    return tuple(dict.fromkeys(frames))


def _validate_request_groups(request: CuroboPlanningRequest) -> None:
    unknown_active = tuple(
        group
        for group in request.active_groups
        if group not in request.spec.motion_groups
    )
    if unknown_active:
        raise ValueError(
            f"active_groups contains unknown motion groups: {unknown_active!r}."
        )
    unknown_locked = tuple(
        group
        for group in request.locked_groups
        if group not in request.spec.motion_groups
    )
    if unknown_locked:
        raise ValueError(
            f"locked_groups contains unknown motion groups: {unknown_locked!r}."
        )
    for step in request.target_steps:
        unknown_step_active = (
            ()
            if step.active_groups is None
            else tuple(
                group
                for group in step.active_groups
                if group not in request.spec.motion_groups
            )
        )
        if unknown_step_active:
            raise ValueError(
                f"TargetStep {step.name!r} contains unknown active groups: {unknown_step_active!r}."
            )
        mismatched_targets = tuple(
            key
            for key, pose in step.target_poses_by_group.items()
            if str(pose.group_name) != str(key)
        )
        if mismatched_targets:
            raise ValueError(
                f"TargetStep {step.name!r} has TargetPose group names that do not match their mapping keys: "
                f"{mismatched_targets!r}."
            )
        unknown_targets = tuple(
            group
            for group in step.target_poses_by_group
            if group not in request.spec.motion_groups
        )
        if unknown_targets:
            raise ValueError(
                f"TargetStep {step.name!r} contains unknown target groups: {unknown_targets!r}."
            )
        unknown_joint_targets = tuple(
            group
            for group in step.joint_targets_by_group
            if group not in request.spec.motion_groups
        )
        if unknown_joint_targets:
            raise ValueError(
                f"TargetStep {step.name!r} contains unknown joint-target groups: {unknown_joint_targets!r}."
            )
        unknown_binary = tuple(
            group
            for group in step.binary_values_by_group
            if group not in request.spec.binary_groups
        )
        if unknown_binary:
            raise ValueError(
                f"TargetStep {step.name!r} contains unknown binary groups: {unknown_binary!r}."
            )


def _infer_num_envs(
    joint_groups: Mapping[str, JointGroupTrajectory],
    binary_groups: Mapping[str, BinaryGroupTrajectory],
    step_success_by_env: np.ndarray,
) -> int:
    if joint_groups:
        return next(iter(joint_groups.values())).positions.shape[1]
    if binary_groups:
        return next(iter(binary_groups.values())).values.shape[1]
    success = np.asarray(step_success_by_env, dtype=bool)
    if success.ndim != 2:
        raise ValueError(
            f"step_success_by_env must have shape (S, N), got {success.shape}."
        )
    return success.shape[1]


@dataclass(slots=True)
class _PlanningContext:
    solver: Any
    current_q: np.ndarray | None = None
