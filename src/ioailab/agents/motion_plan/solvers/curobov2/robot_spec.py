"""Static robot planning specifications for cuRobo v2 helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ioailab.agents.motion_plan.solvers.curobov2.utils.adapter import (
    Curobo2ParallelWBIK,
    Curobo2ParallelWBIKConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_ROBOT_BASE_LINK_NAME = "base_link"


@dataclass(frozen=True, slots=True)
class MotionGroupSpec:
    """Named movable joint group for a robot planning spec."""

    name: str
    joint_names: tuple[str, ...]
    target_frame_name: str | None = None
    group_kind: str = "joint"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self,
            "joint_names",
            tuple(str(joint_name) for joint_name in self.joint_names),
        )
        object.__setattr__(
            self,
            "target_frame_name",
            None if self.target_frame_name is None else str(self.target_frame_name),
        )
        object.__setattr__(self, "group_kind", str(self.group_kind))


@dataclass(frozen=True, slots=True)
class BinaryGroupSpec:
    """Named binary group for metadata such as open/close state."""

    name: str
    value_names: tuple[str, ...] = ("is_open",)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self,
            "value_names",
            tuple(str(value_name) for value_name in self.value_names),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class RobotPlanningSpec:
    """Static planning envelope for one robot family.

    The spec is deliberately data-only. It tells cuRobo which joints and tool
    frames exist, but it does not execute IsaacLab actions or pack tensors.
    """

    robot_name: str
    whole_body_joint_names: tuple[str, ...]
    motion_groups: Mapping[str, MotionGroupSpec]
    binary_groups: Mapping[str, BinaryGroupSpec] = field(default_factory=dict)
    default_joint_positions: Mapping[str, float] = field(default_factory=dict)
    self_collision_ignore_pairs: tuple[tuple[str, str], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        group_map = {str(name): spec for name, spec in self.motion_groups.items()}
        binary_map = {str(name): spec for name, spec in self.binary_groups.items()}
        whole_names = tuple(
            str(joint_name) for joint_name in self.whole_body_joint_names
        )
        _validate_spec_groups(whole_names, group_map)
        object.__setattr__(self, "robot_name", str(self.robot_name))
        object.__setattr__(self, "whole_body_joint_names", whole_names)
        object.__setattr__(self, "motion_groups", group_map)
        object.__setattr__(self, "binary_groups", binary_map)
        object.__setattr__(
            self,
            "default_joint_positions",
            {
                str(name): float(value)
                for name, value in self.default_joint_positions.items()
            },
        )
        object.__setattr__(
            self,
            "self_collision_ignore_pairs",
            tuple(
                (str(left), str(right))
                for left, right in self.self_collision_ignore_pairs
            ),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


def resolve_planning_inputs(
    spec: RobotPlanningSpec,
    *,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve and validate active joints plus tool frames for a planning spec."""

    active_names = tuple(
        str(name) for name in (active_joint_names or spec.whole_body_joint_names)
    )
    validate_joint_subset(
        active_names, spec.whole_body_joint_names, field_name="active_joint_names"
    )
    resolved_tool_frame_names = resolve_tool_frame_names(
        spec, tool_frame_names=tool_frame_names
    )
    return active_names, resolved_tool_frame_names


def make_curobo_robot_config(
    spec: RobotPlanningSpec,
    *,
    urdf_path: str | Path | None = None,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
    cspace_distance_weights: Mapping[str, float] | None = None,
    null_space_weights: Mapping[str, float] | None = None,
    base_link_name: str | None = None,
) -> dict[str, Any]:
    """Build a cuRobo robot config from a static robot planning spec."""

    active_names, resolved_tool_frame_names = resolve_planning_inputs(
        spec,
        active_joint_names=active_joint_names,
        tool_frame_names=tool_frame_names,
    )
    resolved_urdf_path = resolve_robot_urdf_path(
        urdf_path or spec.metadata.get("urdf_path")
    )
    chain_names = required_chain_joint_names(
        spec,
        active_joint_names=active_names,
        tool_frame_names=resolved_tool_frame_names,
    )
    cspace_joint_names = tuple(dict.fromkeys((*chain_names, *active_names)))
    active_set = set(active_names)
    locked_joint_names = tuple(name for name in chain_names if name not in active_set)
    default_positions = spec.default_joint_positions

    return {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": str(resolved_urdf_path),
                "asset_root_path": str(resolved_urdf_path.parents[1]),
                "base_link": str(
                    base_link_name
                    or spec.metadata.get("base_link_name", DEFAULT_ROBOT_BASE_LINK_NAME)
                ),
                "tool_frames": [str(name) for name in resolved_tool_frame_names],
                "lock_joints": {
                    joint_name: float(default_positions.get(joint_name, 0.0))
                    for joint_name in locked_joint_names
                },
                "cspace": {
                    "joint_names": list(cspace_joint_names),
                    "default_joint_position": [
                        float(default_positions.get(joint_name, 0.0))
                        for joint_name in cspace_joint_names
                    ],
                    "cspace_distance_weight": named_cspace_weights(
                        cspace_joint_names, cspace_distance_weights
                    ),
                    "null_space_weight": named_cspace_weights(
                        cspace_joint_names, null_space_weights
                    ),
                    "max_acceleration": [5.0 for _ in cspace_joint_names],
                    "max_jerk": [10.0 for _ in cspace_joint_names],
                },
            }
        }
    }


def make_curobo_parallel_wbik(
    spec: RobotPlanningSpec,
    *,
    urdf_path: str | Path | None = None,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
    cspace_distance_weights: Mapping[str, float] | None = None,
    null_space_weights: Mapping[str, float] | None = None,
    device: str = "cuda:0",
    use_cuda_graph: bool = True,
    run_optimizer: bool = True,
    self_collision_check: bool = False,
    load_collision_spheres: bool = False,
    num_seeds: int = 64,
    return_seeds: int = 8,
    seed_config_noise_std: float = 0.18,
    seed_config_noise_scales: Mapping[str, float] | Sequence[float] | None = None,
    seed_solver_num_seeds: int | None = None,
    position_tolerance: float = 0.005,
    orientation_tolerance: float = 0.05,
    override_optimizer_num_iters: Mapping[str, int | None] | None = None,
) -> Curobo2ParallelWBIK:
    """Create a cuRobo parallel WBIK solver for a robot planning spec."""

    active_names, resolved_tool_frame_names = resolve_planning_inputs(
        spec,
        active_joint_names=active_joint_names,
        tool_frame_names=tool_frame_names,
    )
    return Curobo2ParallelWBIK(
        Curobo2ParallelWBIKConfig(
            robot_config=make_curobo_robot_config(
                spec,
                urdf_path=urdf_path,
                active_joint_names=active_names,
                tool_frame_names=resolved_tool_frame_names,
                cspace_distance_weights=cspace_distance_weights,
                null_space_weights=null_space_weights,
            ),
            whole_body_joint_names=spec.whole_body_joint_names,
            active_joint_names=active_names,
            tool_frame_names=resolved_tool_frame_names,
            device=device,
            use_cuda_graph=use_cuda_graph,
            run_optimizer=run_optimizer,
            self_collision_check=self_collision_check,
            load_collision_spheres=load_collision_spheres,
            num_seeds=num_seeds,
            return_seeds=return_seeds,
            seed_config_noise_std=seed_config_noise_std,
            seed_config_noise_scales=seed_config_noise_scales,
            seed_solver_num_seeds=seed_solver_num_seeds,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            override_optimizer_num_iters=(
                {"lbfgs": None}
                if override_optimizer_num_iters is None
                else {
                    str(name): value
                    for name, value in override_optimizer_num_iters.items()
                }
            ),
        )
    )


def resolve_tool_frame_names(
    spec: RobotPlanningSpec,
    *,
    tool_frame_names: Sequence[str] | None = None,
    group_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve explicit or group-derived tool frame names."""

    if tool_frame_names is not None:
        return tuple(str(name) for name in tool_frame_names)
    groups = (
        spec.motion_groups.values()
        if group_names is None
        else (spec.motion_groups[str(name)] for name in group_names)
    )
    return tuple(
        group.target_frame_name
        for group in groups
        if group.target_frame_name is not None
    )


def required_chain_joint_names(
    spec: RobotPlanningSpec,
    *,
    active_joint_names: Sequence[str],
    tool_frame_names: Sequence[str],
) -> tuple[str, ...]:
    """Return chain joints required by active groups/tool frames."""

    required_by_group = {
        str(group_name): tuple(str(joint_name) for joint_name in joint_names)
        for group_name, joint_names in dict(
            spec.metadata.get("required_chain_joint_names_by_group", {})
        ).items()
    }
    active_set = set(str(name) for name in active_joint_names)
    tool_set = set(str(name) for name in tool_frame_names)
    required: list[str] = []
    for group_name, group in spec.motion_groups.items():
        group_chain = required_by_group.get(group_name, ())
        group_is_active = bool(active_set.intersection(group.joint_names)) or (
            group.target_frame_name is not None
            and group.target_frame_name in tool_set
            and bool(active_set.intersection(group_chain))
        )
        if group_is_active:
            required.extend(group_chain)
    return tuple(dict.fromkeys(required))


def validate_joint_subset(
    joint_names: Sequence[str], allowed_joint_names: Sequence[str], *, field_name: str
) -> None:
    """Validate that all named joints belong to an allowed joint set."""

    allowed = set(str(name) for name in allowed_joint_names)
    unknown = tuple(str(name) for name in joint_names if str(name) not in allowed)
    if unknown:
        raise ValueError(
            f"{field_name} contains joints outside the robot planning spec: {unknown!r}."
        )


def named_cspace_weights(
    joint_names: Sequence[str],
    weights: Mapping[str, float] | None,
    *,
    default: float = 1.0,
) -> list[float]:
    """Return cspace weights in joint order with optional named overrides."""

    overrides = (
        {}
        if weights is None
        else {str(name): float(value) for name, value in weights.items()}
    )
    return [
        float(overrides.get(str(joint_name), default)) for joint_name in joint_names
    ]


def resolve_robot_urdf_path(urdf_path: str | Path | None) -> Path:
    """Resolve a robot URDF path relative to the repository root."""

    if urdf_path is None:
        raise ValueError(
            "RobotPlanningSpec metadata must provide urdf_path or caller must pass urdf_path."
        )
    path = Path(urdf_path)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.is_file():
        return cwd_candidate
    return PROJECT_ROOT / path


def _validate_spec_groups(
    whole_joint_names: Sequence[str], motion_groups: Mapping[str, MotionGroupSpec]
) -> None:
    allowed = set(str(name) for name in whole_joint_names)
    for group_name, group in motion_groups.items():
        unknown = tuple(
            joint_name for joint_name in group.joint_names if joint_name not in allowed
        )
        if unknown:
            raise ValueError(
                f"Motion group {group_name!r} contains joints outside whole_body_joint_names: {unknown!r}."
            )
