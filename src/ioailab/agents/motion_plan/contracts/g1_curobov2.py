"""G1-specific cuRobo v2 planner configuration helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import isaaclab.utils.math as pose_utils
import numpy as np
import torch

from ioailab.utils.asset_utils import (
    G1_MOBILE_BASE_URDF_PATH,
    PROJECT_ROOT,
    get_robot_urdf_path,
)
from ioailab.agents.motion_plan.solvers.curobov2.robot_spec import (
    BinaryGroupSpec,
    MotionGroupSpec,
    RobotPlanningSpec,
    make_curobo_parallel_wbik,
    make_curobo_robot_config,
    resolve_planning_inputs,
)
from ioailab.agents.motion_plan.solvers.curobov2.utils import Curobo2ParallelWBIK
from ioailab.agents.motion_plan.solvers.curobov2.utils import (
    current_curobo_q_from_env,
    merge_group_to_whole_q,
    pose_xyz_wxyz_to_xyz_xyzw,
    pose_xyz_xyzw_to_xyz_wxyz,
)
from ioailab.robots.g1.spec import (
    G1_BASE_WHEEL_DOF_ORDER as G1_BASE_WHEEL_DOF_ORDER,
    G1_LEG_DOF_ORDER as G1_LEG_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER as G1_LEFT_ARM_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER as G1_RIGHT_ARM_DOF_ORDER,
    G1_TOP_DOWN_TCP_WXYZ as G1_TOP_DOWN_TCP_WXYZ,
    MANIPULATION_JOINT_POSITIONS as MANIPULATION_JOINT_POSITIONS,
)

DEFAULT_G1_URDF_PATH = get_robot_urdf_path("galbot_g1", required=False).relative_to(
    PROJECT_ROOT
)
DEFAULT_G1_MOBILE_BASE_URDF_PATH = G1_MOBILE_BASE_URDF_PATH.relative_to(PROJECT_ROOT)
DEFAULT_ROBOT_BASE_LINK_NAME = "base_footprint"
DEFAULT_MOBILE_ROBOT_BASE_LINK_NAME = "world_base_link"
G1_HEAD_DOF_ORDER = ("head_joint1", "head_joint2")


def ensure_g1_mobile_base_urdf(
    *,
    urdf_path: str | Path = DEFAULT_G1_MOBILE_BASE_URDF_PATH,
    source_urdf_path: str | Path = DEFAULT_G1_URDF_PATH,
) -> Path:
    """Ensure the planner-only mobile-base URDF exists and return its path."""

    target_path = _resolve_g1_urdf_asset_path(urdf_path)
    if target_path.is_file():
        return target_path

    source_path = _resolve_g1_urdf_asset_path(source_urdf_path)
    if target_path == source_path:
        raise ValueError(
            "Mobile-base planner URDF path must not overwrite the canonical G1 URDF."
        )
    source_text = source_path.read_text(encoding="utf-8")
    robot_tag_match = re.search(r"<robot\b[^>]*>", source_text)
    if robot_tag_match is None:
        raise ValueError(
            f"Canonical G1 URDF {source_path} is missing a <robot> root tag."
        )
    robot_name_match = re.search(r'\bname="([^"]+)"', robot_tag_match.group(0))
    source_robot_name = robot_name_match.group(1) if robot_name_match else "galbot_g1"
    body = source_text[robot_tag_match.end() :]
    body = _fix_wheel_joints_for_mobile_planner(body)

    mobile_header = f"""<?xml version="1.0" ?>
<!-- =================================================================================== -->
<!-- |    This document was derived from robot.xacro for planner-only mobile cuRobo    | -->
<!-- |    Do not use this URDF as the IsaacLab simulation asset.                       | -->
<!-- =================================================================================== -->
<robot name="{source_robot_name}_mobile_base">
  <!-- Planner-only planar root for cuRobo. IsaacLab still drives the real wheel joints. -->
  <link name="world_base_link"/>
  <link name="base_x_link"/>
  <joint name="base_x" type="prismatic">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="world_base_link"/>
    <child link="base_x_link"/>
    <axis xyz="1 0 0"/>
    <limit effort="1000.0" lower="-10.0" upper="10.0" velocity="1.0"/>
  </joint>
  <link name="base_y_link"/>
  <joint name="base_y" type="prismatic">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="base_x_link"/>
    <child link="base_y_link"/>
    <axis xyz="0 1 0"/>
    <limit effort="1000.0" lower="-10.0" upper="10.0" velocity="1.0"/>
  </joint>
  <joint name="base_yaw" type="revolute">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="base_y_link"/>
    <child link="base_footprint"/>
    <axis xyz="0 0 1"/>
    <limit effort="1000.0" lower="-6.2831853" upper="6.2831853" velocity="1.5"/>
  </joint>"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(mobile_header + body, encoding="utf-8")
    return target_path


def _fix_wheel_joints_for_mobile_planner(urdf_body: str) -> str:
    """Return URDF body with physical wheel joints fixed for planner kinematics."""

    fixed_body = urdf_body
    for joint_name in G1_BASE_WHEEL_DOF_ORDER:
        joint_pattern = re.compile(
            rf'(<joint\s+name="{re.escape(joint_name)}"\s+type=")[^"]+(">\s*)'
            r"(.*?)"
            r"(\s*</joint>)",
            re.DOTALL,
        )

        def _replace_joint(match: re.Match[str]) -> str:
            joint_body = re.sub(
                r"\n\s*<(?:axis|limit)\b[^>]*/>",
                "",
                match.group(3),
            )
            return f"{match.group(1)}fixed{match.group(2)}{joint_body}{match.group(4)}"

        fixed_body, match_count = joint_pattern.subn(_replace_joint, fixed_body)
        if match_count != 1:
            raise ValueError(
                f"Canonical G1 URDF must contain exactly one {joint_name!r} joint."
            )
    return fixed_body


def _resolve_g1_urdf_asset_path(urdf_path: str | Path) -> Path:
    """Resolve a G1 URDF path relative to the ioailab repository root."""

    path = Path(urdf_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


G1_CUROBO_LEFT_LINK_NAME = "left_gripper_tcp_link"
G1_CUROBO_RIGHT_LINK_NAME = "right_gripper_tcp_link"
G1_CUROBO_HEAD_LINK_NAME = "head_end_effector_mount_link"

G1_MOBILE_BASE_DOF_ORDER = ("base_x", "base_y", "base_yaw")
G1_CUROBO_WHOLE_BODY_JOINT_NAMES = (
    *G1_LEG_DOF_ORDER,
    *G1_HEAD_DOF_ORDER,
    *G1_LEFT_ARM_DOF_ORDER,
    *G1_RIGHT_ARM_DOF_ORDER,
)
G1_MOBILE_CUROBO_WHOLE_BODY_JOINT_NAMES = (
    *G1_MOBILE_BASE_DOF_ORDER,
    *G1_CUROBO_WHOLE_BODY_JOINT_NAMES,
)
G1_CUROBO_TOOL_FRAME_NAMES = (
    G1_CUROBO_LEFT_LINK_NAME,
    G1_CUROBO_RIGHT_LINK_NAME,
    G1_CUROBO_HEAD_LINK_NAME,
)

G1_CUROBO_SEED_NOISE_SCALES = {
    **{joint_name: 0.35 for joint_name in G1_LEG_DOF_ORDER},
    **{joint_name: 0.30 for joint_name in G1_HEAD_DOF_ORDER},
}
G1_MOBILE_CUROBO_SEED_NOISE_SCALES = {
    "base_x": 0.25,
    "base_y": 0.25,
    "base_yaw": 0.20,
    **G1_CUROBO_SEED_NOISE_SCALES,
}
G1_CUROBO_DEFAULT_JOINT_POSITIONS = dict(MANIPULATION_JOINT_POSITIONS)
G1_CUROBO_ROBOT_SPEC = RobotPlanningSpec(
    robot_name="g1",
    whole_body_joint_names=G1_CUROBO_WHOLE_BODY_JOINT_NAMES,
    motion_groups={
        "legs": MotionGroupSpec("legs", G1_LEG_DOF_ORDER),
        "head": MotionGroupSpec("head", G1_HEAD_DOF_ORDER, G1_CUROBO_HEAD_LINK_NAME),
        "left_arm": MotionGroupSpec(
            "left_arm", G1_LEFT_ARM_DOF_ORDER, G1_CUROBO_LEFT_LINK_NAME
        ),
        "right_arm": MotionGroupSpec(
            "right_arm", G1_RIGHT_ARM_DOF_ORDER, G1_CUROBO_RIGHT_LINK_NAME
        ),
        "base": MotionGroupSpec("base", ()),
    },
    binary_groups={
        "left_gripper": BinaryGroupSpec("left_gripper"),
        "right_gripper": BinaryGroupSpec("right_gripper"),
    },
    default_joint_positions=dict(G1_CUROBO_DEFAULT_JOINT_POSITIONS),
    metadata={
        "urdf_path": DEFAULT_G1_URDF_PATH,
        "base_link_name": DEFAULT_ROBOT_BASE_LINK_NAME,
        "required_chain_joint_names_by_group": {
            "left_arm": G1_LEG_DOF_ORDER,
            "right_arm": G1_LEG_DOF_ORDER,
            "head": (*G1_LEG_DOF_ORDER, *G1_HEAD_DOF_ORDER),
        },
    },
)

G1_MOBILE_CUROBO_ROBOT_SPEC = RobotPlanningSpec(
    robot_name="g1_mobile_base",
    whole_body_joint_names=G1_MOBILE_CUROBO_WHOLE_BODY_JOINT_NAMES,
    motion_groups={
        "base": MotionGroupSpec("base", G1_MOBILE_BASE_DOF_ORDER),
        "legs": MotionGroupSpec("legs", G1_LEG_DOF_ORDER),
        "head": MotionGroupSpec("head", G1_HEAD_DOF_ORDER, G1_CUROBO_HEAD_LINK_NAME),
        "left_arm": MotionGroupSpec(
            "left_arm", G1_LEFT_ARM_DOF_ORDER, G1_CUROBO_LEFT_LINK_NAME
        ),
        "right_arm": MotionGroupSpec(
            "right_arm", G1_RIGHT_ARM_DOF_ORDER, G1_CUROBO_RIGHT_LINK_NAME
        ),
    },
    binary_groups={
        "left_gripper": BinaryGroupSpec("left_gripper"),
        "right_gripper": BinaryGroupSpec("right_gripper"),
    },
    default_joint_positions=dict(G1_CUROBO_DEFAULT_JOINT_POSITIONS),
    metadata={
        "urdf_path": DEFAULT_G1_MOBILE_BASE_URDF_PATH,
        "base_link_name": DEFAULT_MOBILE_ROBOT_BASE_LINK_NAME,
        "required_chain_joint_names_by_group": {
            "left_arm": (*G1_MOBILE_BASE_DOF_ORDER, *G1_LEG_DOF_ORDER),
            "right_arm": (*G1_MOBILE_BASE_DOF_ORDER, *G1_LEG_DOF_ORDER),
            "head": (*G1_MOBILE_BASE_DOF_ORDER, *G1_LEG_DOF_ORDER, *G1_HEAD_DOF_ORDER),
        },
    },
)


@dataclass(slots=True)
class G1CuroboPlanningContext:
    """Reusable G1 cuRobo v2 solver and seed state for chained plans."""

    solver: Curobo2ParallelWBIK
    current_q: np.ndarray | None = None
    spec: RobotPlanningSpec = G1_CUROBO_ROBOT_SPEC


def current_g1_curobo_q_from_env(
    robot_asset: Any, *, device: torch.device
) -> np.ndarray:
    """Return the current G1 articulation state in cuRobo joint order."""

    return current_curobo_q_from_env(
        robot_asset,
        G1_CUROBO_WHOLE_BODY_JOINT_NAMES,
        device=device,
        default_joint_positions=MANIPULATION_JOINT_POSITIONS,
    )


def current_g1_mobile_curobo_q_from_env(
    robot_asset: Any,
    *,
    device: torch.device,
    base_xy_yaw: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Return current G1 mobile-base state in cuRobo joint order.

    ``base_xy_yaw`` is expected in the planner frame used by the mobile URDF.
    Callers running vectorized IsaacLab scenes should pass env-local base
    coordinates so the virtual ``world_base_link`` is per-env, not global.
    """

    joint_q = current_g1_curobo_q_from_env(robot_asset, device=device)
    base_q = _normalize_base_xy_yaw(base_xy_yaw, rows=joint_q.shape[0])
    return np.concatenate([base_q, joint_q], axis=1).astype(np.float32, copy=False)


def _normalize_base_xy_yaw(
    base_xy_yaw: np.ndarray | torch.Tensor | None, *, rows: int
) -> np.ndarray:
    if base_xy_yaw is None:
        return np.zeros((int(rows), len(G1_MOBILE_BASE_DOF_ORDER)), dtype=np.float32)
    if isinstance(base_xy_yaw, torch.Tensor):
        values = base_xy_yaw.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        values = np.asarray(base_xy_yaw, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    expected_shape = (int(rows), len(G1_MOBILE_BASE_DOF_ORDER))
    if values.shape != expected_shape:
        raise ValueError(
            f"base_xy_yaw must have shape {expected_shape}, got {values.shape}."
        )
    return values.astype(np.float32, copy=True)


def _resolve_g1_curobo_inputs(
    *,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
    left_link_name: str = G1_CUROBO_LEFT_LINK_NAME,
    right_link_name: str = G1_CUROBO_RIGHT_LINK_NAME,
    head_link_name: str = G1_CUROBO_HEAD_LINK_NAME,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    resolved_tool_frame_names = g1_curobo_tool_frame_names(
        tool_frame_names=tool_frame_names,
        left_link_name=left_link_name,
        right_link_name=right_link_name,
        head_link_name=head_link_name,
    )
    try:
        return resolve_planning_inputs(
            G1_CUROBO_ROBOT_SPEC,
            active_joint_names=active_joint_names,
            tool_frame_names=resolved_tool_frame_names,
        )
    except ValueError as exc:
        raise ValueError(
            str(exc).replace("the robot planning spec", "G1 cuRobo joint set")
        ) from exc


def make_g1_curobo_planning_context(
    *,
    urdf_path: str | Path = DEFAULT_G1_URDF_PATH,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
    left_link_name: str = G1_CUROBO_LEFT_LINK_NAME,
    right_link_name: str = G1_CUROBO_RIGHT_LINK_NAME,
    head_link_name: str = G1_CUROBO_HEAD_LINK_NAME,
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
    locked_joint_positions: Mapping[str, float] | None = None,
) -> G1CuroboPlanningContext:
    """Create reusable G1 cuRobo v2 WBIK state."""

    return G1CuroboPlanningContext(
        solver=make_default_g1_curobo_solver(
            urdf_path=urdf_path,
            active_joint_names=active_joint_names,
            tool_frame_names=tool_frame_names,
            left_link_name=left_link_name,
            right_link_name=right_link_name,
            head_link_name=head_link_name,
            cspace_distance_weights=cspace_distance_weights,
            null_space_weights=null_space_weights,
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
            override_optimizer_num_iters=override_optimizer_num_iters,
            locked_joint_positions=locked_joint_positions,
        )
    )


def make_default_g1_curobo_solver(
    *,
    urdf_path: str | Path = DEFAULT_G1_URDF_PATH,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
    left_link_name: str = G1_CUROBO_LEFT_LINK_NAME,
    right_link_name: str = G1_CUROBO_RIGHT_LINK_NAME,
    head_link_name: str = G1_CUROBO_HEAD_LINK_NAME,
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
    locked_joint_positions: Mapping[str, float] | None = None,
) -> Curobo2ParallelWBIK:
    """Build the default G1 cuRobo v2 whole-body IK solver."""

    active_names, resolved_tool_frame_names = _resolve_g1_curobo_inputs(
        active_joint_names=active_joint_names,
        tool_frame_names=tool_frame_names,
        left_link_name=left_link_name,
        right_link_name=right_link_name,
        head_link_name=head_link_name,
    )
    G1_CUROBO_ROBOT_SPEC.default_joint_positions.clear()
    G1_CUROBO_ROBOT_SPEC.default_joint_positions.update(
        G1_CUROBO_DEFAULT_JOINT_POSITIONS
    )
    if locked_joint_positions:
        G1_CUROBO_ROBOT_SPEC.default_joint_positions.update(
            {
                str(joint_name): float(value)
                for joint_name, value in locked_joint_positions.items()
            }
        )
    return make_curobo_parallel_wbik(
        G1_CUROBO_ROBOT_SPEC,
        urdf_path=urdf_path,
        active_joint_names=active_names,
        tool_frame_names=resolved_tool_frame_names,
        cspace_distance_weights=cspace_distance_weights,
        null_space_weights=null_space_weights,
        device=device,
        use_cuda_graph=use_cuda_graph,
        run_optimizer=run_optimizer,
        self_collision_check=self_collision_check,
        load_collision_spheres=load_collision_spheres,
        num_seeds=num_seeds,
        return_seeds=return_seeds,
        seed_config_noise_std=seed_config_noise_std,
        seed_config_noise_scales=(
            G1_CUROBO_SEED_NOISE_SCALES
            if seed_config_noise_scales is None
            else seed_config_noise_scales
        ),
        seed_solver_num_seeds=seed_solver_num_seeds,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        override_optimizer_num_iters=override_optimizer_num_iters,
    )


def make_g1_mobile_curobo_planning_context(
    *,
    urdf_path: str | Path = DEFAULT_G1_MOBILE_BASE_URDF_PATH,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
    left_link_name: str = G1_CUROBO_LEFT_LINK_NAME,
    right_link_name: str = G1_CUROBO_RIGHT_LINK_NAME,
    head_link_name: str = G1_CUROBO_HEAD_LINK_NAME,
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
) -> G1CuroboPlanningContext:
    """Create reusable G1 mobile-base cuRobo v2 WBIK state."""

    return G1CuroboPlanningContext(
        solver=make_default_g1_mobile_curobo_solver(
            urdf_path=urdf_path,
            active_joint_names=active_joint_names,
            tool_frame_names=tool_frame_names,
            left_link_name=left_link_name,
            right_link_name=right_link_name,
            head_link_name=head_link_name,
            cspace_distance_weights=cspace_distance_weights,
            null_space_weights=null_space_weights,
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
            override_optimizer_num_iters=override_optimizer_num_iters,
        ),
        spec=G1_MOBILE_CUROBO_ROBOT_SPEC,
    )


def make_default_g1_mobile_curobo_solver(
    *,
    urdf_path: str | Path = DEFAULT_G1_MOBILE_BASE_URDF_PATH,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
    left_link_name: str = G1_CUROBO_LEFT_LINK_NAME,
    right_link_name: str = G1_CUROBO_RIGHT_LINK_NAME,
    head_link_name: str = G1_CUROBO_HEAD_LINK_NAME,
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
    """Build the G1 cuRobo v2 solver with planner-only planar base joints."""

    resolved_tool_frame_names = g1_curobo_tool_frame_names(
        tool_frame_names=tool_frame_names,
        left_link_name=left_link_name,
        right_link_name=right_link_name,
        head_link_name=head_link_name,
    )
    try:
        active_names, resolved_tool_frame_names = resolve_planning_inputs(
            G1_MOBILE_CUROBO_ROBOT_SPEC,
            active_joint_names=active_joint_names,
            tool_frame_names=resolved_tool_frame_names,
        )
    except ValueError as exc:
        raise ValueError(
            str(exc).replace("the robot planning spec", "G1 mobile cuRobo joint set")
        ) from exc
    return make_curobo_parallel_wbik(
        G1_MOBILE_CUROBO_ROBOT_SPEC,
        urdf_path=ensure_g1_mobile_base_urdf(urdf_path=urdf_path),
        active_joint_names=active_names,
        tool_frame_names=resolved_tool_frame_names,
        cspace_distance_weights=cspace_distance_weights,
        null_space_weights=null_space_weights,
        device=device,
        use_cuda_graph=use_cuda_graph,
        run_optimizer=run_optimizer,
        self_collision_check=self_collision_check,
        load_collision_spheres=load_collision_spheres,
        num_seeds=num_seeds,
        return_seeds=return_seeds,
        seed_config_noise_std=seed_config_noise_std,
        seed_config_noise_scales=(
            G1_MOBILE_CUROBO_SEED_NOISE_SCALES
            if seed_config_noise_scales is None
            else seed_config_noise_scales
        ),
        seed_solver_num_seeds=seed_solver_num_seeds,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        override_optimizer_num_iters=override_optimizer_num_iters,
    )


def make_g1_mobile_curobo_robot_config(
    *,
    urdf_path: str | Path = DEFAULT_G1_MOBILE_BASE_URDF_PATH,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
    left_link_name: str = G1_CUROBO_LEFT_LINK_NAME,
    right_link_name: str = G1_CUROBO_RIGHT_LINK_NAME,
    head_link_name: str = G1_CUROBO_HEAD_LINK_NAME,
    cspace_distance_weights: Mapping[str, float] | None = None,
    null_space_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return the G1 mobile-base cuRobo robot config."""

    resolved_tool_frame_names = g1_curobo_tool_frame_names(
        tool_frame_names=tool_frame_names,
        left_link_name=left_link_name,
        right_link_name=right_link_name,
        head_link_name=head_link_name,
    )
    active_names, resolved_tool_frame_names = resolve_planning_inputs(
        G1_MOBILE_CUROBO_ROBOT_SPEC,
        active_joint_names=active_joint_names,
        tool_frame_names=resolved_tool_frame_names,
    )
    return make_curobo_robot_config(
        G1_MOBILE_CUROBO_ROBOT_SPEC,
        urdf_path=ensure_g1_mobile_base_urdf(urdf_path=urdf_path),
        active_joint_names=active_names,
        tool_frame_names=resolved_tool_frame_names,
        cspace_distance_weights=cspace_distance_weights,
        null_space_weights=null_space_weights,
        base_link_name=DEFAULT_MOBILE_ROBOT_BASE_LINK_NAME,
    )


def make_g1_curobo_robot_config(
    *,
    urdf_path: str | Path = DEFAULT_G1_URDF_PATH,
    active_joint_names: Sequence[str] | None = None,
    tool_frame_names: Sequence[str] | None = None,
    left_link_name: str = G1_CUROBO_LEFT_LINK_NAME,
    right_link_name: str = G1_CUROBO_RIGHT_LINK_NAME,
    head_link_name: str = G1_CUROBO_HEAD_LINK_NAME,
    cspace_distance_weights: Mapping[str, float] | None = None,
    null_space_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return a cuRobo robot config derived from ioailab's G1 constants."""

    active_names, resolved_tool_frame_names = _resolve_g1_curobo_inputs(
        active_joint_names=active_joint_names,
        tool_frame_names=tool_frame_names,
        left_link_name=left_link_name,
        right_link_name=right_link_name,
        head_link_name=head_link_name,
    )
    return make_curobo_robot_config(
        G1_CUROBO_ROBOT_SPEC,
        urdf_path=Path(urdf_path),
        active_joint_names=active_names,
        tool_frame_names=resolved_tool_frame_names,
        cspace_distance_weights=cspace_distance_weights,
        null_space_weights=null_space_weights,
        base_link_name=DEFAULT_ROBOT_BASE_LINK_NAME,
    )


def solve_g1_curobo_group_root_pose_targets(
    env: Any,
    context: G1CuroboPlanningContext,
    *,
    group_name: str,
    target_root_poses: torch.Tensor,
    env_ids: Sequence[int],
    robot_asset_name: str = "robot",
    robot_base_link_name: str = "base_link",
    reference_group_targets: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve one G1 cuRobo group from root-frame TCP target poses.

    Args:
        env: IsaacLab env or wrapper.
        context: Reusable G1 cuRobo planning context.
        group_name: Motion group in ``context.spec`` such as ``"left_arm"``.
        target_root_poses: Batched ``(N, 4, 4)`` target poses in the robot-root
            controller frame used by IsaacLab Mimic.
        env_ids: Environment rows corresponding to ``target_root_poses``.
        robot_asset_name: Scene key for the G1 articulation.
        robot_base_link_name: Robot body used as the cuRobo base frame.
        reference_group_targets: Optional group joint targets used as nullspace bias.

    Returns:
        A pair ``(group_targets, success)`` where ``group_targets`` has shape
        ``(N, group_width)`` and ``success`` has shape ``(N,)``.
    """

    from ioailab.agents.motion_plan.solvers.curobov2.waypoint_plan import (
        CuroboPlanningRequest,
        TargetPose,
        TargetStep,
        compute_curobo_grouped_waypoints,
    )

    group = _curobo_motion_group(context, group_name)
    if group.target_frame_name is None:
        raise ValueError(f"G1 cuRobo group {group_name!r} has no target frame.")

    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene[robot_asset_name]
    device = torch.device(getattr(unwrapped, "device", "cpu"))
    env_id_rows = _validate_env_rows(env_ids)
    target_root_poses = _as_root_pose_batch(
        target_root_poses,
        rows=len(env_id_rows),
        device=device,
    )
    start_q = np.asarray(
        current_g1_curobo_q_from_env(robot, device=device), dtype=np.float32
    )
    _validate_env_id_range(env_id_rows, rows=start_q.shape[0])

    target_pose_xyz_wxyz = np.concatenate(
        [
            _root_pose_matrix_to_robot_base_xyz_wxyz(
                robot,
                target_root_poses[row_index],
                env_id=env_id,
                body_name=robot_base_link_name,
                device=device,
            )
            for row_index, env_id in enumerate(env_id_rows)
        ],
        axis=0,
    )
    env_indices = np.asarray(env_id_rows, dtype=np.int64)
    start_q_rows = start_q[env_indices]
    request = CuroboPlanningRequest(
        spec=context.spec,
        start_q=start_q_rows,
        active_groups=(group.name,),
        nullspace_q=_group_nullspace_q(
            start_q_rows,
            context.spec.whole_body_joint_names,
            group.joint_names,
            reference_group_targets,
        ),
        target_steps=(
            TargetStep(
                f"g1_{group.name}_mimic_target",
                target_poses_by_group={
                    group.name: TargetPose(
                        group.name, target_pose_xyz_wxyz, frame="base"
                    )
                },
            ),
        ),
    )
    plan = compute_curobo_grouped_waypoints(request, context=context)
    trajectory = plan.joint_groups.get(group.name)
    if trajectory is None or trajectory.positions.shape[-1] != len(group.joint_names):
        raise RuntimeError(f"cuRobo did not return a {group.name!r} joint trajectory.")
    targets = torch.as_tensor(
        trajectory.positions[-1], device=device, dtype=target_root_poses.dtype
    )
    success = torch.as_tensor(
        plan.step_success_by_env[-1], device=device, dtype=torch.bool
    )
    if targets.shape[0] != len(env_id_rows):
        raise RuntimeError(
            f"cuRobo returned {targets.shape[0]} rows for {len(env_id_rows)} env ids."
        )
    return targets, success


def g1_curobo_group_targets_to_root_poses(
    env: Any,
    context: G1CuroboPlanningContext,
    group_targets: torch.Tensor | np.ndarray,
    *,
    group_name: str,
    robot_asset_name: str = "robot",
    robot_base_link_name: str = "base_link",
) -> torch.Tensor:
    """Convert G1 group joint targets into robot-root-frame TCP poses."""

    group = _curobo_motion_group(context, group_name)
    if group.target_frame_name is None:
        raise ValueError(f"G1 cuRobo group {group_name!r} has no target frame.")

    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene[robot_asset_name]
    dtype = torch.as_tensor(group_targets).dtype
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32
    device = torch.device(getattr(unwrapped, "device", "cpu"))
    q = current_g1_curobo_q_from_env(robot, device=device).copy()
    targets = _as_group_targets(
        group_targets,
        rows=q.shape[0],
        width=len(group.joint_names),
        field_name=group.name,
    )
    q = merge_group_to_whole_q(
        q, context.spec.whole_body_joint_names, group.joint_names, targets
    )
    pose_xyz_wxyz = context.solver.compute_tool_poses_xyz_wxyz(q)[
        group.target_frame_name
    ]
    target_pos, target_rot = _curobo_pose_xyz_wxyz_to_pos_rot(
        pose_xyz_wxyz, rows=targets.shape[0], device=device, dtype=dtype
    )
    base_pose = robot_body_pose_in_root_frame(
        robot,
        body_name=robot_base_link_name,
        rows=targets.shape[0],
        device=device,
        dtype=dtype,
    )
    if base_pose is not None:
        base_pos, base_quat = base_pose
        base_rot = pose_utils.matrix_from_quat(base_quat)
        target_pos = base_pos + torch.matmul(
            base_rot, target_pos.unsqueeze(-1)
        ).squeeze(-1)
        target_rot = torch.matmul(base_rot, target_rot)
    return pose_utils.make_pose(target_pos, target_rot)


def g1_curobo_tool_frame_names(
    *,
    tool_frame_names: Sequence[str] | None = None,
    left_link_name: str = G1_CUROBO_LEFT_LINK_NAME,
    right_link_name: str = G1_CUROBO_RIGHT_LINK_NAME,
    head_link_name: str = G1_CUROBO_HEAD_LINK_NAME,
    include_left: bool = True,
    include_right: bool = True,
    include_head: bool = True,
) -> tuple[str, ...]:
    """Return G1 cuRobo tool frames from semantic left/right/head link names."""

    if tool_frame_names is not None:
        return tuple(str(name) for name in tool_frame_names)
    frames: list[str] = []
    if include_left:
        frames.append(str(left_link_name))
    if include_right:
        frames.append(str(right_link_name))
    if include_head:
        frames.append(str(head_link_name))
    return tuple(frames)


def build_g1_curobov2_frames(
    env: Any,
    *,
    layout: Any,
    action_tensor: torch.Tensor,
    commands: Sequence[Any],
    robot_asset_name: str,
    base_body_name: str,
    max_joint_step: float,
    target_settle_steps: int,
    position_tolerance: float,
    orientation_tolerance: float,
    default_tcp_wxyz: Sequence[float],
    initial_target_name: str,
    locked_joint_positions: Mapping[str, float] | None = None,
    planner_device: str | None = None,
    use_cuda_graph: bool | None = None,
) -> tuple[tuple[Any, ...], Any]:
    """Build executable G1 action frames from cuRobo v2 grouped waypoints."""

    from ioailab.agents.motion_plan.contracts.g1 import (
        G1_ACTION_GROUP_DOF_ORDER,
        G1ActionFrame,
        with_target_settle_frames,
    )
    from ioailab.agents.motion_plan.solvers.curobov2.utils import (
        resample_grouped_positions_by_max_joint_step,
        select_curobo_joint_targets,
    )
    from ioailab.agents.motion_plan.solvers.curobov2.waypoint_plan import (
        CuroboPlanningRequest,
        TargetPose,
        TargetStep,
        compute_curobo_grouped_waypoints,
    )

    if not commands:
        raise ValueError(
            "G1 motion-plan action source requires at least one motion command."
        )

    unwrapped = getattr(env, "unwrapped", env)
    device = torch.device(unwrapped.device)
    robot = unwrapped.scene[robot_asset_name]
    start_q = current_g1_curobo_q_from_env(robot, device=device)
    active_joint_groups = _active_joint_groups_for_commands(layout, commands)
    tcp_joint_groups = tuple(
        group_name
        for group_name in active_joint_groups
        if any(group_name in command.tcp_targets_w for command in commands)
    )
    tool_frame_names = tuple(
        G1_CUROBO_ROBOT_SPEC.motion_groups[group_name].target_frame_name
        for group_name in tcp_joint_groups
        if G1_CUROBO_ROBOT_SPEC.motion_groups[group_name].target_frame_name is not None
    )
    context = None
    if tcp_joint_groups:
        context = make_g1_curobo_planning_context(
            active_joint_names=tuple(
                joint_name
                for group_name in active_joint_groups
                for joint_name in G1_ACTION_GROUP_DOF_ORDER[group_name]
            ),
            tool_frame_names=tool_frame_names,
            locked_joint_positions=locked_joint_positions,
            device=planner_device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
            use_cuda_graph=torch.cuda.is_available()
            if use_cuda_graph is None
            else bool(use_cuda_graph),
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )
    robot_base_pos, robot_base_xyzw = _robot_base_pose(
        robot, base_link_name=base_body_name, device=device
    )
    base_pose_by_env = _xyz_xyzw_pose_to_xyz_wxyz_numpy(robot_base_pos, robot_base_xyzw)

    def target_pose_for_group(
        group_name: str, position: Any, command: Any
    ) -> TargetPose:
        target_pos, target_wxyz = _as_pose_components(
            position,
            command.tcp_wxyz_by_group.get(group_name, default_tcp_wxyz),
            device=device,
            num_envs=start_q.shape[0],
        )
        return TargetPose(
            group_name,
            _xyz_wxyz_pose_to_numpy(target_pos, target_wxyz),
            command.tcp_frame_by_group.get(group_name, "world"),
        )

    target_steps = tuple(
        TargetStep(
            name=command.name,
            target_poses_by_group={
                group_name: target_pose_for_group(group_name, position, command)
                for group_name, position in command.tcp_targets_w.items()
            },
            joint_targets_by_group=command.joint_targets_by_group,
            binary_values_by_group=command.gripper_open_by_group,
        )
        for command in commands
    )
    grouped_plan = compute_curobo_grouped_waypoints(
        CuroboPlanningRequest(
            spec=G1_CUROBO_ROBOT_SPEC,
            start_q=start_q,
            active_groups=active_joint_groups,
            target_steps=target_steps,
            base_pose_by_env=base_pose_by_env,
            nullspace_q=start_q,
        ),
        context=context,
    )

    start_group_targets = [
        select_curobo_joint_targets(
            start_q,
            G1_CUROBO_WHOLE_BODY_JOINT_NAMES,
            G1_ACTION_GROUP_DOF_ORDER[group_name],
            device=device,
        )
        .detach()
        .cpu()
        .numpy()
        for group_name in active_joint_groups
    ]
    combined_positions = np.concatenate(
        [
            np.concatenate(start_group_targets, axis=1)[None, :, :],
            np.concatenate(
                [
                    grouped_plan.joint_groups[group_name].positions
                    for group_name in active_joint_groups
                ],
                axis=2,
            ),
        ],
        axis=0,
    )
    samples, source_indices = resample_grouped_positions_by_max_joint_step(
        combined_positions, max_joint_step=max_joint_step
    )
    samples = samples[1:]
    source_indices = source_indices[1:]
    names_by_source = (initial_target_name, *grouped_plan.target_step_names)
    frames = []
    split_widths = [
        len(G1_ACTION_GROUP_DOF_ORDER[group_name]) for group_name in active_joint_groups
    ]
    for sample, source_index in zip(samples, source_indices, strict=True):
        columns = np.split(sample, np.cumsum(split_widths)[:-1], axis=1)
        joint_targets_by_group = {
            group_name: torch.as_tensor(
                columns[index], device=action_tensor.device, dtype=action_tensor.dtype
            )
            for index, group_name in enumerate(active_joint_groups)
        }
        binary_values_by_group = {
            group_name: torch.as_tensor(
                _binary_values_for_source(
                    grouped_plan,
                    group_name=group_name,
                    source_index=int(source_index),
                    num_envs=start_q.shape[0],
                ),
                device=action_tensor.device,
                dtype=torch.bool,
            )
            for group_name in layout.binary_group_names
            if _binary_group_declared(commands, group_name)
        }
        frames.append(
            G1ActionFrame(
                name=names_by_source[int(source_index)],
                joint_targets_by_group=joint_targets_by_group,
                binary_values_by_group=binary_values_by_group,
            )
        )
    return with_target_settle_frames(
        tuple(frames), settle_steps=target_settle_steps
    ), grouped_plan


def _curobo_motion_group(
    context: G1CuroboPlanningContext, group_name: str
) -> MotionGroupSpec:
    group = str(group_name)
    try:
        return context.spec.motion_groups[group]
    except KeyError as exc:
        raise ValueError(f"Unknown G1 cuRobo motion group {group!r}.") from exc


def _validate_env_rows(env_ids: Sequence[int]) -> tuple[int, ...]:
    rows = tuple(int(env_id) for env_id in env_ids)
    if not rows:
        raise ValueError("env_ids must contain at least one environment index.")
    if any(env_id < 0 for env_id in rows):
        raise ValueError(f"env_ids must be non-negative, got {rows!r}.")
    return rows


def _validate_env_id_range(env_ids: Sequence[int], *, rows: int) -> None:
    if max(env_ids) >= int(rows):
        raise ValueError(
            f"env_ids {tuple(env_ids)!r} are out of range for {int(rows)} environments."
        )


def _as_root_pose_batch(
    poses: torch.Tensor | np.ndarray, *, rows: int, device: torch.device
) -> torch.Tensor:
    pose_tensor = torch.as_tensor(poses, device=device)
    if pose_tensor.ndim != 3 or pose_tensor.shape[-2:] != (4, 4):
        raise ValueError(
            "target_root_poses must have shape (N, 4, 4), "
            f"got {tuple(pose_tensor.shape)}."
        )
    if pose_tensor.shape[0] != int(rows):
        raise ValueError(
            f"Expected {int(rows)} target poses, got {pose_tensor.shape[0]}."
        )
    return pose_tensor


def _root_pose_matrix_to_robot_base_xyz_wxyz(
    robot: Any,
    pose: torch.Tensor,
    *,
    env_id: int,
    body_name: str,
    device: torch.device,
) -> np.ndarray:
    dtype = pose.dtype
    target_pos, target_rot = pose_utils.unmake_pose(
        torch.as_tensor(pose, device=device)
    )
    base_pose = robot_body_pose_in_root_frame(
        robot, body_name=body_name, rows=int(env_id) + 1, device=device, dtype=dtype
    )
    if base_pose is not None:
        base_pos, base_quat = base_pose
        base_pos = base_pos[int(env_id)]
        base_rot = pose_utils.matrix_from_quat(base_quat[int(env_id)])
        target_pos = torch.matmul(base_rot.transpose(-1, -2), target_pos - base_pos)
        target_rot = torch.matmul(base_rot.transpose(-1, -2), target_rot)

    pose_xyz_xyzw = torch.cat(
        [target_pos, pose_utils.quat_from_matrix(target_rot)], dim=0
    )
    pose_xyz_wxyz = pose_xyz_xyzw_to_xyz_wxyz(
        pose_xyz_xyzw, field_name="target_root_pose_xyz_xyzw"
    )
    return pose_xyz_wxyz.detach().cpu().numpy()[None, :].astype(np.float32, copy=False)


def _group_nullspace_q(
    start_q: np.ndarray,
    whole_body_joint_names: Sequence[str],
    group_joint_names: Sequence[str],
    reference_group_targets: torch.Tensor | np.ndarray | None,
) -> np.ndarray | None:
    if reference_group_targets is None:
        return None
    start = np.asarray(start_q, dtype=np.float32)
    rows = 1 if start.ndim == 1 else start.shape[0]
    reference = _as_group_targets(
        reference_group_targets,
        rows=rows,
        width=len(group_joint_names),
        field_name="reference_group_targets",
    )
    return merge_group_to_whole_q(
        start, whole_body_joint_names, group_joint_names, reference
    )


def _as_group_targets(
    group_targets: torch.Tensor | np.ndarray, *, rows: int, width: int, field_name: str
) -> np.ndarray:
    if isinstance(group_targets, torch.Tensor):
        values = group_targets.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        values = np.asarray(group_targets, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != int(width):
        raise ValueError(
            f"{field_name} must have shape ({int(width)},) or (N, {int(width)}), got {values.shape}."
        )
    if values.shape[0] == 1 and int(rows) != 1:
        values = np.repeat(values, int(rows), axis=0)
    if values.shape[0] != int(rows):
        raise ValueError(f"Expected {int(rows)} target rows, got {values.shape[0]}.")
    return values.astype(np.float32, copy=False)


def _curobo_pose_xyz_wxyz_to_pos_rot(
    pose_xyz_wxyz: torch.Tensor | np.ndarray,
    *,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = torch.as_tensor(pose_xyz_wxyz, device=device, dtype=dtype)
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected cuRobo pose width 7, got {tuple(pose.shape)}.")
    if pose.shape[0] == 1 and int(rows) != 1:
        pose = pose.repeat(int(rows), 1)
    if pose.shape[0] != int(rows):
        raise ValueError(f"Expected {int(rows)} cuRobo pose rows, got {pose.shape[0]}.")
    pose = pose_xyz_wxyz_to_xyz_xyzw(pose, field_name="pose_xyz_wxyz")
    quat_xyzw = pose[:, 3:7]
    quat_xyzw = quat_xyzw / torch.linalg.vector_norm(
        quat_xyzw, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    return pose[:, :3], pose_utils.matrix_from_quat(quat_xyzw)


def _active_joint_groups_for_commands(
    layout: Any, commands: Sequence[Any]
) -> tuple[str, ...]:
    for command in commands:
        for group_name in command.tcp_targets_w:
            layout.term_for_group(group_name)
        for group_name in command.joint_targets_by_group:
            layout.term_for_group(group_name)
        for group_name in command.gripper_open_by_group:
            layout.term_for_group(group_name)
    groups = tuple(
        group_name
        for group_name in layout.joint_group_names
        if any(
            group_name in command.tcp_targets_w
            or group_name in command.joint_targets_by_group
            for command in commands
        )
    )
    if not groups:
        raise ValueError(
            "G1 motion-plan action source requires at least one declared arm pose or joint target."
        )
    return groups


def _binary_group_declared(commands: Sequence[Any], group_name: str) -> bool:
    return any(str(group_name) in command.gripper_open_by_group for command in commands)


def _binary_values_for_source(
    grouped_plan: Any, *, group_name: str, source_index: int, num_envs: int
) -> np.ndarray:
    if source_index <= 0:
        raise ValueError(
            "G1 motion-plan action source should not request binary values for the dropped initial sample."
        )
    if group_name not in grouped_plan.binary_groups:
        raise ValueError(
            f"G1 grouped waypoint plan is missing binary group {group_name!r}; "
            "define the group on every motion command that needs gripper state."
        )
    values = np.asarray(
        grouped_plan.binary_groups[group_name].values[source_index - 1], dtype=bool
    )
    if values.shape != (int(num_envs),):
        raise ValueError(
            f"G1 binary group {group_name!r} has shape {values.shape}, expected ({int(num_envs)},)."
        )
    return values


def _target_wxyz(value: Any, *, batch: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        quat = value.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        quat = np.asarray(value, dtype=np.float32)
    if quat.shape == (4,):
        return np.broadcast_to(quat, (int(batch), 4)).copy()
    if quat.shape == (int(batch), 4):
        return quat.copy()
    raise ValueError(
        f"target wxyz orientation must have shape (4,) or ({int(batch)}, 4), got {quat.shape}."
    )


def _as_world_position_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    tensor = _as_torch(value, device=device).to(dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 3)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(
            f"target position must have shape (num_envs, 3), got {tuple(tensor.shape)}."
        )
    return tensor


def _as_pose_components(
    position: Any, wxyz: Any, *, device: torch.device, num_envs: int
) -> tuple[torch.Tensor, np.ndarray]:
    """Return broadcast frame-qualified xyz and wxyz pose components."""

    position_tensor = _as_world_position_tensor(position, device=device)
    if position_tensor.shape[0] == 1 and int(num_envs) > 1:
        position_tensor = position_tensor.repeat(int(num_envs), 1)
    if position_tensor.shape != (int(num_envs), 3):
        raise ValueError(
            "target position must have shape "
            f"({int(num_envs)}, 3), got {tuple(position_tensor.shape)}."
        )
    return position_tensor, _target_wxyz(wxyz, batch=int(num_envs))


def _xyz_wxyz_pose_to_numpy(position_xyz: torch.Tensor, wxyz: np.ndarray) -> np.ndarray:
    """Return batched ``xyz + wxyz`` cuRobo pose arrays."""

    position = position_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    quat = np.asarray(wxyz, dtype=np.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(
            f"position_xyz must have shape (num_envs, 3), got {position.shape}."
        )
    if quat.shape != (position.shape[0], 4):
        raise ValueError(
            f"wxyz must have shape ({position.shape[0]}, 4), got {quat.shape}."
        )
    return np.concatenate((position, quat), axis=1).astype(np.float32, copy=False)


def _xyz_xyzw_pose_to_xyz_wxyz_numpy(
    position_xyz: torch.Tensor, quat_xyzw: torch.Tensor
) -> np.ndarray:
    """Convert IsaacLab ``xyz + xyzw`` pose tensors to cuRobo ``xyz + wxyz``."""

    pose_xyz_xyzw = torch.cat((position_xyz, quat_xyzw), dim=1)
    pose_xyz_wxyz = pose_xyz_xyzw_to_xyz_wxyz(
        pose_xyz_xyzw, field_name="robot_base_pose_xyz_xyzw"
    )
    return pose_xyz_wxyz.detach().cpu().numpy().astype(np.float32, copy=False)


def _as_torch(value: Any, *, device: torch.device) -> torch.Tensor:
    """Convert tensor-like values to a torch tensor on ``device``."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if hasattr(value, "torch") and hasattr(value, "warp"):
        # IsaacLab 3.0 ProxyArray: the cached ``.torch`` view avoids the implicit
        # conversion deprecation warning.
        return value.torch.to(device=device)
    try:
        import warp as wp
    except ImportError:
        wp = None
    if wp is not None and isinstance(value, wp.array):
        return wp.to_torch(value).to(device=device)
    return torch.as_tensor(value, device=device)


def _body_index(body_names: Sequence[str], body_name: str) -> int | None:
    """Return an exact or unique suffix body-name match."""

    names = list(body_names)
    if body_name in names:
        return names.index(body_name)
    matches = [
        index
        for index, candidate in enumerate(names)
        if str(candidate).rsplit("/", maxsplit=1)[-1] == body_name
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _root_pose(
    asset: Any, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an IsaacLab asset root pose as position and xyzw quaternion."""

    if hasattr(asset.data, "root_pos_w"):
        return (
            _as_torch(asset.data.root_pos_w, device=device).to(dtype=torch.float32),
            _as_torch(asset.data.root_quat_w, device=device).to(dtype=torch.float32),
        )
    root_state = _as_torch(asset.data.root_state_w, device=device).to(
        dtype=torch.float32
    )
    return root_state[:, :3], root_state[:, 3:7]


def robot_body_pose_in_root_frame(
    robot: Any,
    *,
    body_name: str,
    rows: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a robot body pose relative to the articulation root frame.

    IsaacLab stores robot body poses in world coordinates. Mimic action
    conversion uses root-frame EEF poses, while cuRobo expects poses in the
    configured robot base frame. This helper exposes that shared conversion so
    callers do not need local body-index and root/body transform code.
    """

    data = robot.data
    body_index = _body_index(getattr(robot, "body_names", ()), body_name)
    if body_index is None:
        return None

    for pos_attr, quat_attr in (
        ("body_pos_w", "body_quat_w"),
        ("body_link_pos_w", "body_link_quat_w"),
    ):
        if not hasattr(data, pos_attr) or not hasattr(data, quat_attr):
            continue
        body_pos_w = _as_torch(getattr(data, pos_attr), device=device).to(dtype=dtype)
        body_quat_w = _as_torch(getattr(data, quat_attr), device=device).to(dtype=dtype)
        if body_pos_w.ndim != 3 or body_quat_w.ndim != 3:
            continue
        root_pos_w = _as_torch(data.root_pos_w, device=device).to(dtype=dtype)[:rows]
        root_quat_w = _as_torch(data.root_quat_w, device=device).to(dtype=dtype)[:rows]
        return pose_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            body_pos_w[:rows, body_index, :],
            body_quat_w[:rows, body_index, :],
        )

    return None


def _robot_base_pose(
    robot: Any, *, base_link_name: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the world pose for the planner's robot base body."""

    body_idx = _body_index(getattr(robot, "body_names", ()), base_link_name)
    if body_idx is None:
        return _root_pose(robot, device=device)

    data = robot.data
    for pos_attr, quat_attr in (
        ("body_pos_w", "body_quat_w"),
        ("body_link_pos_w", "body_link_quat_w"),
    ):
        if hasattr(data, pos_attr) and hasattr(data, quat_attr):
            pos_all = _as_torch(getattr(data, pos_attr), device=device).to(
                dtype=torch.float32
            )
            quat_all = _as_torch(getattr(data, quat_attr), device=device).to(
                dtype=torch.float32
            )
            if pos_all.ndim == 3 and quat_all.ndim == 3:
                return pos_all[:, body_idx, :], quat_all[:, body_idx, :]
    return _root_pose(robot, device=device)


def _normalize_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Normalize batched xyzw quaternions."""

    return quat / torch.clamp(torch.linalg.norm(quat, dim=1, keepdim=True), min=1.0e-8)


def _quat_rotate_inverse_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse of an IsaacLab xyzw quaternion."""

    quat_vec = quat[:, :3]
    quat_w = quat[:, 3:]
    uv = torch.cross(quat_vec, vec, dim=1)
    uuv = torch.cross(quat_vec, uv, dim=1)
    return vec - 2.0 * (quat_w * uv - uuv)


def _world_position_to_robot_base_matrix(
    *,
    target_world_pos: torch.Tensor,
    robot_base_pos: torch.Tensor,
    robot_base_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Convert world positions to robot-base-frame positions."""

    if target_world_pos.shape != robot_base_pos.shape:
        raise ValueError(
            "target_world_pos and robot_base_pos must have matching shape, "
            f"got {tuple(target_world_pos.shape)} and {tuple(robot_base_pos.shape)}."
        )
    if robot_base_xyzw.shape != (robot_base_pos.shape[0], 4):
        raise ValueError(
            "robot_base_xyzw must have shape "
            f"({robot_base_pos.shape[0]}, 4), got {tuple(robot_base_xyzw.shape)}."
        )
    return _quat_rotate_inverse_xyzw(
        _normalize_quat_xyzw(robot_base_xyzw), target_world_pos - robot_base_pos
    )
