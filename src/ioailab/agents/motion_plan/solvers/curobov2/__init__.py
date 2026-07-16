"""Robot-agnostic cuRobo v2 planning helper package."""

from ioailab.agents.motion_plan.solvers.curobov2.robot_spec import (
    BinaryGroupSpec,
    DEFAULT_ROBOT_BASE_LINK_NAME,
    MotionGroupSpec,
    RobotPlanningSpec,
    make_curobo_parallel_wbik,
    make_curobo_robot_config,
    resolve_planning_inputs,
    resolve_tool_frame_names,
)
from ioailab.agents.motion_plan.solvers.curobov2.utils import (
    current_curobo_q_from_env,
    curobo_pose_from_robot_base_position,
    map_q_to_group,
    merge_group_to_whole_q,
    resample_grouped_positions_by_max_joint_step,
    select_curobo_joint_targets,
)
from ioailab.agents.motion_plan.solvers.curobov2.waypoint_plan import (
    BinaryGroupTrajectory,
    CuroboPlanningRequest,
    GroupedWaypointPlan,
    JointGroupTrajectory,
    PoseGroupTrajectory,
    TargetPose,
    TargetStep,
    compute_curobo_grouped_waypoints,
)

__all__ = [
    "BinaryGroupSpec",
    "BinaryGroupTrajectory",
    "CuroboPlanningRequest",
    "DEFAULT_ROBOT_BASE_LINK_NAME",
    "GroupedWaypointPlan",
    "JointGroupTrajectory",
    "MotionGroupSpec",
    "PoseGroupTrajectory",
    "RobotPlanningSpec",
    "TargetPose",
    "TargetStep",
    "compute_curobo_grouped_waypoints",
    "current_curobo_q_from_env",
    "curobo_pose_from_robot_base_position",
    "make_curobo_parallel_wbik",
    "make_curobo_robot_config",
    "map_q_to_group",
    "merge_group_to_whole_q",
    "resample_grouped_positions_by_max_joint_step",
    "resolve_planning_inputs",
    "resolve_tool_frame_names",
    "select_curobo_joint_targets",
]
