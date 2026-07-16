"""Categorized cuRobo v2 utility helpers.

The package keeps the historical ``ioailab.agents.motion_plan.solvers.curobov2.utils`` import
path while separating plain pose/tensor helpers from cuRobo backend code and
IsaacLab interop helpers.
"""

from ioailab.agents.motion_plan.solvers.curobov2.utils.adapter import (
    Curobo2ParallelWBIK,
    Curobo2ParallelWBIKConfig,
    Curobo2WBIKRequest,
    Curobo2WBIKResult,
)
from ioailab.agents.motion_plan.solvers.curobov2.utils.backend import (
    Curobo2UnavailableError,
    require_curobo_public_api,
)
from ioailab.agents.motion_plan.solvers.curobov2.utils.isaac import (
    current_curobo_q_from_env,
    select_curobo_joint_targets,
)
from ioailab.agents.motion_plan.solvers.curobov2.utils.pose import (
    curobo_pose_from_robot_base_position,
    normalize_pose_xyz_wxyz,
    pose_xyz_wxyz_to_xyz_xyzw,
    pose_xyz_xyzw_to_xyz_wxyz,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    resolve_target_pose_xyz_wxyz,
)
from ioailab.agents.motion_plan.solvers.curobov2.utils.tensors import (
    expand_binary_values,
    map_q_to_group,
    merge_group_to_whole_q,
    resample_grouped_positions_by_max_joint_step,
    validate_sample_target_indices,
    validate_step_success,
)

__all__ = [
    "Curobo2ParallelWBIK",
    "Curobo2ParallelWBIKConfig",
    "Curobo2UnavailableError",
    "Curobo2WBIKRequest",
    "Curobo2WBIKResult",
    "current_curobo_q_from_env",
    "curobo_pose_from_robot_base_position",
    "expand_binary_values",
    "map_q_to_group",
    "merge_group_to_whole_q",
    "normalize_pose_xyz_wxyz",
    "pose_xyz_wxyz_to_xyz_xyzw",
    "pose_xyz_xyzw_to_xyz_wxyz",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "require_curobo_public_api",
    "resolve_target_pose_xyz_wxyz",
    "resample_grouped_positions_by_max_joint_step",
    "select_curobo_joint_targets",
    "validate_sample_target_indices",
    "validate_step_success",
]
