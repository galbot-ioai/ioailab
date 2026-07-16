"""Canonical Galbot G1 robot-specific config package."""

from ioailab.robots.g1.spec import (
    G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES,
    G1_BASE_WHEEL_DOF_ORDER,
    G1_GRIPPER_VELOCITY_LIMIT_SIM,
    G1_LEG_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER,
    G1_LEFT_ARM_FOLDED_JOINT_POSITIONS,
    G1_LEFT_GRIPPER_DOF_ORDER,
    G1_POSTURE_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER,
    G1_RIGHT_ARM_FOLDED_JOINT_POSITIONS,
    G1_RIGHT_GRIPPER_DOF_ORDER,
)

_ARTICULATION_EXPORTS = {
    "CONTROLLED_JOINT_NAMES",
    "DEFAULT_END_EFFECTOR_LINK",
    "DEFAULT_PRIM_PATH",
    "DISPLAY_NAME",
    "G1_ACTION_JOINT_NAMES",
    "G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES",
    "G1Articulation",
    "G1_FIXED_BASE_BODY_CANDIDATES",
    "G1_GRIPPER_VELOCITY_LIMIT_SIM",
    "G1_MOBILE_BASE_BODY_NAME",
    "G1_MOBILE_BASE_RESET_ROOT_BODY_NAME",
    "G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE",
    "G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW",
    "G1_POSTURE_DOF_ORDER",
    "G1_TOP_DOWN_TCP_WXYZ",
    "GALBOT_G1_CFG",
    "MANIPULATION_ASSET_MIN_Z",
    "MANIPULATION_BASE_CLEARANCE",
    "MANIPULATION_BASE_FOOTPRINT_Z",
    "MANIPULATION_BASE_LINK_Z",
    "MANIPULATION_GROUND_Z",
    "MANIPULATION_JOINT_POSITIONS",
    "MANIPULATION_POSTURE_JOINT_NAMES",
    "MANIPULATION_ROOT_ORIENTATION",
    "MANIPULATION_ROOT_POSITION",
    "ROBOT_NAME",
    "is_galbot_g1_asset_available",
    "make_galbot_g1_articulation_cfg",
    "make_galbot_g1_manipulation_articulation_cfg",
    "make_galbot_g1_mobile_base_articulation_cfg",
    "mobile_base_root_pose_from_base_pose",
    "resolve_galbot_g1_usd_path",
    "spawn_galbot_g1_usd",
    "spawn_galbot_g1_usd_mobile_base",
    "spawn_galbot_g1_usd_mobile_base_with_controller_graphs",
    "spawn_galbot_g1_usd_with_controller_graphs",
}
_ROBOT_EXPORTS = {"G1", "G1Robot", "g1"}
_ACTION_EXPORTS = {"G1Actions"}
_SENSOR_EXPORTS = {"G1Sensors"}


def __getattr__(name: str):
    """Load heavier G1 config objects only when requested."""

    if name in _ARTICULATION_EXPORTS:
        from ioailab.robots.g1 import articulation

        value = getattr(articulation, name)
        globals()[name] = value
        return value
    if name in _ROBOT_EXPORTS:
        from ioailab.robots.g1 import assemble

        value = getattr(assemble, name)
        globals()[name] = value
        return value
    if name in _ACTION_EXPORTS:
        from ioailab.robots.g1.actions.core import G1Actions

        globals()[name] = G1Actions
        return G1Actions
    if name in _SENSOR_EXPORTS:
        from ioailab.robots.g1.sensors.core import G1Sensors

        globals()[name] = G1Sensors
        return G1Sensors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "G1",
    "G1Actions",
    "G1Articulation",
    "G1Robot",
    "G1Sensors",
    "g1",
    "CONTROLLED_JOINT_NAMES",
    "DEFAULT_END_EFFECTOR_LINK",
    "DEFAULT_PRIM_PATH",
    "DISPLAY_NAME",
    "G1_ACTION_JOINT_NAMES",
    "G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES",
    "G1_BASE_WHEEL_DOF_ORDER",
    "G1_FIXED_BASE_BODY_CANDIDATES",
    "G1_GRIPPER_VELOCITY_LIMIT_SIM",
    "G1_LEG_DOF_ORDER",
    "G1_LEFT_ARM_DOF_ORDER",
    "G1_LEFT_ARM_FOLDED_JOINT_POSITIONS",
    "G1_LEFT_GRIPPER_DOF_ORDER",
    "G1_MOBILE_BASE_BODY_NAME",
    "G1_MOBILE_BASE_RESET_ROOT_BODY_NAME",
    "G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE",
    "G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW",
    "G1_POSTURE_DOF_ORDER",
    "G1_RIGHT_ARM_DOF_ORDER",
    "G1_RIGHT_ARM_FOLDED_JOINT_POSITIONS",
    "G1_RIGHT_GRIPPER_DOF_ORDER",
    "G1_TOP_DOWN_TCP_WXYZ",
    "GALBOT_G1_CFG",
    "MANIPULATION_ASSET_MIN_Z",
    "MANIPULATION_BASE_CLEARANCE",
    "MANIPULATION_BASE_FOOTPRINT_Z",
    "MANIPULATION_BASE_LINK_Z",
    "MANIPULATION_GROUND_Z",
    "MANIPULATION_JOINT_POSITIONS",
    "MANIPULATION_POSTURE_JOINT_NAMES",
    "MANIPULATION_ROOT_ORIENTATION",
    "MANIPULATION_ROOT_POSITION",
    "ROBOT_NAME",
    "is_galbot_g1_asset_available",
    "make_galbot_g1_articulation_cfg",
    "make_galbot_g1_manipulation_articulation_cfg",
    "make_galbot_g1_mobile_base_articulation_cfg",
    "mobile_base_root_pose_from_base_pose",
    "resolve_galbot_g1_usd_path",
    "spawn_galbot_g1_usd",
    "spawn_galbot_g1_usd_mobile_base",
    "spawn_galbot_g1_usd_mobile_base_with_controller_graphs",
    "spawn_galbot_g1_usd_with_controller_graphs",
]
