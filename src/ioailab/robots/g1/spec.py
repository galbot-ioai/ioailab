"""Pure-data Galbot G1 robot facts.

This module intentionally contains no IsaacLab, cuRobo, asset-registry, task, or
runtime imports. IsaacLab config construction belongs in neighboring modules.
"""

from __future__ import annotations

from dataclasses import dataclass

ROBOT_NAME = "galbot_g1"
DISPLAY_NAME = "Galbot G1"
DEFAULT_PRIM_PATH = "/World/GalbotG1"
DEFAULT_END_EFFECTOR_LINK = "left_arm_link7"

G1_BASE_WHEEL_DOF_ORDER = (
    "wheel1_joint",
    "wheel2_joint",
    "wheel3_joint",
    "wheel4_joint",
)
G1_LEG_DOF_ORDER = (
    "leg_joint1",
    "leg_joint2",
    "leg_joint3",
    "leg_joint4",
    "leg_joint5",
)
G1_LEFT_ARM_DOF_ORDER = (
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
)
G1_RIGHT_ARM_DOF_ORDER = (
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
)


def _folded_arm_joint_positions(joint_names: tuple[str, ...]) -> dict[str, float]:
    """Return the shared G1 folded-arm posture for one arm DOF order."""

    positions = {joint_name: 0.0 for joint_name in joint_names}
    positions[joint_names[1]] = -1.0
    positions[joint_names[3]] = -1.5
    return positions


G1_LEFT_ARM_FOLDED_JOINT_POSITIONS = _folded_arm_joint_positions(G1_LEFT_ARM_DOF_ORDER)
G1_RIGHT_ARM_FOLDED_JOINT_POSITIONS = _folded_arm_joint_positions(
    G1_RIGHT_ARM_DOF_ORDER
)
G1_LEFT_GRIPPER_DOF_ORDER = ("left_gripper_joint",)
G1_RIGHT_GRIPPER_DOF_ORDER = ("right_gripper_joint",)
G1_POSTURE_DOF_ORDER = (
    "head_joint1",
    "head_joint2",
)
G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES = tuple(
    f"wheel_{wheel_index}_passive_{roller_index}_joint"
    for wheel_index in range(1, 5)
    for roller_index in range(10)
)
G1_GRIPPER_VELOCITY_LIMIT_SIM = 2.0
CONTROLLED_JOINT_NAMES = G1_LEFT_ARM_DOF_ORDER
G1_ACTION_JOINT_NAMES = (
    *G1_LEG_DOF_ORDER,
    *G1_LEFT_ARM_DOF_ORDER,
    *G1_RIGHT_ARM_DOF_ORDER,
    *G1_LEFT_GRIPPER_DOF_ORDER,
    *G1_RIGHT_GRIPPER_DOF_ORDER,
    *G1_BASE_WHEEL_DOF_ORDER,
)

G1_FIXED_BASE_BODY_CANDIDATES = ("base_footprint", "base_link")
G1_TOP_DOWN_TCP_WXYZ = (
    0.0,
    0.7071067811865476,
    0.0,
    -0.7071067811865475,
)

MANIPULATION_POSTURE_JOINT_NAMES = G1_POSTURE_DOF_ORDER
MANIPULATION_GROUND_Z = 0.0
MANIPULATION_BASE_FOOTPRINT_Z = 0.0
MANIPULATION_BASE_LINK_Z = 0.028165999799966812
MANIPULATION_ASSET_MIN_Z = -0.005105130374431638
MANIPULATION_BASE_CLEARANCE = 0.005
MANIPULATION_ROOT_POSITION = (-1.0, 0.0, MANIPULATION_BASE_FOOTPRINT_Z)
MANIPULATION_ROOT_ORIENTATION = (0.0, 0.0, 0.0, 1.0)
MANIPULATION_JOINT_POSITIONS = {joint_name: 0.0 for joint_name in G1_ACTION_JOINT_NAMES}
MANIPULATION_JOINT_POSITIONS.update(
    {joint_name: 0.0 for joint_name in MANIPULATION_POSTURE_JOINT_NAMES}
)

G1_MOBILE_BASE_BODY_NAME = "base_footprint"
G1_MOBILE_BASE_RESET_ROOT_BODY_NAME = "base_footprint"
G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE = (0.0, 0.0, 0.0)
G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW = (0.0, 0.0, 0.0, 1.0)

DEFAULT_ROBOT_ASSET_NAME = "robot"
DEFAULT_BASE_WHEEL_RADIUS = 0.1
DEFAULT_BASE_WHEEL_X = 0.176493
DEFAULT_BASE_WHEEL_Y = 0.176493
DEFAULT_GRIPPER_OPEN_POSITION = 0.0
DEFAULT_GRIPPER_CLOSED_POSITION = 1.2

ROBOT_PRIM_PATH = "{ENV_REGEX_NS}/Robot"
ROBOT_USD_ROOT_PRIM_PATH = ROBOT_PRIM_PATH
G1_TORSO_BASE_PRIM_PATH = (
    f"{ROBOT_USD_ROOT_PRIM_PATH}/leg_link5/leg_end_effector_mount_link/torso_base_link"
)
FRONT_HEAD_CAMERA_PARENT_PRIM_PATH = (
    f"{ROBOT_USD_ROOT_PRIM_PATH}/head_link2/head_end_effector_mount_link"
)
LEFT_WRIST_CAMERA_PARENT_PRIM_PATH = (
    f"{ROBOT_USD_ROOT_PRIM_PATH}/left_arm_link7/left_arm_end_effector_mount_link"
)
RIGHT_WRIST_CAMERA_PARENT_PRIM_PATH = (
    f"{ROBOT_USD_ROOT_PRIM_PATH}/right_arm_link7/right_arm_end_effector_mount_link"
)

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_UPDATE_PERIOD = 0.0

DEFAULT_PINHOLE_CAMERA_KWARGS = {
    "focal_length": 24.0,
    "focus_distance": 400.0,
    "f_stop": 0.0,
    "horizontal_aperture": 20.955,
    "clipping_range": (0.05, 15.0),
}


@dataclass(frozen=True, slots=True)
class G1CameraMountSpec:
    """Pure-data camera mount facts for one G1 USD attachment point."""

    parent_prim_path: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]


FRONT_HEAD_CAMERA_POS = (
    0.0860441614606322,
    -0.04430213071916153,
    0.03775394593541334,
)
FRONT_HEAD_CAMERA_ROT = (
    -0.16830090763876662,
    0.686891777200189,
    0.174601740354762,
    0.6851048993897368,
)
LEFT_WRIST_CAMERA_POS = (
    -0.028503262323055674,
    0.010121006758704362,
    0.06923672234517289,
)
LEFT_WRIST_CAMERA_ROT = (
    0.5181547595278811,
    -0.4896793386321121,
    0.47473422704547164,
    -0.5160980567362753,
)
RIGHT_WRIST_CAMERA_POS = (
    -0.027569458572299,
    0.007515698932418123,
    0.06927524658358669,
)
RIGHT_WRIST_CAMERA_ROT = (
    -0.5102597706172789,
    0.4892647456674752,
    -0.479785589522129,
    0.5196737084204334,
)
G1_CAMERA_MOUNT_SPECS = {
    "front_head": G1CameraMountSpec(
        parent_prim_path=FRONT_HEAD_CAMERA_PARENT_PRIM_PATH,
        pos=FRONT_HEAD_CAMERA_POS,
        rot=FRONT_HEAD_CAMERA_ROT,
    ),
    "left_wrist": G1CameraMountSpec(
        parent_prim_path=LEFT_WRIST_CAMERA_PARENT_PRIM_PATH,
        pos=LEFT_WRIST_CAMERA_POS,
        rot=LEFT_WRIST_CAMERA_ROT,
    ),
    "right_wrist": G1CameraMountSpec(
        parent_prim_path=RIGHT_WRIST_CAMERA_PARENT_PRIM_PATH,
        pos=RIGHT_WRIST_CAMERA_POS,
        rot=RIGHT_WRIST_CAMERA_ROT,
    ),
}
G1_CAMERA_DATA_TYPES = {
    "rgb": ("rgb",),
    "depth": ("distance_to_image_plane",),
    "rgbd": ("rgb", "distance_to_image_plane"),
    "rgb_semantic": ("rgb", "semantic_segmentation"),
    "rgbd_semantic": ("rgb", "distance_to_image_plane", "semantic_segmentation"),
}
