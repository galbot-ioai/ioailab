"""Manager-based Galbot G1 pick-cube task config."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from ioailab.datasets.mimic import MimicCfg
from ioailab.robots.g1 import g1
from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER
from ioailab.robots.g1.articulation import (
    make_galbot_g1_manipulation_articulation_cfg,
)
from ioailab.robots.g1.converters import G1ArmEefActionConverter
from ioailab.robots.g1.sensors.camera import make_g1_camera_cfg
from ioailab.tasks.common.defaults import DefaultEnvCfg
from ioailab.tasks.pick_cube.config.g1.mdp_cfg import (
    PickCubeMdpCfg,
    PickCubeTeleopObservationsCfg,
)
from ioailab.tasks.pick_cube.mdp import PickCubeMimicSuccessCfg
from ioailab.tasks.pick_cube.mdp import terminations as pick_cube_terminations
from ioailab.tasks.pick_cube.scene import PickCubeSceneCfg

G1_PICK_CUBE_LEFT_ARM_INITIAL_JOINT_POS = {
    "left_arm_joint1": 0.6544984694978736,
    "left_arm_joint2": -0.5218534463463045,
    "left_arm_joint3": -0.12391837689159762,
    "left_arm_joint4": -1.53588974175501,
    "left_arm_joint5": -1.0890854532444616,
    "left_arm_joint6": 0.2897246558310587,
    "left_arm_joint7": 0.08552113334772216,
}
"""Initial G1 left-arm joint positions applied when the pick-cube env resets."""

G1_PICK_CUBE_LEFT_ARM_INITIAL_ACTION_VALUES = tuple(
    G1_PICK_CUBE_LEFT_ARM_INITIAL_JOINT_POS[joint_name]
    for joint_name in G1_LEFT_ARM_DOF_ORDER
)
"""Initial left-arm posture ordered to match the left-arm action tensor."""

G1_PICK_CUBE_HEAD_INITIAL_JOINT_POS = {
    "head_joint1": 0.0,
    "head_joint2": 0.45,
}
"""Initial G1 head posture for front-head camera table visibility."""

G1_PICK_CUBE_FRONT_HEAD_CAMERA_WIDTH = 298
G1_PICK_CUBE_FRONT_HEAD_CAMERA_HEIGHT = 224


@configclass
class G1PickCubeSceneCfg(PickCubeSceneCfg):
    """Pick-cube world with the G1 robot and front-head camera inserted."""

    robot = make_galbot_g1_manipulation_articulation_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        required_asset=False,
    )
    robot.init_state.joint_pos.update(G1_PICK_CUBE_LEFT_ARM_INITIAL_JOINT_POS)
    robot.init_state.joint_pos.update(G1_PICK_CUBE_HEAD_INITIAL_JOINT_POS)
    front_head_rgb_camera = make_g1_camera_cfg(
        mount="front_head",
        data="rgb",
        width=G1_PICK_CUBE_FRONT_HEAD_CAMERA_WIDTH,
        height=G1_PICK_CUBE_FRONT_HEAD_CAMERA_HEIGHT,
    )


@configclass
class G1PickCubeMimicSceneCfg(G1PickCubeSceneCfg):
    """Pick-cube scene cfg with TCP frame required by IsaacLab Mimic."""

    tcp_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_arm_link7",
                name="left_tcp",
                offset=OffsetCfg(
                    pos=(-0.25573010977804644, 0.0, 0.0),
                    rot=(0.0, 0.0, 1.0, 0.0),
                ),
            ),
        ],
    )


@configclass
class G1PickCubeTeleopSceneCfg(G1PickCubeSceneCfg):
    """Pick-cube scene cfg with G1 RGB cameras for GP001 teleop."""

    left_wrist_rgb_camera = g1.sensors.camera("left_wrist")
    front_head_rgb_camera = g1.sensors.camera("front_head")


@configclass
class GalbotG1PickCubeBaseEnvCfg(PickCubeMdpCfg, DefaultEnvCfg):
    """Motion-planning env cfg for data collection."""

    gripper_joint_names = ["left_gripper_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 1.2

    scene: G1PickCubeSceneCfg = G1PickCubeSceneCfg(env_spacing=2.5)
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    evaluation_success = pick_cube_terminations.make_pick_cube_evaluation_success_term()


@configclass
class GalbotG1PickCubeMimicEnvCfg(GalbotG1PickCubeBaseEnvCfg, MimicEnvCfg):
    """Mimic-compatible env cfg with success termination metadata."""

    scene: G1PickCubeMimicSceneCfg = G1PickCubeMimicSceneCfg(env_spacing=2.5)
    terminations: PickCubeMimicSuccessCfg = PickCubeMimicSuccessCfg()
    mimic: MimicCfg = MimicCfg(
        eef_name="left_tcp",
        converter=G1ArmEefActionConverter.left(eef_name="left_tcp"),
        stages={
            "grasp_cube": {
                "object": "cube",
                "done": pick_cube_terminations.grasped_cube,
                "offset_range": (5, 15),
                "next": "pick_cube_success",
            },
            # Final stage has no subtask term signal. IsaacLab Mimic uses
            # ``terminations.success`` to validate whole-task completion and
            # treats the final stage as ending at episode end.
            "pick_cube_success": {
                "object": "cube",
            },
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.concatenate_terms = False
        self.events.randomize_ground_material = None
        self.events.randomize_table_material = None
        self.events.randomize_hdri_texture = None
        self.mimic.apply_to(self)


@configclass
class GalbotG1PickCubeTeleopEnvCfg(GalbotG1PickCubeBaseEnvCfg):
    """Teleop env cfg for ``GalbotG1-PickCube-Teleop-v0``."""

    observations: PickCubeTeleopObservationsCfg = PickCubeTeleopObservationsCfg()
    scene: G1PickCubeTeleopSceneCfg = G1PickCubeTeleopSceneCfg(env_spacing=2.5)


@configclass
class GalbotG1PickCubeEnvCfg(GalbotG1PickCubeBaseEnvCfg):
    """Registered env cfg for ``GalbotG1-PickCube-v0``."""


__all__ = [
    "G1_PICK_CUBE_FRONT_HEAD_CAMERA_HEIGHT",
    "G1_PICK_CUBE_FRONT_HEAD_CAMERA_WIDTH",
    "G1_PICK_CUBE_HEAD_INITIAL_JOINT_POS",
    "G1_PICK_CUBE_LEFT_ARM_INITIAL_ACTION_VALUES",
    "G1_PICK_CUBE_LEFT_ARM_INITIAL_JOINT_POS",
    "G1PickCubeMimicSceneCfg",
    "G1PickCubeSceneCfg",
    "G1PickCubeTeleopSceneCfg",
    "GalbotG1PickCubeBaseEnvCfg",
    "GalbotG1PickCubeEnvCfg",
    "GalbotG1PickCubeMimicEnvCfg",
    "GalbotG1PickCubeTeleopEnvCfg",
]
