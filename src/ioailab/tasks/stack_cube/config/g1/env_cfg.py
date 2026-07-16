"""Manager-based Galbot G1 stack-cube task config."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from ioailab.datasets.mimic import MimicCfg
from ioailab.robots.g1.articulation import (
    make_galbot_g1_manipulation_articulation_cfg,
)
from ioailab.robots.g1.converters import G1ArmEefActionConverter
from ioailab.tasks.common.defaults import DefaultEnvCfg
from ioailab.tasks.stack_cube.config.g1.mdp_cfg import StackCubeMdpCfg
from ioailab.tasks.stack_cube.mdp import StackCubeMimicSuccessCfg
from ioailab.tasks.stack_cube.mdp import terminations as stack_cube_terminations
from ioailab.tasks.stack_cube.scene import StackCubeSceneCfg

G1_BASE_LINK_PRIM_PATH = "{ENV_REGEX_NS}/Robot/base_footprint"
G1_LEFT_END_EFFECTOR_PRIM_PATH = "{ENV_REGEX_NS}/Robot/left_arm_link7"

G1_STACK_CUBE_LEFT_ARM_INITIAL_JOINT_POS = {
    "left_arm_joint1": 0.6544984694978736,
    "left_arm_joint2": -0.5218534463463045,
    "left_arm_joint3": -0.12391837689159762,
    "left_arm_joint4": -1.53588974175501,
    "left_arm_joint5": -1.0890854532444616,
    "left_arm_joint6": 0.2897246558310587,
    "left_arm_joint7": 0.08552113334772216,
}
"""Initial G1 left-arm joint positions applied when the stack-cube env resets."""


@configclass
class G1StackCubeSceneCfg(StackCubeSceneCfg):
    """Stack-cube world with the G1 robot and end-effector frame inserted."""

    robot = make_galbot_g1_manipulation_articulation_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        required_asset=False,
    )
    robot.init_state.joint_pos.update(G1_STACK_CUBE_LEFT_ARM_INITIAL_JOINT_POS)
    ee_frame = FrameTransformerCfg(
        prim_path=G1_BASE_LINK_PRIM_PATH,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=G1_LEFT_END_EFFECTOR_PRIM_PATH,
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )


@configclass
class G1StackCubeMimicSceneCfg(G1StackCubeSceneCfg):
    """Stack-cube scene cfg with the TCP frame required by IsaacLab Mimic."""

    tcp_frame = FrameTransformerCfg(
        prim_path=G1_BASE_LINK_PRIM_PATH,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=G1_LEFT_END_EFFECTOR_PRIM_PATH,
                name="left_tcp",
                offset=OffsetCfg(
                    pos=(-0.25573010977804644, 0.0, 0.0),
                    rot=(0.0, 0.0, 1.0, 0.0),
                ),
            ),
        ],
    )


@configclass
class GalbotG1StackCubeEnvCfg(StackCubeMdpCfg, DefaultEnvCfg):
    """G1 stack-cube env cfg registered as ``GalbotG1-StackCube-v0``."""

    scene: G1StackCubeSceneCfg = G1StackCubeSceneCfg(env_spacing=2.5)
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()

    gripper_joint_names = ["left_gripper_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 0.10


@configclass
class GalbotG1StackCubeMimicEnvCfg(GalbotG1StackCubeEnvCfg, MimicEnvCfg):
    """Mimic-compatible env cfg for StackCube data expansion."""

    scene: G1StackCubeMimicSceneCfg = G1StackCubeMimicSceneCfg(env_spacing=2.5)
    terminations: StackCubeMimicSuccessCfg = StackCubeMimicSuccessCfg()
    mimic: MimicCfg = MimicCfg(
        eef_name="left_tcp",
        converter=G1ArmEefActionConverter.left(eef_name="left_tcp"),
        stages={
            "grasp_cube_2": {
                "object": "cube_2",
                "done": stack_cube_terminations.cube_2_grasped,
                "offset_range": (5, 15),
                "next": "place_cube_2_on_cube_1",
            },
            "place_cube_2_on_cube_1": {
                "object": "cube_1",
                "done": stack_cube_terminations.cube_2_on_cube_1,
                "offset_range": (0, 5),
                "next": "grasp_cube_3",
            },
            "grasp_cube_3": {
                "object": "cube_3",
                "done": stack_cube_terminations.cube_3_grasped,
                "offset_range": (5, 15),
                "next": "place_cube_3_on_cube_2",
            },
            "place_cube_3_on_cube_2": {
                "object": "cube_2",
                "done": stack_cube_terminations.cube_3_on_cube_2,
                "offset_range": (0, 5),
                "next": "stack_cube_success",
            },
            # Final stage has no subtask term signal. IsaacLab Mimic uses
            # ``terminations.success`` to validate whole-task completion and
            # treats the final stage as ending at episode end.
            "stack_cube_success": {
                "object": "cube_2",
            },
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.concatenate_terms = False
        self.mimic.apply_to(self)
