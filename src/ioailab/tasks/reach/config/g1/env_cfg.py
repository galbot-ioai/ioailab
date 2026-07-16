"""G1 reach task config."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from ioailab.robots.g1.actions import g1_action_cfg
from ioailab.robots.g1.articulation import (
    DEFAULT_END_EFFECTOR_LINK,
    make_galbot_g1_manipulation_articulation_cfg,
)


@configclass
class GalbotG1ReachEnvCfg(ReachEnvCfg):
    """IsaacLab-native G1 left-arm reach config."""

    def __post_init__(self) -> None:
        """Replace the upstream robot/actions with G1 left-arm task defaults."""

        super().__post_init__()
        self.scene.robot = make_galbot_g1_manipulation_articulation_cfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            required_asset=False,
        )
        self.actions.arm_action = g1_action_cfg("left_arm", "absolute")
        if hasattr(self.actions, "gripper_action"):
            self.actions.gripper_action = None
        self.commands.ee_pose.body_name = DEFAULT_END_EFFECTOR_LINK
        self.commands.ee_pose.ranges.pitch = (0.0, 0.0)
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [
            DEFAULT_END_EFFECTOR_LINK
        ]
        self.rewards.end_effector_position_tracking_fine_grained.params[
            "asset_cfg"
        ].body_names = [DEFAULT_END_EFFECTOR_LINK]
        self.rewards.end_effector_orientation_tracking.params[
            "asset_cfg"
        ].body_names = [DEFAULT_END_EFFECTOR_LINK]
        self.rewards.end_effector_orientation_tracking.weight = 0.0
        self.observations.policy.enable_corruption = False
