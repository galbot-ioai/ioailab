"""Manager-based Galbot G1 base navigation task config."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.articulation import (
    G1_MOBILE_BASE_BODY_NAME,
    make_galbot_g1_mobile_base_articulation_cfg,
)
from ioailab.tasks.base_nav.config.g1.mdp_cfg import BaseNavMdpCfg
from ioailab.tasks.common.defaults import DefaultEnvCfg
from ioailab.tasks.common.defaults import DefaultSceneCfg
from ioailab.tasks.common.defaults import make_default_ground_cfg


@configclass
class BaseNavSceneCfg(DefaultSceneCfg):
    """Task-local scene for G1 base navigation."""

    robot = make_galbot_g1_mobile_base_articulation_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        required_asset=False,
        base_position=(0.0, 0.0, 0.0),
        base_orientation=(0.0, 0.0, 0.0, 1.0),
        use_usd_controller_graphs=False,
    )
    plane = make_default_ground_cfg(size=(4.0, 4.0, 0.02))


@configclass
class GalbotG1BaseNavEnvCfg(BaseNavMdpCfg, DefaultEnvCfg):
    """G1 base navigation env from A=(0, 0) to B=(2, 0)."""

    scene: BaseNavSceneCfg = BaseNavSceneCfg(num_envs=16)
    episode_length_s: float = 12.0
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()

    goal_position: tuple[float, float, float] = (2.0, 0.0, 0.0)
    success_radius: float = 0.15
    max_command_speed: float = 0.45
    base_body_name: str = G1_MOBILE_BASE_BODY_NAME
