"""Env cfg for the SortToShelf navigation phase task."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.articulation import G1_MOBILE_BASE_BODY_NAME
from ioailab.tasks.common.defaults import DefaultEnvCfg
from ioailab.tasks.sort_to_shelf.scene import (
    SORTING_SHELF_NAV_YAW,
    sorting_place_base_position_for_object,
)
from ioailab.tasks.sort_to_shelf_nav.config.g1.mdp_cfg import (
    SortToShelfNavMdpCfg,
    make_nav_success_term,
)
from ioailab.tasks.sort_to_shelf_pick.config.g1.env_cfg import (
    G1SortToShelfCarrySceneCfg,
    apply_sort_to_shelf_task_options,
)


@configclass
class GalbotG1SortToShelfNavEnvCfg(SortToShelfNavMdpCfg, DefaultEnvCfg):
    """Standalone navigation phase env for moving a carried object to the shelf."""

    goal_position = sorting_place_base_position_for_object("red_cube")
    goal_yaw = float(SORTING_SHELF_NAV_YAW)
    success_radius: float = 0.02
    max_command_speed: float = 0.45
    base_body_name: str = G1_MOBILE_BASE_BODY_NAME

    selected_sorting_object = "red_cube"
    gripper_joint_names = ["left_gripper_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 1.2

    scene: G1SortToShelfCarrySceneCfg = G1SortToShelfCarrySceneCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    evaluation_success = make_nav_success_term("red_cube")

    def apply_task_options(self, task_options: dict) -> None:
        """Apply selected sorting object and optional init scenario."""

        apply_sort_to_shelf_task_options(self, task_options)


__all__ = ["GalbotG1SortToShelfNavEnvCfg"]
