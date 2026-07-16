"""Env cfg for the SortToShelf place phase task."""

from __future__ import annotations

from collections.abc import Mapping

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.defaults import DefaultEnvCfg
from ioailab.tasks.sort_to_shelf_pick.config.g1.env_cfg import (
    G1SortToShelfShelfFacingSceneCfg,
    apply_sort_to_shelf_task_options,
)
from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
    make_place_success_term,
)
from ioailab.tasks.sort_to_shelf_place.config.g1.mdp_cfg import (
    SortToShelfPlaceMdpCfg,
)


@configclass
class GalbotG1SortToShelfPlaceEnvCfg(SortToShelfPlaceMdpCfg, DefaultEnvCfg):
    """Standalone place phase env for sorting data collection and evaluation."""

    selected_sorting_object = "red_cube"
    gripper_joint_names = ["left_gripper_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 1.2

    scene: G1SortToShelfShelfFacingSceneCfg = G1SortToShelfShelfFacingSceneCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    evaluation_success = make_place_success_term("red_cube")

    def apply_task_options(self, task_options: Mapping[str, object]) -> None:
        """Apply selected sorting object from ``make_env(..., task_options=...)``."""

        apply_sort_to_shelf_task_options(self, task_options)


__all__ = ["GalbotG1SortToShelfPlaceEnvCfg"]
