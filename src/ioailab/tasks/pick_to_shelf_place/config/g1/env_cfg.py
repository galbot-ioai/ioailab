"""Env cfg for the PickToShelf place phase task."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.defaults import DefaultEnvCfg
from ioailab.tasks.common.scenario import load_scenario
from ioailab.tasks.pick_to_shelf_place.config.g1.mdp_cfg import (
    PickToShelfPlaceMdpCfg,
)
from ioailab.tasks.pick_to_shelf_place.mdp.terminations import (
    make_shelf_place_success_term,
)
from ioailab.tasks.pick_to_shelf_pick.config.g1.env_cfg import G1PickToShelfSceneCfg


@configclass
class GalbotG1PickToShelfPlaceEnvCfg(PickToShelfPlaceMdpCfg, DefaultEnvCfg):
    """Standalone place phase env for policy collection, training, and eval."""

    gripper_joint_names = ["left_gripper_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 1.2

    scene: G1PickToShelfSceneCfg = G1PickToShelfSceneCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    evaluation_success = make_shelf_place_success_term()

    def apply_task_options(self, options: dict) -> None:
        """Apply task-local options before env construction."""

        _apply_init_scenario_option(self, options)


def _apply_init_scenario_option(
    cfg: GalbotG1PickToShelfPlaceEnvCfg, options: dict
) -> None:
    unknown = set(options) - {"init_scenario"}
    if unknown:
        raise ValueError(
            f"Unknown PickToShelf Place task option(s): {tuple(sorted(unknown))}."
        )
    scenario_path = options.get("init_scenario")
    if scenario_path:
        cfg.events.reset_all.params["scenario"] = load_scenario(scenario_path)


__all__ = ["GalbotG1PickToShelfPlaceEnvCfg"]
