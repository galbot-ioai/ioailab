"""Env cfg for the PickToShelf navigation phase task."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.articulation import G1_MOBILE_BASE_BODY_NAME
from ioailab.tasks.base_nav.mdp.terminations import goal_reached
from ioailab.tasks.common.defaults import DefaultEnvCfg
from ioailab.tasks.common.scenario import load_scenario
from ioailab.tasks.pick_to_shelf_nav.config.g1.mdp_cfg import (
    PickToShelfNavMdpCfg,
)
from ioailab.tasks.pick_to_shelf_nav.mdp.goals import SHELF_NAV_XY, SHELF_NAV_YAW
from ioailab.tasks.pick_to_shelf_pick.config.g1.env_cfg import G1PickToShelfSceneCfg


@configclass
class GalbotG1PickToShelfNavEnvCfg(PickToShelfNavMdpCfg, DefaultEnvCfg):
    """Standalone navigation phase env for moving the carried cube to the shelf."""

    goal_position = (SHELF_NAV_XY[0], SHELF_NAV_XY[1], 0.0)
    goal_yaw = float(SHELF_NAV_YAW)
    success_radius: float = 0.02
    max_command_speed: float = 0.45
    base_body_name: str = G1_MOBILE_BASE_BODY_NAME

    scene: G1PickToShelfSceneCfg = G1PickToShelfSceneCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    evaluation_success = DoneTerm(func=goal_reached)

    def apply_task_options(self, options: dict) -> None:
        """Apply task-local options before env construction."""

        _apply_init_scenario_option(self, options)


def _apply_init_scenario_option(
    cfg: GalbotG1PickToShelfNavEnvCfg, options: dict
) -> None:
    unknown = set(options) - {"init_scenario"}
    if unknown:
        raise ValueError(
            f"Unknown PickToShelf Nav task option(s): {tuple(sorted(unknown))}."
        )
    scenario_path = options.get("init_scenario")
    if scenario_path:
        cfg.events.reset_all.params["scenario"] = load_scenario(scenario_path)


__all__ = ["GalbotG1PickToShelfNavEnvCfg"]
