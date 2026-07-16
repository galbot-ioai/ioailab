"""Env cfg and G1 scene bindings for SortToShelf phase tasks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import DomeLightCfg
from isaaclab.utils.configclass import configclass

from ioailab.agents.flow import TaskFlowSpec, TaskPhaseSpec
import ioailab.envs.flow as env_flow
from ioailab.robots.g1.actions import (
    DEFAULT_GRIPPER_CLOSED_POSITION,
)
from ioailab.robots.g1.articulation import (
    make_galbot_g1_manipulation_articulation_cfg,
    make_galbot_g1_mobile_base_articulation_cfg,
)
from ioailab.robots.g1.sensors.camera import make_g1_camera_cfg
from ioailab.tasks.common.defaults import DefaultEnvCfg
from ioailab.tasks.common.scenario import (
    load_scenario,
    scenario_reset_event,
)
from ioailab.tasks.sort_to_shelf.scene import (
    SORTING_SHELF_BASE_ORIENTATION,
    SORTING_SHELF_NAV_XY,
    SortToShelfSceneCfg,
    sorting_object_name,
    sorting_object_pick_lift_min_z,
    sorting_object_requires_leg_lift,
    sorting_place_base_position_for_object,
    sorting_place_board_asset_name_for_object,
    sorting_place_target_offset_from_board_for_object,
    sorting_place_upright_z_axis_min_dot_for_object,
)
from ioailab.tasks.sort_to_shelf_nav.config.g1.mdp_cfg import (
    sorting_place_start_joint_pos_for_object,
)
from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
    G1_SORT_TO_SHELF_HEAD_INITIAL_JOINT_POS,
    G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
    G1_SORT_TO_SHELF_RIGHT_ARM_INITIAL_JOINT_POS,
    SORTING_A_CELL_LEG_LIFT_JOINT_POS,
    SORTING_DEFAULT_LEG_JOINT_POS,
    SortToShelfPickMdpCfg,
    make_pick_success_term,
    make_place_success_term,
    sorting_place_approach_left_arm_joint_pos_for_object,
)

SORTING_PICK_BASE_POSITION = (-1.2, 0.0, 0.0)
SORTING_PICK_BASE_ORIENTATION = (0.0, 0.0, 0.0, 1.0)
_TASKS_ROOT = Path(__file__).resolve().parents[3]


def _make_sort_to_shelf_mobile_robot_cfg():
    """Return the mobile G1 cfg reset upright at the sorting pick pose."""

    return make_galbot_g1_mobile_base_articulation_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        required_asset=False,
        base_position=SORTING_PICK_BASE_POSITION,
        base_orientation=SORTING_PICK_BASE_ORIENTATION,
        use_usd_controller_graphs=False,
    )


@configclass
class G1SortToShelfSceneCfg(SortToShelfSceneCfg):
    """Sort-to-shelf world with mobile-base G1 and front-head camera."""

    light = AssetBaseCfg(
        prim_path="/World/ioailabLight",
        spawn=DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )
    robot = _make_sort_to_shelf_mobile_robot_cfg()
    robot.init_state.joint_pos.update(G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS)
    robot.init_state.joint_pos.update(G1_SORT_TO_SHELF_RIGHT_ARM_INITIAL_JOINT_POS)
    robot.init_state.joint_pos.update({"head_joint1": 0.0, "head_joint2": 0.25})
    front_head_rgb_camera = make_g1_camera_cfg(
        mount="front_head",
        data="rgbd_semantic",
        width=298,
        height=224,
    )


@configclass
class G1SortToShelfPickSceneCfg(G1SortToShelfSceneCfg):
    """Sort-to-shelf tabletop pick scene with the mobile base enabled."""


@configclass
class G1SortToShelfCarrySceneCfg(G1SortToShelfSceneCfg):
    """Sort-to-shelf mobile scene with G1 staged to carry the selected object."""

    robot = _make_sort_to_shelf_mobile_robot_cfg()
    robot.init_state.joint_pos.update(G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS)
    robot.init_state.joint_pos["left_gripper_joint"] = DEFAULT_GRIPPER_CLOSED_POSITION
    robot.init_state.joint_pos.update(G1_SORT_TO_SHELF_RIGHT_ARM_INITIAL_JOINT_POS)
    robot.init_state.joint_pos.update(G1_SORT_TO_SHELF_HEAD_INITIAL_JOINT_POS)


@configclass
class G1SortToShelfShelfFacingSceneCfg(G1SortToShelfSceneCfg):
    """Sort-to-shelf scene with the G1 pre-staged at the shelf."""

    robot = make_galbot_g1_manipulation_articulation_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        required_asset=False,
        root_position=(SORTING_SHELF_NAV_XY[0], SORTING_SHELF_NAV_XY[1], 0.0),
        root_orientation=SORTING_SHELF_BASE_ORIENTATION,
    )
    robot.init_state.joint_pos.update(G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS)
    robot.init_state.joint_pos["left_gripper_joint"] = DEFAULT_GRIPPER_CLOSED_POSITION
    robot.init_state.joint_pos.update(G1_SORT_TO_SHELF_RIGHT_ARM_INITIAL_JOINT_POS)
    robot.init_state.joint_pos.update(G1_SORT_TO_SHELF_HEAD_INITIAL_JOINT_POS)
    robot.init_state.joint_pos.update(SORTING_A_CELL_LEG_LIFT_JOINT_POS)


_SORTING_TASK_OPTION_KEYS = {
    "init_scenario",
    "sorting_object",
}


def apply_sort_to_shelf_task_options(
    cfg: Any, task_options: Mapping[str, object]
) -> None:
    """Apply task-level sorting object selection to cfg-owned terms."""

    options = dict(task_options)
    unknown = tuple(sorted(set(options) - _SORTING_TASK_OPTION_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown sort-to-shelf task option(s): {unknown}. "
            f"Allowed options: {tuple(sorted(_SORTING_TASK_OPTION_KEYS))}."
        )
    selected_object = sorting_object_name(
        options.get(
            "sorting_object", getattr(cfg, "selected_sorting_object", "red_cube")
        )
    )
    cfg.selected_sorting_object = selected_object
    _configure_sorting_success_terms(cfg, selected_object)
    _configure_sorting_place_events(cfg, selected_object)
    _configure_sorting_nav_goal(cfg, selected_object)
    _configure_sorting_place_leg_posture(cfg, selected_object)
    _configure_sorting_place_left_arm_posture(cfg, selected_object)
    _configure_sorting_place_base(cfg, selected_object)
    _configure_sorting_task_flow(cfg, selected_object)
    _configure_sorting_reset_scenario(
        cfg, selected_object, options.get("init_scenario")
    )


def _configure_sorting_success_terms(cfg: Any, object_name: str) -> None:
    terms = [getattr(cfg, "evaluation_success", None)]
    terminations = getattr(cfg, "terminations", None)
    if terminations is not None:
        terms.extend(
            getattr(terminations, term_name, None)
            for term_name in ("at_carry", "at_place_start", "placed")
        )

    for term in terms:
        _configure_sorting_success_term_params(term, object_name)


def _configure_sorting_success_term_params(term: Any, object_name: str) -> None:
    params = getattr(term, "params", None)
    if not isinstance(params, dict):
        return
    if "object_cfg" in params:
        params["object_cfg"] = SceneEntityCfg(object_name)
    if "min_object_center_z" in params:
        params["min_object_center_z"] = sorting_object_pick_lift_min_z(object_name)
    if "target_asset_cfg" in params:
        params["target_asset_cfg"] = SceneEntityCfg(
            sorting_place_board_asset_name_for_object(object_name)
        )
    if "target_offset_xyz" in params:
        params["target_offset_xyz"] = sorting_place_target_offset_from_board_for_object(
            object_name
        )
    if "upright_z_axis_min_dot" in params:
        params["upright_z_axis_min_dot"] = (
            sorting_place_upright_z_axis_min_dot_for_object(object_name)
        )
    if "target_joint_names" in params and "target_joint_pos_by_name" in params:
        params["target_joint_pos_by_name"] = sorting_place_start_joint_pos_for_object(
            object_name
        )


def _configure_sorting_place_events(cfg: Any, object_name: str) -> None:
    events = getattr(cfg, "events", None)
    if events is None:
        return
    for event_name in ("randomize_pick_and_place_positions",):
        event = getattr(events, event_name, None)
        params = getattr(event, "params", None)
        if not isinstance(params, dict):
            continue
        if "object_cfg" in params:
            params["object_cfg"] = SceneEntityCfg(object_name)


def _configure_sorting_reset_scenario(
    cfg: Any, object_name: str, scenario: object | None
) -> None:
    if scenario:
        _set_sorting_reset_scenario(cfg, scenario)
        return
    phase = _default_scenario_phase_for_cfg(cfg)
    if phase is None:
        return
    _set_sorting_reset_scenario(cfg, _sorting_phase_scenario(phase, object_name))


def _default_scenario_phase_for_cfg(cfg: Any) -> str | None:
    scene = getattr(cfg, "scene", None)
    if isinstance(scene, G1SortToShelfCarrySceneCfg):
        return "nav"
    if isinstance(scene, G1SortToShelfShelfFacingSceneCfg):
        return "place"
    return None


def _sorting_phase_scenario(phase: str, object_name: str | None):
    phase_name = str(phase)
    if phase_name not in {"nav", "place"}:
        raise ValueError(f"Unknown SortToShelf scenario phase {phase!r}.")
    scenario_path = (
        _TASKS_ROOT
        / f"sort_to_shelf_{phase_name}"
        / "config"
        / "g1"
        / "scenarios"
        / f"{sorting_object_name(object_name)}.yaml"
    )
    return load_scenario(scenario_path)


def _set_sorting_reset_scenario(cfg: Any, scenario: object) -> None:
    events = getattr(cfg, "events", None)
    if events is not None:
        events.reset_all = scenario_reset_event(scenario)
        for event_name in ("randomize_pick_and_place_positions",):
            if hasattr(events, event_name):
                setattr(events, event_name, None)


def _configure_sorting_nav_goal(cfg: Any, object_name: str) -> None:
    """Set nav success goal to the standalone place start base position."""

    if hasattr(cfg, "goal_position"):
        cfg.goal_position = sorting_place_base_position_for_object(object_name)


def _configure_sorting_place_leg_posture(cfg: Any, object_name: str) -> None:
    scene = getattr(cfg, "scene", None)
    if not isinstance(scene, G1SortToShelfShelfFacingSceneCfg):
        return
    joint_pos = scene.robot.init_state.joint_pos
    posture = (
        SORTING_A_CELL_LEG_LIFT_JOINT_POS
        if sorting_object_requires_leg_lift(object_name)
        else SORTING_DEFAULT_LEG_JOINT_POS
    )
    joint_pos.update(posture)


def _configure_sorting_place_left_arm_posture(cfg: Any, object_name: str) -> None:
    """Start standalone place tasks at the target-outside aligned left-arm pose."""

    scene = getattr(cfg, "scene", None)
    if not isinstance(scene, G1SortToShelfShelfFacingSceneCfg):
        return
    scene.robot.init_state.joint_pos.update(
        sorting_place_approach_left_arm_joint_pos_for_object(object_name)
    )


def _configure_sorting_place_base(cfg: Any, object_name: str) -> None:
    """Shift the standalone place base toward the object's target column."""

    scene = getattr(cfg, "scene", None)
    if not isinstance(scene, G1SortToShelfShelfFacingSceneCfg):
        return
    scene.robot.init_state.pos = sorting_place_base_position_for_object(object_name)


def _configure_sorting_task_flow(cfg: Any, object_name: str) -> None:
    flow = getattr(cfg, "task_flow", None)
    if not isinstance(flow, TaskFlowSpec):
        return

    pick_success = make_pick_success_term(object_name)
    place_success = make_place_success_term(object_name)
    phases: list[TaskPhaseSpec] = []
    for phase in flow.phases:
        success = phase.success
        if phase.name == "pick":
            success = _success_func(pick_success)
        elif phase.name == "nav":
            # Full-task nav is a nested agent sequence: drive to the shelf, then
            # settle legs/left arm into the place-start posture. Let the
            # sequence own its latched completion state instead of rechecking a
            # non-latched MDP predicate that can fail if the base drifts after
            # the drive substep finishes.
            success = None
        elif phase.name == "place":
            success = _success_func(place_success)
        phases.append(
            TaskPhaseSpec(
                name=phase.name,
                phase_task_id=phase.phase_task_id,
                success=success,
                default_agent=phase.default_agent,
                action_terms=phase.action_terms,
                fixed_base=phase.fixed_base,
            )
        )
    cfg.task_flow = TaskFlowSpec(
        phases=tuple(phases),
        final_phase=flow.final_phase,
        phase_state_getter=flow.phase_state_getter,
    )

    terminations = getattr(cfg, "terminations", None)
    if terminations is not None and hasattr(terminations, "placed"):
        terminations.placed = DoneTerm(
            func=env_flow.phase_gated_success("place", place_success)
        )
        cfg.evaluation_success = terminations.placed


def _success_func(term: Any):
    def _predicate(env: Any) -> Any:
        return term.func(env, **dict(getattr(term, "params", {}) or {}))

    _predicate.__name__ = getattr(term.func, "__name__", "phase_success")
    return _predicate


@configclass
class GalbotG1SortToShelfPickEnvCfg(SortToShelfPickMdpCfg, DefaultEnvCfg):
    """Standalone pick phase env for sorting data collection and evaluation."""

    selected_sorting_object = "red_cube"
    gripper_joint_names = ["left_gripper_joint"]
    gripper_open_val = 0.0
    gripper_threshold = 1.2

    scene: G1SortToShelfPickSceneCfg = G1SortToShelfPickSceneCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    evaluation_success = make_pick_success_term("red_cube")

    def apply_task_options(self, task_options: Mapping[str, object]) -> None:
        """Apply selected sorting object from ``make_env(..., task_options=...)``."""

        apply_sort_to_shelf_task_options(self, task_options)


__all__ = [
    "G1SortToShelfCarrySceneCfg",
    "G1SortToShelfPickSceneCfg",
    "G1SortToShelfSceneCfg",
    "G1SortToShelfShelfFacingSceneCfg",
    "GalbotG1SortToShelfPickEnvCfg",
    "apply_sort_to_shelf_task_options",
]
