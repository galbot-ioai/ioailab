from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_TASK_IDS = (
    "GalbotG1-Reach-v0",
    "GalbotG1-PickCube-v0",
    "GalbotG1-PickCube-Teleop-v0",
    "GalbotG1-PickCube-Mimic-v0",
    "GalbotG1-StackCube-v0",
    "GalbotG1-BaseNav-v0",
    "GalbotG1-PickToShelf-v0",
    "GalbotG1-PickToShelf-Pick-v0",
    "GalbotG1-PickToShelf-Nav-v0",
    "GalbotG1-PickToShelf-Place-v0",
    "GalbotG1-SortToShelf-v0",
    "GalbotG1-SortToShelf-Pick-v0",
    "GalbotG1-SortToShelf-Nav-v0",
    "GalbotG1-SortToShelf-Place-v0",
)

DEFERRED_TASK_IDS = (
    "GalbotG1PickCube-v0",
    "GalbotG1WholeBodyPickCube-v0",
    "GalbotG1DualArmStackCube-v0",
    "GalbotG1WholeBodyStackCube-v0",
    "GalbotG1MobileStackCube-v0",
)


def source_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return env


def _assert_ground_top_surface_at_world_zero(plane_cfg) -> None:
    assert plane_cfg.init_state.pos[2] + plane_cfg.spawn.size[2] / 2.0 == pytest.approx(
        0.0
    )


def test_common_scene_defaults_keep_values_inline_without_global_constants() -> None:
    source = (ROOT / "src" / "ioailab" / "tasks" / "common" / "defaults.py").read_text(
        encoding="utf-8"
    )

    assert "DEFAULT_GROUND" not in source
    assert "DEFAULT_LIGHT" not in source
    # The single-use light factory was inlined into DefaultSceneCfg.light.
    assert "make_default_light_cfg" not in source


def test_base_nav_constants_module_is_folded_into_env_cfg() -> None:
    """base_nav has no standalone constants module after the fold."""

    assert not (
        ROOT / "src" / "ioailab" / "tasks" / "base_nav" / "constants.py"
    ).exists()


def test_task_spec_exposes_only_runtime_consumed_fields() -> None:
    from ioailab.tasks import TaskSpec

    entry = TaskSpec(
        task_id="GalbotExample-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        isaaclab_kwargs={
            "env_cfg_entry_point": "pkg.env:Cfg",
            "rsl_rl_cfg_entry_point": "pkg.agents:RunnerCfg",
        },
        motion_plan_entry_point="pkg.motion:ExampleMotionPlan",
        requires_cameras=True,
        reset_randomization_events=("randomize_pick_and_place_positions",),
    )

    assert entry.env_cfg_entry_point == "pkg.env:Cfg"
    # IsaacLab RL training reads agent cfg entry points straight off the gym kwargs.
    assert entry.gym_kwargs() == {
        "env_cfg_entry_point": "pkg.env:Cfg",
        "rsl_rl_cfg_entry_point": "pkg.agents:RunnerCfg",
    }
    assert entry.motion_plan_entry_point == "pkg.motion:ExampleMotionPlan"
    assert entry.requires_cameras is True
    assert entry.reset_randomization_events == ("randomize_pick_and_place_positions",)
    assert entry.task_id == "GalbotExample-v0"

    # Removed metadata that nothing consumes at runtime must stay gone.
    for removed in (
        "env_id",
        "workflow_name",
        "workflow_tags",
        "teleop_entry_point",
        "synthetic_data_entry_point",
        "rsl_rl_cfg_entry_point",
        "rl_cfg_entry_points",
        "workflow_config_types",
        "motion_planning_cfg_entry_point",
        "subtask_resolver_entry_point",
        "subtask_motion_plan_resolver_entry_point",
        "subtask_nav_agent_resolver_entry_point",
        "subtask_names",
    ):
        assert not hasattr(entry, removed), removed


def test_task_registry_public_names_use_task_id_not_env_id() -> None:
    import ioailab.tasks as tasks

    assert "BUILTIN_TASK_IDS" in tasks.__all__
    assert "DEFAULT_TASK_ID" in tasks.__all__
    assert "task_entry_for_task_id" in tasks.__all__
    assert "nav_agent_for_task" in tasks.__all__
    for stale_name in (
        "BUILTIN_ENV_IDS",
        "DEFAULT_ENV_ID",
        "task_entry_for_env_id",
        "GALBOT_G1_PICK_CUBE_ENV_ID",
        "GALBOT_G1_PICK_CUBE_ENV_IDS",
    ):
        assert stale_name not in tasks.__all__
        assert not hasattr(tasks, stale_name)


def test_top_level_registry_registers_task_specs() -> None:
    registry_source = (ROOT / "src" / "ioailab" / "tasks" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert "gym.register(" in registry_source
    assert "_register_task_spec" in registry_source


def test_task_modules_only_declare_task_specs() -> None:
    task_module_paths = (
        ROOT / "src" / "ioailab" / "tasks" / "reach" / "__init__.py",
        ROOT / "src" / "ioailab" / "tasks" / "pick_cube" / "__init__.py",
        ROOT / "src" / "ioailab" / "tasks" / "stack_cube" / "__init__.py",
        ROOT / "src" / "ioailab" / "tasks" / "base_nav" / "__init__.py",
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf" / "__init__.py",
        ROOT / "src" / "ioailab" / "tasks" / "sort_to_shelf" / "__init__.py",
    )
    assert not (ROOT / "src" / "ioailab" / "tasks" / "_registry.py").exists()
    for path in task_module_paths:
        source = path.read_text(encoding="utf-8")
        assert "def register_tasks(" not in source, path
        assert "gym.register(" not in source, path
        assert "from gymnasium" not in source, path


def test_task_registry_modules_do_not_import_motion_plans() -> None:
    """Task registries stay manifests; motion-plan code is loaded from entry points."""

    reach_module = ROOT / "src" / "ioailab" / "tasks" / "reach" / "__init__.py"
    pick_cube_module = ROOT / "src" / "ioailab" / "tasks" / "pick_cube" / "__init__.py"
    stack_cube_module = (
        ROOT / "src" / "ioailab" / "tasks" / "stack_cube" / "__init__.py"
    )
    pick_to_shelf_module = (
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf" / "__init__.py"
    )
    pick_phase_module = (
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf_pick" / "__init__.py"
    )
    nav_phase_module = (
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf_nav" / "__init__.py"
    )
    place_phase_module = (
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf_place" / "__init__.py"
    )
    sort_to_shelf_module = (
        ROOT / "src" / "ioailab" / "tasks" / "sort_to_shelf" / "__init__.py"
    )
    task_module_paths = (
        reach_module,
        pick_cube_module,
        stack_cube_module,
        pick_to_shelf_module,
        pick_phase_module,
        nav_phase_module,
        place_phase_module,
        sort_to_shelf_module,
    )
    for path in task_module_paths:
        source = path.read_text(encoding="utf-8")
        assert "from dataclasses import dataclass" not in source, path
        assert ".motion_plan import" not in source, path

    # Motion-plan task modules declare entry points without importing plan code.
    for path in (
        reach_module,
        pick_cube_module,
        stack_cube_module,
        pick_phase_module,
        place_phase_module,
    ):
        assert "motion_plan_entry_point=" in path.read_text(encoding="utf-8"), path

    full_source = pick_to_shelf_module.read_text(encoding="utf-8")
    nav_source = nav_phase_module.read_text(encoding="utf-8")
    assert "task_flow_entry_point=" in full_source
    assert "motion_plan_entry_point=" not in full_source
    assert "nav_agent_entry_point=" in nav_source
    assert "motion_plan_entry_point=" not in nav_source
    assert not (
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf" / "subtasks"
    ).exists()


def test_public_manager_based_task_ids_use_isaaclab_entry_point() -> None:
    from ioailab.tasks import BUILTIN_TASKS

    for task in BUILTIN_TASKS:
        if task.task_id.endswith("-Mimic-v0"):
            continue
        assert task.entry_point == "isaaclab.envs:ManagerBasedRLEnv"


def test_gym_registration_matches_task_specs() -> None:
    gym = pytest.importorskip("gymnasium")

    import ioailab.tasks as tasks

    tasks.register_tasks()
    for task in tasks.BUILTIN_TASKS:
        spec = gym.spec(task.task_id)
        assert spec.entry_point == task.entry_point
        assert spec.kwargs == task.gym_kwargs()


def test_builtin_task_specs_cover_scoped_motion_planning_workflows() -> None:
    from ioailab.tasks import (
        BUILTIN_TASK_IDS,
        GALBOT_G1_BASE_NAV_TASKS,
        GALBOT_G1_PICK_TO_SHELF_TASKS,
        GALBOT_G1_SORT_TO_SHELF_TASKS,
        GALBOT_G1_SORT_TO_SHELF_NAV_TASKS,
        GALBOT_G1_PICK_CUBE_TASKS,
        GALBOT_G1_REACH_TASKS,
        GALBOT_G1_SORT_TO_SHELF_PICK_TASKS,
        GALBOT_G1_SORT_TO_SHELF_PLACE_TASKS,
        GALBOT_G1_STACK_CUBE_TASKS,
    )

    assert len(GALBOT_G1_REACH_TASKS) == 1
    assert len(GALBOT_G1_PICK_CUBE_TASKS) == 3
    assert len(GALBOT_G1_STACK_CUBE_TASKS) == 1
    assert len(GALBOT_G1_BASE_NAV_TASKS) == 1
    # The PickToShelf family has one coherent task plus standalone component task IDs.
    assert len(GALBOT_G1_PICK_TO_SHELF_TASKS) == 1
    assert len(GALBOT_G1_SORT_TO_SHELF_TASKS) == 1
    assert len(GALBOT_G1_SORT_TO_SHELF_PICK_TASKS) == 1
    assert len(GALBOT_G1_SORT_TO_SHELF_NAV_TASKS) == 1
    assert len(GALBOT_G1_SORT_TO_SHELF_PLACE_TASKS) == 1
    assert {task.task_id for task in GALBOT_G1_PICK_TO_SHELF_TASKS} == {
        "GalbotG1-PickToShelf-v0",
    }
    assert {
        "GalbotG1-PickToShelf-v0",
        "GalbotG1-PickToShelf-Pick-v0",
        "GalbotG1-PickToShelf-Nav-v0",
        "GalbotG1-PickToShelf-Place-v0",
    }.issubset(set(BUILTIN_TASK_IDS))
    assert {
        "GalbotG1-SortToShelf-v0",
        "GalbotG1-SortToShelf-Pick-v0",
        "GalbotG1-SortToShelf-Nav-v0",
        "GalbotG1-SortToShelf-Place-v0",
    }.issubset(set(BUILTIN_TASK_IDS))
    for task in (
        *GALBOT_G1_REACH_TASKS,
        GALBOT_G1_PICK_CUBE_TASKS[0],
        *GALBOT_G1_STACK_CUBE_TASKS,
    ):
        assert task.motion_plan_factory is not None
        assert ".tasks.g1." not in task.env_cfg_entry_point
        assert ".tasks.g1." not in task.motion_plan_factory.__module__
        # The plan bundles its own config.
        assert task.motion_plan_factory().config is not None

    def _plan_name(task: object) -> str:
        return task.motion_plan_factory.__name__

    def _cfg_name(task: object) -> str:
        return type(task.motion_plan_factory().config).__name__

    assert _cfg_name(GALBOT_G1_REACH_TASKS[0]) == "GalbotG1ReachMotionPlanningCfg"
    assert _plan_name(GALBOT_G1_REACH_TASKS[0]) == "ReachMotionPlan"
    assert (
        _cfg_name(GALBOT_G1_PICK_CUBE_TASKS[0]) == "GalbotG1PickCubeMotionPlanningCfg"
    )
    assert _plan_name(GALBOT_G1_PICK_CUBE_TASKS[0]) == "PickCubeMotionPlan"
    assert GALBOT_G1_PICK_CUBE_TASKS[0].requires_cameras is True
    assert GALBOT_G1_PICK_CUBE_TASKS[1].task_id == "GalbotG1-PickCube-Teleop-v0"
    assert GALBOT_G1_PICK_CUBE_TASKS[1].requires_cameras is True
    assert GALBOT_G1_PICK_CUBE_TASKS[2].task_id == "GalbotG1-PickCube-Mimic-v0"
    assert GALBOT_G1_PICK_CUBE_TASKS[2].entry_point.endswith(
        "datasets.mimic.env:ioailabMimicEnv"
    )
    assert GALBOT_G1_PICK_CUBE_TASKS[2].motion_plan_factory is None
    assert (
        _cfg_name(GALBOT_G1_STACK_CUBE_TASKS[0]) == "GalbotG1StackCubeMotionPlanningCfg"
    )
    assert _plan_name(GALBOT_G1_STACK_CUBE_TASKS[0]) == "StackCubeMotionPlan"
    # The motion plan is a G1-scoped expert recipe, co-located with the G1 agent cfgs.
    assert GALBOT_G1_STACK_CUBE_TASKS[0].motion_plan_factory.__module__.endswith(
        "config.g1.agent_cfg.motion_plan"
    )
    assert GALBOT_G1_BASE_NAV_TASKS[0].task_id == "GalbotG1-BaseNav-v0"
    assert GALBOT_G1_BASE_NAV_TASKS[0].motion_plan_factory is None
    assert GALBOT_G1_PICK_TO_SHELF_TASKS[0].task_id == "GalbotG1-PickToShelf-v0"
    assert GALBOT_G1_PICK_TO_SHELF_TASKS[0].motion_plan_factory is None
    assert GALBOT_G1_PICK_TO_SHELF_TASKS[0].requires_cameras is True
    assert GALBOT_G1_SORT_TO_SHELF_TASKS[0].task_id == "GalbotG1-SortToShelf-v0"
    assert GALBOT_G1_SORT_TO_SHELF_TASKS[0].motion_plan_factory is None
    assert GALBOT_G1_SORT_TO_SHELF_TASKS[0].requires_cameras is True
    assert (
        GALBOT_G1_SORT_TO_SHELF_PICK_TASKS[0].task_id == "GalbotG1-SortToShelf-Pick-v0"
    )
    assert GALBOT_G1_SORT_TO_SHELF_NAV_TASKS[0].task_id == "GalbotG1-SortToShelf-Nav-v0"
    assert (
        GALBOT_G1_SORT_TO_SHELF_PLACE_TASKS[0].task_id
        == "GalbotG1-SortToShelf-Place-v0"
    )
    # Pick/place motion plans are direct component task plans.
    from ioailab.tasks import motion_plan_for_task

    assert (
        type(motion_plan_for_task("GalbotG1-PickToShelf-Pick-v0")).__name__
        == "YamlMotionPlan"
    )
    assert (
        type(motion_plan_for_task("GalbotG1-PickToShelf-Place-v0")).__name__
        == "YamlMotionPlan"
    )
    assert (
        type(motion_plan_for_task("GalbotG1-SortToShelf-Pick-v0")).__name__
        == "YamlMotionPlan"
    )
    assert (
        type(motion_plan_for_task("GalbotG1-SortToShelf-Place-v0")).__name__
        == "YamlMotionPlan"
    )
    with pytest.raises(ValueError, match="does not define a motion plan"):
        motion_plan_for_task("GalbotG1-PickToShelf-Nav-v0")
    with pytest.raises(ValueError, match="does not define a motion plan"):
        motion_plan_for_task("GalbotG1-SortToShelf-Nav-v0")


def test_pick_cube_mimic_tcp_frame_is_declared_inline_and_default_off() -> None:
    from ioailab.tasks.pick_cube.config.g1.env_cfg import (
        GalbotG1PickCubeMimicEnvCfg,
    )

    assert GalbotG1PickCubeMimicEnvCfg().scene.tcp_frame.debug_vis is False

    source = Path("src/ioailab/tasks/pick_cube/config/g1/env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "_tcp_frame_" not in source
    assert "TCP_FRAME_VIS" not in source


def test_pick_to_shelf_policy_phase_planner_metadata_uses_task_local_yaml() -> None:
    from ioailab.agents import CuroboPlannerAgent
    from ioailab.agents.motion_plan.yaml_motion_plan import YamlMotionPlan
    from ioailab.tasks.pick_to_shelf_pick.motion_plan import (
        PickToShelfPickMotionPlanningCfg,
    )
    from ioailab.tasks.pick_to_shelf_place.motion_plan import (
        PickToShelfPlaceMotionPlanningCfg,
    )

    pick_agent = CuroboPlannerAgent.from_task("GalbotG1-PickToShelf-Pick-v0")
    place_agent = CuroboPlannerAgent.from_task("GalbotG1-PickToShelf-Place-v0")

    assert isinstance(pick_agent.motion_cfg, PickToShelfPickMotionPlanningCfg)
    assert isinstance(place_agent.motion_cfg, PickToShelfPlaceMotionPlanningCfg)
    assert isinstance(pick_agent.motion_plan, YamlMotionPlan)
    assert isinstance(place_agent.motion_plan, YamlMotionPlan)
    assert pick_agent.robot_asset_name == "robot"
    assert place_agent.robot_asset_name == "robot"

    for motion_plan_path in (
        Path("src/ioailab/tasks/pick_to_shelf_pick/motion_plan.py"),
        Path("src/ioailab/tasks/pick_to_shelf_place/motion_plan.py"),
    ):
        source = motion_plan_path.read_text(encoding="utf-8")
        assert "env.step" not in source
        assert "fallback" not in source.lower()
    assert not Path("src/ioailab/tasks/pick_to_shelf/motion_plan.py").exists()
    assert Path("src/ioailab/tasks/pick_to_shelf_pick/motion_plan.yaml").exists()
    assert Path("src/ioailab/tasks/pick_to_shelf_place/motion_plan.yaml").exists()
    assert not Path("src/ioailab/tasks/pick_to_shelf_nav/motion_plan.py").exists()


def test_sort_to_shelf_policy_phase_planner_metadata_uses_task_local_plans() -> None:
    from ioailab.agents import CuroboPlannerAgent
    from ioailab.agents.motion_plan.yaml_motion_plan import YamlMotionPlan
    from ioailab.tasks.sort_to_shelf_pick.motion_plan import (
        SortToShelfPickMotionPlanningCfg,
    )
    from ioailab.tasks.sort_to_shelf_place.motion_plan import (
        SortToShelfPlaceMotionPlanningCfg,
    )

    pick_agent = CuroboPlannerAgent.from_task("GalbotG1-SortToShelf-Pick-v0")
    place_agent = CuroboPlannerAgent.from_task("GalbotG1-SortToShelf-Place-v0")

    assert isinstance(pick_agent.motion_cfg, SortToShelfPickMotionPlanningCfg)
    assert isinstance(place_agent.motion_cfg, SortToShelfPlaceMotionPlanningCfg)
    assert isinstance(pick_agent.motion_plan, YamlMotionPlan)
    assert isinstance(place_agent.motion_plan, YamlMotionPlan)
    assert pick_agent.robot_asset_name == "robot"
    assert place_agent.robot_asset_name == "robot"
    assert not hasattr(pick_agent.motion_cfg, "subtask")
    assert not hasattr(place_agent.motion_cfg, "subtask")

    for motion_plan_path in (
        Path("src/ioailab/tasks/sort_to_shelf_pick/motion_plan.py"),
        Path("src/ioailab/tasks/sort_to_shelf_place/motion_plan.py"),
    ):
        source = motion_plan_path.read_text(encoding="utf-8")
        assert "env.step" not in source
        assert "fallback" not in source.lower()
    assert Path("src/ioailab/tasks/sort_to_shelf_pick/motion_plan.yaml").exists()
    assert Path("src/ioailab/tasks/sort_to_shelf_place/motion_plan.yaml").exists()


def test_pick_to_shelf_phase_policy_observations_do_not_expose_scene_truth() -> None:
    from ioailab.tasks.pick_to_shelf.config.g1.env_cfg import (
        GalbotG1PickToShelfEnvCfg,
    )
    from ioailab.tasks.pick_to_shelf_pick.config.g1.env_cfg import (
        GalbotG1PickToShelfPickEnvCfg,
    )
    from ioailab.tasks.pick_to_shelf_place.config.g1.env_cfg import (
        GalbotG1PickToShelfPlaceEnvCfg,
    )

    expected_terms = ("actions", "robot_joint_pos", "front_head_rgb")
    forbidden_terms = (
        "blue_block_pos",
        "blue_block_quat",
        "cube_pos",
        "cube_quat",
        "shelf_deck_pos",
        "shelf_deck_quat",
    )
    for cfg_cls in (
        GalbotG1PickToShelfPickEnvCfg,
        GalbotG1PickToShelfPlaceEnvCfg,
        GalbotG1PickToShelfEnvCfg,
    ):
        policy = cfg_cls().observations.policy
        for term_name in expected_terms:
            assert hasattr(policy, term_name)
        for term_name in forbidden_terms:
            assert not hasattr(policy, term_name)


def test_tasks_resolve_motion_plan_types() -> None:
    import ioailab.tasks as tasks

    entry = tasks.task_entry_for_task_id("GalbotG1-PickCube-v0")

    assert "motion_plan_for_task" in tasks.__all__
    assert entry.task_id == "GalbotG1-PickCube-v0"

    def _plan(task_id: str, **kwargs: object) -> object:
        return tasks.motion_plan_for_task(task_id, **kwargs)

    reach_plan = _plan("GalbotG1-Reach-v0")
    assert type(reach_plan).__name__ == "ReachMotionPlan"
    assert type(reach_plan.config).__name__ == "GalbotG1ReachMotionPlanningCfg"
    assert type(_plan("GalbotG1-PickCube-v0")).__name__ == "PickCubeMotionPlan"
    assert (
        type(_plan("GalbotG1-PickCube-v0").config).__name__
        == "GalbotG1PickCubeMotionPlanningCfg"
    )
    assert type(_plan("GalbotG1-StackCube-v0")).__name__ == "StackCubeMotionPlan"

    # A caller may override the bundled config.
    from ioailab.tasks.reach.config.g1.agent_cfg.motion_plan import (
        GalbotG1ReachMotionPlanningCfg,
    )

    override = GalbotG1ReachMotionPlanningCfg(robot_asset_name="other_robot")
    assert _plan("GalbotG1-Reach-v0", config=override).config is override

    assert (
        tasks.task_entry_for_task_id("GalbotG1-BaseNav-v0").env_cfg_entry_point
        == "ioailab.tasks.base_nav.config.g1.env_cfg:GalbotG1BaseNavEnvCfg"
    )
    assert (
        tasks.task_entry_for_task_id("GalbotG1-PickToShelf-v0").env_cfg_entry_point
        == "ioailab.tasks.pick_to_shelf.config.g1.env_cfg:GalbotG1PickToShelfEnvCfg"
    )
    assert (
        tasks.task_entry_for_task_id("GalbotG1-SortToShelf-v0").env_cfg_entry_point
        == "ioailab.tasks.sort_to_shelf.config.g1.env_cfg:GalbotG1SortToShelfEnvCfg"
    )
    # Pick/place plans are addressed by standalone component task IDs.
    pick_plan = _plan("GalbotG1-PickToShelf-Pick-v0")
    assert type(pick_plan).__name__ == "YamlMotionPlan"
    assert type(pick_plan.config).__name__ == "PickToShelfPickMotionPlanningCfg"
    place_plan = _plan("GalbotG1-PickToShelf-Place-v0")
    assert type(place_plan.config).__name__ == "PickToShelfPlaceMotionPlanningCfg"

    with pytest.raises(ValueError, match="does not define a motion plan"):
        _plan("GalbotG1-BaseNav-v0")
    sort_pick_plan = _plan("GalbotG1-SortToShelf-Pick-v0")
    assert type(sort_pick_plan).__name__ == "YamlMotionPlan"
    assert type(sort_pick_plan.config).__name__ == "SortToShelfPickMotionPlanningCfg"
    sort_place_plan = _plan("GalbotG1-SortToShelf-Place-v0")
    assert type(sort_place_plan).__name__ == "YamlMotionPlan"
    assert type(sort_place_plan.config).__name__ == "SortToShelfPlaceMotionPlanningCfg"

    with pytest.raises(ValueError, match="does not define a motion plan"):
        _plan("GalbotG1-PickToShelf-v0")
    with pytest.raises(ValueError, match="does not define a motion plan"):
        _plan("GalbotG1-PickToShelf-Nav-v0")
    with pytest.raises(ValueError, match="does not define a motion plan"):
        _plan("GalbotG1-SortToShelf-Nav-v0")
    with pytest.raises(ValueError, match="does not define a motion plan"):
        _plan("GalbotG1-SortToShelf-v0")
    with pytest.raises(ValueError, match="Unknown ioailab task ID"):
        tasks.task_entry_for_task_id("GalbotMissing-v0")


def test_motion_planning_script_bridge_is_removed() -> None:
    def non_cache_files(root: Path) -> list[str]:
        if not root.exists():
            return []
        return sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )

    # The motion-planning / imitation-learning Python script bridges are removed.
    # (A non-bridge docs builder such as scripts/build_versioned_docs.sh may
    # legitimately live under scripts/ and is allowed.)
    remaining_py_scripts = [
        path for path in non_cache_files(Path("scripts")) if path.endswith(".py")
    ]
    assert remaining_py_scripts == []
    assert not Path("scripts/motion_planning/run.py").exists()
    assert not Path("scripts/imitation_learning/record_curobo_demos.py").exists()
    assert not Path("scripts/imitation_learning/record_demos.py").exists()
    assert not Path("scripts/imitation_learning/replay_demos.py").exists()
    assert not Path("scripts/imitation_learning/teleop_se3_agent.py").exists()
    assert not Path("scripts/imitation_learning/robomimic_train.py").exists()
    assert not Path("scripts/imitation_learning/robomimic_play.py").exists()
    assert '"scripts/motion_planning/*.py"' not in Path("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"scripts/imitation_learning/*.py"' not in Path("pyproject.toml").read_text(
        encoding="utf-8"
    )


def test_task_common_package_is_not_a_helper_facade() -> None:
    import ioailab.tasks.common as common
    from ioailab.tasks.common.props import rigid_cuboid, static_cuboid

    assert not hasattr(common, "__all__")
    assert callable(rigid_cuboid)
    assert callable(static_cuboid)
    assert not Path("src/ioailab/tasks/common/uv_cuboid.py").exists()
    assert not hasattr(common, "workflows")
    assert not hasattr(common, "run_motion_planning_env")
    assert not Path("src/ioailab/tasks/common/lightweight_scene.py").exists()


def test_randomizers_package_keeps_task_policy_local() -> None:
    randomizers_root = ROOT / "src" / "ioailab" / "randomizers"
    assert not (
        ROOT / "src" / "ioailab" / "tasks" / "common" / "randomization.py"
    ).exists()

    from ioailab.randomizers import (
        DomeLightTextureRandomizer,
        ObjectPoseRandomizer,
        Randomizer,
        VisualMaterialRandomizer,
    )

    for randomizer in (
        ObjectPoseRandomizer,
        VisualMaterialRandomizer,
        DomeLightTextureRandomizer,
    ):
        assert issubclass(randomizer, Randomizer)
        assert callable(randomizer.apply)

    forbidden_policy_fragments = (
        "list_hdri_paths",
        "list_visual_material_paths",
        "PickCube",
        "StackCube",
        "blue_block",
        "cube_1",
        "cube_2",
        "cube_3",
    )
    imported_modules: set[str] = set()
    for path in sorted(randomizers_root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_policy_fragments:
            assert fragment not in source, f"{path.name} leaks task policy: {fragment}"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)

    assert "ioailab.utils.asset_utils" not in imported_modules
    assert not any(module.startswith("ioailab.tasks") for module in imported_modules)
    assert all(not module.startswith("ioailab.tasks.") for module in imported_modules)


def test_g1_task_env_cfgs_use_current_final_variant_classes() -> None:
    from ioailab.tasks.pick_cube.config.g1.env_cfg import (
        GalbotG1PickCubeEnvCfg,
        GalbotG1PickCubeBaseEnvCfg,
        GalbotG1PickCubeTeleopEnvCfg,
    )
    from ioailab.tasks.pick_to_shelf.config.g1.env_cfg import (
        GalbotG1PickToShelfEnvCfg,
    )
    from ioailab.tasks.stack_cube.config.g1.env_cfg import GalbotG1StackCubeEnvCfg

    assert issubclass(GalbotG1PickCubeBaseEnvCfg, object)
    assert issubclass(GalbotG1PickCubeEnvCfg, GalbotG1PickCubeBaseEnvCfg)
    assert issubclass(GalbotG1PickToShelfEnvCfg, object)
    assert issubclass(GalbotG1PickCubeTeleopEnvCfg, GalbotG1PickCubeBaseEnvCfg)
    teleop_cfg = GalbotG1PickCubeTeleopEnvCfg()
    assert hasattr(teleop_cfg.scene, "left_wrist_rgb_camera")
    assert hasattr(teleop_cfg.scene, "front_head_rgb_camera")
    assert hasattr(teleop_cfg.observations.policy, "left_wrist_rgb")
    assert hasattr(teleop_cfg.observations.policy, "front_head_rgb")
    assert (
        teleop_cfg.observations.policy.left_wrist_rgb.params["sensor_cfg"].name
        == "left_wrist_rgb_camera"
    )
    assert (
        teleop_cfg.observations.policy.front_head_rgb.params["sensor_cfg"].name
        == "front_head_rgb_camera"
    )
    assert teleop_cfg.observations.policy.left_wrist_rgb.params["normalize"] is False
    assert teleop_cfg.observations.policy.front_head_rgb.params["normalize"] is False
    normal_cfg = GalbotG1PickCubeEnvCfg()
    assert (
        normal_cfg.terminations.released_on_blue_block.func.__name__
        == "cube_released_on_blue_block"
    )
    assert normal_cfg.evaluation_success.func.__name__ == "cube_released_on_blue_block"
    assert not hasattr(normal_cfg.scene, "left_wrist_rgb_camera")
    assert hasattr(normal_cfg.scene, "front_head_rgb_camera")
    assert not hasattr(normal_cfg.observations.policy, "left_wrist_rgb")
    assert hasattr(normal_cfg.observations.policy, "front_head_rgb")
    assert issubclass(GalbotG1StackCubeEnvCfg, object)


def test_task_registry_import_does_not_load_env_cfg_modules_in_fresh_process() -> None:
    code = textwrap.dedent(
        """
        import json
        import sys

        import ioailab.tasks

        print(json.dumps({
            "registered_reach": "GalbotG1-Reach-v0" in __import__("gymnasium").registry,
            "reach_env_cfg": "ioailab.tasks.reach.config.g1.env_cfg" in sys.modules,
            "pick_env_cfg": "ioailab.tasks.pick_cube.config.g1.env_cfg" in sys.modules,
            "base_nav_env_cfg": "ioailab.tasks.base_nav.config.g1.env_cfg" in sys.modules,
            "pick_to_shelf_env_cfg": "ioailab.tasks.pick_to_shelf.config.g1.env_cfg" in sys.modules,
            "stack_cube_env_cfg": "ioailab.tasks.stack_cube.config.g1.env_cfg" in sys.modules,
            "stack_cube_motion_planning": "ioailab.tasks.stack_cube.joint_motion_planning" in sys.modules,
            "stack_env_cfg": "ioailab.tasks.stack_cube.config.g1.env_cfg" in sys.modules,
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=source_subprocess_env(),
        text=True,
    )
    data = json.loads(result.stdout.strip())

    assert data == {
        "registered_reach": False,
        "reach_env_cfg": False,
        "pick_env_cfg": False,
        "base_nav_env_cfg": False,
        "pick_to_shelf_env_cfg": False,
        "stack_cube_env_cfg": False,
        "stack_cube_motion_planning": False,
        "stack_env_cfg": False,
    }


def test_task_env_cfg_imports_do_not_load_pxr_before_simulation_app() -> None:
    code = textwrap.dedent(
        """
        import json
        import sys

        import ioailab.tasks.stack_cube.config.g1.env_cfg
        import ioailab.tasks.stack_cube.config.g1.env_cfg

        print(json.dumps({
            "pxr_modules": [name for name in sys.modules if name.startswith("pxr")],
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=source_subprocess_env(),
        text=True,
    )
    data = json.loads(result.stdout.strip())

    assert data == {"pxr_modules": []}


def test_reach_motion_plan_is_task_local_outside_agents() -> None:
    source_path = Path("src/ioailab/tasks/reach/config/g1/agent_cfg/motion_plan.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    import ioailab.tasks.reach.config.g1.agent_cfg.motion_plan as motion_plan

    assert not Path("src/ioailab/tasks/reach/agents").exists()
    assert Path("src/ioailab/tasks/reach/config/g1/agent_cfg/motion_plan.py").exists()
    assert hasattr(motion_plan, "ReachMotionPlan")
    assert "G1TaskMotionPlan" in source
    assert "MotionStep(" in source
    assert "WorldTarget(" in source
    assert "move_to_target" not in source
    assert 'if __name__ == "__ioailab_motion_plan__"' not in source
    forbidden = (
        "def main",
        "env.step",
        "import curobo",
        "from curobo",
        "make_g1_curobo_motion_plan_action_source",
    )
    for pattern in forbidden:
        assert pattern not in source
    assert len(_all_calls_named(tree, "MotionStep")) == 1


def test_pick_cube_motion_plan_is_task_local_outside_agents() -> None:
    assert not Path("src/ioailab/tasks/pick_cube/agents").exists()
    assert Path(
        "src/ioailab/tasks/pick_cube/config/g1/agent_cfg/motion_plan.py"
    ).exists()
    assert not Path("src/ioailab/tasks/pick_cube/targets.py").exists()

    code = textwrap.dedent(
        """
        import json
        import sys

        import ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan as motion_plan

        print(json.dumps({
            "has_plan": hasattr(motion_plan, "PickCubeMotionPlan"),
            "gripper_hold_steps": motion_plan.GRIPPER_CLOSE_HOLD_STEPS,
            "curobo_loaded": "ioailab.agents.motion_plan.contracts.g1_curobov2" in sys.modules,
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=source_subprocess_env(),
        text=True,
    )
    data = json.loads(result.stdout.strip())

    assert data == {
        "has_plan": True,
        "gripper_hold_steps": 25,
        "curobo_loaded": False,
    }


def test_pick_cube_motion_plan_file_is_typed_task_class() -> None:
    source_path = Path("src/ioailab/tasks/pick_cube/config/g1/agent_cfg/motion_plan.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    import ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan as motion_plan

    assert not Path("src/ioailab/tasks/pick_cube/config.py").exists()
    # The robot-agnostic world now lives in a task-level scene module.
    assert Path("src/ioailab/tasks/pick_cube/scene.py").exists()
    assert hasattr(motion_plan, "PickCubeMotionPlan")
    assert "G1TaskMotionPlan" in source
    assert "MotionStep(" in source
    assert "AssetRelativeTarget(" in source
    assert "WorldTarget(" in source
    assert "move_to_target" not in source
    assert 'if __name__ == "__ioailab_motion_plan__"' not in source

    forbidden = (
        "_as_torch",
        "_asset_root_pos",
        "_z_offset_like",
        "_place_approach_position",
        "ioailab.tasks.common.defaults_state",
        "asset_root_pos_w",
        "configclass",
        "PickCubePlannerAgent",
        "make_pick_cube_planner_agent",
        "MotionPlanAgent",
        "make_g1_curobo_motion_plan_action_source",
        "make_g1_curobo_motion_plan_agent",
        "_make_writer",
        "_make_action_layout",
        "_make_motion_targets",
        "_prepare_initial_action",
        "_validate_action_order",
        "PICK_CUBE_ACTION_TERM_ORDER",
        '"arm_action"',
        '"gripper_action"',
        "G1ActionTensorLayout",
        "make_g1_curobo_action_writer",
        "ioailab.agents.motion_plan.contracts.g1_curobov2.action_writer",
        "env.step",
        "import curobo",
        "from curobo",
        "make_g1_motion_target",
        "G1MotionTarget",
        "pick_cube_motion_points",
        "def pick_cube_motion_targets",
        "compute_pick_cube_motion_targets",
        "PickCubeMotionTargets",
    )
    for pattern in forbidden:
        assert pattern not in source

    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "main" not in function_names
    assert "build" in function_names
    assert len(_all_calls_named(tree, "MotionStep")) == 7


def test_pick_cube_motion_plan_normal_import_does_not_execute_commands() -> None:
    code = textwrap.dedent(
        """
        import json
        import ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan as motion_plan

        print(json.dumps({"has_plan": hasattr(motion_plan, "PickCubeMotionPlan")}))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=source_subprocess_env(),
        text=True,
    )
    assert json.loads(result.stdout.strip()) == {"has_plan": True}


def test_pick_cube_motion_plan_executor_records_direct_commands() -> None:
    from ioailab.agents.motion_plan.commands import execute_motion_plan
    from ioailab.agents.motion_plan.commands import g1_motion_command_context
    from ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan import (
        GalbotG1PickCubeMotionPlanningCfg,
    )
    import ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan as motion_plan

    cfg = GalbotG1PickCubeMotionPlanningCfg()
    env = _fake_pick_cube_env(
        cube_pos=torch.tensor([[1.0, 2.0, 0.10], [1.5, 2.5, 0.20]]),
        target_pos=torch.tensor([[3.0, 4.0, 0.20], [3.5, 4.5, 0.30]]),
    )

    with g1_motion_command_context(
        env=env,
        motion_cfg=cfg,
        available_joint_groups=("left_arm",),
        available_binary_groups=("left_gripper",),
    ) as context:
        execute_motion_plan(motion_plan.PickCubeMotionPlan(config=cfg), env=env)

    names = tuple(command.name for command in context.commands)

    assert names[:2] == ("approach_cube", "descend_to_cube")
    assert names[2 : 2 + 25] == ("close_left_gripper",) * 25
    assert names[-4:] == (
        "lift_cube",
        "move_above_target",
        "descend_to_target",
        "open_left_gripper",
    )
    assert all(
        set(command.tcp_targets_w).issubset({"left_arm"})
        for command in context.commands
    )
    assert all(
        set(command.gripper_open_by_group) == {"left_gripper"}
        for command in context.commands
    )
    assert [
        command.gripper_open_by_group["left_gripper"]
        for command in context.commands[:2]
    ] == [True, True]
    assert all(
        not command.gripper_open_by_group["left_gripper"]
        for command in context.commands[2:-1]
    )
    assert context.commands[-1].gripper_open_by_group["left_gripper"] is True


def test_pick_cube_motion_plan_computes_explicit_tcp_poses() -> None:
    from ioailab.agents.motion_plan.commands import execute_motion_plan
    from ioailab.agents.motion_plan.commands import g1_motion_command_context
    from ioailab.robots.g1.articulation import G1_TOP_DOWN_TCP_WXYZ
    from ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan import (
        GalbotG1PickCubeMotionPlanningCfg,
    )
    import ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan as motion_plan

    cfg = GalbotG1PickCubeMotionPlanningCfg()
    cube_pos = torch.tensor([[1.0, 2.0, 0.10]], dtype=torch.float32)
    target_pos = torch.tensor([[3.0, 4.0, 0.20]], dtype=torch.float32)
    env = _fake_pick_cube_env(cube_pos=cube_pos, target_pos=target_pos)

    with g1_motion_command_context(
        env=env,
        motion_cfg=cfg,
        available_joint_groups=("left_arm",),
        available_binary_groups=("left_gripper",),
    ) as context:
        execute_motion_plan(motion_plan.PickCubeMotionPlan(config=cfg), env=env)

    # Offsets are inlined in the plan as literal z-clearances above each asset.
    approach = cube_pos + torch.tensor([[0.0, 0.0, 0.205]], dtype=torch.float32)
    grasp = cube_pos + torch.tensor([[0.0, 0.0, 0.035]], dtype=torch.float32)
    lift = cube_pos + torch.tensor([[0.0, 0.0, 0.135]], dtype=torch.float32)
    place = target_pos + torch.tensor(
        [
            [
                0.0,
                0.0,
                0.07,
            ]
        ],
        dtype=torch.float32,
    )
    top_down_wxyz = torch.tensor([G1_TOP_DOWN_TCP_WXYZ], dtype=torch.float32)

    assert torch.allclose(context.commands[0].tcp_targets_w["left_arm"], approach)
    assert torch.allclose(context.commands[1].tcp_targets_w["left_arm"], grasp)
    assert torch.allclose(context.commands[27].tcp_targets_w["left_arm"], lift)
    assert torch.allclose(context.commands[-2].tcp_targets_w["left_arm"], place)
    assert torch.allclose(
        context.commands[0].tcp_wxyz_by_group["left_arm"], top_down_wxyz
    )
    assert torch.allclose(
        context.commands[-2].tcp_wxyz_by_group["left_arm"], top_down_wxyz
    )


def _all_calls_named(tree: ast.Module, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _fake_pick_cube_env(
    *, cube_pos: torch.Tensor, target_pos: torch.Tensor
) -> SimpleNamespace:
    def asset(pos: torch.Tensor) -> SimpleNamespace:
        quat_xyzw = (
            pos.new_tensor((0.0, 0.0, 0.0, 1.0)).reshape(1, 4).repeat(pos.shape[0], 1)
        )
        return SimpleNamespace(
            data=SimpleNamespace(root_pos_w=pos, root_quat_w=quat_xyzw)
        )

    return SimpleNamespace(
        unwrapped=SimpleNamespace(
            device="cpu",
            scene={
                "cube": asset(cube_pos),
                "blue_block": asset(target_pos),
            },
        )
    )


def _fake_stack_cube_env(
    *,
    cube_positions: dict[str, torch.Tensor],
) -> SimpleNamespace:
    def asset(pos: torch.Tensor) -> SimpleNamespace:
        quat_xyzw = (
            pos.new_tensor((0.0, 0.0, 0.0, 1.0)).reshape(1, 4).repeat(pos.shape[0], 1)
        )
        return SimpleNamespace(
            data=SimpleNamespace(root_pos_w=pos, root_quat_w=quat_xyzw)
        )

    scene = {name: asset(pos) for name, pos in cube_positions.items()}
    return SimpleNamespace(unwrapped=SimpleNamespace(device="cpu", scene=scene))


def test_stack_cube_scene_is_task_local_without_legacy_surfaces() -> None:
    # The task owns a robot-agnostic world scene; no top-level/legacy surfaces.
    assert Path("src/ioailab/tasks/stack_cube/scene.py").exists()
    assert not Path("src/ioailab/scenes").exists()
    assert not Path("src/ioailab/tasks/stack_cube/config.py").exists()
    assert not Path("src/ioailab/tasks/stack_cube/specs.py").exists()
    assert not Path("src/ioailab/tasks/stack_cube/runtime/motion_plan.py").exists()


def test_stack_cube_env_cfg_assembles_scene_robot_and_mdp() -> None:
    from isaaclab.envs import ManagerBasedRLEnvCfg

    from ioailab.randomizers import ObjectPoseRandomizer
    from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER
    from ioailab.tasks.stack_cube.config.g1 import env_cfg as stack_env_cfg
    from ioailab.tasks.stack_cube.config.g1.env_cfg import (
        GalbotG1StackCubeEnvCfg,
        G1StackCubeSceneCfg,
    )
    from ioailab.tasks.stack_cube.config.g1.mdp_cfg import StackCubeMdpCfg
    from ioailab.tasks.stack_cube.mdp import rewards as stack_rewards

    cfg = GalbotG1StackCubeEnvCfg()

    assert hasattr(stack_env_cfg, "G1StackCubeSceneCfg")
    assert hasattr(stack_env_cfg, "GalbotG1StackCubeEnvCfg")
    assert not hasattr(stack_env_cfg, "G1_STACK_CUBE_RIGHT_ARM_STOW_JOINT_POS")
    assert not hasattr(stack_env_cfg, "G1_STACK_CUBE_POSTURE_STOW_JOINT_POS")
    assert not hasattr(stack_env_cfg, "G1_STACK_CUBE_LEFT_ARM_INITIAL_ACTION_VALUES")
    assert isinstance(cfg, ManagerBasedRLEnvCfg)
    assert isinstance(cfg, StackCubeMdpCfg)
    assert isinstance(cfg.scene, G1StackCubeSceneCfg)
    assert cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    _assert_ground_top_surface_at_world_zero(cfg.scene.plane)
    assert (
        cfg.scene.ee_frame.target_frames[0].prim_path
        == "{ENV_REGEX_NS}/Robot/left_arm_link7"
    )
    assert cfg.actions.arm_action.joint_names == list(G1_LEFT_ARM_DOF_ORDER)
    assert cfg.actions.gripper_action.joint_names == ["left_gripper_joint"]
    assert cfg.events.randomize_cube_positions.func is ObjectPoseRandomizer.apply
    assert cfg.observations.policy.eef_pos.params["ee_frame_cfg"].name == "ee_frame"
    # The Mimic subtask_terms observation group was vestigial (superseded by the
    # get_subtask_term_signals override) and has been removed.
    assert not hasattr(cfg.observations, "subtask")
    assert cfg.rewards.stack_success.func is stack_rewards.stack_success_reward
    assert cfg.terminations.success is None


def test_stack_cube_motion_plan_file_is_typed_task_class() -> None:
    source_path = Path(
        "src/ioailab/tasks/stack_cube/config/g1/agent_cfg/motion_plan.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    import ioailab.tasks.stack_cube.config.g1.agent_cfg.motion_plan as motion_plan

    assert not Path("src/ioailab/tasks/stack_cube/agents").exists()
    assert Path(
        "src/ioailab/tasks/stack_cube/config/g1/agent_cfg/motion_plan.py"
    ).exists()
    assert hasattr(motion_plan, "StackCubeMotionPlan")
    assert "G1TaskMotionPlan" in source
    assert "MotionStep(" in source
    assert "WorldTarget(" in source
    assert "move_to_target" not in source
    assert 'if __name__ == "__ioailab_motion_plan__"' not in source

    forbidden = (
        "ioailab.tasks.stack_cube.config",
        "ioailab.tasks.stack_cube.runtime",
        "env.step",
        "import curobo",
        "from curobo",
        "make_g1_curobo_motion_plan_action_source",
        "def main",
    )
    for pattern in forbidden:
        assert pattern not in source

    assert len(_all_calls_named(tree, "MotionStep")) == 9


def test_stack_cube_motion_plan_executor_records_direct_commands() -> None:
    from ioailab.agents.motion_plan.commands import execute_motion_plan
    from ioailab.agents.motion_plan.commands import g1_motion_command_context
    from ioailab.tasks.stack_cube.config.g1.agent_cfg.motion_plan import (
        GalbotG1StackCubeMotionPlanningCfg,
    )
    from ioailab.tasks.stack_cube.config.g1.agent_cfg.motion_plan import (
        StackCubeMotionPlan,
    )

    cube_size = 0.05

    cfg = GalbotG1StackCubeMotionPlanningCfg()
    assert not hasattr(cfg, "cube_asset_names")
    cube_positions = {
        "cube_1": torch.tensor([[-0.30, 0.12, 0.125]], dtype=torch.float32),
        "cube_2": torch.tensor([[-0.30, 0.18, 0.125]], dtype=torch.float32),
        "cube_3": torch.tensor([[-0.30, 0.24, 0.125]], dtype=torch.float32),
    }
    env = _fake_stack_cube_env(cube_positions=cube_positions)

    with g1_motion_command_context(
        env=env,
        motion_cfg=cfg,
        available_joint_groups=("left_arm",),
        available_binary_groups=("left_gripper",),
    ) as context:
        execute_motion_plan(StackCubeMotionPlan(config=cfg), env=env)

    names = tuple(command.name for command in context.commands)
    commands_per_cube = 2 + cfg.gripper_close_hold_steps + 6

    assert cfg.stack_release_clearance_m == pytest.approx(0.060)
    assert len(context.commands) == commands_per_cube * len(cfg.stack_steps)
    assert names[:2] == ("approach_cube_2", "descend_to_cube_2")
    assert (
        names[2 : 2 + cfg.gripper_close_hold_steps]
        == ("close_gripper_on_cube_2",) * cfg.gripper_close_hold_steps
    )
    assert names[commands_per_cube : commands_per_cube + 2] == (
        "approach_cube_3",
        "descend_to_cube_3",
    )
    assert names[2 + cfg.gripper_close_hold_steps + 1] == "raise_cube_2_before_transfer"
    assert (
        names[commands_per_cube + 2 + cfg.gripper_close_hold_steps + 1]
        == "raise_cube_3_before_transfer"
    )
    assert all(
        set(command.tcp_targets_w).issubset({"left_arm"})
        for command in context.commands
    )
    assert all(
        set(command.gripper_open_by_group) == {"left_gripper"}
        for command in context.commands
    )
    assert [
        context.commands[index * commands_per_cube].gripper_open_by_group[
            "left_gripper"
        ]
        for index in range(len(cfg.stack_steps))
    ] == [True, True]

    for picked_cube_name, base_cube_name, stack_level in cfg.stack_steps:
        lift_command = next(
            command
            for command in context.commands
            if command.name == f"lift_{picked_cube_name}"
        )
        raise_command = next(
            command
            for command in context.commands
            if command.name == f"raise_{picked_cube_name}_before_transfer"
        )
        move_command = next(
            command
            for command in context.commands
            if command.name == f"move_{picked_cube_name}_above_{base_cube_name}"
        )
        stack_command = next(
            command
            for command in context.commands
            if command.name == f"descend_{picked_cube_name}_to_stack"
        )
        expected_z = (
            cube_positions[base_cube_name][:, 2]
            + cube_size * float(stack_level)
            + cfg.stack_release_clearance_m
        )
        assert torch.allclose(
            raise_command.tcp_targets_w["left_arm"][:, :2],
            lift_command.tcp_targets_w["left_arm"][:, :2],
        )
        assert torch.allclose(
            raise_command.tcp_targets_w["left_arm"][:, 2],
            move_command.tcp_targets_w["left_arm"][:, 2],
        )
        assert torch.allclose(stack_command.tcp_targets_w["left_arm"][:, 2], expected_z)


def test_stack_cube_mimic_cfg_uses_compact_stage_api() -> None:
    from ioailab.datasets.mimic import MimicCfg
    from ioailab.tasks.stack_cube.config.g1.env_cfg import (
        GalbotG1StackCubeMimicEnvCfg,
    )

    source = Path("src/ioailab/tasks/stack_cube/config/g1/env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "SubTaskConfig" not in source
    assert "ioailab.tasks.stack_cube.config.g1.mimic_env_cfg" not in source

    cfg = GalbotG1StackCubeMimicEnvCfg()
    assert isinstance(cfg.mimic, MimicCfg)
    assert cfg.datagen_config.name == "galbot_g1_stack_cube_mimic"
    assert cfg.mimic.object_names == ("cube_2", "cube_1", "cube_3")
    assert set(cfg.mimic.stage_signals) == {
        "grasp_cube_2",
        "place_cube_2_on_cube_1",
        "grasp_cube_3",
        "place_cube_3_on_cube_2",
    }
    assert set(cfg.subtask_configs) == {"left_tcp"}
    stages = cfg.subtask_configs["left_tcp"]
    assert [stage.object_ref for stage in stages] == [
        "cube_2",
        "cube_1",
        "cube_3",
        "cube_2",
        "cube_2",
    ]
    assert [stage.subtask_term_signal for stage in stages] == [
        "grasp_cube_2",
        "place_cube_2_on_cube_1",
        "grasp_cube_3",
        "place_cube_3_on_cube_2",
        None,
    ]
    assert [stage.subtask_term_offset_range for stage in stages] == [
        (5, 15),
        (0, 5),
        (5, 15),
        (0, 5),
        (0, 0),
    ]
    assert [stage.action_noise for stage in stages] == [0.005] * 5
    assert [stage.num_interpolation_steps for stage in stages] == [15] * 5


def test_stack_cube_task_imports_are_planner_lazy() -> None:
    script = r"""
import importlib
import sys

importlib.import_module('ioailab.tasks.stack_cube')
for module_name in sys.modules:
    if module_name.startswith('ioailab.tasks.' + 'g1'):
        raise SystemExit(f'eager G1 task import: {module_name}')
    if module_name.startswith('ioailab.agents.motion_plan.solvers.curobov2'):
        raise SystemExit(f'eager cuRobo planner import: {module_name}')
    if module_name.startswith('ioailab.agents.motion_plan.contracts.g1_curobov2'):
        raise SystemExit(f'eager G1 cuRobo planner import: {module_name}')
"""

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run([sys.executable, "-c", script], cwd=repo_root, env=env, check=True)


def test_task_config_import_does_not_load_concrete_tasks() -> None:
    script = r"""
import importlib
import sys

importlib.import_module('ioailab.tasks.common')
for module_name in sys.modules:
    if module_name.startswith('ioailab.tasks.' + 'g1'):
        raise SystemExit(f'eager G1 task import: {module_name}')
    if module_name.startswith('ioailab.task_kitchen'):
        raise SystemExit(f'eager task_kitchen import: {module_name}')
    if module_name.startswith('ioailab.legacy'):
        raise SystemExit(f'eager legacy import: {module_name}')
    if module_name.startswith('ioailab.agents.motion_plan.solvers.curobov2'):
        raise SystemExit(f'eager cuRobo planner import: {module_name}')
"""

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run([sys.executable, "-c", script], cwd=repo_root, env=env, check=True)


def test_pick_cube_completion_terms_live_in_terminations_module() -> None:
    """Keep PickCube completion checks in task-local terminations, not success/predicates modules."""

    task_root = ROOT / "src" / "ioailab" / "tasks" / "pick_cube"
    assert not (task_root / "mdp" / "success.py").exists()
    assert not (task_root / "mdp" / "predicates.py").exists()
    assert not (task_root / "mdp" / "signals.py").exists()
    env_cfg = (task_root / "config" / "g1" / "env_cfg.py").read_text(encoding="utf-8")
    assert "make_pick_cube_evaluation_success_term" in env_cfg
    assert "evaluation_success =" in env_cfg

    from ioailab.tasks.pick_cube.mdp import terminations

    assert callable(terminations.grasped_cube)
    assert callable(terminations.cube_released_on_blue_block)


def test_pick_cube_env_cfg_uses_event_randomization_and_motion_cfg() -> None:
    from ioailab.randomizers import (
        DomeLightTextureRandomizer,
        ObjectPoseRandomizer,
        VisualMaterialRandomizer,
    )
    from ioailab.tasks.pick_cube.config.g1.env_cfg import GalbotG1PickCubeEnvCfg
    from ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan import (
        GalbotG1PickCubeMotionPlanningCfg,
    )
    from ioailab.tasks.stack_cube.config.g1.env_cfg import (
        G1_STACK_CUBE_LEFT_ARM_INITIAL_JOINT_POS,
        G1StackCubeSceneCfg,
    )

    expected_light_prim_path = "/World/ioailabLight"
    cfg = GalbotG1PickCubeEnvCfg()
    motion_cfg = GalbotG1PickCubeMotionPlanningCfg()

    assert (
        cfg.events.randomize_pick_and_place_positions.func is ObjectPoseRandomizer.apply
    )
    assert "asset_pose_ranges" in cfg.events.randomize_pick_and_place_positions.params
    assert cfg.events.randomize_ground_material.func is VisualMaterialRandomizer.apply
    assert cfg.events.randomize_table_material.func is VisualMaterialRandomizer.apply
    assert cfg.events.randomize_hdri_texture.func is DomeLightTextureRandomizer.apply
    assert cfg.events.randomize_ground_material.params["asset_cfg"].name == "plane"
    assert cfg.events.randomize_table_material.params["asset_cfg"].name == "table"
    assert isinstance(
        cfg.events.randomize_ground_material.params["material_paths"], tuple
    )
    assert isinstance(
        cfg.events.randomize_table_material.params["material_paths"], tuple
    )
    assert (
        cfg.events.randomize_hdri_texture.params["light_prim_path"]
        == expected_light_prim_path
    )
    assert isinstance(cfg.events.randomize_hdri_texture.params["texture_paths"], tuple)
    assert cfg.actions.leg_action is None
    assert cfg.scene.robot.init_state.joint_pos["left_arm_joint4"] == pytest.approx(
        -1.53588974175501
    )
    assert motion_cfg.planner == "curobov2"
    assert not hasattr(motion_cfg, "agent")
    assert not hasattr(motion_cfg, "use_whole_body")
    assert not hasattr(motion_cfg, "place_approach_lateral_y_offset")
    stack_scene = G1StackCubeSceneCfg(num_envs=1)
    assert (
        cfg.scene.table.init_state.pos[0]
        == stack_scene.table.init_state.pos[0]
        == -0.30
    )
    assert G1_STACK_CUBE_LEFT_ARM_INITIAL_JOINT_POS["left_arm_joint4"] == pytest.approx(
        -1.53588974175501
    )
    assert cfg.scene.table.init_state.pos == pytest.approx((-0.30, 0.0, 0.075))
    assert cfg.scene.blue_block.init_state.pos == pytest.approx((-0.30, -0.08, 0.11))
    assert cfg.scene.light.prim_path == expected_light_prim_path
    assert cfg.scene.light.spawn.intensity == pytest.approx(4000.0)
    assert cfg.scene.replicate_physics is False
    assert cfg.scene.plane.prim_path == "{ENV_REGEX_NS}/GroundPlane"
    assert cfg.scene.plane.spawn.size == pytest.approx((7.0, 7.0, 0.02))
    assert cfg.scene.plane.spawn.__class__.__name__ == "MeshCuboidCfg"
    assert cfg.scene.table.spawn.__class__.__name__ == "MeshCuboidCfg"
    assert cfg.scene.plane.spawn.func.__name__ == "spawn_mesh_cuboid"
    assert cfg.scene.table.spawn.func.__name__ == "spawn_mesh_cuboid"
    _assert_ground_top_surface_at_world_zero(cfg.scene.plane)
    assert cfg.scene.blue_block.init_state.pos[2] == pytest.approx(0.11)
    assert cfg.scene.blue_block.spawn.rigid_props.kinematic_enabled is True
    assert cfg.scene.blue_block.spawn.rigid_props.disable_gravity is True
    assert [
        asset_cfg.name
        for asset_cfg in cfg.events.randomize_pick_and_place_positions.params[
            "asset_cfgs"
        ]
    ] == ["cube"]
    assert set(
        cfg.events.randomize_pick_and_place_positions.params["asset_pose_ranges"]
    ) == {"cube"}
    assert cfg.events.randomize_pick_and_place_positions.params["asset_pose_ranges"][
        "cube"
    ] == {
        "x": (-0.38, -0.22),
        "y": (0.12, 0.26),
        "z": (0.125, 0.125),
        "yaw": (0.0, 0.0),
    }


def test_pick_cube_agent_resolves_world_target_into_base_frame() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2 import TargetPose
    from ioailab.agents.motion_plan.solvers.curobov2.utils.pose import (
        resolve_target_pose_xyz_wxyz,
    )

    target_base_pose = resolve_target_pose_xyz_wxyz(
        TargetPose("left_arm", [2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0], frame="world"),
        num_envs=1,
        base_pose_by_env=[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
    )

    assert torch.allclose(
        torch.as_tensor(target_base_pose[:, :3]),
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
    )


def test_base_nav_env_cfg_owns_mobile_base_task_surface() -> None:
    from ioailab.robots.g1.actions import G1_BASE_WHEEL_DOF_ORDER
    from ioailab.robots.g1.actions import DEFAULT_BASE_WHEEL_RADIUS
    from ioailab.robots.g1.articulation import mobile_base_root_pose_from_base_pose
    from ioailab.tasks.base_nav.config.g1.env_cfg import GalbotG1BaseNavEnvCfg

    cfg = GalbotG1BaseNavEnvCfg()
    max_wheel_velocity = cfg.max_command_speed / DEFAULT_BASE_WHEEL_RADIUS
    expected_root_position, expected_root_orientation = (
        mobile_base_root_pose_from_base_pose(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    assert cfg.goal_position == (2.0, 0.0, 0.0)
    assert cfg.scene.robot.init_state.pos == pytest.approx(expected_root_position)
    assert cfg.scene.robot.init_state.rot == pytest.approx(expected_root_orientation)
    _assert_ground_top_surface_at_world_zero(cfg.scene.plane)
    assert cfg.scene.robot.spawn.func.__name__ == "spawn_galbot_g1_usd_mobile_base"
    assert cfg.actions.base_action.asset_name == "robot"
    assert tuple(cfg.actions.base_action.joint_names) == G1_BASE_WHEEL_DOF_ORDER
    assert cfg.actions.base_action.clip == {
        joint_name: (-max_wheel_velocity, max_wheel_velocity)
        for joint_name in G1_BASE_WHEEL_DOF_ORDER
    }
    assert cfg.observations.policy.concatenate_terms is True
    assert cfg.terminations.goal_reached.func.__name__ == "goal_reached"
    assert cfg.rewards.distance_to_goal.func.__name__ == "negative_distance_reward"


def test_pick_to_shelf_env_cfg_declares_shelf_scene_and_action_contracts() -> None:
    from ioailab.robots.g1.actions import (
        G1_BASE_WHEEL_DOF_ORDER,
        G1_LEFT_ARM_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
        G1_RIGHT_ARM_DOF_ORDER,
    )
    from ioailab.tasks.common.defaults import DefaultSceneCfg
    from ioailab.tasks.pick_to_shelf.config.g1.env_cfg import (
        G1PickToShelfSceneCfg,
        GalbotG1PickToShelfEnvCfg,
    )
    from ioailab.tasks.pick_to_shelf_place.mdp.terminations import (
        SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT,
    )
    from ioailab.tasks.pick_to_shelf.scene import (
        CUBE_POSITION,
        CUBE_SIZE,
        SHELF_BACK_BAFFLE_POSITION,
        SHELF_BACK_BAFFLE_SIZE,
        SHELF_BAFFLE_BOTTOM_Z,
        SHELF_BAFFLE_THICKNESS,
        SHELF_DECK_POSITION,
        SHELF_DECK_SIZE,
        SHELF_LEFT_BAFFLE_POSITION,
        SHELF_RIGHT_BAFFLE_POSITION,
        SHELF_SIDE_BAFFLE_SIZE,
        TABLE_POSITION,
    )
    from ioailab.tasks.pick_to_shelf_nav.config.g1.env_cfg import (
        GalbotG1PickToShelfNavEnvCfg,
    )
    from ioailab.tasks.pick_to_shelf_pick.config.g1.env_cfg import (
        GalbotG1PickToShelfPickEnvCfg,
    )
    from ioailab.tasks.pick_to_shelf_place.config.g1.env_cfg import (
        GalbotG1PickToShelfPlaceEnvCfg,
    )

    cfg = GalbotG1PickToShelfEnvCfg()
    pick_cfg = GalbotG1PickToShelfPickEnvCfg()
    nav_cfg = GalbotG1PickToShelfNavEnvCfg()
    place_cfg = GalbotG1PickToShelfPlaceEnvCfg()
    policy = cfg.observations.policy

    assert issubclass(G1PickToShelfSceneCfg, DefaultSceneCfg)
    assert cfg.scene.env_spacing == pytest.approx(4.0)
    assert cfg.scene.plane.spawn.size == pytest.approx((7.0, 7.0, 0.02))
    assert cfg.scene.plane.spawn.func.__name__ == "spawn_mesh_cuboid"
    assert cfg.scene.plane.spawn.visual_material.diffuse_color == pytest.approx(
        (0.35, 0.35, 0.32)
    )
    assert cfg.scene.cube.init_state.pos == pytest.approx(CUBE_POSITION)
    assert cfg.scene.cube.spawn.size == pytest.approx(CUBE_SIZE)
    assert cfg.scene.table.init_state.pos == pytest.approx(TABLE_POSITION)
    assert cfg.scene.shelf_deck.init_state.pos == pytest.approx(SHELF_DECK_POSITION)
    assert cfg.scene.shelf_deck.spawn.size == pytest.approx(SHELF_DECK_SIZE)
    assert cfg.scene.shelf_back_baffle.init_state.pos == pytest.approx(
        SHELF_BACK_BAFFLE_POSITION
    )
    assert cfg.scene.shelf_back_baffle.spawn.size == pytest.approx(
        SHELF_BACK_BAFFLE_SIZE
    )
    assert cfg.scene.shelf_left_baffle.init_state.pos == pytest.approx(
        SHELF_LEFT_BAFFLE_POSITION
    )
    assert cfg.scene.shelf_right_baffle.init_state.pos == pytest.approx(
        SHELF_RIGHT_BAFFLE_POSITION
    )
    assert cfg.scene.shelf_left_baffle.spawn.size == pytest.approx(
        SHELF_SIDE_BAFFLE_SIZE
    )
    assert cfg.scene.shelf_right_baffle.spawn.size == pytest.approx(
        SHELF_SIDE_BAFFLE_SIZE
    )
    assert SHELF_BAFFLE_THICKNESS == pytest.approx(0.06)
    for baffle_cfg in (
        cfg.scene.shelf_back_baffle,
        cfg.scene.shelf_left_baffle,
        cfg.scene.shelf_right_baffle,
    ):
        assert baffle_cfg.init_state.pos[2] - baffle_cfg.spawn.size[
            2
        ] / 2.0 == pytest.approx(SHELF_BAFFLE_BOTTOM_Z)
    assert cfg.scene.shelf_deck.init_state.pos[0] < cfg.scene.table.init_state.pos[0]
    assert cfg.scene.shelf_deck.init_state.pos[1] < cfg.scene.table.init_state.pos[1]
    assert cfg.scene.shelf_deck.spawn.rigid_props.kinematic_enabled is True
    assert cfg.scene.shelf_deck.spawn.rigid_props.disable_gravity is True
    assert cfg.scene.front_head_rgb_camera.width == 298
    assert cfg.scene.front_head_rgb_camera.height == 224

    expected_right_arm_joint_pos = (-1.91, 1.46, 0.57, 2.10, 0.0, -0.71, -0.03)
    assert tuple(
        cfg.scene.robot.init_state.joint_pos[joint_name]
        for joint_name in G1_RIGHT_ARM_DOF_ORDER
    ) == pytest.approx(expected_right_arm_joint_pos)

    expected_left_arm_joint_pos = (
        1.910009444500404,
        -1.460010959112611,
        -0.4741512242415168,
        -2.467893642457805,
        -0.0016785070526536992,
        -0.1221698763763522,
        -0.09424494344765931,
    )
    assert tuple(
        cfg.scene.robot.init_state.joint_pos[joint_name]
        for joint_name in G1_LEFT_ARM_DOF_ORDER
    ) == pytest.approx(expected_left_arm_joint_pos)
    assert cfg.scene.robot.init_state.joint_pos["head_joint2"] == pytest.approx(0.25)

    assert tuple(pick_cfg.actions.arm_action.joint_names) == G1_LEFT_ARM_DOF_ORDER
    assert (
        tuple(pick_cfg.actions.gripper_action.joint_names) == G1_LEFT_GRIPPER_DOF_ORDER
    )
    assert pick_cfg.actions.leg_action is None
    assert not hasattr(pick_cfg.actions, "base_action")
    assert tuple(place_cfg.actions.arm_action.joint_names) == G1_LEFT_ARM_DOF_ORDER
    assert (
        tuple(place_cfg.actions.gripper_action.joint_names) == G1_LEFT_GRIPPER_DOF_ORDER
    )
    assert place_cfg.actions.leg_action is None
    assert not hasattr(place_cfg.actions, "base_action")
    assert tuple(nav_cfg.actions.base_action.joint_names) == G1_BASE_WHEEL_DOF_ORDER
    assert not hasattr(nav_cfg.actions, "arm_action")
    assert not hasattr(nav_cfg.actions, "gripper_action")
    assert tuple(cfg.actions.base_action.joint_names) == G1_BASE_WHEEL_DOF_ORDER
    assert tuple(cfg.actions.arm_action.joint_names) == G1_LEFT_ARM_DOF_ORDER
    assert tuple(cfg.actions.gripper_action.joint_names) == G1_LEFT_GRIPPER_DOF_ORDER

    expected_shelf_nav_xy = (
        SHELF_DECK_POSITION[0],
        SHELF_DECK_POSITION[1] + SHELF_DECK_SIZE[1] / 2.0 + 0.65,
    )
    assert cfg.goal_position[:2] == pytest.approx(expected_shelf_nav_xy)
    assert nav_cfg.goal_position[:2] == pytest.approx(expected_shelf_nav_xy)
    assert cfg.goal_yaw == pytest.approx(-1.5707963267948966)
    assert nav_cfg.goal_yaw == pytest.approx(cfg.goal_yaw)
    assert cfg.success_radius == pytest.approx(0.02)
    assert nav_cfg.success_radius == pytest.approx(0.02)

    assert pick_cfg.scene.robot.init_state.pos == place_cfg.scene.robot.init_state.pos
    assert pick_cfg.scene.robot.init_state.rot == place_cfg.scene.robot.init_state.rot
    assert not hasattr(place_cfg.events, "place_cube_in_left_gripper")
    assert not hasattr(nav_cfg.events, "randomize_pick_and_place_positions")
    assert not hasattr(place_cfg.events, "randomize_pick_and_place_positions")
    assert nav_cfg.events.reset_all.func.__name__ == "_apply_scenario_event"
    assert place_cfg.events.reset_all.func.__name__ == "_apply_scenario_event"
    assert (
        nav_cfg.events.reset_all.params["scenario"].metadata["task_id"]
        == "GalbotG1-PickToShelf-Nav-v0"
    )
    assert (
        place_cfg.events.reset_all.params["scenario"].metadata["task_id"]
        == "GalbotG1-PickToShelf-Place-v0"
    )

    assert (
        pick_cfg.evaluation_success.func.__name__ == "cube_lifted_and_left_arm_at_carry"
    )
    assert cfg.evaluation_success.func.__name__ == "final_phase_success"
    assert cfg.terminations.cube_on_shelf.func.__name__ == ("final_phase_success")
    assert nav_cfg.evaluation_success.func.__name__ == "goal_reached"
    assert place_cfg.evaluation_success.func.__name__ == "cube_placed_on_shelf"

    assert pick_cfg.events.randomize_pick_and_place_positions.params[
        "asset_pose_ranges"
    ]["cube"] == {
        "x": (-0.58, -0.42),
        "y": (0.10, 0.30),
        "z": (CUBE_POSITION[2], CUBE_POSITION[2]),
        "yaw": (0.0, 0.0),
    }
    assert place_cfg.evaluation_success.params[
        "upright_z_axis_min_dot"
    ] == pytest.approx(SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT)

    assert hasattr(policy, "actions")
    assert hasattr(policy, "robot_joint_pos")
    assert hasattr(policy, "front_head_rgb")
    assert policy.front_head_rgb.params["sensor_cfg"].name == "front_head_rgb_camera"
    assert not hasattr(policy, "cube_pos")
    assert not hasattr(policy, "cube_quat")
    assert not hasattr(policy, "shelf_deck_pos")
    assert policy.concatenate_terms is False


def test_pick_to_shelf_shelf_success_rejects_tipped_cube() -> None:
    from ioailab.tasks.pick_to_shelf_place.mdp.terminations import (
        SHELF_TOP_TO_CUBE_CENTER,
        cube_placed_on_shelf,
    )
    from ioailab.tasks.pick_to_shelf.scene import SHELF_DECK_POSITION

    cube_center = torch.tensor(
        [
            [
                SHELF_DECK_POSITION[0],
                SHELF_DECK_POSITION[1],
                SHELF_DECK_POSITION[2] + SHELF_TOP_TO_CUBE_CENTER,
            ]
        ],
        dtype=torch.float32,
    )
    shelf_center = torch.tensor([SHELF_DECK_POSITION], dtype=torch.float32)

    def asset(
        pos: torch.Tensor, quat_xyzw: tuple[float, float, float, float]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=pos,
                root_quat_w=pos.new_tensor(quat_xyzw).reshape(1, 4),
            )
        )

    class Robot:
        joint_names = ("left_gripper_joint",)
        data = SimpleNamespace(joint_pos=torch.zeros((1, 1), dtype=torch.float32))

        def find_joints(self, joint_names):
            return [0], list(joint_names)

    def environment(quat_xyzw: tuple[float, float, float, float]):
        env = SimpleNamespace(
            device="cpu",
            cfg=SimpleNamespace(
                gripper_joint_names=("left_gripper_joint",),
                gripper_open_val=0.0,
            ),
            scene={
                "robot": Robot(),
                "cube": asset(cube_center, quat_xyzw),
                "shelf_deck": asset(shelf_center, (0.0, 0.0, 0.0, 1.0)),
            },
        )
        env.unwrapped = env
        return env

    upright_env = environment((0.0, 0.0, 0.0, 1.0))
    tipped_env = environment((0.70710678, 0.0, 0.0, 0.70710678))

    assert cube_placed_on_shelf(upright_env, min_success_steps=1).tolist() == [True]
    assert cube_placed_on_shelf(tipped_env, min_success_steps=1).tolist() == [False]


def test_base_nav_terms_compute_local_goal_progress() -> None:
    from ioailab.tasks.base_nav.mdp import (
        distance_to_goal,
        goal_position_xy,
        goal_reached,
        vector_to_goal_xy,
    )

    robot = SimpleNamespace(
        body_names=("base_footprint",),
        data=SimpleNamespace(
            body_pos_w=torch.tensor([[[0.0, 0.0, 0.0]], [[2.1, 0.0, 0.0]]])
        ),
    )

    class FakeScene:
        env_origins = torch.zeros((2, 3))

        def __getitem__(self, name: str):
            assert name == "robot"
            return robot

    scene = FakeScene()
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            goal_position=(2.0, 0.0, 0.0),
            success_radius=0.15,
            base_body_name="base_footprint",
        ),
        device="cpu",
        scene=scene,
    )

    assert torch.allclose(goal_position_xy(env), torch.tensor([[2.0, 0.0], [2.0, 0.0]]))
    assert torch.allclose(
        vector_to_goal_xy(env), torch.tensor([[2.0, 0.0], [-0.1, 0.0]])
    )
    assert torch.allclose(distance_to_goal(env), torch.tensor([2.0, 0.1]))
    assert goal_reached(env).tolist() == [False, True]


@pytest.mark.skip(
    reason="Deferred RL/IL or non-scoped stack-cube coverage in this motion-planning milestone."
)
def test_state_stack_cube_exposes_sb3_sac_cfg() -> None:
    from pathlib import Path

    from ioailab.tasks.stack_cube import GALBOT_G1_STACK_CUBE_TASK

    assert GALBOT_G1_STACK_CUBE_TASK.task_id == "GalbotG1-StackCube-v0"
    assert not Path("src/ioailab/tasks/stack_cube/agents").exists()
    assert Path("src/ioailab/tasks/stack_cube/config/g1/agent_cfg").is_dir()


def test_builtin_registry_matches_exact_migrated_scope() -> None:
    import ioailab.tasks as tasks

    assert set(tasks.BUILTIN_TASK_IDS) == set(SUPPORTED_TASK_IDS)
    for task_id in SUPPORTED_TASK_IDS:
        entry = tasks.task_entry_for_task_id(task_id)
        assert entry.task_id == task_id
        assert "ioailab.tasks.g1" not in entry.entry_point
        assert "ioailab.tasks.g1" not in entry.env_cfg_entry_point
        for value in entry.gym_kwargs().values():
            assert "ioailab.tasks.g1" not in str(value)
        assert "ioailab.tasks.g1" not in str(entry.motion_plan_entry_point)

    for task_id in DEFERRED_TASK_IDS:
        with pytest.raises(ValueError, match="Unknown ioailab task ID"):
            tasks.task_entry_for_task_id(task_id)


def test_register_tasks_exposes_exact_scope_without_deferred_ids() -> None:
    gym = pytest.importorskip("gymnasium")

    import ioailab.tasks as tasks

    tasks.register_tasks()
    for task_id in SUPPORTED_TASK_IDS:
        spec = gym.spec(task_id)
        assert spec.kwargs is not None
        assert "ioailab.tasks.g1" not in str(spec.entry_point)
        assert "ioailab.tasks.g1" not in str(spec.kwargs)
    for task_id in DEFERRED_TASK_IDS:
        with pytest.raises(gym.error.NameNotFound):
            gym.spec(task_id)


def test_pick_cube_mimic_cfg_uses_compact_stage_api() -> None:
    import ioailab.datasets.mimic as mimic_api
    from ioailab.datasets.mimic import MimicCfg
    from ioailab.tasks.pick_cube.config.g1.env_cfg import (
        GalbotG1PickCubeMimicEnvCfg,
    )

    assert mimic_api.__all__ == ["MimicCfg"]
    source = Path("src/ioailab/tasks/pick_cube/config/g1/env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "PICK_CUBE_MIMIC" not in source
    assert "MimicPhaseCfg" not in source

    cfg = GalbotG1PickCubeMimicEnvCfg()
    assert isinstance(cfg.mimic, MimicCfg)
    assert "grasp_cube" in cfg.mimic.stages
    assert cfg.mimic.object_names == ("cube",)
    assert set(cfg.mimic.stage_signals) == {"grasp_cube"}
    assert cfg.datagen_config.name == "galbot_g1_pick_cube_mimic"
    assert cfg.mimic.converter is not None
    assert set(cfg.subtask_configs) == {"left_tcp"}
    phases = cfg.subtask_configs["left_tcp"]
    assert [phase.object_ref for phase in phases] == ["cube", "cube"]
    assert [phase.subtask_term_signal for phase in phases] == ["grasp_cube", None]
    assert [phase.subtask_term_offset_range for phase in phases] == [(5, 15), (0, 0)]
    assert [phase.action_noise for phase in phases] == [0.005, 0.005]
    assert [phase.num_interpolation_steps for phase in phases] == [15, 15]
