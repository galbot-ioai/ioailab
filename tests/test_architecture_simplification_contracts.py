from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

_ALLOWED_ALL_EXPORT_FILES = {
    "src/ioailab/__init__.py",
    "src/ioailab/agents/__init__.py",
    "src/ioailab/agents/flow/__init__.py",
    "src/ioailab/agents/flow/sequence.py",
    "src/ioailab/agents/flow/task_flow.py",
    "src/ioailab/agents/nav/__init__.py",
    "src/ioailab/agents/motion_plan/__init__.py",
    "src/ioailab/agents/motion_plan/action_source.py",
    "src/ioailab/agents/motion_plan/commands.py",
    "src/ioailab/agents/motion_plan/motion_plan.py",
    "src/ioailab/agents/motion_plan/targets.py",
    "src/ioailab/agents/motion_plan/yaml_motion_plan.py",
    "src/ioailab/agents/motion_plan/contracts/g1.py",
    "src/ioailab/agents/motion_plan/contracts/base.py",
    "src/ioailab/agents/motion_plan/solvers/types.py",
    "src/ioailab/agents/motion_plan/contracts/__init__.py",
    "src/ioailab/agents/motion_plan/solvers/__init__.py",
    "src/ioailab/agents/policy/__init__.py",
    "src/ioailab/agents/teleop/__init__.py",
    "src/ioailab/datasets/__init__.py",
    "src/ioailab/datasets/mimic/__init__.py",
    "src/ioailab/datasets/mimic/config.py",
    "src/ioailab/datasets/mimic/env.py",
    "src/ioailab/datasets/refs.py",
    "src/ioailab/envs/__init__.py",
    "src/ioailab/randomizers/__init__.py",
    "src/ioailab/randomizers/base.py",
    "src/ioailab/randomizers/camera.py",
    "src/ioailab/randomizers/lighting.py",
    "src/ioailab/randomizers/material.py",
    "src/ioailab/randomizers/pose.py",
    "src/ioailab/robots/__init__.py",
    "src/ioailab/robots/common/__init__.py",
    "src/ioailab/robots/common/actions/__init__.py",
    "src/ioailab/agents/motion_plan/solvers/curobov2/__init__.py",
    "src/ioailab/agents/motion_plan/solvers/curobov2/utils/__init__.py",
    "src/ioailab/robots/common/sensors/__init__.py",
    "src/ioailab/robots/g1/__init__.py",
    "src/ioailab/robots/g1/actions/__init__.py",
    "src/ioailab/robots/g1/converters.py",
    "src/ioailab/robots/g1/sensors/__init__.py",
    "src/ioailab/tasks/__init__.py",
    "src/ioailab/tasks/common/composition.py",
    "src/ioailab/tasks/common/mdp.py",
    "src/ioailab/tasks/common/props.py",
    "src/ioailab/tasks/base_nav/__init__.py",
    "src/ioailab/tasks/base_nav/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/base_nav/mdp/__init__.py",
    "src/ioailab/tasks/pick_cube/__init__.py",
    "src/ioailab/tasks/pick_cube/scene.py",
    "src/ioailab/tasks/stack_cube/scene.py",
    "src/ioailab/tasks/pick_to_shelf/scene.py",
    "src/ioailab/tasks/pick_cube/config/g1/env_cfg.py",
    "src/ioailab/tasks/pick_cube/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/pick_cube/mdp/__init__.py",
    "src/ioailab/tasks/pick_cube/mdp/terminations.py",
    "src/ioailab/tasks/pick_to_shelf/__init__.py",
    "src/ioailab/tasks/pick_to_shelf/config/g1/env_cfg.py",
    "src/ioailab/tasks/pick_to_shelf/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/pick_to_shelf_pick/__init__.py",
    "src/ioailab/tasks/pick_to_shelf_pick/config/g1/env_cfg.py",
    "src/ioailab/tasks/pick_to_shelf_pick/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/pick_to_shelf_pick/mdp/__init__.py",
    "src/ioailab/tasks/pick_to_shelf_pick/mdp/events.py",
    "src/ioailab/tasks/pick_to_shelf_pick/mdp/observations.py",
    "src/ioailab/tasks/pick_to_shelf_pick/mdp/rewards.py",
    "src/ioailab/tasks/pick_to_shelf_pick/mdp/terminations.py",
    "src/ioailab/tasks/pick_to_shelf_pick/motion_plan.py",
    "src/ioailab/tasks/pick_to_shelf_nav/__init__.py",
    "src/ioailab/tasks/pick_to_shelf_nav/agent.py",
    "src/ioailab/tasks/pick_to_shelf_nav/config/g1/env_cfg.py",
    "src/ioailab/tasks/pick_to_shelf_nav/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/pick_to_shelf_nav/mdp/__init__.py",
    "src/ioailab/tasks/pick_to_shelf_nav/mdp/events.py",
    "src/ioailab/tasks/pick_to_shelf_nav/mdp/goals.py",
    "src/ioailab/tasks/pick_to_shelf_place/__init__.py",
    "src/ioailab/tasks/pick_to_shelf_place/config/g1/env_cfg.py",
    "src/ioailab/tasks/pick_to_shelf_place/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/pick_to_shelf_place/mdp/__init__.py",
    "src/ioailab/tasks/pick_to_shelf_place/mdp/events.py",
    "src/ioailab/tasks/pick_to_shelf_place/mdp/terminations.py",
    "src/ioailab/tasks/pick_to_shelf_place/motion_plan.py",
    "src/ioailab/tasks/sort_to_shelf/__init__.py",
    "src/ioailab/tasks/sort_to_shelf/scene.py",
    "src/ioailab/tasks/sort_to_shelf/config/g1/env_cfg.py",
    "src/ioailab/tasks/sort_to_shelf/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/sort_to_shelf/mdp/__init__.py",
    "src/ioailab/tasks/sort_to_shelf/mdp/events.py",
    "src/ioailab/tasks/sort_to_shelf/mdp/terminations.py",
    "src/ioailab/tasks/sort_to_shelf_pick/__init__.py",
    "src/ioailab/tasks/sort_to_shelf_pick/config/g1/env_cfg.py",
    "src/ioailab/tasks/sort_to_shelf_pick/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/sort_to_shelf_pick/mdp/__init__.py",
    "src/ioailab/tasks/sort_to_shelf_pick/mdp/events.py",
    "src/ioailab/tasks/sort_to_shelf_pick/mdp/observations.py",
    "src/ioailab/tasks/sort_to_shelf_pick/mdp/terminations.py",
    "src/ioailab/tasks/sort_to_shelf_pick/motion_plan.py",
    "src/ioailab/tasks/sort_to_shelf_nav/__init__.py",
    "src/ioailab/tasks/sort_to_shelf_nav/agent.py",
    "src/ioailab/tasks/sort_to_shelf_nav/config/g1/env_cfg.py",
    "src/ioailab/tasks/sort_to_shelf_nav/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/sort_to_shelf_nav/mdp/__init__.py",
    "src/ioailab/tasks/sort_to_shelf_nav/mdp/events.py",
    "src/ioailab/tasks/sort_to_shelf_nav/mdp/terminations.py",
    "src/ioailab/tasks/sort_to_shelf_place/__init__.py",
    "src/ioailab/tasks/sort_to_shelf_place/config/g1/env_cfg.py",
    "src/ioailab/tasks/sort_to_shelf_place/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/sort_to_shelf_place/mdp/__init__.py",
    "src/ioailab/tasks/sort_to_shelf_place/mdp/events.py",
    "src/ioailab/tasks/sort_to_shelf_place/mdp/terminations.py",
    "src/ioailab/tasks/sort_to_shelf_place/motion_plan.py",
    "src/ioailab/tasks/reach/__init__.py",
    "src/ioailab/tasks/stack_cube/__init__.py",
    "src/ioailab/tasks/stack_cube/config/g1/mdp_cfg.py",
    "src/ioailab/tasks/stack_cube/mdp/__init__.py",
}


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )


def _text_files() -> list[Path]:
    roots = [ROOT / "src", ROOT / "tests", ROOT / "docs", ROOT / "examples"]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    for extra in (ROOT / "README.md", ROOT / "CHANGELOG.md"):
        if extra.exists():
            files.append(extra)
    return sorted(files)


def _run_fresh_process(code: str) -> dict[str, object]:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not old_pythonpath else f"{SRC}:{old_pythonpath}"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        text=True,
    )
    return json.loads(result.stdout.strip())


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _text_file_paths_for_architecture_cleanup() -> list[Path]:
    return [
        path
        for path in _text_files()
        if path != Path(__file__)
        and ".omx" not in path.parts
        and "__pycache__" not in path.parts
    ]


def _grep_architecture_cleanup_tokens(tokens: tuple[str, ...]) -> list[str]:
    offenders: list[str] = []
    for path in _text_file_paths_for_architecture_cleanup():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in tokens:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: {token}")
    return offenders


def test_g1_spec_module_is_pure_data_when_present() -> None:
    spec_path = ROOT / "src" / "ioailab" / "robots" / "g1" / "spec.py"
    if not spec_path.exists():
        return

    tree = ast.parse(spec_path.read_text(encoding="utf-8"), filename=str(spec_path))
    imports = _imported_modules(tree)
    forbidden_prefixes = (
        "isaaclab",
        "curobo",
        "ioailab.utils.asset_utils",
        "ioailab.envs",
        "ioailab.tasks",
        "ioailab.agents.motion_plan.contracts.g1_curobov2",
    )

    forbidden = sorted(
        module
        for module in imports
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )
    assert forbidden == []


def test_all_exports_are_explicitly_allowlisted() -> None:
    files_with_all = {
        path.relative_to(ROOT).as_posix()
        for path in _python_files(ROOT / "src")
        if "__all__" in path.read_text(encoding="utf-8")
    }

    assert files_with_all <= _ALLOWED_ALL_EXPORT_FILES


def test_top_level_import_stays_side_effect_free_for_architecture_refactors() -> None:
    data = _run_fresh_process(
        """
        import json
        import sys

        import ioailab

        loaded = set(sys.modules)
        print(json.dumps({
            "all": ioailab.__all__,
            "tasks_loaded": "ioailab.tasks" in loaded,
            "g1_planner_loaded": any(name.startswith("ioailab.agents.motion_plan.contracts.g1_curobov2") for name in loaded),
            "generic_curobo_loaded": any(name.startswith("ioailab.agents.motion_plan.solvers.curobov2") for name in loaded),
            "isaaclab_app_loaded": "isaaclab.app" in loaded,
            "gymnasium_loaded": "gymnasium" in loaded,
        }))
        """
    )

    assert data == {
        "all": ["__version__"],
        "tasks_loaded": False,
        "g1_planner_loaded": False,
        "generic_curobo_loaded": False,
        "isaaclab_app_loaded": False,
        "gymnasium_loaded": False,
    }


def test_public_task_ids_remain_stable() -> None:
    import ioailab.tasks as tasks

    assert set(tasks.BUILTIN_TASK_IDS) == {
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
    }


def test_envs_package_exposes_only_public_workflow_api() -> None:
    assert not (ROOT / "src" / "ioailab" / "envs" / "handle.py").exists()

    import ioailab.envs as envs

    assert envs.__all__ == [
        "ioailabEnv",
        "make_env",
    ]
    assert not (ROOT / "src" / "ioailab" / "envs" / "snapshot.py").exists()
    assert not (ROOT / "src" / "ioailab" / "envs" / "_rollout.py").exists()


def test_combined_task_runtime_hooks_live_under_envs_layer() -> None:
    assert not (ROOT / "src" / "ioailab" / "tasks" / "flow.py").exists()
    task_composition_src = (
        ROOT / "src" / "ioailab" / "tasks" / "common" / "composition.py"
    ).read_text(encoding="utf-8")

    assert "def current_task_phases" not in task_composition_src
    assert "def set_task_phases" not in task_composition_src
    assert "def reset_task_phases" not in task_composition_src
    assert "def phase_gated_success" not in task_composition_src
    assert "import ioailab.envs.flow as env_flow" in task_composition_src

    import ioailab.envs.flow as env_flow
    import ioailab.tasks.common.composition as task_composition

    for name in (
        "current_task_phases",
        "set_task_phases",
        "reset_task_phases",
        "phase_gated_success",
    ):
        assert hasattr(env_flow, name)
        assert not hasattr(task_composition, name)


def test_mdp_terms_buckets_are_removed_and_semantic_owners_are_documented() -> None:
    terms_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src" / "ioailab" / "tasks").glob("*/mdp/terms.py")
    )
    assert terms_files == []

    offenders = _grep_architecture_cleanup_tokens(
        (
            ".mdp.terms",
            "from ioailab.tasks.base_nav.mdp import terms",
            "from ioailab.tasks.pick_cube.mdp import terms",
            "from ioailab.tasks.pick_to_shelf.mdp import terms",
            "from ioailab.tasks.stack_cube.mdp import terms",
        )
    )
    assert offenders == []

    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "docs").rglob("*.md"))
    )
    assert "`terms.py` contains runtime helper functions" not in docs
    assert "Do not add a generic `terms.py` bucket" in docs
    assert "predicates.py" not in docs
    assert "success.py" not in docs


def test_example_01_documents_empty_reward_and_curriculum_managers() -> None:
    script = (ROOT / "examples" / "01_collect.py").read_text(encoding="utf-8")
    docs = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "docs" / "examples.md", ROOT / "docs" / "tutorial.md")
        if path.exists()
    )
    combined = script + "\n" + docs

    assert "CuroboPlannerAgent" in script
    assert "env.collect(" in script
    assert "empty reward" in combined.lower()
    assert "curriculum" in combined.lower()
    assert "expert motion" in combined.lower() or "motion-planning" in combined.lower()

    pick_cube_mdp_cfg = (
        ROOT
        / "src"
        / "ioailab"
        / "tasks"
        / "pick_cube"
        / "config"
        / "g1"
        / "mdp_cfg.py"
    ).read_text(encoding="utf-8")
    assert "rewards = None" in pick_cube_mdp_cfg
    assert "curriculum = None" in pick_cube_mdp_cfg


def test_task_scenes_split_robot_agnostic_world_from_g1_layer() -> None:
    # Manipulation tasks own a robot-agnostic world scene in scene.py; the G1
    # scene cfg in config/g1/env_cfg.py subclasses it and inserts the robot.
    for task_name, scene_cls in (
        ("pick_cube", "PickCubeSceneCfg"),
        ("stack_cube", "StackCubeSceneCfg"),
        ("pick_to_shelf", "PickToShelfSceneCfg"),
        ("sort_to_shelf", "SortToShelfSceneCfg"),
    ):
        scene_src = (
            ROOT / "src" / "ioailab" / "tasks" / task_name / "scene.py"
        ).read_text(encoding="utf-8")
        assert f"class {scene_cls}(DefaultSceneCfg)" in scene_src
        # The world scene is robot-agnostic: no robot or sensors declared.
        assert "robot =" not in scene_src
        assert "make_g1_camera_cfg" not in scene_src

        env_cfg_path = (
            ROOT
            / "src"
            / "ioailab"
            / "tasks"
            / task_name
            / "config"
            / "g1"
            / "env_cfg.py"
        )
        if task_name in {"pick_to_shelf", "sort_to_shelf"}:
            env_cfg_path = (
                ROOT
                / "src"
                / "ioailab"
                / "tasks"
                / f"{task_name}_pick"
                / "config"
                / "g1"
                / "env_cfg.py"
            )
        env_src = env_cfg_path.read_text(encoding="utf-8")
        assert f"({scene_cls})" in env_src
        assert "robot = make_galbot_g1" in env_src

    # base_nav is robot-only (mobile base); it keeps its scene inline.
    assert not (ROOT / "src" / "ioailab" / "tasks" / "base_nav" / "scene.py").exists()
    assert not (
        ROOT / "src" / "ioailab" / "tasks" / "pick_cube" / "demo_scene.py"
    ).exists()
    assert not (
        ROOT / "src" / "ioailab" / "tasks" / "common" / "lightweight_scene.py"
    ).exists()

    docs = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "docs/development.md",
            "docs/tasks.md",
            "docs/architecture.md",
        )
    )
    assert "demo_scene.py" not in docs
    assert "make_*_cfg" in docs

    env_cfg_path = (
        ROOT
        / "src"
        / "ioailab"
        / "tasks"
        / "pick_cube"
        / "config"
        / "g1"
        / "env_cfg.py"
    )
    tree = ast.parse(
        env_cfg_path.read_text(encoding="utf-8"), filename=str(env_cfg_path)
    )
    cfg_constructor_names = {"AssetBaseCfg", "RigidObjectCfg", "FrameTransformerCfg"}
    shared_cfg_names: set[str] = set()

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        constructor = node.value.func
        if (
            isinstance(constructor, ast.Name)
            and constructor.id in cfg_constructor_names
        ):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.isupper()
                    and target.id.endswith("_CFG")
                ):
                    shared_cfg_names.add(target.id)

    assert shared_cfg_names == set()

    mutations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(
                    target.value, ast.Name
                ):
                    if target.value.id in shared_cfg_names:
                        mutations.append(f"{target.value.id}.{target.attr}")

    assert mutations == []


def test_motion_plan_has_canonical_agent_package_boundary() -> None:
    canonical = ROOT / "src" / "ioailab" / "agents" / "motion_plan"
    assert (canonical / "agent.py").exists()
    assert (canonical / "action_source.py").exists()
    assert (canonical / "commands.py").exists()
    assert (canonical / "joint_target_agent.py").exists()
    assert (canonical / "contracts" / "g1.py").exists()
    assert (canonical / "contracts" / "g1_curobov2.py").exists()
    assert (canonical / "solvers" / "curobov2").is_dir()

    planner_dir = ROOT / "src" / "ioailab" / "agents" / "planner"
    planner_py_files = (
        []
        if not planner_dir.exists()
        else [path.name for path in planner_dir.glob("*.py")]
    )
    assert planner_py_files == []


def test_tasks_have_no_task_root_constants_module() -> None:
    """Shared values live with their owner, not in a separate constants module.

    Tasks expose no dedicated constants dump (no task-root ``spec.py``, no
    ``mdp/layout.py``): pick-to-shelf world geometry lives in ``scene.py`` and the
    place/pick thresholds live in ``mdp/terminations.py``.
    """

    tasks_root = ROOT / "src" / "ioailab" / "tasks"
    task_root_specs = sorted(
        path.relative_to(ROOT).as_posix() for path in tasks_root.glob("*/spec.py")
    )
    assert task_root_specs == []

    # Mimic phases/hooks live in env_cfg and robot conversion lives in the G1
    # converter layer; task-local Mimic runtime packages stay gone.
    assert not (tasks_root / "stack_cube" / "mimic").exists()
    assert not (tasks_root / "pick_cube" / "mimic").exists()

    # No standalone geometry-constants module; geometry is homed in scene.py.
    assert not (tasks_root / "pick_to_shelf" / "mdp" / "layout.py").exists()


def test_task_common_does_not_import_task_specific_packages() -> None:
    """Shared task helpers stay mechanical and do not depend on task packages."""

    common_root = ROOT / "src" / "ioailab" / "tasks" / "common"
    offenders: list[str] = []

    for path in sorted(common_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                module_names.append(node.module)

            for module_name in module_names:
                if module_name.startswith(
                    "ioailab.tasks."
                ) and not module_name.startswith("ioailab.tasks.common"):
                    offenders.append(
                        f"{path.relative_to(ROOT).as_posix()}: imports {module_name}"
                    )

    assert offenders == []


def test_deleted_subtask_packages_do_not_return() -> None:
    """Task composition uses component task packages, not subtask packages."""

    tasks_root = ROOT / "src" / "ioailab" / "tasks"
    assert not (ROOT / "src" / "ioailab" / "subtasks").exists()
    assert not (ROOT / "src" / "ioailab" / "agents" / "subtasks").exists()
    assert not (ROOT / "src" / "ioailab" / "agents" / "flow" / "subtasks.py").exists()
    assert not (
        ROOT / "src" / "ioailab" / "agents" / "flow" / "subtask_agent.py"
    ).exists()

    pick_to_shelf = tasks_root / "pick_to_shelf"
    assert not (pick_to_shelf / "subtasks").exists()
    assert not (pick_to_shelf / "flow.py").exists()
    assert not list((pick_to_shelf / "mdp").glob("*.py"))
    assert (tasks_root / "pick_to_shelf_pick").is_dir()
    assert (tasks_root / "pick_to_shelf_nav").is_dir()
    assert (tasks_root / "pick_to_shelf_place").is_dir()


def test_pick_to_shelf_phase_starts_use_scenario_files() -> None:
    """PickToShelf standalone components reset from task scenario YAML files."""

    tasks_root = ROOT / "src" / "ioailab" / "tasks"
    assert not (tasks_root / "pick_to_shelf" / "snapshots.py").exists()
    assert not (ROOT / "examples" / "07_pick_to_shelf_generate_snapshots.py").exists()
    collect_example = (ROOT / "examples" / "01_collect.py").read_text(encoding="utf-8")
    assert "--init-scenario" in collect_example
    assert "--save-end-scenario" in collect_example
    assert "load_snapshot" not in collect_example
    assert "--init-snapshot" not in collect_example
    assert "init_snapshot" not in collect_example

    for scenario_path in (
        tasks_root
        / "pick_to_shelf_nav"
        / "config"
        / "g1"
        / "scenarios"
        / "nav_default.yaml",
        tasks_root
        / "pick_to_shelf_place"
        / "config"
        / "g1"
        / "scenarios"
        / "place_default.yaml",
    ):
        assert scenario_path.exists()
        assert len(scenario_path.read_text(encoding="utf-8").splitlines()) < 100
        scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        assert scenario["schema"] == "ioailabScenario-v0"
        assert scenario["frame"] == "env"
        robot = scenario["assets"]["articulation"]["robot"]
        assert "base_pose" in robot
        assert "root_pose" not in robot
        assert "root_velocity" not in robot
        assert "joint_velocity" not in robot
        assert robot["joint_position"]["left_gripper_joint"] > 1.0
        assert "cube" in scenario["assets"]["rigid_object"]
        for asset_fields in scenario["assets"]["rigid_object"].values():
            assert "root_pose" in asset_fields
            assert "root_velocity" not in asset_fields


def test_robots_do_not_import_motion_plan_agents() -> None:
    forbidden = (
        "ioailab.agents.motion_plan",
        "ioailab.agents.planner",
        "curobo",
        "curobov2",
    )
    offenders: list[str] = []
    allowed_bridge = ROOT / "src" / "ioailab" / "robots" / "g1" / "converters.py"
    for path in _python_files(ROOT / "src" / "ioailab" / "robots"):
        if path == allowed_bridge:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {token!r}")
    assert offenders == []


def test_navigation_agents_do_not_write_manipulation_action_groups() -> None:
    """Navigation agents own base motion, not arm/gripper/leg action targets."""

    forbidden_write_calls = {
        "write_g1_binary_values",
        "write_g1_frame_action",
        "write_g1_initial_action",
        "write_g1_joint_targets",
    }
    non_base_groups = {
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
        "legs",
    }
    offenders: list[str] = []
    for path in _python_files(ROOT / "src" / "ioailab"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for class_node in (
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ):
            if not _is_nav_agent_class(class_node):
                continue
            for node in ast.walk(class_node):
                if isinstance(node, ast.Call):
                    call_name = _call_name(node.func)
                    if call_name in forbidden_write_calls:
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} "
                            f"{class_node.name} calls {call_name}"
                        )
                    if call_name.startswith("pack_g1_") and (
                        call_name != "pack_g1_base_velocity_command"
                    ):
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} "
                            f"{class_node.name} calls {call_name}"
                        )
                    for keyword in node.keywords:
                        if (
                            keyword.arg == "group_name"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value in non_base_groups
                        ):
                            offenders.append(
                                f"{path.relative_to(ROOT)}:{node.lineno} "
                                f"{class_node.name} writes {keyword.value.value!r}"
                            )
    assert offenders == []


def _is_nav_agent_class(class_node: ast.ClassDef) -> bool:
    if class_node.name in {"BaseNavAgent", "ProportionalNavAgent"}:
        return True
    if class_node.name.endswith("NavAgent"):
        return True
    return any(_call_name(base).endswith("NavAgent") for base in class_node.bases)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _handles_import_error(handler: ast.ExceptHandler) -> bool:
    """Return whether an except handler catches ImportError/ModuleNotFoundError."""

    exc_type = handler.type
    if exc_type is None:
        return False
    candidates = list(exc_type.elts) if isinstance(exc_type, ast.Tuple) else [exc_type]
    return any(
        isinstance(candidate, ast.Name)
        and candidate.id in {"ImportError", "ModuleNotFoundError"}
        for candidate in candidates
    )


def test_no_silent_import_error_swallow_in_source() -> None:
    """``except ImportError`` must not silently skip work via a bare return/pass.

    Optional-backend handlers may fall back to a value (for example ``wp = None``
    for the lazy warp backend) or re-raise with a clear message. A bare ``return``
    or ``pass`` silently changes control flow and must not reappear.
    """

    offenders: list[str] = []
    for path in _python_files(SRC / "ioailab"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not _handles_import_error(
                node
            ):
                continue
            body = node.body
            if len(body) == 1 and (
                isinstance(body[0], ast.Pass)
                or (isinstance(body[0], ast.Return) and body[0].value is None)
            ):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")

    assert offenders == []


def test_no_deprecated_isaaclab_schema_cfg_imports() -> None:
    """PhysX schema cfgs come from ``isaaclab_physx.sim.schemas``, not the aliases.

    IsaacLab 3.0 split ``RigidBodyPropertiesCfg`` / ``CollisionPropertiesCfg`` into
    base + PhysX classes and relocated them to ``isaaclab_physx.sim.schemas``. The
    old names still resolve from ``isaaclab.sim`` via a deprecation shim that warns
    on construction. Source code must import the ``Physx*`` classes directly so the
    deprecated path cannot creep back in.
    """

    deprecated = {"RigidBodyPropertiesCfg", "CollisionPropertiesCfg"}
    offenders: list[str] = []
    for path in _python_files(SRC / "ioailab"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "isaaclab.sim":
                continue
            for alias in node.names:
                if alias.name in deprecated:
                    offenders.append(
                        f"{path.relative_to(ROOT).as_posix()}:{node.lineno} "
                        f"imports {alias.name} from isaaclab.sim"
                    )

    assert offenders == []


def test_g1_camera_cfg_factory_is_public_not_private() -> None:
    """The G1 camera cfg factory is a public keep-me surface, not a private import."""

    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in _python_files(SRC / "ioailab")
        if "_make_g1_camera_cfg" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []

    from ioailab.robots.g1.sensors.camera import make_g1_camera_cfg

    assert callable(make_g1_camera_cfg)


def test_policy_code_is_consolidated_under_agents_policy() -> None:
    """All policy code lives under ``ioailab.agents.policy`` — one home, no cycle.

    The runtime ``PolicyAgent`` and the offline train/checkpoint adapters
    (``Policy`` / ``PolicyCheckpoint`` / ``PolicyTrainCfg`` / ``RobomimicDiffusionPolicy``)
    sit in the same package, so there is no separate top-level ``policies`` /
    ``learning`` package and no agents <-> learning import cycle.
    """

    policy_pkg = SRC / "ioailab" / "agents" / "policy"
    assert (policy_pkg / "action_source.py").exists()
    assert (policy_pkg / "checkpoint.py").exists()
    assert (policy_pkg / "train.py").exists()
    assert (policy_pkg / "backends" / "robomimic_diffusion.py").exists()
    assert not (SRC / "ioailab" / "policies").exists()
    assert not (SRC / "ioailab" / "learning").exists()

    offenders = _grep_architecture_cleanup_tokens(
        ("ioailab" + ".policies", "ioailab" + ".learning")
    )
    assert offenders == []

    from ioailab.agents.policy import (
        RobomimicDiffusionPolicy,
        Policy,
        PolicyAgent,
        PolicyCheckpoint,
        PolicyTrainCfg,
    )

    assert all(
        obj is not None
        for obj in (
            Policy,
            PolicyCheckpoint,
            PolicyTrainCfg,
            RobomimicDiffusionPolicy,
            PolicyAgent,
        )
    )


def test_cuboids_are_authored_as_meshes_not_shape_prims() -> None:
    """Cuboid props spawn via ``MeshCuboidCfg``; bare ``CuboidCfg`` shape prims are gone.

    IsaacLab 3.0 favors mesh-backed geometry (uniform collision/material handling
    across PhysX/Newton), so ioailab authors cuboids as meshes everywhere.
    """

    import re

    bare_cuboid = re.compile(r"(?<!Mesh)\bCuboidCfg\b")
    offenders: list[str] = []
    for path in _python_files(SRC / "ioailab"):
        text = path.read_text(encoding="utf-8")
        if bare_cuboid.search(text):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_env_uses_qualified_private_helper_imports_not_aliases() -> None:
    """envs/env.py reads its private helpers as ``_module.func``, not ``... as _foo``.

    The factory/recorder/mask helpers are imported as module namespaces so the
    call sites stay self-describing. The old ``from ._recorder import (foo as
    _foo, ...)`` alias wall must not creep back in.
    """

    env_path = SRC / "ioailab" / "envs" / "env.py"
    tree = ast.parse(env_path.read_text(encoding="utf-8"), filename=str(env_path))
    private_modules = {
        "ioailab.envs._factory",
        "ioailab.envs._masks",
        "ioailab.envs._recorder",
    }

    aliased_from_imports = [
        f"{node.module}.{alias.name} as {alias.asname}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in private_modules
        for alias in node.names
    ]
    assert aliased_from_imports == []

    namespace_imports = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name in private_modules
    }
    assert namespace_imports == {"_factory", "_masks", "_recorder"}
