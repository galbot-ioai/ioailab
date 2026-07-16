from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_G1_USD = "assets/galbot_one_golf_description/usd/galbot_one_golf.usda"
STALE_G1_USD = "assets/galbot_g1/usd/usd/galbot_g1.usd"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_old_project_name_is_not_reintroduced() -> None:
    old_lower = "galbot" + "lab"
    old_title = "Galbot" + "Lab"
    old_upper = "GALBOT" + "LAB"
    forbidden = (
        old_lower,
        old_title,
        old_upper,
        old_lower.replace("bot", "bot-"),
        old_lower.replace("bot", "bot_"),
        old_lower.replace("bot", "bot "),
        old_lower.replace("bot", "bot."),
    )
    checked_suffixes = {
        "",
        ".cfg",
        ".css",
        ".hbs",
        ".json",
        ".lock",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    skipped_dirs = {
        ".git",
        ".mypy_cache",
        ".omx",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "book",
    }
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT}",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    offenders: list[str] = []
    paths = [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    for path in paths:
        if not path.is_file():
            continue
        if any(part in skipped_dirs for part in path.relative_to(ROOT).parts):
            continue
        if (
            path.name not in {"Dockerfile", "Makefile"}
            and path.suffix not in checked_suffixes
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(name in text for name in forbidden):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_docker_compose_lives_under_docker_directory() -> None:
    makefile = read("Makefile")
    compose = read("docker/compose.yaml")
    dockerfile = read("docker/Dockerfile")
    shell_gui = read("docker/shell_gui.sh")
    development = read("docs/development.md")

    assert not (ROOT / "docker_compose.yaml").exists()
    assert (ROOT / "docker" / "compose.yaml").exists()
    assert "docker compose -f docker/compose.yaml" in makefile
    assert "ioailab_PACKAGE_VERSION :=" in makefile
    assert "ioailab_IMAGE_TAG ?= $(ioailab_PACKAGE_VERSION)" in makefile
    assert "image: ${ioailab_IMAGE:-ioailab:1.0.0a1}" in compose
    assert "ioailab:dev" not in compose
    assert "FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2" in dockerfile
    assert "USER root" in dockerfile
    assert "/usr/local/bin/python" in dockerfile
    assert "DISPLAY: ${DISPLAY:-}" in compose
    assert "ioailab_VERSION_FILE" in shell_gui
    assert "export ioailab_IMAGE" in shell_gui
    assert "python -m ruff check ." in makefile
    assert "python -m ruff format ." in makefile
    assert "python -m ty check src examples tests" in makefile
    assert "--warn all --quiet" in makefile
    assert "typecheck:" in makefile
    assert "version-tagged Docker image" in development
    assert "docker_compose.yaml" not in makefile + development


def test_docs_build_with_mdbook_without_mkdocs_or_ci_and_with_precommit() -> None:
    makefile = read("Makefile")
    pyproject = read("pyproject.toml")
    precommit = read(".pre-commit-config.yaml")
    development = read("docs/development.md")

    assert not (ROOT / ".gitlab-ci.yml").exists()
    assert not (ROOT / ".ci").exists()
    assert (ROOT / ".pre-commit-config.yaml").exists()

    # Docs migrated from MkDocs to mdBook.
    assert not (ROOT / "mkdocs.yml").exists()
    assert not (ROOT / "docs" / "stylesheets").exists()
    assert (ROOT / "book.toml").exists()
    assert (ROOT / "docs" / "SUMMARY.md").exists()
    assert 'src = "docs"' in read("book.toml")

    assert "mdbook build" in makefile
    assert "mdbook serve" in makefile
    assert "mkdocs" not in makefile
    assert "mkdocs" not in pyproject
    assert "ruff-pre-commit" in precommit
    assert "pre-commit>=4.0" in pyproject
    assert "## Pre-commit" in development
    assert "GitLab CI" not in development


def test_scenes_are_task_local_without_top_level_scene_package() -> None:
    source_and_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for root in ("src", "tests", "docs")
        for path in (ROOT / root).rglob("*.py" if root != "docs" else "*.md")
    )

    assert not (ROOT / "src" / "ioailab" / "scenes").exists()
    stale_scene_import = "ioailab" + ".scenes"
    stale_import_patterns = (
        f"from {stale_scene_import}",
        f"import {stale_scene_import}",
    )
    for pattern in stale_import_patterns:
        assert pattern not in source_and_docs
    assert "ioailab.tasks.stack_cube.config.g1.env_cfg" in source_and_docs
    assert "ioailab.tasks.pick_cube.config.g1.env_cfg" in source_and_docs


def test_architecture_docs_and_agents_are_the_design_source() -> None:
    agents = read("AGENTS.md")
    architecture = read("docs/architecture.md")
    index = read("docs/index.md")
    development = read("docs/development.md")

    assert not (ROOT / "ioailab_DESIGN.md").exists()
    assert not (ROOT / ".codex" / "skills" / "ioailab-architecture").exists()
    assert "ioailab Architecture" in architecture
    assert "GalbotG1-BaseNav-v0" in read("docs/tasks.md")
    assert "docker/compose.yaml" in development
    stale_scene_package = "ioailab" + ".scenes"
    assert f"no top-level `{stale_scene_package}`" in architecture
    assert "AGENTS.md" in index
    assert "AGENTS.md" in development
    assert "ioailab_DESIGN.md" not in index + development
    assert "ioailab-architecture" not in index + development
    assert "Architecture Hygiene" not in agents
    assert not (ROOT / "docs" / "pick_to_shelf_taskflow_redesign.md").exists()
    assert "pick_to_shelf_taskflow_redesign.md" not in architecture
    assert "pick_to_shelf_taskflow_redesign.md" not in read("docs/SUMMARY.md")


def test_docs_use_canonical_asset_path_and_no_stale_layout() -> None:
    checked_paths = [
        "README.md",
        "AGENTS.md",
        "docs/index.md",
        "docs/development.md",
        "docs/reference.md",
        "docs/tasks.md",
        "docs/galbot_sensors.md",
    ]
    combined = "\n".join(read(path) for path in checked_paths)

    assert CANONICAL_G1_USD in combined
    assert STALE_G1_USD not in combined
    assert "temporary IsaacLab host-env scaffolding" not in combined


def test_data_docs_cover_pipeline_and_lerobot_helpers() -> None:
    readme = read("README.md")
    summary = read("docs/SUMMARY.md")
    data = read("docs/data.md")
    dockerfile = read("docker/Dockerfile")

    assert "[Data & Datasets](data.md)" in summary
    assert "docs/data.md" in readme
    assert "ARG LEROBOT_VERSION=0.5.1" in dockerfile
    assert (
        'uv pip install --system --no-deps "lerobot==${LEROBOT_VERSION}"' in dockerfile
    )
    # The data page documents the public collect -> mimic -> train -> evaluate
    # path and the retained LeRobot exporter.
    assert "env.collect(" in data
    assert "dataset = mimic(" in data
    assert "Policy.from_backend" in data
    assert "env.evaluate(" in data
    assert "GalbotG1-PickCube-v0" in data
    assert "MotionPlanLeRobotExporter" in data
    assert (
        "from ioailab.datasets.motion_plan_lerobot import MotionPlanLeRobotExporter"
        in data
    )
    assert "from lerobot.datasets.lerobot_dataset import LeRobotDataset" in data
    assert "observation.state" in data
    assert "observation.images" in data
    assert "robomimic_diffusion" in readme
    assert "mimic(dataset, episodes=...)" in readme
    # No resurrected runner record flags.
    assert "--record-lerobot-root" not in data
    assert "/tmp/galbot_lerobot" not in data


def test_docs_document_public_api_import_and_registry_boundaries() -> None:
    readme = read("README.md")
    development = read("docs/development.md")
    sensors = read("docs/galbot_sensors.md")
    tasks_doc = read("docs/tasks.md")
    architecture = read("docs/architecture.md")

    # The side-effect-free import and registry boundaries are documented in the
    # architecture and tasks references; the README stays minimal.
    assert "Top-level imports stay side-effect free" in architecture
    assert "explicit IsaacLab-style registry" in tasks_doc
    assert "make_env" in readme
    old_handle = "WorkflowEnv" + "Handle"
    old_session = "RuntimeEnv" + "Session"
    assert old_handle not in readme
    assert old_session not in readme
    assert "cuRobo v2 (`curobov2`)" in readme
    assert "For motion-planning examples and docs, use cuRobo v2" in development
    assert "ioailab_DESIGN.md" not in readme + development
    assert "make_env" in sensors
    assert "GalbotG1-PickCube-Teleop-v0" in sensors
    assert "GalbotG1-PickCube-v0" in readme
    # The full task-ID list lives in the tasks reference.
    assert "GalbotG1-StackCube-v0" in tasks_doc
    assert "GalbotG1-BaseNav-v0" in tasks_doc
    assert "Keep task packages task-first" in development
    public_camera_docs = "\n".join(
        (
            sensors,
            read("docs/architecture.md"),
            read("docs/examples.md"),
            read("examples/01_collect.py"),
        )
    )
    for stale_camera_option in (
        "camera_mounts",
        "camera_data",
        "camera_width",
        "camera_height",
    ):
        assert stale_camera_option not in public_camera_docs


def test_docs_document_scoped_task_component_convention() -> None:
    readme = read("README.md")
    agents = read("AGENTS.md")
    architecture = read("docs/architecture.md")
    tasks_doc = read("docs/tasks.md")
    development = read("docs/development.md")

    combined = "\n".join((readme, agents, architecture, tasks_doc, development))
    assert "GalbotG1-PickCube-v0" in combined
    assert "GalbotG1-StackCube-v0" in combined
    assert "GalbotG1-BaseNav-v0" in combined
    assert "ioailab.tasks.<task>" in combined
    assert "ioailab.tasks.<task>.config.g1.agent_cfg" in combined
    assert "tasks/<task>/config/g1/agent_cfg/motion_plan.py" in combined
    assert (
        "Task packages stay task-first" in combined
        or "Keep task packages task-first" in combined
    )


def test_examples_that_parse_registered_tasks_register_task_registry() -> None:
    for path in sorted((ROOT / "examples").glob("**/*.py")):
        script = path.read_text(encoding="utf-8")
        if 'parse_env_cfg("GalbotG1' in script or "parse_env_cfg(TASK_ID" in script:
            assert "import ioailab.tasks" in script, path
            assert "ioailab.tasks.register_tasks()" in script, path


def test_motion_planning_runner_uses_single_curobov2_backend() -> None:
    assert not (ROOT / "scripts" / "motion_planning" / "run.py").exists()
    script = read("examples/01_collect.py")

    assert '"--task"' in script
    assert 'default="GalbotG1-PickCube-v0"' in script
    assert '"--planner"' not in script
    assert 'choices=("curobov2",)' not in script
    assert "args.planner" not in script
    assert "_planner_label" not in script
    assert "CuroboPlannerAgent" in script
    assert "env.collect(" in script


def test_motion_planning_run_accepts_scoped_task_variants() -> None:
    assert not (ROOT / "scripts" / "motion_planning" / "run.py").exists()
    registry = (
        read("src/ioailab/tasks/pick_cube/__init__.py")
        + read("src/ioailab/tasks/stack_cube/__init__.py")
        + read("src/ioailab/tasks/base_nav/__init__.py")
    )
    readme = read("README.md")
    tasks_doc = read("docs/tasks.md")

    assert "GalbotG1-PickCube-v0" in registry
    assert "GalbotG1-StackCube-v0" in registry
    assert "GalbotG1-BaseNav-v0" in registry
    assert "GalbotG1-PickCube-v0" in readme
    assert "GalbotG1-StackCube-v0" in tasks_doc
    assert "GalbotG1-BaseNav-v0" in tasks_doc
    assert "make_env" in readme
    assert "env.collect(" in readme or "env.evaluate(" in readme
    assert "env.set_agent(" not in readme
    assert "from ioailab.tasks.common import Agent" not in readme
    assert "MOTION_PLAN" not in readme
    assert "examples/01_collect.py" in readme or "examples/02_mimic.py" in readme
    assert "GalbotG1DualArmStackCube-v0" not in registry
    assert "GalbotG1WholeBodyStackCube-v0" not in registry
    assert "GalbotG1MobileStackCube-v0" not in registry
    assert not (ROOT / "src/ioailab/tasks/stack_cube/runtime/motion_plan.py").exists()


def test_task_tutorial_documents_pick_cube_motion_teleop_and_mimic_workflows() -> None:
    tutorial = read("docs/tutorial.md")
    chapter_1 = read("docs/tutorials/01_build_and_activate_simple_task.md")
    index = read("docs/index.md")
    summary = read("docs/SUMMARY.md")
    combined_tutorial = "\n".join((tutorial, chapter_1))

    assert "# Tutorial" in tutorial
    assert "Chapter 1: Build and Activate a Simple Task" in tutorial
    # Uses the pick-cube family as the reference task across motion plan,
    # teleop, and Mimic.
    assert "GalbotG1-PickCube-v0" in combined_tutorial
    assert "GalbotG1-PickCube-Teleop-v0" in combined_tutorial
    assert "GalbotG1-PickCube-Mimic-v0" in combined_tutorial
    assert "optional teleop" in tutorial
    assert "optional mimic dataset expansion" in tutorial
    assert "DatasetRef.task_id" in combined_tutorial
    # Authoring order and example commands.
    assert "src/ioailab/tasks/pick_cube/" in chapter_1
    assert "scene.py" in chapter_1
    assert "config/g1/" in chapter_1
    assert "agent_cfg/motion_plan.py" in chapter_1
    assert "TaskSpec(" in chapter_1
    assert "python examples/01_collect.py" in chapter_1
    assert "python examples/02_mimic.py" in chapter_1
    assert "TeleopAgent.from_device" in chapter_1
    assert "--episodes 36" in chapter_1
    # Navigation references resolve.
    assert "[Tutorial](tutorial.md)" in summary
    assert "tutorial.md" in index


def test_collect_example_restores_parser_and_uses_review_hook() -> None:
    script = read("examples/01_collect.py")

    assert "argparse.ArgumentParser" in script
    assert '"--task"' in script
    assert '"--episodes"' in script
    assert '"--episodes-per-env"' not in script
    assert '"--num-envs"' in script
    assert "default=1" in script
    assert "CuroboPlannerAgent.from_task" in script
    assert '# agent = TeleopAgent.from_device("gp001", task=task_id)' in script
    assert "dataset = env.collect(" in script
    assert "path=args.dataset_path" in script
    assert "episodes=args.episodes" in script
    assert "max_steps=args.max_steps" in script
    assert "#     decision = agent.review_demo()" in script
    assert "#         dataset.drop()" in script
    assert "export_decision" not in script
    assert "ask_keep_drop_exit" not in script
    assert not (ROOT / "scripts" / "motion_planning" / "run.py").exists()


def test_motion_planning_randomize_flag_controls_reset_randomization() -> None:
    registry = read("src/ioailab/tasks/pick_cube/__init__.py") + read(
        "src/ioailab/tasks/stack_cube/__init__.py"
    )

    assert '"randomize_pick_and_place_positions"' in registry
    assert '"randomize_cube_positions"' in registry
    assert '"randomize_ground_material"' in registry
    assert '"randomize_table_material"' in registry
    assert '"randomize_hdri_texture"' in registry


def test_task_flow_example_docs_cover_scenario_and_compound_workflows() -> None:
    readme = read("README.md")
    examples = read("docs/examples.md")

    combined = "\n".join((readme, examples))
    assert "examples/07_compound_task.py --task GalbotG1-PickToShelf-v0" in combined
    assert "examples/07_compound_task.py --task GalbotG1-SortToShelf-v0" in combined
    assert "python examples/06_collect_component_task.py" in combined
    assert "GalbotG1-PickToShelf-Pick-v0" in combined
    assert "GalbotG1-SortToShelf-Nav-v0" in combined
    assert "--save-end-scenario" in combined
    assert "--init-scenario" in combined
    assert "nav_sequence_agent" in combined
    assert "--sorting-object red_cube" in combined
    assert "GalbotG1-PickToShelf-Place-v0" in combined
    assert "scenario" in combined
    assert "--pick-policy-checkpoint" not in combined
    assert "--place-policy-checkpoint" not in combined


def test_examples_have_root_numbered_tutorial_surface() -> None:
    canonical_examples = {
        "examples/01_collect.py",
        "examples/02_mimic.py",
        "examples/03_train.py",
        "examples/04_eval.py",
        "examples/05_custom_agent.py",
        "examples/06_collect_component_task.py",
        "examples/07_compound_task.py",
    }

    assert {
        str(path.relative_to(ROOT)) for path in (ROOT / "examples").glob("*.py")
    } == canonical_examples

    for directory in ("tutorials", "basic", "tasks", "motion_control", "sensors"):
        assert not (ROOT / "examples" / directory).exists(), directory

    assert (ROOT / "docs" / "examples.md").exists()
    assert not (ROOT / "examples" / "utils").exists()
    assert (ROOT / "src" / "ioailab" / "utils" / "rerun_utils.py").exists()
    assert not (ROOT / "scripts" / "motion_planning" / "run.py").exists()

    collect = read("examples/01_collect.py")
    mimic_example = read("examples/02_mimic.py")
    train = read("examples/03_train.py")
    evaluate = read("examples/04_eval.py")
    custom = read("examples/05_custom_agent.py")
    component_collect = read("examples/06_collect_component_task.py")
    compound = read("examples/07_compound_task.py")
    examples_readme = read("docs/examples.md")
    nav_scenario = read(
        "src/ioailab/tasks/pick_to_shelf_nav/config/g1/scenarios/nav_default.yaml"
    )
    place_scenario = read(
        "src/ioailab/tasks/pick_to_shelf_place/config/g1/scenarios/place_default.yaml"
    )

    assert "CuroboPlannerAgent" in collect
    assert "make_env" in collect
    assert "env.collect(" in collect
    assert "expert_agent(task_id)" not in collect
    assert "agent = CuroboPlannerAgent.from_task(task_id)" in collect
    assert 'if task_id.startswith("GalbotG1-SortToShelf-")' not in collect
    assert "--sorting-object" not in collect
    assert "camera_mounts" not in collect
    assert "camera_data" not in collect
    assert "TrajectoryNavAgent" in component_collect
    assert "nav_sequence_agent" in component_collect
    assert "COMPONENT_PRESET" in component_collect
    assert "--sorting-object" in component_collect
    assert "choices=SORTING_OBJECT_CHOICES" in component_collect
    assert 'sort_options = {"sorting_object": args.sorting_object}' in component_collect
    assert (
        "CuroboPlannerAgent.from_task(task_id, task_options=sort_options)"
        in component_collect
    )
    assert "nav_sequence_agent(sorting_object=args.sorting_object)" in component_collect
    assert '"GalbotG1-SortToShelf-Pick-v0"' in component_collect

    assert "DatasetRef" in mimic_example
    assert "dataset = mimic(" in mimic_example
    assert "task=" not in mimic_example.split("mimic(", 1)[1].split(")", 1)[0]

    assert "Policy.from_backend" in train
    assert "policy.train(" in train
    assert "Policy.from_backend" in evaluate
    assert "env.evaluate(" in evaluate

    assert "BaseAgent" in custom
    assert "act" in custom
    assert "TaskFlowAgent" in compound
    assert "SubtaskFlow(" not in compound
    assert "env.collect(" in compound
    assert "env.evaluate(" in compound
    assert "TaskFlowAgent.from_env(env)" in compound
    assert "TaskFlowAgent.from_env(env, agents=phase_agents)" in compound
    assert "save_end_scenario" in component_collect
    assert "end_scenario_name" not in collect
    assert "--save-end-scenario" in collect
    assert "--init-scenario" in collect
    assert "--sorting-object" in component_collect
    assert "--sorting-object" in compound
    assert "TrajectoryNavAgent(" not in compound
    assert "GALBOT_G1_PICK_TO_SHELF_TASK_ID" in compound
    assert "--pick-policy-checkpoint" not in compound
    assert "ioailabScenario-v0" in nav_scenario
    assert "ioailabScenario-v0" in place_scenario
    assert "left_gripper_joint:" in nav_scenario
    assert "left_gripper_joint:" in place_scenario
    assert "rigid_object:" in nav_scenario
    assert "rigid_object:" in place_scenario
    combined_examples = collect + component_collect + compound
    assert "load_snapshot(" not in combined_examples
    assert "--init-snapshot" not in combined_examples
    assert "init_snapshot" not in combined_examples
    assert "--subtask" not in combined_examples
    assert "subtask" not in combined_examples
    assert "G1ManipulationPolicyActionAdapter" in compound
    assert "PolicyAgent.from_checkpoint" in compound
    assert "TaskFlowAgent.from_env" in compound
    assert "task_options" in compound

    assert "TeleopAgent" in examples_readme
    assert "PolicyAgent" in examples_readme
    assert "examples/06_collect_component_task.py" in examples_readme
    assert "examples/07_compound_task.py" in examples_readme
    assert "examples/08_pick_to_shelf_collect_phase.py" not in examples_readme
    assert "examples/15_sort_to_shelf_eval_full.py" not in examples_readme
    assert "empty reward and curriculum managers" in examples_readme
    assert "dummy reward or curriculum terms" in examples_readme
    assert "component task" in examples_readme.lower()

    tutorial = read("docs/tutorials/01_build_and_activate_simple_task.md")
    assert "Do not add a generic `terms.py` bucket" in tutorial
    assert "`predicates.py`" not in tutorial
    assert "`success.py`" not in tutorial
    assert "`terms.py` contains runtime helper functions" not in tutorial

    removed_examples = {
        "examples/01_motion_planning.py",
        "examples/02_collect_train_evaluate.py",
        "examples/03_teleop_collect.py",
        "examples/04_subtask_flow.py",
        "examples/06_mobile_base_drive.py",
        "examples/07_pick_to_shelf.py",
        "examples/06_pick_to_shelf_collect_full.py",
        "examples/08_pick_to_shelf_collect_phase.py",
        "examples/09_pick_to_shelf_train_dp.py",
        "examples/10_pick_to_shelf_eval_full.py",
        "examples/11_pick_to_shelf_eval_phase.py",
        "examples/12_sort_to_shelf_save_scenario.py",
        "examples/13_sort_to_shelf_collect_phase.py",
        "examples/14_sort_to_shelf_eval_phase.py",
        "examples/15_sort_to_shelf_eval_full.py",
        "examples/g1-pick-cube.py",
        "examples/g1-pick-cube-motion-planning.py",
        "examples/g1-stack-cubes-motion-planning.py",
        "examples/g1-motion-planning.py",
        "examples/stack_cubes.py",
        "examples/g1_camera_sensors.py",
        "examples/g1_left_wrist_camera_stack_scene.py",
        "examples/g1_base_motion.py",
        "examples/g1_check_camera_render.py",
        "examples/g1_left_arm_joint_motion.py",
        "examples/g1_leg_joint_motion.py",
        "examples/g1_multi_env_arm_joint_motion.py",
        "examples/g1_multi_env_leg_joint_motion.py",
        "examples/g1_stack_cube_task.py",
        "examples/basic/" + "g1_robot" + "_camera_rerun.py",
    }
    for path in removed_examples:
        assert not (ROOT / path).exists(), path


def test_rerun_utils_convert_tile_and_encode_web_urls() -> None:
    import numpy as np

    from ioailab.utils.rerun_utils import (
        as_uint8_rgb_array,
        rerun_web_url,
        tile_rgb_batch,
    )

    single = as_uint8_rgb_array(np.ones((2, 3, 4), dtype=np.float32))
    assert single.shape == (1, 2, 3, 3)
    assert single.dtype == np.uint8
    assert single.max() == 255

    batch = np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)
    tiled = tile_rgb_batch(batch)
    assert tiled.shape == (4, 6, 3)
    assert np.array_equal(tiled[:2, :3], batch[0])
    assert np.array_equal(tiled[:2, 3:6], batch[1])
    assert np.array_equal(tiled[2:4, :3], batch[2])
    assert np.array_equal(tiled[2:4, 3:6], batch[3])

    assert rerun_web_url(9090, "rerun+http://127.0.0.1:9876/proxy") == (
        "http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy"
    )


def test_rerun_utils_import_is_side_effect_free_in_fresh_process() -> None:
    import json
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import json
        import sys

        import ioailab.utils.rerun_utils as rerun_utils

        print(json.dumps({
            "has_log": hasattr(rerun_utils, "log_rerun_rgb"),
            "has_tile": hasattr(rerun_utils, "tile_rgb_batch"),
            "has_uint8": hasattr(rerun_utils, "as_uint8_rgb_array"),
            "has_url": hasattr(rerun_utils, "rerun_web_url"),
            "rerun_loaded": "rerun" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
            "isaaclab_loaded": any(name == "isaaclab" or name.startswith("isaaclab.") for name in sys.modules),
        }))
        """
    )
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT / "src") if not old_pythonpath else f"{ROOT / 'src'}:{old_pythonpath}"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    data = json.loads(result.stdout.strip())

    assert data == {
        "has_log": True,
        "has_tile": True,
        "has_uint8": True,
        "has_url": True,
        "rerun_loaded": False,
        "torch_loaded": False,
        "isaaclab_loaded": False,
    }


def test_stale_motion_control_and_sensor_examples_are_removed() -> None:
    readme = read("README.md")
    index = read("docs/index.md")
    development = read("docs/development.md")
    examples_readme = read("docs/examples.md")
    sensors = read("docs/galbot_sensors.md")

    assert not (ROOT / "examples" / "motion_control").exists()
    assert not (ROOT / "examples" / "sensors").exists()
    assert not (ROOT / "examples" / "tutorials").exists()
    assert not (ROOT / "examples" / "basic").exists()
    assert not (ROOT / "examples" / "tasks").exists()
    assert "examples/02_mimic.py" in examples_readme
    all_docs = readme + examples_readme + index + development + sensors
    assert "python examples/sensors/g1_check_camera_render.py" not in all_docs


def test_tutorial_placeholders_expose_agent_interfaces_without_fake_runtime() -> None:
    teleop = read("examples/01_collect.py")
    full_task = read("examples/07_compound_task.py")
    custom = read("examples/05_custom_agent.py")

    assert "TeleopAgent.from_device" in teleop
    assert "dataset.drop()" in teleop
    assert "camera_mounts" not in teleop
    assert "camera_data" not in teleop
    assert "from ioailab.agents import" in full_task
    assert "PolicyAgent.from_checkpoint" in full_task
    assert "TaskFlowAgent" in full_task
    assert "class SinusoidAgent(BaseAgent)" in custom
    assert "def act(self" in custom


def test_docs_ship_an_mdbook_version_switcher() -> None:
    head = read("docs/theme/head.hbs")
    book_config = read("book.toml")
    build_script_path = ROOT / "scripts" / "build_versioned_docs.sh"
    build_script = read("scripts/build_versioned_docs.sh")
    makefile = read("Makefile")
    development = read("docs/development.md")

    # The switcher is injected via the mdBook theme head and reads the manifest.
    assert build_script_path.exists()
    assert "{{ path_to_root }}versions.json" in head
    assert ".menu-bar .left-buttons" in head
    assert 'theme = "docs/theme"' in book_config

    # The multi-version build writes the manifest and a root redirect, and only
    # publishes mdBook-era tags (0.0.1 used MkDocs and is excluded).
    import os

    assert os.access(build_script_path, os.X_OK)
    assert "versions.json" in build_script
    assert 'RELEASED_TAGS=("v1.0.0a0")' in build_script
    assert "MDBOOK_OUTPUT__HTML__SITE_URL" in build_script
    assert "git worktree add" in build_script
    assert "index.html" in build_script
    assert 'CURRENT_PATH="latest"' in build_script
    assert 'DEV_PATH="dev"' not in build_script

    # The workflow is exposed as a make target and documented.
    assert "docs-versions:" in makefile
    assert "build_versioned_docs.sh" in makefile
    assert "make docs-versions" in development
    assert "`latest`, plus each released tag" in development


def test_changelog_is_single_alpha_release_describing_current_features() -> None:
    changelog = read("CHANGELOG.md")
    version = read("src/ioailab/__init__.py")

    # Alpha entry plus current Unreleased changes, matching the package version.
    assert '__version__ = "1.0.0a1"' in version
    assert "## [1.0.0a1]" in changelog
    assert "## Unreleased" in changelog
    assert "## [0.1.0]" not in changelog
    # Describes the current feature surface.
    assert "make_env" in changelog
    assert "cuRobo v2" in changelog
    assert "robomimic Diffusion Policy" in changelog
    assert "GalbotG1-PickCube-v0" in changelog
    assert "TaskFlowAgent" in changelog


def test_changelog_and_docs_describe_pick_to_shelf_task_flow_surface() -> None:
    """Docs/changelog describe the coherent task-flow surface."""

    changelog = read("CHANGELOG.md")
    tasks_doc = read("docs/tasks.md")
    agents_doc = read("docs/agents.md")

    # Removed legacy policy task IDs must not be advertised anywhere.
    for removed_id in ("PickToShelf-PickPolicy", "PickToShelf-PlacePolicy"):
        assert removed_id not in changelog
        assert removed_id not in tasks_doc
        assert removed_id not in agents_doc

    # Removed runtime APIs must not be advertised as current features.
    assert "per-subtask MDP overlays" not in changelog
    assert "subtask init states" not in changelog
    for removed_api in ("SubtaskFlowAgent", "SubtaskMdpOverlay", "SubtaskInitState"):
        assert removed_api not in agents_doc

    # The current task-flow and component task surface is documented.
    assert "TaskFlowAgent" in tasks_doc
    assert "GalbotG1-PickToShelf-Pick-v0" in tasks_doc
    assert "GalbotG1-PickToShelf-Nav-v0" in tasks_doc
    assert "GalbotG1-PickToShelf-Place-v0" in tasks_doc
