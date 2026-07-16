from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _run_fresh_process(code: str) -> dict[str, Any]:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT / "src") if not old_pythonpath else f"{ROOT / 'src'}:{old_pythonpath}"
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    return json.loads(result.stdout.strip())


def test_curobo_planner_agent_exports_standard_base_agent_contract() -> None:
    from ioailab.agents import BaseAgent, CuroboPlannerAgent, PlannerAgent

    assert issubclass(CuroboPlannerAgent, PlannerAgent)
    assert issubclass(CuroboPlannerAgent, BaseAgent)
    assert (
        "CuroboPlannerAgent"
        in __import__("ioailab.agents", fromlist=["__all__"]).__all__
    )

    assert list(inspect.signature(CuroboPlannerAgent.reset).parameters)[:2] == [
        "self",
        "env",
    ]
    assert list(inspect.signature(CuroboPlannerAgent.act).parameters)[:2] == [
        "self",
        "env",
    ]
    assert list(inspect.signature(CuroboPlannerAgent.done).parameters)[:2] == [
        "self",
        "env",
    ]
    assert hasattr(CuroboPlannerAgent, "from_task")


def test_curobo_planner_agent_import_is_side_effect_free_in_fresh_process() -> None:
    data = _run_fresh_process(
        """
        import json
        import sys

        from ioailab.agents import BaseAgent, CuroboPlannerAgent, PlannerAgent

        print(json.dumps({
            "agent": CuroboPlannerAgent.__name__,
            "is_planner_agent": issubclass(CuroboPlannerAgent, PlannerAgent),
            "is_base_agent": issubclass(CuroboPlannerAgent, BaseAgent),
            "isaaclab_app_loaded": "isaaclab.app" in sys.modules,
            "gymnasium_loaded": "gymnasium" in sys.modules,
            "tasks_loaded": "ioailab.tasks" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
            "external_curobo_loaded": any(name == "curobo" or name.startswith("curobo.") for name in sys.modules),
            "rerun_loaded": "rerun" in sys.modules,
        }))
        """
    )

    assert data == {
        "agent": "CuroboPlannerAgent",
        "is_planner_agent": True,
        "is_base_agent": True,
        "isaaclab_app_loaded": False,
        "gymnasium_loaded": False,
        "tasks_loaded": False,
        "torch_loaded": False,
        "external_curobo_loaded": False,
        "rerun_loaded": False,
    }


def test_curobo_planner_agent_does_not_own_or_step_envs() -> None:
    import ioailab.agents as agents

    source = inspect.getsource(agents.CuroboPlannerAgent)
    forbidden_snippets = (
        "gym.make",
        "register_tasks(",
        ".step(",
        "env.step",
        "env_create",
        "ioailabEnv",
        "make_env(",
        "task_entry_for_task_id",
    )
    for snippet in forbidden_snippets:
        assert snippet not in source


class _FakeActionManager:
    total_action_dim = 3


class _FakeUnwrappedEnv:
    action_manager = _FakeActionManager()
    device = "cpu"
    num_envs = 2

    def __init__(self) -> None:
        self.step_calls = 0

    def step(self, action: Any) -> None:
        self.step_calls += 1
        raise AssertionError("CuroboPlannerAgent must not step environments")


class _FakeEnv:
    num_envs = 2

    def __init__(self) -> None:
        self.unwrapped = _FakeUnwrappedEnv()

    @property
    def step_calls(self) -> int:
        return self.unwrapped.step_calls


def test_curobo_planner_agent_reset_act_requires_task_metadata_without_stepping() -> (
    None
):
    from ioailab.agents import CuroboPlannerAgent

    env = _FakeEnv()
    agent = CuroboPlannerAgent()

    with pytest.raises(ValueError, match="planner metadata|motion-planning|task"):
        agent.reset(env)
    assert env.step_calls == 0

    with pytest.raises(RuntimeError, match="reset|planner metadata|action source"):
        agent.act(env)
    assert env.step_calls == 0


def test_registered_planner_tasks_have_final_ids_and_metadata() -> None:
    import ioailab.tasks as tasks

    task_ids = tasks.BUILTIN_TASK_IDS

    assert "GalbotG1-Reach-v0" in task_ids
    assert "GalbotG1-PickCube-v0" in task_ids
    assert "GalbotG1-StackCube-v0" in task_ids
    assert "GalbotG1-BaseNav-v0" in task_ids
    assert "GalbotG1DualArmsStackCube-v1" not in task_ids

    forbidden_ids = {
        "GalbotG1DualArmStackCube-v0",
        "GalbotG1WholeBodyStackCube-v0",
        "GalbotG1MobileStackCube-v0",
    }
    assert not forbidden_ids.intersection(task_ids)

    for task_id in task_ids:
        entry = tasks.task_entry_for_task_id(task_id)
        assert entry.env_cfg_entry_point
        if entry.motion_plan_factory is not None:
            # A plan bundles its own config behind one entry point.
            assert entry.motion_plan_factory().config is not None, task_id
        assert "compat" not in entry.env_cfg_entry_point.lower()


def test_task_config_modules_do_not_import_forbidden_runtime_or_fallback_apis() -> None:
    task_roots = [ROOT / "src" / "ioailab" / "tasks"]
    forbidden = (
        "env_create",
        "Agent.MOTION_PLAN",
        "from ioailab.tasks.common import Agent",
        "make_g1_curobo_motion_plan_agent",
        "fallback action source",
        "compatibility alias",
        "env.step(",
    )

    for task_root in task_roots:
        for path in task_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in source, (
                    f"{token!r} found in {path.relative_to(ROOT)}"
                )


def test_motion_plan_surface_has_one_vocabulary_and_one_entry_point_grammar() -> None:
    """The retired motion-plan surfaces must not reappear anywhere in src/."""

    retired_tokens = (
        # Old single target type, replaced by WorldTarget/AssetRelativeTarget.
        "MotionTarget",
        # Old dual entry-point field, collapsed into the plan's bundled config.
        "motion_planning_cfg_entry_point",
        # Old magic YAML entry-point dialect, replaced by module:object factories.
        "yaml:",
        # Old multi-resolver registry surface.
        "planner_metadata_for_task",
        "PlannerMetadata",
    )
    for path in (ROOT / "src" / "ioailab").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in retired_tokens:
            assert token not in source, f"{token!r} found in {path.relative_to(ROOT)}"


def test_sensor_docs_delegate_rerun_helpers_after_example_fold() -> None:
    source = read("docs/galbot_sensors.md")

    assert "ioailab.utils.rerun_utils" in source
    assert '.data.output["rgb"]' in source
    assert "distance_to_image_plane" in source


def test_docs_and_collect_example_use_make_env_agent_save() -> None:
    checked_paths = (
        "README.md",
        "docs/architecture.md",
        "docs/agents.md",
        "examples/01_collect.py",
    )
    combined = "\n".join(read(path) for path in checked_paths)

    assert "from ioailab.envs import make_env" in combined
    assert "make_env(" in combined
    assert "CuroboPlannerAgent.from_task" in combined
    assert "env.collect(" in combined

    # No resurrected legacy/compatibility agent-mode surface.
    for token in (
        "env_create",
        "Agent.MOTION_PLAN",
        "env.set_agent(",
        "compatibility alias",
    ):
        assert token not in combined


def test_smoke_diagnostics_script_documents_required_runtime_commands() -> None:
    smoke = read("tests/smoke_g1_planner.py")

    assert "python -m pytest" in smoke
    assert "python -m ruff check ." in smoke
    assert "python -m ty check src examples tests" in smoke
    assert "--warn all --quiet" in smoke
    assert "examples/01_basic_control.py --headless --num-envs 1" in smoke
    assert "failure_category" in smoke
    assert "action_tensor_shape" in smoke
    assert "CuroboPlannerAgent.from_task" in smoke
