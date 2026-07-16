"""Tests for PickToShelf task-flow and component task registry."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_pick_to_shelf_registers_coherent_and_phase_task_ids():
    import ioailab.tasks as tasks
    from ioailab.tasks.pick_to_shelf import GALBOT_G1_PICK_TO_SHELF_TASK

    expected = {
        "GalbotG1-PickToShelf-v0",
        "GalbotG1-PickToShelf-Pick-v0",
        "GalbotG1-PickToShelf-Nav-v0",
        "GalbotG1-PickToShelf-Place-v0",
    }
    assert expected.issubset(set(tasks.BUILTIN_TASK_IDS))
    assert GALBOT_G1_PICK_TO_SHELF_TASK.task_flow_entry_point is not None
    assert not hasattr(GALBOT_G1_PICK_TO_SHELF_TASK, "subtask_resolver_entry_point")
    assert not hasattr(GALBOT_G1_PICK_TO_SHELF_TASK, "subtask_names")


def test_task_flow_for_task_returns_ordered_metadata():
    from ioailab.agents import TaskFlowSpec
    from ioailab.tasks import task_flow_for_task
    from ioailab.tasks.pick_to_shelf.config.g1.env_cfg import (
        GalbotG1PickToShelfEnvCfg,
    )

    flow = task_flow_for_task("GalbotG1-PickToShelf-v0")

    assert isinstance(flow, TaskFlowSpec)
    assert flow is GalbotG1PickToShelfEnvCfg.task_flow
    assert flow.phase_names == ("pick", "nav", "place")
    assert flow.phase("pick").phase_task_id == "GalbotG1-PickToShelf-Pick-v0"
    assert flow.phase("nav").phase_task_id == "GalbotG1-PickToShelf-Nav-v0"
    assert flow.phase("place").phase_task_id == "GalbotG1-PickToShelf-Place-v0"


def test_combined_task_public_api_supports_simple_phase_task_list():
    pytest.importorskip("isaaclab")

    from ioailab.tasks.common.composition import (
        PhaseTask,
        combined_task,
        combined_task_definition,
        phase,
        task_sequence,
    )

    explicit = phase("pick", "GalbotG1-PickToShelf-Pick-v0", action_terms=("arm",))

    assert isinstance(explicit, PhaseTask)
    assert explicit.name == "pick"
    assert explicit.task_id == "GalbotG1-PickToShelf-Pick-v0"
    assert explicit.action_terms == ("arm",)

    env_cfg_cls = combined_task(
        name="ExampleEnvCfg",
        task_id="Example-Combined-v0",
        phases=task_sequence(
            "GalbotG1-PickToShelf-Pick-v0",
            "GalbotG1-PickToShelf-Nav-v0",
            "GalbotG1-PickToShelf-Place-v0",
        ),
        module=__name__,
    )
    definition = combined_task_definition(env_cfg_cls)

    assert definition.task_spec.task_id == "Example-Combined-v0"
    assert definition.task_flow.phase_names == ("pick", "nav", "place")
    assert definition.task_spec.env_cfg_entry_point == f"{__name__}:ExampleEnvCfg"
    assert definition.task_spec.task_flow_entry_point == (
        f"{__name__}:ExampleEnvCfg.task_flow"
    )
    assert hasattr(env_cfg_cls().terminations, "cube_on_shelf")
    assert hasattr(env_cfg_cls().events, "reset_task_phase")


def test_task_flow_spec_agent_provider_supports_object_and_factory_forms():
    from ioailab.agents import BaseAgent, TaskFlowAgent, TaskFlowSpec, TaskPhaseSpec

    class Agent(BaseAgent):
        def act(self, env, env_ids=None):
            return []

    object_agent = Agent()
    factory_agent = Agent()
    flow = TaskFlowSpec(
        phases=(
            TaskPhaseSpec(
                name="pick",
                phase_task_id="pick-task",
                default_agent=object_agent,
            ),
            TaskPhaseSpec(
                name="place",
                phase_task_id="place-task",
                default_agent=lambda _env: factory_agent,
            ),
        ),
        final_phase="place",
    )
    env = _TinyEnv()

    agent = TaskFlowAgent(flow, env=env)

    assert agent.phase_agent("pick") is object_agent
    assert agent.phase_agent("place") is factory_agent
    assert flow.phase("pick").agent is object_agent


def test_phase_tasks_resolve_default_expert_agents():
    pytest.importorskip("torch")

    from ioailab.agents import CuroboPlannerAgent, TrajectoryNavAgent

    pick_agent = CuroboPlannerAgent.from_task("GalbotG1-PickToShelf-Pick-v0")
    nav_agent = TrajectoryNavAgent.from_task("GalbotG1-PickToShelf-Nav-v0")
    place_agent = CuroboPlannerAgent.from_task("GalbotG1-PickToShelf-Place-v0")

    assert type(pick_agent).__name__ == "CuroboPlannerAgent"
    assert type(nav_agent).__name__ == "TrajectoryNavAgent"
    assert type(place_agent).__name__ == "CuroboPlannerAgent"


def test_pick_to_shelf_phase_motion_plans_are_direct_task_plans():
    from ioailab.tasks import motion_plan_for_task
    from ioailab.tasks.pick_to_shelf_pick.motion_plan import (
        PickToShelfPickMotionPlanningCfg,
    )
    from ioailab.tasks.pick_to_shelf_place.motion_plan import (
        PickToShelfPlaceMotionPlanningCfg,
    )

    pick_plan = motion_plan_for_task("GalbotG1-PickToShelf-Pick-v0")
    place_plan = motion_plan_for_task("GalbotG1-PickToShelf-Place-v0")

    assert isinstance(pick_plan.config, PickToShelfPickMotionPlanningCfg)
    assert isinstance(place_plan.config, PickToShelfPlaceMotionPlanningCfg)
    with pytest.raises(ValueError, match="motion plan"):
        motion_plan_for_task("GalbotG1-PickToShelf-Nav-v0")


def test_pick_to_shelf_has_no_deleted_subtask_source_package():
    assert not (
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf" / "subtasks"
    ).exists()


def test_pick_to_shelf_phase_packages_do_not_import_coherent_cfg_or_mdp():
    phase_roots = (
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf_pick",
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf_nav",
        ROOT / "src" / "ioailab" / "tasks" / "pick_to_shelf_place",
    )
    forbidden = (
        "ioailab.tasks.pick_to_shelf.config.g1",
        "ioailab.tasks.pick_to_shelf.mdp",
    )
    offenders: list[str] = []
    for root in phase_roots:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(ROOT)} imports {token}")
    assert offenders == []


class _TinyEnv:
    task_id = "tiny"
    num_envs = 1
    unwrapped = None

    def __init__(self) -> None:
        self.unwrapped = self
        self.phases = ["pick"]

    def current_task_phases(self, env_ids=None):
        return tuple(self.phases)

    def set_task_phases(self, *, env_ids=None, phase="pick"):
        self.phases = [str(phase)]
