from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from ioailab.datasets import DatasetRef

ROOT = Path(__file__).resolve().parents[1]


def load_example(name: str):
    path = ROOT / "examples" / name
    spec = importlib.util.spec_from_file_location(f"_ioailab_example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeEnv:
    def __init__(self) -> None:
        self.closed = False
        self.collect_calls = []
        self.evaluate_calls = []
        self.reset_calls = 0
        self.step_calls = []
        self.num_envs = 1

    def collect(self, **kwargs):
        self.collect_calls.append(kwargs)
        return DatasetRef(kwargs["path"], task_id="Task-v0")

    def evaluate(self, **kwargs):
        self.evaluate_calls.append(kwargs)
        return {
            "success_rate": 1.0,
            "success_count": 12,
            "steps": 12,
            "total_episodes": 12,
        }

    def reset(self):
        self.reset_calls += 1
        return {}, {}

    def step(self, action):
        self.step_calls.append(action)
        return {}, 0.0, False, False, {}

    def close(self):
        self.closed = True


class FakePlanner:
    calls = []

    @staticmethod
    def from_task(task, *, task_options=None):
        FakePlanner.calls.append((task, task_options))
        return FakePlannerAgent(task)


class FakeTrajectoryNavAgent:
    calls = []

    @staticmethod
    def from_task(task):
        FakeTrajectoryNavAgent.calls.append(task)
        return FakePlannerAgent(f"nav:{task}")


class FakePlannerAgent:
    def __init__(self, task):
        self.task = task
        self.actions = 0

    def reset(self, env):
        del env
        self.actions = 0

    def done(self, env):
        return [self.actions >= 1] * int(env.num_envs)

    def act(self, env):
        del env
        self.actions += 1
        return f"planner-action:{self.task}:{self.actions}"


class FakePolicy:
    def train(self, dataset, train_cfg):
        del dataset, train_cfg
        return SimpleNamespace(path="ckpt.pth", backend="fake", metadata={})

    def load_checkpoint(self, checkpoint):
        return f"policy:{checkpoint}"


class FakePolicyFactory:
    @staticmethod
    def from_backend(backend):
        assert backend == "robomimic_diffusion"
        return FakePolicy()


class FakePolicyAgent:
    @staticmethod
    def from_checkpoint(checkpoint):
        return f"policy:{checkpoint}"


def test_pipeline_examples_are_compact_root_numbered_files() -> None:
    expected = {
        "01_collect.py",
        "02_mimic.py",
        "03_train.py",
        "04_eval.py",
        "05_custom_agent.py",
        "06_collect_component_task.py",
        "07_compound_task.py",
    }

    assert {path.name for path in (ROOT / "examples").glob("*.py")} == expected
    assert not (ROOT / "src" / "ioailab" / "workflows.py").exists()


def test_parallel_example_defaults_are_shared() -> None:
    for name in (
        "02_mimic.py",
        "04_eval.py",
        "05_custom_agent.py",
        "07_compound_task.py",
    ):
        source = (ROOT / "examples" / name).read_text(encoding="utf-8")

        assert '"--episodes"' in source
        assert '"--episodes-per-env"' not in source
        assert "default=36" in source
        assert '"--num-envs"' in source
        assert "default=9" in source


def test_collect_example_uses_motion_planner_collect(monkeypatch) -> None:
    module = load_example("01_collect.py")
    env = FakeEnv()
    make_env_calls = []
    FakePlanner.calls = []

    def fake_make_env(*args, **kwargs):
        make_env_calls.append((args, kwargs))
        env.num_envs = int(kwargs["num_envs"])
        return env

    monkeypatch.setattr(module, "make_env", fake_make_env)
    monkeypatch.setattr(module, "CuroboPlannerAgent", FakePlanner)

    module.main(["--episodes", "1"])

    assert FakePlanner.calls == [("GalbotG1-PickCube-v0", None)]
    assert make_env_calls == [
        (("GalbotG1-PickCube-v0",), {"num_envs": 1, "headless": False})
    ]
    call = env.collect_calls[0]
    assert isinstance(call["agent"], FakePlannerAgent)
    assert call["path"] == "data/pick_cube_demos.hdf5"
    assert call["episodes"] == 1
    assert call["max_steps"] == 1000
    assert env.closed


def test_collect_example_threads_task_options_and_scenarios(
    monkeypatch, tmp_path
) -> None:
    module = load_example("01_collect.py")
    env = FakeEnv()
    make_env_calls = []
    FakePlanner.calls = []

    def fake_make_env(*args, **kwargs):
        make_env_calls.append((args, kwargs))
        return env

    monkeypatch.setattr(module, "make_env", fake_make_env)
    monkeypatch.setattr(module, "CuroboPlannerAgent", FakePlanner)

    module.main(
        [
            "--task",
            "GalbotG1-PickToShelf-Pick-v0",
            "--init-scenario",
            str(tmp_path / "pick_start.yaml"),
            "--save-end-scenario",
            str(tmp_path / "nav_start.yaml"),
        ]
    )

    task_options = {"init_scenario": str(tmp_path / "pick_start.yaml")}
    assert FakePlanner.calls == [("GalbotG1-PickToShelf-Pick-v0", None)]
    assert make_env_calls == [
        (
            ("GalbotG1-PickToShelf-Pick-v0",),
            {"num_envs": 1, "headless": False, "task_options": task_options},
        )
    ]
    assert env.collect_calls[0]["save_end_scenario"] == str(tmp_path / "nav_start.yaml")


def test_component_collect_example_supports_sort_to_shelf_motion_plan_presets() -> None:
    source = (ROOT / "examples" / "06_collect_component_task.py").read_text(
        encoding="utf-8"
    )

    assert '# COMPONENT_PRESET = "sort_to_shelf_pick"' in source
    assert '# COMPONENT_PRESET = "sort_to_shelf_place"' in source
    assert '"--sorting-object"' in source
    assert '"red_cube"' in source
    assert 'sort_options = {"sorting_object": args.sorting_object}' in source
    assert "task_options = {**task_options, **sort_options}" in source
    assert "CuroboPlannerAgent.from_task(task_id, task_options=sort_options)" in source


def test_component_collect_example_supports_sort_to_shelf_nav_sequence() -> None:
    source = (ROOT / "examples" / "06_collect_component_task.py").read_text(
        encoding="utf-8"
    )

    assert '# COMPONENT_PRESET = "sort_to_shelf_nav"' in source
    assert (
        "from ioailab.tasks.sort_to_shelf_nav.agent import nav_sequence_agent" in source
    )
    assert 'task_id = "GalbotG1-SortToShelf-Nav-v0"' in source
    assert "nav_sequence_agent(sorting_object=args.sorting_object)" in source


def test_collect_example_runs_motion_plan_and_comments_teleop_swap() -> None:
    source = (ROOT / "examples" / "01_collect.py").read_text(encoding="utf-8")

    assert "from ioailab.agents import CuroboPlannerAgent" in source
    assert 'if task_id.startswith("GalbotG1-SortToShelf-")' not in source
    assert "# from ioailab.agents import TeleopAgent" in source
    assert 'default="GalbotG1-PickCube-v0"' in source
    assert "dataset = env.collect(" in source
    assert "agent=agent" in source
    assert "--init-scenario" in source
    assert "--save-end-scenario" in source
    assert "--sorting-object" not in source
    assert '# agent = TeleopAgent.from_device("gp001", task=task_id)' in source
    assert "#     decision = agent.review_demo()" in source
    assert "#         dataset.drop()" in source
    assert "run_candidate" not in source


def test_mimic_example_uses_dataset_task_metadata(monkeypatch) -> None:
    module = load_example("02_mimic.py")
    calls = []

    def fake_mimic(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return DatasetRef(kwargs["output_path"], task_id=dataset.task_id)

    monkeypatch.setattr(module, "mimic", fake_mimic)
    module.main(
        [
            "--task",
            "Task-v0",
            "--dataset-path",
            "data/demo.hdf5",
            "--episodes",
            "40",
            "--num-envs",
            "10",
        ]
    )

    dataset, kwargs = calls[0]
    assert dataset.path == Path("data/demo.hdf5")
    assert dataset.task_id == "Task-v0"
    assert kwargs["episodes"] == 40
    assert kwargs["num_envs"] == 10
    assert kwargs["headless"] is False
    assert "task" not in kwargs


def test_train_example_keeps_training_in_policy_layer(monkeypatch) -> None:
    module = load_example("03_train.py")
    calls = []

    class CapturingPolicy:
        def train(self, dataset, train_cfg):
            calls.append((dataset, train_cfg))
            return SimpleNamespace(path="ckpt.pth", backend="fake", metadata={})

    class CapturingPolicyFactory:
        @staticmethod
        def from_backend(backend):
            assert backend == "robomimic_diffusion"
            return CapturingPolicy()

    monkeypatch.setattr(module, "Policy", CapturingPolicyFactory)

    module.main(
        [
            "--task",
            "Task-v0",
            "--dataset-path",
            "data/mimic.hdf5",
            "--epochs",
            "2",
            "--num-data-workers",
            "3",
        ]
    )

    dataset, train_cfg = calls[0]
    assert dataset.path == Path("data/mimic.hdf5")
    assert dataset.task_id == "Task-v0"
    assert train_cfg.output_dir == Path("outputs/pick_cube")
    assert train_cfg.epochs == 2
    assert train_cfg.num_data_workers == 3
    assert train_cfg.optimizer.learning_rate == 1.0e-4


def test_eval_example_turns_checkpoint_into_agent(monkeypatch) -> None:
    module = load_example("04_eval.py")
    env = FakeEnv()
    make_env_calls = []

    def fake_make_env(*args, **kwargs):
        make_env_calls.append((args, kwargs))
        return env

    monkeypatch.setattr(module, "make_env", fake_make_env)
    monkeypatch.setattr(module, "Policy", FakePolicyFactory)

    module.main(
        [
            "--task",
            "Task-v0",
            "--checkpoint",
            "ckpt.pth",
            "--episodes",
            "5",
            "--num-envs",
            "3",
        ]
    )

    assert make_env_calls == [(("Task-v0",), {"num_envs": 3, "headless": False})]
    assert env.evaluate_calls == [
        {"agent": "policy:ckpt.pth", "episodes": 5, "max_steps": 1000}
    ]
    assert env.closed


def test_custom_agent_example_exposes_episode_and_frequency_args(monkeypatch) -> None:
    module = load_example("05_custom_agent.py")
    env = FakeEnv()
    make_env_calls = []

    def fake_make_env(*args, **kwargs):
        make_env_calls.append((args, kwargs))
        return env

    monkeypatch.setattr(module, "make_env", fake_make_env)

    module.main(
        [
            "--task",
            "Task-v0",
            "--num-envs",
            "4",
            "--episodes",
            "12",
            "--max-steps",
            "7",
            "--frequency",
            "2.5",
            "--headless",
        ]
    )

    assert make_env_calls == [(("Task-v0",), {"num_envs": 4, "headless": True})]
    call = env.evaluate_calls[0]
    assert call["episodes"] == 12
    assert call["max_steps"] == 7
    assert call["agent"]._frequency == 2.5
    assert env.closed


def test_custom_agent_returns_one_action_per_requested_row() -> None:
    module = load_example("05_custom_agent.py")
    env = SimpleNamespace(
        action_space=SimpleNamespace(shape=(2,)),
        device="cpu",
        num_envs=3,
    )
    agent = module.SinusoidAgent(frequency=1.0)

    full_action = agent.act(env)
    subset_action = agent.act(env, env_ids=(1,))

    assert full_action.shape == (3, 2)
    assert subset_action.shape == (1, 2)
    assert torch.allclose(full_action[0], subset_action[0])


def test_compound_task_example_evaluates_default_policy(monkeypatch) -> None:
    module = load_example("07_compound_task.py")
    env = FakeEnv()
    make_env_calls = []

    def fake_make_env(*args, **kwargs):
        make_env_calls.append((args, kwargs))
        return env

    class FakeTaskFlowAgent:
        @staticmethod
        def from_env(agent_env, *, agents=None):
            return SimpleNamespace(env=agent_env, agents=agents)

    monkeypatch.setattr(module, "make_env", fake_make_env)
    monkeypatch.setattr(module, "TaskFlowAgent", FakeTaskFlowAgent)

    module.main(
        [
            "--task",
            "GalbotG1-PickToShelf-v0",
            "--episodes",
            "5",
            "--num-envs",
            "3",
            "--max-steps",
            "8",
        ]
    )

    assert make_env_calls == [
        (("GalbotG1-PickToShelf-v0",), {"num_envs": 3, "headless": False})
    ]
    call = env.evaluate_calls[0]
    assert call["episodes"] == 5
    assert call["max_steps"] == 8
    assert call["agent"].env is env
    assert set(call["agent"].agents) == {"pick", "place"}
    assert env.closed


def test_compound_task_example_collect_mode(monkeypatch) -> None:
    module = load_example("07_compound_task.py")
    env = FakeEnv()

    class FakeTaskFlowAgent:
        @staticmethod
        def from_env(agent_env, *, agents=None):
            return SimpleNamespace(env=agent_env, agents=agents)

    monkeypatch.setattr(module, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(module, "TaskFlowAgent", FakeTaskFlowAgent)

    module.main(
        [
            "--mode",
            "collect",
            "--dataset-path",
            "full.hdf5",
            "--episodes",
            "2",
        ]
    )

    call = env.collect_calls[0]
    assert call["path"] == "full.hdf5"
    assert call["episodes"] == 2
    assert call["max_steps"] == 1500
    assert call["metadata"] == {
        "collection": "compound_task",
        "task": "GalbotG1-PickToShelf-v0",
    }


def test_compound_task_example_threads_sorting_options(monkeypatch) -> None:
    module = load_example("07_compound_task.py")
    monkeypatch.setattr(module, "COMPOUND_AGENT_PRESET", "task_default")
    env = FakeEnv()
    make_env_calls = []

    def fake_make_env(*args, **kwargs):
        make_env_calls.append((args, kwargs))
        return env

    class FakeTaskFlowAgent:
        @staticmethod
        def from_env(agent_env, *, agents=None):
            return SimpleNamespace(env=agent_env, agents=agents)

    monkeypatch.setattr(module, "make_env", fake_make_env)
    monkeypatch.setattr(module, "TaskFlowAgent", FakeTaskFlowAgent)

    module.main(
        [
            "--task",
            "GalbotG1-SortToShelf-v0",
            "--sorting-object",
            "blue_cuboid",
            "--episodes",
            "2",
        ]
    )

    assert make_env_calls == [
        (
            ("GalbotG1-SortToShelf-v0",),
            {
                "num_envs": 9,
                "headless": False,
                "task_options": {"sorting_object": "blue_cuboid"},
            },
        )
    ]
    assert env.evaluate_calls[0]["agent"].agents is None


def test_compound_task_example_comments_phase_agent_overrides() -> None:
    source = (ROOT / "examples" / "07_compound_task.py").read_text(encoding="utf-8")

    assert "COMPOUND_AGENT_PRESET" in source
    assert '"task_default"' in source
    assert '"pick_to_shelf_experts"' in source
    assert '"pick_to_shelf_policy"' in source
    assert "phase_agents = {" in source
    assert "agent = TaskFlowAgent.from_env(env, agents=phase_agents)" in source
    assert '"nav": TrajectoryNavAgent.from_task' in source
    assert "from ioailab.agents import G1ManipulationPolicyActionAdapter" in source
