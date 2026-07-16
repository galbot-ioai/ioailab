from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import inspect
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeApp:
    def __init__(self) -> None:
        self.closed = False
        self.updated = 0

    def is_running(self) -> bool:
        return True

    def update(self) -> None:
        self.updated += 1

    def close(self) -> None:
        self.closed = True


class FakeRawEnv:
    def __init__(self, *, num_envs: int, device: str = "cuda:0") -> None:
        self.unwrapped = self
        self.num_envs = num_envs
        self.device = device
        self.scene = {"robot": object()}
        self.action_space = object()
        self.reset_calls = []
        self.step_calls = []
        self.render_calls = 0
        self.closed = False
        self.cfg = SimpleNamespace(recorders=None)
        self.recorder_manager = None
        self.observation_manager = None
        self.step_results: list[Any] = []

    def reset(
        self, *args: Any, **kwargs: Any
    ) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        self.reset_calls.append((args, kwargs))
        return "reset", args, kwargs

    def step(self, action: Any) -> Any:
        self.step_calls.append(action)
        if self.step_results:
            return self.step_results.pop(0)
        # Default to an IsaacLab-shaped no-op 5-tuple so loops advance to max_steps.
        return {}, 0.0, False, False, {}

    def render(self) -> str:
        self.render_calls += 1
        return "rendered"

    def close(self) -> None:
        self.closed = True


class FakeGymWrapper:
    """Minimal Gymnasium wrapper that keeps IsaacLab env behind ``unwrapped``."""

    def __init__(self, env: FakeRawEnv) -> None:
        self.unwrapped = env
        self.reset_calls = []

    @property
    def num_envs(self) -> int:
        return self.unwrapped.num_envs

    @num_envs.setter
    def num_envs(self, value: int) -> None:
        self.unwrapped.num_envs = value

    @property
    def device(self) -> str:
        return self.unwrapped.device

    @property
    def scene(self) -> Any:
        return self.unwrapped.scene

    @property
    def action_space(self) -> Any:
        return self.unwrapped.action_space

    @property
    def cfg(self) -> Any:
        return self.unwrapped.cfg

    @cfg.setter
    def cfg(self, value: Any) -> None:
        self.unwrapped.cfg = value

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if "env_ids" in kwargs:
            raise TypeError(
                "OrderEnforcing.reset() got an unexpected keyword argument 'env_ids'"
            )
        self.reset_calls.append((args, kwargs))
        return self.unwrapped.reset(*args, **kwargs)

    def step(self, action: Any) -> Any:
        return self.unwrapped.step(action)

    def render(self) -> str:
        return self.unwrapped.render()

    def close(self) -> None:
        self.unwrapped.close()


def _patch_make_env_runtime(
    monkeypatch: pytest.MonkeyPatch, *, raw_env: Any | None = None
) -> dict[str, Any]:
    import ioailab.envs._factory as factory_module

    calls: dict[str, Any] = {}
    fake_app = FakeApp()
    fake_raw_env = raw_env or FakeRawEnv(num_envs=4)
    fake_env_cfg = SimpleNamespace(
        task_id=None, num_envs=None, recorders=fake_raw_env.cfg.recorders
    )

    def build(
        task_id: str,
        num_envs: int,
        options: dict[str, Any],
    ) -> tuple[FakeApp, SimpleNamespace]:
        calls["build"] = {
            "task_id": task_id,
            "num_envs": num_envs,
            "options": dict(options),
        }
        fake_raw_env.num_envs = num_envs
        fake_env_cfg.task_id = task_id
        fake_env_cfg.num_envs = num_envs
        fake_env_cfg.recorders = fake_raw_env.cfg.recorders
        return fake_app, fake_env_cfg

    def make_raw(task_id: str, env_cfg: SimpleNamespace) -> FakeRawEnv:
        calls["make_raw"] = {"task_id": task_id, "env_cfg": env_cfg}
        fake_raw_env.cfg = env_cfg
        return fake_raw_env

    monkeypatch.setattr(factory_module, "build_isaaclab_app_and_cfg", build)
    monkeypatch.setattr(factory_module, "make_gym_env", make_raw)
    calls["app"] = fake_app
    calls["raw_env"] = fake_raw_env
    calls["env_cfg"] = fake_env_cfg
    return calls


def test_make_env_returns_direct_workflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from ioailab.envs import ioailabEnv, make_env

    calls = _patch_make_env_runtime(monkeypatch)

    env = make_env("GalbotG1-PickCube-v0", num_envs=4, device="cuda:0", headless=False)

    assert isinstance(env, ioailabEnv)
    assert env.task_id == "GalbotG1-PickCube-v0"
    assert "env_id" not in ioailabEnv.__dataclass_fields__
    assert not hasattr(env, "env_id")
    assert env.num_envs == 4
    assert env.options == {"device": "cuda:0", "headless": False}
    assert calls["build"] == {
        "task_id": "GalbotG1-PickCube-v0",
        "num_envs": 4,
        "options": {"device": "cuda:0", "headless": False},
    }
    assert calls["make_raw"] == {
        "task_id": "GalbotG1-PickCube-v0",
        "env_cfg": calls["env_cfg"],
    }
    assert calls["env_cfg"].task_id == "GalbotG1-PickCube-v0"
    assert calls["env_cfg"].num_envs == 4
    assert hasattr(env, "step")
    assert hasattr(env, "collect")
    assert hasattr(env, "evaluate")
    assert hasattr(env, "get_scenario")
    assert not hasattr(env, "get_snapshot")
    assert not hasattr(env, "set_snapshot")
    assert not hasattr(env, "replay")
    legacy_collect_name = "collect" + "_data"
    assert not hasattr(env, legacy_collect_name)


def test_collect_api_exposes_compact_public_options() -> None:
    from ioailab.envs import ioailabEnv

    parameters = inspect.signature(ioailabEnv.collect).parameters
    assert "init_snapshot" not in parameters
    assert "end_scenario_path" not in parameters
    assert "end_scenario_name" not in parameters
    assert "save_end_scenario" in parameters


def test_make_env_validates_static_inputs() -> None:
    from ioailab.envs import make_env

    with pytest.raises(ValueError, match="task_id"):
        make_env("")
    with pytest.raises(ValueError, match="num_envs"):
        make_env("GalbotG1-PickCube-v0", num_envs=0)
    with pytest.raises(ValueError, match="Unknown make_env option"):
        make_env("GalbotG1-PickCube-v0", unsupported=True)
    for stale_option in (
        "camera_mounts",
        "camera_data",
        "camera_width",
        "camera_height",
    ):
        with pytest.raises(ValueError, match="Unknown make_env option"):
            make_env("GalbotG1-PickCube-v0", **{stale_option: "stale"})


def test_make_env_task_options_are_allowed_and_delegated() -> None:
    from ioailab.envs._factory import (
        configure_task_options,
        validate_make_env_options,
    )

    calls = []

    class Cfg:
        def apply_task_options(self, options):
            calls.append(options)
            options["mutated"] = True

    task_options = {"sorting_object": "green_cylinder"}
    validate_make_env_options({"task_options": task_options})
    configure_task_options(Cfg(), {"task_options": task_options})

    assert calls == [{"sorting_object": "green_cylinder", "mutated": True}]
    assert task_options == {"sorting_object": "green_cylinder"}


def test_ioailab_env_task_options_returns_copy() -> None:
    from ioailab.envs.env import ioailabEnv

    env = ioailabEnv(
        task_id="GalbotG1-SortToShelf-v0",
        raw_env=object(),
        app=None,
        num_envs=1,
        options={"task_options": {"sorting_object": "green_cylinder"}},
    )

    task_options = env.task_options
    task_options["sorting_object"] = "red_cube"

    assert env.task_options == {"sorting_object": "green_cylinder"}
    assert (
        ioailabEnv(
            task_id="GalbotG1-PickCube-v0",
            raw_env=object(),
            app=None,
            num_envs=1,
        ).task_options
        == {}
    )


def test_make_env_task_options_reject_invalid_targets() -> None:
    from ioailab.envs._factory import configure_task_options

    with pytest.raises(ValueError, match="must be a mapping"):
        configure_task_options(object(), {"task_options": "green_cylinder"})
    with pytest.raises(ValueError, match="does not accept task_options"):
        configure_task_options(
            object(), {"task_options": {"sorting_object": "red_cube"}}
        )


def test_make_env_app_kwargs_use_task_camera_metadata() -> None:
    from ioailab.envs._factory import make_app_kwargs

    assert make_app_kwargs({}, requires_cameras=True)["enable_cameras"] is True
    assert (
        make_app_kwargs({"enable_cameras": True}, requires_cameras=True)[
            "enable_cameras"
        ]
        is True
    )
    with pytest.raises(ValueError, match="requires cameras"):
        make_app_kwargs({"enable_cameras": False}, requires_cameras=True)
    assert "enable_cameras" not in make_app_kwargs({}, requires_cameras=False)


def test_pick_cube_task_requests_cameras_by_default() -> None:
    from ioailab.envs._factory import make_app_kwargs
    from ioailab.tasks import task_entry_for_task_id

    task_entry = task_entry_for_task_id("GalbotG1-PickCube-v0")

    assert task_entry.requires_cameras is True
    assert (
        make_app_kwargs({}, requires_cameras=task_entry.requires_cameras)[
            "enable_cameras"
        ]
        is True
    )
    assert (
        make_app_kwargs(
            {"headless": True}, requires_cameras=task_entry.requires_cameras
        )["enable_cameras"]
        is True
    )
    with pytest.raises(ValueError, match="requires cameras"):
        make_app_kwargs(
            {"enable_cameras": False}, requires_cameras=task_entry.requires_cameras
        )


def test_env_does_not_own_agent_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from ioailab.envs import make_env

    _patch_make_env_runtime(monkeypatch, raw_env=FakeRawEnv(num_envs=2))
    env = make_env("GalbotG1-PickCube-v0", num_envs=2)

    assert not hasattr(env, "set_agent")
    assert not hasattr(env, "agent")


def test_direct_env_delegates_live_env_reset_step_render_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.envs import make_env

    raw_env = FakeRawEnv(num_envs=2)
    calls = _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)

    raw_env.step_results = [("obs", 0.5, False, True, {"k": 1})]
    assert env.reset() == ("reset", (), {})
    assert env.step("action") == ("obs", 0.5, False, True, {"k": 1})
    assert env.render() == "rendered"
    assert raw_env.step_calls == ["action"]
    assert raw_env.render_calls == 1
    assert calls["app"].updated == 1
    assert env.scene is raw_env.scene
    assert env.is_running() is True

    env.close()
    assert raw_env.closed is True
    assert calls["app"].closed is True


def test_direct_env_reset_env_ids_uses_isaaclab_row_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.envs import make_env

    isaac_env = FakeRawEnv(num_envs=2)
    raw_env = FakeGymWrapper(isaac_env)
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)

    env.reset(env_ids=(1,))

    assert raw_env.reset_calls == []
    assert tuple(int(env_id) for env_id in isaac_env.reset_calls[0][1]["env_ids"]) == (
        1,
    )
    assert isaac_env.reset_calls[0][1]["env_ids"].shape == (1,)


def test_workflow_env_does_not_wrap_step_calls() -> None:
    from ioailab.envs import ioailabEnv

    source = inspect.getsource(ioailabEnv)

    assert "def step(" not in source
    assert "raw_env.step(action)" in source


def test_envs_import_is_runtime_lazy_in_fresh_process() -> None:
    code = textwrap.dedent(
        """
        import json
        import sys

        from ioailab.envs import ioailabEnv, make_env

        print(json.dumps({
            "exports": [ioailabEnv.__name__, make_env.__name__],
            "isaaclab_app_loaded": "isaaclab.app" in sys.modules,
            "gymnasium_loaded": "gymnasium" in sys.modules,
            "tasks_loaded": "ioailab.tasks" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
            "curobo_loaded": "ioailab.agents.motion_plan.contracts.g1_curobov2" in sys.modules,
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
        cwd=ROOT,
        env=env,
    )
    data = json.loads(result.stdout.strip())

    assert data == {
        "exports": ["ioailabEnv", "make_env"],
        "isaaclab_app_loaded": False,
        "gymnasium_loaded": False,
        "tasks_loaded": False,
        "torch_loaded": False,
        "curobo_loaded": False,
    }


def test_envs_module_exports_only_make_env_factory() -> None:
    import ioailab.envs as envs

    assert envs.__all__ == [
        "ioailabEnv",
        "make_env",
    ]
    assert "make_env" in envs.__all__
    legacy_factory_name = "env" + "_create"
    assert legacy_factory_name not in envs.__all__
    assert not hasattr(envs, legacy_factory_name)
    old_handle = "WorkflowEnv" + "Handle"
    old_session = "RuntimeEnv" + "Session"
    assert old_handle not in envs.__all__
    assert old_session not in envs.__all__
    assert "GalbotEnv" not in envs.__all__


def test_direct_env_collect_uses_recorder_manager_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ioailab.agents import PlannerAgent
    from ioailab.datasets import DatasetRef
    from ioailab.envs import make_env

    class FakeRecorderManager:
        def __init__(self) -> None:
            self.exports = []

        def export_episodes(self, env_ids=None, demo_ids=None):
            self.exports.append(
                {
                    "env_ids": tuple(env_ids),
                    "demo_ids": None if demo_ids is None else tuple(demo_ids),
                }
            )
            return {"file_handler": "fake_robomimic"}

    raw_env = FakeRawEnv(num_envs=2)
    raw_env.cfg.recorders = SimpleNamespace(
        dataset_export_dir_path="", dataset_filename=""
    )
    raw_env.recorder_manager = FakeRecorderManager()
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)
    agent = PlannerAgent(lambda sim, env_ids: [[1.0, 2.0], [3.0, 4.0]])

    dataset = env.collect(
        agent=agent,
        path=tmp_path / "demo.hdf5",
        episodes=3,
        max_steps=3,
        metadata={"source": "test"},
    )

    assert isinstance(dataset, DatasetRef)
    assert dataset.path == tmp_path / "demo.hdf5"
    assert dataset.format == "robomimic_hdf5"
    assert dataset.task_id == "GalbotG1-PickCube-v0"
    assert dataset.metadata["episodes"] == 3
    assert dataset.metadata["num_envs"] == 2
    assert dataset.metadata["source"] == "test"
    assert dataset.metadata["file_handler"] == "fake_robomimic"
    assert raw_env.cfg.recorders.dataset_export_dir_path == str(tmp_path)
    assert raw_env.cfg.recorders.dataset_filename == "demo"
    assert dataset.metadata["total_demos"] == 3
    assert dataset.metadata["saved_demo_ids"] == (0, 1, 2)
    assert dataset.metadata["collection_rounds"] == 2
    assert raw_env.recorder_manager.exports == [
        {"env_ids": (0, 1), "demo_ids": None},
        {"env_ids": (0,), "demo_ids": None},
    ]
    assert len(raw_env.step_calls) == 6


def test_direct_env_collect_appends_after_existing_hdf5_demo_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import h5py

    from ioailab.agents import PlannerAgent
    from ioailab.envs import make_env

    class FakeRecorderManager:
        def __init__(self) -> None:
            self.exports = []

        def export_episodes(self, env_ids=None, demo_ids=None):
            self.exports.append(
                {
                    "env_ids": tuple(env_ids),
                    "demo_ids": None if demo_ids is None else tuple(demo_ids),
                }
            )
            return {}

    dataset_path = tmp_path / "demo.hdf5"
    with h5py.File(dataset_path, "w") as file:
        file.create_group("data/demo_0")

    raw_env = FakeRawEnv(num_envs=2)
    raw_env.cfg.recorders = SimpleNamespace(
        dataset_export_dir_path="", dataset_filename=""
    )
    raw_env.recorder_manager = FakeRecorderManager()
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)
    agent = PlannerAgent(lambda sim, env_ids: [[0.0], [0.0]])

    dataset = env.collect(
        agent=agent,
        path=dataset_path,
        episodes=2,
        max_steps=1,
    )

    assert dataset.metadata["saved_demo_ids"] == (1, 2)
    assert raw_env.recorder_manager.exports == [{"env_ids": (0, 1), "demo_ids": None}]


def test_direct_env_collect_uses_explicit_exit_requested_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ioailab.agents.base import BaseAgent
    from ioailab.envs import make_env

    class FakeRecorderManager:
        def __init__(self) -> None:
            self.exports = []

        def export_episodes(self, env_ids=None, demo_ids=None):
            self.exports.append(
                {
                    "env_ids": tuple(env_ids),
                    "demo_ids": None if demo_ids is None else tuple(demo_ids),
                }
            )
            return {}

    class ExitAfterTwoActionsAgent(BaseAgent):
        def __init__(self) -> None:
            self.actions = 0

        def act(self, env, env_ids=None):
            del env, env_ids
            self.actions += 1
            return [[0.0]]

        def done(self, env, env_ids=None):
            raise AssertionError("collect must not use agent.done as a boundary")

        def exit_requested(self) -> bool:
            return self.actions >= 2

    raw_env = FakeRawEnv(num_envs=1)
    raw_env.cfg.recorders = SimpleNamespace(
        dataset_export_dir_path="", dataset_filename=""
    )
    raw_env.recorder_manager = FakeRecorderManager()
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-Teleop-v0", num_envs=1, headless=True)
    agent = ExitAfterTwoActionsAgent()

    dataset = env.collect(
        agent=agent,
        path=tmp_path / "teleop.hdf5",
        episodes=1,
        max_steps=10,
    )

    assert len(raw_env.step_calls) == 2
    assert dataset.metadata["episode_lengths"] == (2,)
    assert dataset.metadata["termination_reasons"] == ("user_exit",)
    assert dataset.metadata["saved_demo_ids"] == (0,)
    assert raw_env.recorder_manager.exports == [{"env_ids": (0,), "demo_ids": None}]


def test_direct_env_save_demo_exports_current_recorded_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ioailab.datasets import DatasetRef
    from ioailab.envs import make_env

    class FakeRecorderManager:
        def __init__(self) -> None:
            self.exports = []

        def export_episodes(self, env_ids=None, demo_ids=None):
            self.exports.append(
                {
                    "env_ids": tuple(env_ids),
                    "demo_ids": None if demo_ids is None else tuple(demo_ids),
                }
            )
            return {"file_handler": "fake_robomimic"}

    raw_env = FakeRawEnv(num_envs=2)
    raw_env.cfg.recorders = SimpleNamespace(
        dataset_export_dir_path="", dataset_filename=""
    )
    raw_env.recorder_manager = FakeRecorderManager()
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)

    dataset = env.save_demo(
        tmp_path / "manual.hdf5",
        env_ids=(1,),
        demo_ids=(7,),
        metadata={"source": "manual-loop"},
    )

    assert isinstance(dataset, DatasetRef)
    assert dataset.path == tmp_path / "manual.hdf5"
    assert dataset.task_id == "GalbotG1-PickCube-v0"
    assert dataset.metadata["saved_env_ids"] == (1,)
    assert dataset.metadata["saved_demo_ids"] == (7,)
    assert dataset.metadata["saved_demos"] == 1
    assert dataset.metadata["source"] == "manual-loop"
    assert dataset.metadata["file_handler"] == "fake_robomimic"
    assert raw_env.cfg.recorders.dataset_export_dir_path == str(tmp_path)
    assert raw_env.cfg.recorders.dataset_filename == "manual"
    assert raw_env.recorder_manager.exports == [
        {"env_ids": (1,), "demo_ids": (7,)},
    ]


def test_direct_env_drop_demo_discards_current_recorded_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.envs import make_env
    from isaaclab.utils.datasets import EpisodeData

    class FakeRecorderManager:
        def __init__(self) -> None:
            self._episodes = {0: object(), 1: object()}

    raw_env = FakeRawEnv(num_envs=2)
    raw_env.recorder_manager = FakeRecorderManager()
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)

    original_env_1_episode = raw_env.recorder_manager._episodes[1]
    env.drop_demo(env_ids=(0,))

    assert isinstance(raw_env.recorder_manager._episodes[0], EpisodeData)
    assert raw_env.recorder_manager._episodes[1] is original_env_1_episode


def test_direct_env_collect_does_not_read_agent_done_until_collection_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ioailab.agents.base import BaseAgent
    from ioailab.envs import make_env

    class FakeRecorderManager:
        def __init__(self) -> None:
            self.exports = []

        def export_episodes(self, env_ids=None, demo_ids=None):
            self.exports.append(
                {
                    "env_ids": tuple(env_ids),
                    "demo_ids": None if demo_ids is None else tuple(demo_ids),
                }
            )
            return {}

    class AgentDoneMustNotBeCalled(BaseAgent):
        def act(self, env, env_ids=None):
            del env, env_ids
            return [[0.0], [0.0]]

        def done(self, env, env_ids=None):
            raise AssertionError("collect must not use agent.done as a boundary")

    raw_env = FakeRawEnv(num_envs=2)
    raw_env.cfg.recorders = SimpleNamespace(
        dataset_export_dir_path="", dataset_filename=""
    )
    raw_env.recorder_manager = FakeRecorderManager()
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)

    dataset = env.collect(
        agent=AgentDoneMustNotBeCalled(),
        path=tmp_path / "demo.hdf5",
        episodes=5,
        max_steps=3,
    )

    assert dataset.metadata["episodes"] == 5
    assert dataset.metadata["total_demos"] == 5
    assert dataset.metadata["attempted_demos"] == 5
    assert dataset.metadata["collection_rounds"] == 3
    assert dataset.metadata["planned_collection_rounds"] == 3
    assert dataset.metadata["episode_lengths"] == (3, 3, 3, 3, 3)
    assert dataset.metadata["termination_reasons"] == (
        "max_steps",
        "max_steps",
        "max_steps",
        "max_steps",
        "max_steps",
    )
    assert raw_env.recorder_manager.exports == [
        {"env_ids": (0, 1), "demo_ids": None},
        {"env_ids": (0, 1), "demo_ids": None},
        {"env_ids": (0,), "demo_ids": None},
    ]
    assert len(raw_env.step_calls) == 9


def test_direct_env_evaluate_does_not_read_agent_done_as_completion_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.agents.base import BaseAgent
    from ioailab.envs import make_env

    class AgentDoneMustNotBeCalled(BaseAgent):
        def act(self, env, env_ids=None):
            del env, env_ids
            return [[0.0], [0.0]]

        def done(self, env, env_ids=None):
            raise AssertionError("evaluate must not use agent.done as a boundary")

    raw_env = FakeRawEnv(num_envs=2)
    raw_env.step_results = [
        ({}, [0.0, 0.0], [False, False], [False, False], {}),
        ({}, [0.0, 0.0], [True, True], [False, False], {}),
    ]
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)

    metrics = env.evaluate(
        agent=AgentDoneMustNotBeCalled(),
        episodes=2,
        max_steps=5,
    )

    assert metrics["episode_lengths"] == (2, 2)
    assert metrics["termination_reasons"] == ("terminated", "terminated")


def test_direct_env_collect_exports_and_resets_independent_env_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ioailab.agents.base import BaseAgent
    from ioailab.envs import make_env

    class FakeRecorderManager:
        def __init__(self) -> None:
            self.exports = []

        def export_episodes(self, env_ids=None, demo_ids=None):
            self.exports.append(
                {
                    "env_ids": tuple(env_ids),
                    "demo_ids": None if demo_ids is None else tuple(demo_ids),
                }
            )
            return {}

    class ResetTrackingAgent(BaseAgent):
        def __init__(self) -> None:
            self.reset_calls: list[tuple[int, ...] | None] = []

        def reset(self, env, env_ids=None) -> None:
            del env
            self.reset_calls.append(None if env_ids is None else tuple(env_ids))

        def act(self, env, env_ids=None):
            del env, env_ids
            return [[0.0], [0.0]]

    isaac_env = FakeRawEnv(num_envs=2)
    raw_env = FakeGymWrapper(isaac_env)
    isaac_env.cfg.recorders = SimpleNamespace(
        dataset_export_dir_path="", dataset_filename=""
    )
    isaac_env.recorder_manager = FakeRecorderManager()
    isaac_env.step_results = [
        ({}, [1.0, 2.0], [True, False], [False, False], {"success": [True, False]}),
        ({}, [3.0, 4.0], [False, True], [False, False], {"success": [False, True]}),
        ({}, [5.0, 6.0], [True, False], [False, False], {"success": [True, False]}),
    ]
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)
    agent = ResetTrackingAgent()

    dataset = env.collect(
        agent=agent,
        path=tmp_path / "demo.hdf5",
        episodes=3,
        max_steps=10,
    )

    assert dataset.metadata["total_demos"] == 3
    assert dataset.metadata["collection_rounds"] == 3
    assert dataset.metadata["steps"] == 3
    assert dataset.metadata["row_steps"] == 5
    assert dataset.metadata["episode_lengths"] == (1, 2, 2)
    assert dataset.metadata["termination_reasons"] == (
        "terminated",
        "terminated",
        "terminated",
    )
    assert isaac_env.recorder_manager.exports == [
        {"env_ids": (0,), "demo_ids": None},
        {"env_ids": (1,), "demo_ids": None},
        {"env_ids": (0,), "demo_ids": None},
    ]
    assert raw_env.reset_calls == [((), {})]
    assert isaac_env.reset_calls[0] == ((), {})
    assert tuple(int(i) for i in isaac_env.reset_calls[1][1]["env_ids"]) == (0,)
    assert tuple(int(i) for i in isaac_env.reset_calls[2][1]["env_ids"]) == (1,)
    assert agent.reset_calls == [None, (0,), (1,)]


def test_direct_env_collect_resets_curobo_planner_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ioailab.agents import CuroboPlannerAgent
    from ioailab.envs import make_env

    class FakeRecorderManager:
        def __init__(self) -> None:
            self.exports = []

        def export_episodes(self, env_ids=None, demo_ids=None):
            self.exports.append(
                {
                    "env_ids": tuple(env_ids),
                    "demo_ids": None if demo_ids is None else tuple(demo_ids),
                }
            )
            return {}

    class RowAwarePlannerSource:
        is_complete = False

        def __init__(self) -> None:
            self.reset_calls: list[tuple[int, ...] | None] = []

        def reset(self, env, env_ids=None) -> None:
            self.reset_calls.append(None if env_ids is None else tuple(env_ids))

        def act(self, env, env_ids=None):
            rows = int(env.num_envs) if env_ids is None else len(env_ids)
            return [[0.0] for _ in range(rows)]

        def done(self, env, env_ids=None):
            rows = int(env.num_envs) if env_ids is None else len(env_ids)
            return [False] * rows

    isaac_env = FakeRawEnv(num_envs=2)
    raw_env = FakeGymWrapper(isaac_env)
    isaac_env.cfg.recorders = SimpleNamespace(
        dataset_export_dir_path="", dataset_filename=""
    )
    isaac_env.recorder_manager = FakeRecorderManager()
    isaac_env.step_results = [
        ({}, [1.0, 2.0], [True, False], [False, False], {"success": [True, False]}),
        ({}, [3.0, 4.0], [False, True], [False, False], {"success": [False, True]}),
    ]
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)
    source = RowAwarePlannerSource()
    agent = CuroboPlannerAgent(action_source=source)

    dataset = env.collect(
        agent=agent,
        path=tmp_path / "demo.hdf5",
        episodes=2,
        max_steps=10,
    )

    assert dataset.metadata["total_demos"] == 2
    assert isaac_env.recorder_manager.exports == [
        {"env_ids": (0,), "demo_ids": None},
        {"env_ids": (1,), "demo_ids": None},
    ]
    assert source.reset_calls == [None, (0,)]


def test_direct_env_collect_requires_configured_recorder_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ioailab.agents import PlannerAgent
    from ioailab.envs import make_env

    raw_env = FakeRawEnv(num_envs=1)
    raw_env.cfg.recorders = SimpleNamespace(
        dataset_export_dir_path="", dataset_filename=""
    )
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=1, headless=True)
    agent = PlannerAgent(lambda sim, env_ids: [[1.0]])

    with pytest.raises(RuntimeError, match="recorder_manager"):
        env.collect(agent=agent, path=tmp_path / "demo.hdf5", episodes=1, max_steps=1)

    raw_env.recorder_manager = SimpleNamespace(
        export_episodes=lambda env_ids=None, demo_ids=None: None
    )
    raw_env.cfg.recorders = None
    env.env_cfg.recorders = None
    with pytest.raises(RuntimeError, match="cfg.recorders"):
        env.collect(agent=agent, path=tmp_path / "demo.hdf5", episodes=1, max_steps=1)


def test_direct_env_evaluate_completes_on_configured_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from ioailab.agents import PlannerAgent
    from ioailab.envs import make_env

    raw_env = FakeRawEnv(num_envs=2)
    raw_env.step_results = [
        ({}, [0.0, 0.0], [False, False], [False, False], {}),
        ({}, [0.0, 0.0], [False, False], [False, False], {}),
    ]
    calls = _patch_make_env_runtime(monkeypatch, raw_env=raw_env)

    def success_metric(env):
        assert env is raw_env
        return np.array([True, True], dtype=bool)

    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)
    calls["env_cfg"].evaluation_success = SimpleNamespace(
        func=success_metric, params={}
    )
    agent = PlannerAgent(lambda sim, env_ids: [[0.0], [0.0]])

    metrics = env.evaluate(agent=agent, episodes=2, max_steps=2)

    assert metrics["total_episodes"] == 2
    assert metrics["success_count"] == 2
    assert metrics["success_rate"] == 1.0
    assert metrics["success_masks"] == ((True,), (True,))
    assert metrics["episode_lengths"] == (1, 1)
    assert metrics["termination_reasons"] == ("success", "success")


def test_direct_env_evaluate_requires_explicit_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.agents import PlannerAgent
    from ioailab.envs import make_env

    raw_env = FakeRawEnv(num_envs=1)
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=1, headless=True)
    agent = PlannerAgent(lambda sim, env_ids: [[0.0]])

    metrics = env.evaluate(agent=agent, episodes=2, max_steps=2)

    assert metrics["episodes"] == 2
    assert metrics["num_envs"] == 1
    assert metrics["average_length"] == 2.0
    assert metrics["reward_total_mean"] == 0.0
    assert metrics["action_validation_failures"] == 0


def test_direct_env_evaluate_reports_task_success_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.agents import PlannerAgent
    from ioailab.envs import make_env

    raw_env = FakeRawEnv(num_envs=2)
    raw_env.step_results = [
        ({}, [1.0, 2.0], [False, False], [False, False], {"success": [False, True]}),
        ({}, [3.0, 4.0], [False, False], [False, False], {"success": [True, True]}),
    ]
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)
    agent = PlannerAgent(lambda sim, env_ids: [[0.0], [0.0]])

    metrics = env.evaluate(agent=agent, episodes=2, max_steps=2)

    assert metrics["total_episodes"] == 2
    assert metrics["success_count"] == 2
    assert metrics["success_rate"] == 1.0
    assert metrics["success_masks"] == ((True,), (True,))
    assert metrics["episode_lengths"] == (1, 2)
    assert metrics["episode_lengths_by_env"] == ((2,), (1,))
    assert metrics["reward_totals"] == (2.0, 4.0)
    assert metrics["reward_totals_by_env"] == ((4.0,), (2.0,))
    assert metrics["termination_reasons"] == ("success", "success")


def test_direct_env_evaluate_counts_async_vector_env_row_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.agents.base import BaseAgent
    from ioailab.envs import make_env

    class ResetTrackingAgent(BaseAgent):
        def __init__(self) -> None:
            self.reset_calls: list[tuple[int, ...] | None] = []

        def reset(self, env, env_ids=None) -> None:
            del env
            self.reset_calls.append(None if env_ids is None else tuple(env_ids))

        def act(self, env, env_ids=None):
            del env, env_ids
            return [[0.0], [0.0]]

    raw_env = FakeRawEnv(num_envs=2)
    raw_env.step_results = [
        ({}, [1.0, 2.0], [True, False], [False, False], {"success": [True, False]}),
        ({}, [3.0, 4.0], [False, True], [False, False], {"success": [False, True]}),
        ({}, [5.0, 6.0], [True, False], [False, False], {"success": [True, False]}),
        ({}, [7.0, 8.0], [False, True], [False, False], {"success": [False, True]}),
    ]
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)
    agent = ResetTrackingAgent()

    metrics = env.evaluate(agent=agent, episodes=3, max_steps=10)

    assert metrics["episodes"] == 3
    assert metrics["total_episodes"] == 3
    assert metrics["success_count"] == 3
    assert metrics["success_rate"] == 1.0
    assert metrics["vector_steps"] == 3
    assert metrics["row_steps"] == 5
    assert metrics["episode_lengths"] == (1, 2, 2)
    assert metrics["episode_lengths_by_env"] == ((1, 2), (2,))
    assert metrics["reward_totals"] == (1.0, 6.0, 8.0)
    assert metrics["reward_totals_by_env"] == ((1.0, 8.0), (6.0,))
    assert metrics["success_masks"] == ((True, True), (True,))
    assert metrics["termination_reasons"] == (
        "terminated",
        "terminated",
        "terminated",
    )
    assert metrics["termination_reasons_by_env"] == (
        ("terminated", "terminated"),
        ("terminated",),
    )
    assert agent.reset_calls == [None, (0,), (1,)]


def test_direct_env_evaluate_exposes_compact_public_options() -> None:
    from ioailab.envs import ioailabEnv

    parameters = inspect.signature(ioailabEnv.evaluate).parameters
    assert "init_snapshot" not in parameters


def test_direct_env_evaluate_resets_rows_after_nonterminal_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.agents import PlannerAgent
    from ioailab.envs import make_env

    isaac_env = FakeRawEnv(num_envs=2)
    raw_env = FakeGymWrapper(isaac_env)
    isaac_env.step_results = [
        ({}, [1.0, 2.0], [False, False], [False, False], {"success": [True, False]}),
        ({}, [3.0, 4.0], [False, False], [False, False], {"success": [False, True]}),
    ]
    _patch_make_env_runtime(monkeypatch, raw_env=raw_env)
    env = make_env("GalbotG1-PickCube-v0", num_envs=2, headless=True)
    agent = PlannerAgent(lambda sim, env_ids: [[0.0], [0.0]])

    metrics = env.evaluate(agent=agent, episodes=2, max_steps=2)

    assert metrics["success_count"] == 2
    assert metrics["success_rate"] == 1.0
    assert metrics["success_masks"] == ((True,), (True,))
    assert metrics["episode_lengths"] == (1, 2)
    assert metrics["termination_reasons"] == ("success", "success")
    assert raw_env.reset_calls == [((), {})]
    assert len(isaac_env.reset_calls) == 2
    assert isaac_env.reset_calls[0] == ((), {})
    assert tuple(int(i) for i in isaac_env.reset_calls[1][1]["env_ids"]) == (0,)


class _RuntimeFakeActionBuilder:
    def __init__(self) -> None:
        self.targets = {}

    def joint_position(self, targets: dict[str, float]) -> "_RuntimeFakeActionBuilder":
        self.targets = dict(targets)
        return self

    def build(self) -> dict[str, Any]:
        return {"joint_position": self.targets}


class _RuntimeFakeCamera:
    def read_rgb(self) -> str:
        return "rgb"

    def read_rgb_depth(self) -> tuple[str, str]:
        return "rgb", "depth"


class _RuntimeFakeRobot:
    def action_builder(self) -> _RuntimeFakeActionBuilder:
        return _RuntimeFakeActionBuilder()

    def get_sensor(self, name: str) -> _RuntimeFakeCamera:
        assert name == "front_camera"
        return _RuntimeFakeCamera()


def test_robot_sensor_runtime_protocols_accept_basic_robot_shape() -> None:
    from ioailab.robots.common.interfaces import (
        CameraSensor,
        RobotHandle,
        TaskActionBuilder,
    )

    robot = _RuntimeFakeRobot()
    builder = robot.action_builder()
    camera = robot.get_sensor("front_camera")

    assert isinstance(robot, RobotHandle)
    assert isinstance(builder, TaskActionBuilder)
    assert isinstance(camera, CameraSensor)
    assert builder.joint_position({"left_arm_joint_1": 0.4}).build() == {
        "joint_position": {"left_arm_joint_1": 0.4}
    }
    assert camera.read_rgb() == "rgb"
    assert camera.read_rgb_depth() == ("rgb", "depth")


def test_workflow_env_scene_access_delegates_to_live_env() -> None:
    from ioailab.envs import ioailabEnv

    raw_env = type(
        "RawEnv",
        (),
        {"unwrapped": None, "scene": {"robot": _RuntimeFakeRobot()}, "device": "cpu"},
    )()
    raw_env.unwrapped = raw_env
    env = ioailabEnv("GalbotG1-PickCube-v0", raw_env=raw_env, app=None, num_envs=1)

    assert isinstance(env.scene["robot"], _RuntimeFakeRobot)


def test_collect_root_tutorial_uses_make_env_agent_save() -> None:
    tutorial_script = (ROOT / "examples" / "01_collect.py").read_text(encoding="utf-8")

    module = ast.parse(tutorial_script)
    top_level_imports = {
        alias.name
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "rerun" not in top_level_imports
    assert "from ioailab.envs import make_env" in tutorial_script
    assert "make_env(" in tutorial_script
    assert "from ioailab.agents import CuroboPlannerAgent" in tutorial_script
    assert "# from ioailab.agents import TeleopAgent" in tutorial_script
    assert "# agent = TeleopAgent.from_device" in tutorial_script
    assert "dataset = env.collect(" in tutorial_script
    assert "argparse.ArgumentParser" in tutorial_script
    assert '"--task"' in tutorial_script
    assert '"--episodes"' in tutorial_script
    assert '"--num-envs"' in tutorial_script
    assert "run_candidate" not in tutorial_script
    assert "from ioailab.agents import CuroboPlannerAgent" in tutorial_script
    assert "episodes=args.episodes" in tutorial_script
    assert "while accepted < args.episodes" in tutorial_script
    assert "all(agent.done(env))" not in tutorial_script
    assert "#     decision = agent.review_demo()" in tutorial_script
    assert "export_decision" not in tutorial_script
    assert "ask_keep_drop_exit" not in tutorial_script
    assert ("with handle" + ".runtime(") not in tutorial_script
    assert not (ROOT / "examples" / "basic").exists()
    assert not (ROOT / "examples" / "tutorials").exists()
