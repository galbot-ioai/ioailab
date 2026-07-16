"""Tests for human-readable scenario reset configs."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class _FakeArticulation:
    def __init__(self, joint_names):
        self.joint_names = list(joint_names)


class _FakeScene:
    def __init__(self, num_envs: int = 3, num_joints: int = 4):
        self.num_envs = num_envs
        self.cfg = SimpleNamespace(
            robot=SimpleNamespace(
                scenario_base_pose_from_root_pose=lambda pose: pose,
                scenario_root_pose_from_base_pose=lambda pose: pose,
            )
        )
        self.joint_names = [f"joint_{i}" for i in range(num_joints)]
        self.state = {
            "articulation": {
                "robot": {
                    "root_pose": torch.zeros(num_envs, 7),
                    "root_velocity": torch.ones(num_envs, 6),
                    "joint_position": torch.arange(
                        num_envs * num_joints, dtype=torch.float32
                    ).reshape(num_envs, num_joints),
                    "joint_velocity": torch.zeros(num_envs, num_joints),
                }
            },
            "rigid_object": {
                "cube": {
                    "root_pose": torch.full((num_envs, 7), 2.0),
                    "root_velocity": torch.zeros(num_envs, 6),
                }
            },
        }
        self.reset_to_calls: list[dict] = []

    def __getitem__(self, name: str):
        if name == "robot":
            return _FakeArticulation(self.joint_names)
        raise KeyError(name)

    def get_state(self, is_relative: bool = False):
        return {
            category: {
                entity: {
                    field_name: tensor.clone() for field_name, tensor in fields.items()
                }
                for entity, fields in entities.items()
            }
            for category, entities in self.state.items()
        }

    def reset_to(self, state, env_ids=None, is_relative: bool = False):
        self.reset_to_calls.append(
            {"state": state, "env_ids": env_ids, "is_relative": is_relative}
        )


def _fake_env(scene: _FakeScene, *, cfg=None):
    if cfg is not None and hasattr(cfg, "scene"):
        scene.cfg = cfg.scene
    raw = {
        "scene": scene,
        "num_envs": scene.num_envs,
        "device": torch.device("cpu"),
    }
    if cfg is not None:
        raw["cfg"] = cfg
    return SimpleNamespace(unwrapped=SimpleNamespace(**raw))


def test_scenario_yaml_roundtrip(tmp_path):
    from ioailab.tasks.common.scenario import (
        Scenario,
        load_scenario,
        save_scenario,
    )

    scenario = Scenario(
        name="unit",
        assets={
            "articulation": {
                "robot": {
                    "base_pose": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                    "joint_position": {"joint_1": 0.25},
                }
            }
        },
        metadata={"task_id": "demo"},
    )

    path = save_scenario(tmp_path / "scenario.yaml", scenario)
    loaded = load_scenario(path)

    assert loaded.name == "unit"
    assert loaded.frame == "env"
    assert loaded.metadata["task_id"] == "demo"
    assert loaded.assets["articulation"]["robot"]["joint_position"] == {"joint_1": 0.25}


def test_scenario_rejects_stale_root_convention_metadata():
    from ioailab.tasks.common.scenario import Scenario

    with pytest.raises(ValueError, match="base_pose"):
        Scenario.from_dict(
            {
                "schema": "ioailabScenario-v0",
                "frame": "env",
                "metadata": {
                    "asset_conventions": {"articulation.robot.root_pose": "x"}
                },
                "assets": {},
            }
        )


def test_scenario_rejects_raw_articulation_state_fields():
    from ioailab.tasks.common.scenario import Scenario

    for field_name in ("root_pose", "root_velocity", "joint_velocity"):
        with pytest.raises(ValueError, match="base_pose|joint_position"):
            Scenario(
                assets={
                    "articulation": {
                        "robot": {
                            field_name: [0.0],
                        }
                    }
                }
            )


def test_apply_scenario_resets_default_then_overlays_named_joints(monkeypatch):
    from ioailab.tasks.common import scenario as module
    from ioailab.tasks.common.scenario import Scenario, apply_scenario

    scene = _FakeScene()
    env = _fake_env(scene)
    default_reset_calls = []

    def fake_reset_scene_to_default(env_arg, env_ids=None, reset_joint_targets=False):
        default_reset_calls.append(
            {
                "env": env_arg,
                "env_ids": env_ids,
                "reset_joint_targets": reset_joint_targets,
            }
        )

    monkeypatch.setattr(
        module.base_mdp, "reset_scene_to_default", fake_reset_scene_to_default
    )
    scenario = Scenario(
        assets={
            "articulation": {
                "robot": {
                    "base_pose": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                    "joint_position": {"joint_1": 10.0, "joint_3": 30.0},
                }
            },
            "rigid_object": {
                "cube": {
                    "root_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                }
            },
        }
    )
    scene.state["rigid_object"]["shelf"] = {
        "root_pose": torch.full((scene.num_envs, 7), 3.0),
        "root_velocity": torch.zeros(scene.num_envs, 6),
    }

    apply_scenario(env, env_ids=[1, 2], scenario=scenario)

    assert default_reset_calls == [
        {"env": env, "env_ids": [1, 2], "reset_joint_targets": True}
    ]
    call = scene.reset_to_calls[-1]
    assert call["env_ids"] == [1, 2]
    assert call["is_relative"] is True
    root_pose = call["state"]["articulation"]["robot"]["root_pose"]
    assert torch.allclose(
        root_pose,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    joint_position = call["state"]["articulation"]["robot"]["joint_position"]
    expected = scene.state["articulation"]["robot"]["joint_position"][[1, 2]].clone()
    expected[:, 1] = 10.0
    expected[:, 3] = 30.0
    assert torch.allclose(joint_position, expected)
    assert "shelf" in call["state"]["rigid_object"]


def test_apply_scenario_converts_base_pose_to_raw_root_pose(monkeypatch):
    from ioailab.tasks.common import scenario as module
    from ioailab.tasks.common.scenario import Scenario, apply_scenario

    scene = _FakeScene()
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            robot=SimpleNamespace(
                scenario_root_pose_from_base_pose=lambda pose: [
                    pose[0] + 10.0,
                    pose[1] + 20.0,
                    pose[2] + 30.0,
                    pose[3],
                    pose[4],
                    pose[5],
                    pose[6],
                ],
            )
        )
    )
    env = _fake_env(scene, cfg=cfg)

    monkeypatch.setattr(
        module.base_mdp,
        "reset_scene_to_default",
        lambda env_arg, env_ids=None, reset_joint_targets=False: None,
    )
    scenario = Scenario(
        assets={
            "articulation": {
                "robot": {
                    "base_pose": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                }
            }
        },
    )

    apply_scenario(env, env_ids=[0], scenario=scenario)

    root_pose = scene.reset_to_calls[-1]["state"]["articulation"]["robot"]["root_pose"]
    assert torch.allclose(
        root_pose,
        torch.tensor([[11.0, 22.0, 33.0, 0.0, 0.0, 0.0, 1.0]]),
    )


def test_get_scenario_captures_base_pose_and_named_qpos():
    from ioailab.tasks.common.scenario import get_scenario

    scene = _FakeScene()
    scene.state["articulation"]["robot"]["root_pose"][0] = torch.tensor(
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    )
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            robot=SimpleNamespace(
                scenario_base_pose_from_root_pose=lambda pose: [
                    pose[0] - 1.0,
                    pose[1] - 2.0,
                    pose[2] - 3.0,
                    pose[3],
                    pose[4],
                    pose[5],
                    pose[6],
                ],
            ),
        )
    )

    scenario = get_scenario(_fake_env(scene, cfg=cfg).unwrapped)
    robot = scenario.assets["articulation"]["robot"]

    assert robot["base_pose"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert robot["joint_position"]["joint_2"] == 2.0
    assert "root_pose" not in robot
    assert "root_velocity" not in robot
    assert "joint_velocity" not in robot


def test_scenario_reset_event_uses_isaaclab_event_compatible_signature():
    from ioailab.tasks.common.scenario import Scenario, scenario_reset_event

    term = scenario_reset_event(Scenario())
    params = inspect.signature(term.func).parameters

    assert params["env_ids"].default is inspect.Parameter.empty
    assert set(term.params) == {"scenario", "reset_to_default_first"}


def test_apply_scenario_rejects_unknown_joint():
    from ioailab.tasks.common.scenario import Scenario, apply_scenario

    scene = _FakeScene()
    env = _fake_env(scene)
    scenario = Scenario(
        assets={
            "articulation": {
                "robot": {
                    "joint_position": {"missing_joint": 1.0},
                }
            }
        }
    )

    with pytest.raises(ValueError, match="missing_joint"):
        apply_scenario(env, scenario=scenario, reset_to_default_first=False)


def test_apply_scenario_rejects_unknown_field():
    from ioailab.tasks.common.scenario import Scenario, apply_scenario

    scene = _FakeScene()
    env = _fake_env(scene)
    scenario = Scenario(
        assets={
            "rigid_object": {
                "cube": {
                    "missing_field": [0.0],
                }
            }
        }
    )

    with pytest.raises(ValueError, match="missing_field"):
        apply_scenario(env, scenario=scenario, reset_to_default_first=False)


def test_apply_scenario_rejects_scene_topology_from_yaml():
    from ioailab.tasks.common.scenario import Scenario, apply_scenario

    scene = _FakeScene()
    env = _fake_env(scene)
    scenario = Scenario(
        assets={
            "rigid_object": {
                "new_cube": {
                    "root_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                }
            }
        }
    )

    with pytest.raises(ValueError, match="new_cube"):
        apply_scenario(env, scenario=scenario, reset_to_default_first=False)


def test_galbot_env_get_scenario_captures_named_joint_values():
    from ioailab.envs import ioailabEnv

    scene = _FakeScene()
    raw_env = _fake_env(scene)
    env = ioailabEnv(
        task_id="GalbotG1-Demo-v0",
        raw_env=raw_env,
        app=None,
        num_envs=scene.num_envs,
    )

    scenario = env.get_scenario(env_id=1, name="captured")

    assert scenario.name == "captured"
    assert scenario.metadata["task_id"] == "GalbotG1-Demo-v0"
    assert scenario.metadata["source_env_id"] == 1
    assert scenario.assets["articulation"]["robot"]["joint_position"]["joint_2"] == 6.0
    assert scenario.assets["rigid_object"]["cube"]["root_pose"] == [
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ]


def test_galbot_env_get_scenario_defaults_to_first_env_row():
    from ioailab.envs import ioailabEnv

    scene = _FakeScene()
    raw_env = _fake_env(scene)
    env = ioailabEnv(
        task_id="GalbotG1-Demo-v0",
        raw_env=raw_env,
        app=None,
        num_envs=scene.num_envs,
    )

    scenario = env.get_scenario(name="first")

    assert scenario.name == "first"
    assert scenario.metadata["source_env_id"] == 0
    assert scenario.assets["articulation"]["robot"]["joint_position"]["joint_2"] == 2.0


def test_pick_to_shelf_phase_init_scenario_option_replaces_default(tmp_path):
    from ioailab.tasks.common.scenario import Scenario, save_scenario
    from ioailab.tasks.pick_to_shelf_nav.config.g1.env_cfg import (
        GalbotG1PickToShelfNavEnvCfg,
    )
    from ioailab.tasks.pick_to_shelf_place.config.g1.env_cfg import (
        GalbotG1PickToShelfPlaceEnvCfg,
    )

    path = save_scenario(
        tmp_path / "custom.yaml",
        Scenario(name="custom_start", metadata={"task_id": "custom"}),
    )

    for cfg_cls in (GalbotG1PickToShelfNavEnvCfg, GalbotG1PickToShelfPlaceEnvCfg):
        cfg = cfg_cls()
        cfg.apply_task_options({"init_scenario": path})

        scenario = cfg.events.reset_all.params["scenario"]
        assert scenario.name == "custom_start"
        assert scenario.metadata["task_id"] == "custom"


def test_collect_end_scenario_uses_pre_reset_terminal_state(tmp_path):
    from ioailab.agents.base import BaseAgent
    from ioailab.envs import ioailabEnv
    from ioailab.tasks.common.scenario import load_scenario

    class OneStepAgent(BaseAgent):
        def act(self, env, env_ids=None):
            del env, env_ids
            return torch.zeros((1, 1))

    class FakeRecorderManager:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(export_in_record_pre_reset=True)
            self.record_pre_reset_calls = []

        def record_pre_reset(self, env_ids):
            self.record_pre_reset_calls.append(tuple(int(row) for row in env_ids))

    class AutoResetRawEnv:
        def __init__(self, scene: _FakeScene) -> None:
            self.scene = scene
            self.recorders = SimpleNamespace(
                dataset_export_dir_path=str(tmp_path),
                dataset_filename="old",
            )
            self.recorder_manager = FakeRecorderManager()
            self.cfg = SimpleNamespace(recorders=self.recorders)
            self.unwrapped = SimpleNamespace(
                scene=scene,
                num_envs=1,
                device=torch.device("cpu"),
                cfg=self.cfg,
                recorder_manager=self.recorder_manager,
            )
            self.reset_calls = 0

        def reset(self, *args, **kwargs):
            del args, kwargs
            self.reset_calls += 1
            return {}, {}

        def step(self, action):
            del action
            self.scene.state["rigid_object"]["cube"]["root_pose"][0, 0] = 42.0
            self.recorder_manager.record_pre_reset(torch.tensor([0]))
            self.scene.state["rigid_object"]["cube"]["root_pose"][0, 0] = -1.0
            return (
                {},
                torch.tensor([1.0]),
                torch.tensor([True]),
                torch.tensor([False]),
                {},
            )

    scene = _FakeScene(num_envs=1)
    raw_env = AutoResetRawEnv(scene)
    env = ioailabEnv(
        task_id="GalbotG1-Demo-v0",
        raw_env=raw_env,
        app=None,
        num_envs=1,
        env_cfg=raw_env.cfg,
    )

    scenario_path = tmp_path / "nav_start.yaml"
    dataset = env.collect(
        agent=OneStepAgent(),
        path=tmp_path / "demo.hdf5",
        episodes=1,
        max_steps=10,
        save_end_scenario=scenario_path,
    )

    scenario = load_scenario(scenario_path)
    assert dataset.metadata["saved_end_scenario_path"] == str(scenario_path)
    assert scenario.name == "GalbotG1-Demo_end"
    assert scenario.metadata["termination_reason"] == "terminated"
    assert scenario.assets["rigid_object"]["cube"]["root_pose"][0] == 42.0
    assert scene.state["rigid_object"]["cube"]["root_pose"][0, 0] == -1.0
    assert raw_env.recorder_manager.record_pre_reset_calls == [(0,)]
