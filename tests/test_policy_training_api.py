from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_policy_backend_resolves_only_robomimic_diffusion() -> None:
    from ioailab.agents.policy import RobomimicDiffusionPolicy, Policy

    policy = Policy.from_backend("robomimic_diffusion")

    assert isinstance(policy, RobomimicDiffusionPolicy)
    assert policy.backend == "robomimic_diffusion"

    with pytest.raises(ValueError, match="Unsupported policy backend"):
        Policy.from_backend("act")


def test_diffusion_policy_train_delegates_to_injected_trainer(tmp_path: Path) -> None:
    from ioailab.datasets import DatasetRef
    from ioailab.agents.policy import (
        RobomimicDiffusionPolicy,
        PolicyCheckpoint,
        RobomimicDiffusionTrainCfg,
    )

    calls = []

    class FakeTrainer:
        def train(
            self,
            train_config: RobomimicDiffusionTrainCfg,
            dataset: DatasetRef,
            *,
            backend: str,
        ) -> PolicyCheckpoint:
            calls.append((train_config, dataset, backend))
            return PolicyCheckpoint(
                path=tmp_path / "model.pth",
                backend=backend,
                metadata={"epochs": train_config.epochs},
            )

    dataset = DatasetRef(path=tmp_path / "demo.hdf5", format="robomimic_hdf5")
    train_config = RobomimicDiffusionTrainCfg(
        output_dir=tmp_path / "runs", epochs=2, batch_size=8
    )
    policy = RobomimicDiffusionPolicy(trainer=FakeTrainer())

    checkpoint = policy.train(dataset, train_config)

    assert checkpoint.path == tmp_path / "model.pth"
    assert checkpoint.backend == "robomimic_diffusion"
    assert checkpoint.metadata == {"epochs": 2}
    assert calls == [(train_config, dataset, "robomimic_diffusion")]


def test_policy_train_cfg_validates_common_fields() -> None:
    from ioailab.agents.policy import OptimizerCfg, PolicyTrainCfg

    with pytest.raises(ValueError, match="learning_rate"):
        OptimizerCfg(learning_rate=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        OptimizerCfg(weight_decay=-1.0)
    with pytest.raises(ValueError, match="grad_clip_norm"):
        OptimizerCfg(grad_clip_norm=0.0)
    with pytest.raises(ValueError, match="epochs"):
        PolicyTrainCfg(epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        PolicyTrainCfg(batch_size=0)
    with pytest.raises(ValueError, match="seed"):
        PolicyTrainCfg(seed=-1)


def test_robomimic_diffusion_train_cfg_defaults_and_validation() -> None:
    from ioailab.agents.policy import OptimizerCfg, RobomimicDiffusionTrainCfg

    cfg = RobomimicDiffusionTrainCfg(
        output_dir="runs",
        optimizer=OptimizerCfg(learning_rate=3.0e-4, weight_decay=1.0e-6),
    )

    assert cfg.output_dir == Path("runs")
    assert cfg.optimizer.learning_rate == 3.0e-4
    assert cfg.optimizer.weight_decay == 1.0e-6
    assert cfg.normalize_actions is True
    assert cfg.rollout_enabled is False

    with pytest.raises(ValueError, match="num_data_workers"):
        RobomimicDiffusionTrainCfg(num_data_workers=-1)
    with pytest.raises(ValueError, match="prediction_horizon"):
        RobomimicDiffusionTrainCfg(prediction_horizon=0)


def test_robomimic_trainer_action_normalization_returns_inference_metadata(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from ioailab.agents.policy.backends.robomimic_diffusion import (
        _normalize_actions,
        _read_obs_keys_from_hdf5,
    )

    dataset_path = tmp_path / "demo.hdf5"
    with h5py.File(dataset_path, "w") as f:
        demo = f.create_group("data/demo_0")
        demo.create_dataset(
            "actions", data=np.array([[0.0, 10.0], [2.0, 14.0]], dtype=np.float32)
        )
        obs = demo.create_group("obs")
        obs.create_dataset("robot_state", data=np.zeros((2, 3), dtype=np.float32))

    normalized_path, metadata = _normalize_actions(dataset_path, tmp_path / "run")

    assert normalized_path == tmp_path / "run" / "demo_normalized.hdf5"
    assert metadata == {
        "action_min": [0.0, 10.0],
        "action_max": [2.0, 14.0],
        "action_shape": (2,),
    }
    assert _read_obs_keys_from_hdf5(normalized_path) == ["robot_state"]
    with h5py.File(normalized_path, "r") as f:
        assert np.allclose(f["data/demo_0/actions"][...], [[-1.0, -1.0], [1.0, 1.0]])


def test_robomimic_obs_modalities_route_rgb_and_exclude_object_truth(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from ioailab.agents.policy.backends.robomimic_diffusion import (
        _read_obs_keys_from_hdf5,
        _read_obs_modalities_from_hdf5,
    )

    dataset_path = tmp_path / "vision_demo.hdf5"
    with h5py.File(dataset_path, "w") as f:
        demo = f.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((2, 8), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("actions", data=np.zeros((2, 8), dtype=np.float32))
        obs.create_dataset("robot_joint_pos", data=np.zeros((2, 77), dtype=np.float32))
        obs.create_dataset(
            "front_head_rgb", data=np.zeros((2, 224, 298, 3), dtype=np.uint8)
        )
        obs.create_dataset("cube_pos", data=np.zeros((2, 3), dtype=np.float32))
        obs.create_dataset("cube_quat", data=np.zeros((2, 4), dtype=np.float32))
        obs.create_dataset("blue_block_pos", data=np.zeros((2, 3), dtype=np.float32))
        obs.create_dataset("blue_block_quat", data=np.zeros((2, 4), dtype=np.float32))

    assert _read_obs_modalities_from_hdf5(dataset_path) == {
        "low_dim": ["robot_joint_pos"],
        "rgb": ["front_head_rgb"],
    }
    assert _read_obs_keys_from_hdf5(dataset_path) == [
        "robot_joint_pos",
        "front_head_rgb",
    ]


def test_robomimic_diffusion_config_disables_next_obs_loading() -> None:
    from types import SimpleNamespace

    from ioailab.agents.policy.backends.robomimic_diffusion import (
        _configure_dataset_loading,
    )

    config = SimpleNamespace(
        train=SimpleNamespace(hdf5_load_next_obs=True, hdf5_cache_mode="all")
    )

    _configure_dataset_loading(config)

    assert config.train.hdf5_load_next_obs is False
    assert config.train.hdf5_cache_mode == "all"


def test_robomimic_diffusion_config_streams_rgb_from_hdf5() -> None:
    from types import SimpleNamespace

    from ioailab.agents.policy.backends.robomimic_diffusion import (
        _configure_dataset_loading,
    )

    config = SimpleNamespace(
        train=SimpleNamespace(hdf5_load_next_obs=True, hdf5_cache_mode="all")
    )

    _configure_dataset_loading(config, has_rgb_observations=True)

    assert config.train.hdf5_load_next_obs is False
    assert config.train.hdf5_cache_mode == "low_dim"


def test_robomimic_diffusion_train_cfg_maps_to_backend_config() -> None:
    from types import SimpleNamespace

    from ioailab.agents.policy import OptimizerCfg, RobomimicDiffusionTrainCfg
    from ioailab.agents.policy.backends.robomimic_diffusion import (
        _apply_robomimic_diffusion_train_cfg,
    )

    config = SimpleNamespace(
        experiment=SimpleNamespace(rollout=SimpleNamespace(enabled=True)),
        train=SimpleNamespace(
            num_epochs=100,
            batch_size=64,
            num_data_workers=0,
            seed=None,
            max_grad_norm=None,
        ),
        algo=SimpleNamespace(
            horizon=SimpleNamespace(
                prediction_horizon=16,
                observation_horizon=2,
                action_horizon=8,
            ),
            optim_params={
                "policy": {
                    "learning_rate": {"initial": 1.0e-4},
                    "regularization": {"L2": 0.0},
                }
            },
        ),
    )
    train_cfg = RobomimicDiffusionTrainCfg(
        epochs=3,
        batch_size=11,
        seed=7,
        num_data_workers=2,
        prediction_horizon=12,
        observation_horizon=4,
        action_horizon=6,
        rollout_enabled=False,
        optimizer=OptimizerCfg(
            learning_rate=3.0e-4,
            weight_decay=1.0e-6,
            grad_clip_norm=10.0,
        ),
    )

    _apply_robomimic_diffusion_train_cfg(config, train_cfg)

    assert config.experiment.rollout.enabled is False
    assert config.train.num_epochs == 3
    assert config.train.batch_size == 11
    assert config.train.num_data_workers == 2
    assert config.train.seed == 7
    assert config.train.max_grad_norm == 10.0
    assert config.algo.horizon.prediction_horizon == 12
    assert config.algo.horizon.observation_horizon == 4
    assert config.algo.horizon.action_horizon == 6
    assert config.algo.optim_params["policy"]["learning_rate"]["initial"] == 3.0e-4
    assert config.algo.optim_params["policy"]["regularization"]["L2"] == 1.0e-6


def test_action_normalization_adds_missing_robomimic_env_kwargs(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from ioailab.agents.policy.backends.robomimic_diffusion import _normalize_actions

    dataset_path = tmp_path / "mimic.hdf5"
    with h5py.File(dataset_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"env_name": "GalbotG1-PickCube-Mimic-v0", "type": 2}
        )
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.array([[0.0], [1.0]], dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("robot_state", data=np.zeros((2, 3), dtype=np.float32))

    normalized_path, _metadata = _normalize_actions(dataset_path, tmp_path / "run")

    with h5py.File(normalized_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    assert env_args["env_kwargs"] == {}


def test_robomimic_diffusion_load_checkpoint_returns_policy_agent(
    tmp_path: Path,
) -> None:
    from ioailab.agents import PolicyAgent
    from ioailab.agents.policy import RobomimicDiffusionPolicy, PolicyCheckpoint

    checkpoint = PolicyCheckpoint(
        path=tmp_path / "model.pth", backend="robomimic_diffusion"
    )
    agent = RobomimicDiffusionPolicy().load_checkpoint(checkpoint)

    assert isinstance(agent, PolicyAgent)
    assert agent.metadata["checkpoint_path"] == tmp_path / "model.pth"
    assert agent.metadata["backend"] == "robomimic_diffusion"


def test_robomimic_diffusion_checkpoint_creates_backend_ref(tmp_path: Path) -> None:
    from ioailab.agents.policy import RobomimicDiffusionPolicy

    checkpoint = RobomimicDiffusionPolicy().checkpoint(
        tmp_path / "model.pth", tag="grasp"
    )

    assert checkpoint.path == tmp_path / "model.pth"
    assert checkpoint.backend == "robomimic_diffusion"
    assert checkpoint.metadata == {"tag": "grasp"}


def test_robomimic_diffusion_load_checkpoint_recovers_artifact_metadata(
    tmp_path: Path,
) -> None:
    from ioailab.agents.policy import RobomimicDiffusionPolicy

    checkpoint_path = tmp_path / "last.pth"
    checkpoint_path.write_bytes(b"fixture")
    (tmp_path / "action_norm_params.json").write_text(
        json.dumps(
            {
                "action_shape": [2],
                "action_min": [0.0, 1.0],
                "action_max": [2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "observation": {
                    "modalities": {"obs": {"low_dim": ["robot_joint_pos"], "rgb": []}}
                }
            }
        ),
        encoding="utf-8",
    )

    agent = RobomimicDiffusionPolicy().load_checkpoint(checkpoint_path)

    assert agent.metadata["backend"] == "robomimic_diffusion"
    assert agent.metadata["obs_keys"] == ["robot_joint_pos"]
    assert agent.metadata["action_shape"] == [2]


def test_policies_import_and_backend_are_robomimic_lazy_in_fresh_process() -> None:
    code = textwrap.dedent(
        """
        import json
        import sys

        from ioailab.agents.policy import Policy

        policy = Policy.from_backend("robomimic_diffusion")
        print(json.dumps({
            "policy_type": type(policy).__name__,
            "isaaclab_app_loaded": "isaaclab.app" in sys.modules,
            "gymnasium_loaded": "gymnasium" in sys.modules,
            "robomimic_loaded": "robomimic" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
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
        "policy_type": "RobomimicDiffusionPolicy",
        "isaaclab_app_loaded": False,
        "gymnasium_loaded": False,
        "robomimic_loaded": False,
        "torch_loaded": False,
    }


def test_robomimic_diffusers_ema_compat_adapts_model_keyword(monkeypatch):
    """Robomimic EMA compatibility lives in the robomimic backend."""

    from types import ModuleType

    robomimic = ModuleType("robomimic")
    algo = ModuleType("robomimic.algo")
    diffusion_policy = ModuleType("robomimic.algo.diffusion_policy")

    class FakeNetwork:
        def __init__(self):
            self.params = [object()]
            self.evaluated = False
            self.requires_grad = True

        def parameters(self):
            return self.params

        def eval(self):
            self.evaluated = True
            return self

        def requires_grad_(self, value):
            self.requires_grad = value
            return self

    class FakeEMAModel:
        def __init__(self, parameters, *args, **kwargs):
            self.parameters_arg = list(parameters)
            self.kwargs = kwargs
            self.copied_to = None
            self.stepped_with = None

        def copy_to(self, parameters):
            self.copied_to = list(parameters)

        def step(self, parameters):
            self.stepped_with = list(parameters)
            return "stepped"

    diffusion_policy.EMAModel = FakeEMAModel
    monkeypatch.setitem(sys.modules, "robomimic", robomimic)
    monkeypatch.setitem(sys.modules, "robomimic.algo", algo)
    monkeypatch.setitem(
        sys.modules, "robomimic.algo.diffusion_policy", diffusion_policy
    )
    monkeypatch.setattr(robomimic, "algo", algo, raising=False)
    monkeypatch.setattr(algo, "diffusion_policy", diffusion_policy, raising=False)

    from ioailab.agents.policy.backends.robomimic_diffusion import (
        _install_robomimic_diffusers_ema_compat,
    )

    _install_robomimic_diffusers_ema_compat()

    network = FakeNetwork()
    ema = diffusion_policy.EMAModel(model=network, power=0.75)

    assert diffusion_policy.EMAModel._ioailab_patched is True
    assert ema.parameters_arg
    assert ema.copied_to
    assert ema.step(network) == "stepped"
    assert ema.stepped_with == network.params


def test_docker_image_has_no_site_package_patch_directory() -> None:
    """Robomimic compatibility lives in package code, not Docker site patches."""

    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert not (ROOT / "docker" / "patches").exists()
    assert "docker/patches" not in dockerfile
    assert "robomimic_lang_utils.py" not in dockerfile


def test_robomimic_diffusion_checkpoint_agent_runs_lazy_inference(
    tmp_path: Path,
) -> None:
    import numpy as np
    import torch

    from ioailab.agents.policy import RobomimicDiffusionPolicy, PolicyCheckpoint

    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"fixture")
    loader_calls = []

    class FakeEnv:
        num_envs = 2

        def _get_obs_dict(self):
            return {"robot_state": np.array([[-1.0], [1.0]], dtype=np.float32)}

    def loader(path: Path):
        loader_calls.append(path)

        def policy(obs):
            assert set(obs) == {"robot_state"}
            return np.array([obs["robot_state"][0], 0.0], dtype=np.float32)

        return policy

    checkpoint = PolicyCheckpoint(
        checkpoint_path,
        "robomimic_diffusion",
        metadata={
            "policy_loader": loader,
            "obs_keys": ("robot_state",),
            "action_shape": (2,),
            "action_min": [-2.0, 10.0],
            "action_max": [2.0, 14.0],
        },
    )
    agent = RobomimicDiffusionPolicy().load_checkpoint(checkpoint)

    action = agent.act(FakeEnv())
    second = agent.act(FakeEnv())

    assert loader_calls == [checkpoint_path, checkpoint_path]
    assert isinstance(action, torch.Tensor)
    assert action.dtype == torch.float32
    assert action.shape == (2, 2)
    assert np.allclose(action, [[-2.0, 12.0], [2.0, 12.0]])
    assert np.allclose(second, action)


def test_checkpoint_observations_prefers_config_modalities_over_stale_obs_keys(
    tmp_path: Path,
) -> None:
    import numpy as np

    from ioailab.agents.policy import PolicyCheckpoint
    from ioailab.agents.policy.backends.robomimic_diffusion import (
        _checkpoint_observations,
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "observation": {
                    "modalities": {
                        "obs": {
                            "low_dim": ["robot_joint_pos"],
                            "rgb": ["front_head_rgb"],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeEnv:
        num_envs = 1

        def _get_obs_dict(self):
            return {
                "robot_joint_pos": np.zeros((1, 77), dtype=np.float32),
                "front_head_rgb": np.zeros((1, 224, 298, 3), dtype=np.uint8),
            }

    checkpoint = PolicyCheckpoint(
        tmp_path / "model.pth",
        "robomimic_diffusion",
        metadata={
            "config_path": str(config_path),
            "obs_keys": ("blue_block_pos", "blue_block_quat", "cube_pos", "cube_quat"),
        },
    )

    obs = _checkpoint_observations(checkpoint, FakeEnv(), env_id=0)

    assert set(obs) == {"robot_joint_pos", "front_head_rgb"}
    assert obs["robot_joint_pos"].shape == (77,)
    assert obs["front_head_rgb"].shape == (224, 298, 3)


def test_robomimic_vector_observation_selects_requested_env_row() -> None:
    import numpy as np

    from ioailab.agents.policy.backends.robomimic_diffusion import (
        _env_row_observation,
    )

    obs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    row = _env_row_observation(obs, env_id=1, num_envs=2)

    assert row.dtype == np.float32
    assert row.shape == (2,)
    assert np.allclose(row, [3.0, 4.0])


def test_robomimic_observation_horizon_reads_config(tmp_path: Path) -> None:
    import json

    from ioailab.agents.policy import PolicyCheckpoint
    from ioailab.agents.policy.backends.robomimic_diffusion import (
        _observation_horizon,
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"algo": {"horizon": {"observation_horizon": 3}}}),
        encoding="utf-8",
    )
    checkpoint = PolicyCheckpoint(
        tmp_path / "model.pth",
        "robomimic_diffusion",
        metadata={"config_path": str(config_path)},
    )

    assert _observation_horizon(checkpoint) == 3


def test_diffusion_policy_agent_keeps_per_env_history_and_episode_state(
    tmp_path: Path,
) -> None:
    import numpy as np

    torch = pytest.importorskip("torch")

    from ioailab.agents.policy import RobomimicDiffusionPolicy, PolicyCheckpoint

    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"fixture")
    seen_obs = []
    starts = []
    policies = []

    class FakeEnv:
        num_envs = 3

        def _get_obs_dict(self):
            return {"robot_state": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}

    class FakeRolloutPolicy:
        def __init__(self) -> None:
            self.index = len(policies)
            policies.append(self)

        def start_episode(self):
            starts.append(self.index)

        def __call__(self, obs):
            seen_obs.append((self.index, obs["robot_state"].copy()))
            return np.array([float(self.index) - 1.0, 0.0], dtype=np.float32)

    checkpoint = PolicyCheckpoint(
        checkpoint_path,
        "robomimic_diffusion",
        metadata={
            "policy_loader": lambda _path: FakeRolloutPolicy(),
            "obs_keys": ("robot_state",),
            "action_shape": (2,),
            "action_min": [-2.0, 10.0],
            "action_max": [2.0, 14.0],
            "observation_horizon": 2,
        },
    )
    agent = RobomimicDiffusionPolicy().load_checkpoint(checkpoint)

    agent.reset(FakeEnv())
    action = agent.act(FakeEnv())
    second = agent.act(FakeEnv())
    agent.reset(FakeEnv(), env_ids=(1,))
    third = agent.act(FakeEnv())

    assert starts == [0, 1, 2, 1]
    assert [policy.index for policy in policies] == [0, 1, 2]
    assert seen_obs[0][0] == 0
    assert seen_obs[0][1].shape == (2, 2)
    assert np.allclose(seen_obs[0][1], [[1.0, 2.0], [1.0, 2.0]])
    assert seen_obs[1][0] == 1
    assert np.allclose(seen_obs[1][1], [[3.0, 4.0], [3.0, 4.0]])
    assert seen_obs[2][0] == 2
    assert np.allclose(seen_obs[2][1], [[5.0, 6.0], [5.0, 6.0]])
    assert action.shape == (3, 2)
    assert np.allclose(second, action)
    assert np.allclose(third, action)


def test_robomimic_diffusion_checkpoint_agent_requires_metadata(tmp_path: Path) -> None:
    from ioailab.agents.policy import RobomimicDiffusionPolicy, PolicyCheckpoint

    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"fixture")
    agent = RobomimicDiffusionPolicy().load_checkpoint(
        PolicyCheckpoint(
            checkpoint_path,
            "robomimic_diffusion",
            metadata={"policy_loader": lambda path: lambda obs: [0.0]},
        )
    )

    class FakeEnv:
        num_envs = 1

        def _get_obs_dict(self):
            return {"obs": [1.0]}

    with pytest.raises(RuntimeError, match="obs_keys"):
        agent.act(FakeEnv())
