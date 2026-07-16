"""Robomimic-backed policy adapters."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Protocol

from ioailab.agents.policy.action_source import PolicyAgent
from ioailab.datasets import DatasetRef, ensure_dataset_ref
from ioailab.agents.policy.checkpoint import PolicyCheckpoint
from ioailab.agents.policy.train import OptimizerCfg, Policy, PolicyTrainCfg


@dataclass(frozen=True, slots=True)
class RobomimicDiffusionTrainCfg(PolicyTrainCfg):
    """Training settings for robomimic Diffusion Policy."""

    num_data_workers: int | None = None
    prediction_horizon: int | None = None
    observation_horizon: int | None = None
    action_horizon: int | None = None
    normalize_actions: bool = True
    rollout_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate robomimic Diffusion Policy settings."""

        PolicyTrainCfg.__post_init__(self)
        if self.num_data_workers is not None and self.num_data_workers < 0:
            raise ValueError(
                "RobomimicDiffusionTrainCfg.num_data_workers must not be negative."
            )
        for field_name in (
            "prediction_horizon",
            "observation_horizon",
            "action_horizon",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(
                    f"RobomimicDiffusionTrainCfg.{field_name} must be greater "
                    "than zero when provided."
                )


class RobomimicDiffusionTrainer(Protocol):
    """Training backend protocol for robomimic Diffusion Policy."""

    def train(
        self,
        train_config: RobomimicDiffusionTrainCfg,
        dataset: DatasetRef,
        *,
        backend: str,
    ) -> PolicyCheckpoint:
        """Train on ``dataset`` and return a checkpoint reference."""


def checkpoint_metadata_from_artifacts(checkpoint_path: str | Path) -> dict[str, Any]:
    """Recover inference metadata from files emitted next to a robomimic run."""

    path = Path(checkpoint_path)
    for root in (path.parent, *path.parents):
        action_path = root / "action_norm_params.json"
        config_path = root / "config.json"
        if not action_path.is_file() or not config_path.is_file():
            continue

        action_metadata = json.loads(action_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        obs_modalities = (
            config.get("observation", {}).get("modalities", {}).get("obs", {})
        )
        low_dim_obs_keys = list(obs_modalities.get("low_dim", []))
        rgb_obs_keys = list(obs_modalities.get("rgb", []))
        obs_keys = [*low_dim_obs_keys, *rgb_obs_keys]
        metadata = {
            "obs_keys": obs_keys,
            "low_dim_obs_keys": low_dim_obs_keys,
            "rgb_obs_keys": rgb_obs_keys,
            "config_path": str(config_path),
            **action_metadata,
        }
        train_data = config.get("train", {}).get("data")
        if train_data:
            metadata["dataset_path"] = str(train_data)
        return metadata
    return {}


def _install_robomimic_diffusers_ema_compat() -> None:
    """Adapt robomimic 0.5 Diffusion Policy to the installed Diffusers EMA API.

    Robomimic 0.5 constructs ``diffusers.training_utils.EMAModel`` with
    ``EMAModel(model=nets, ...)``. Diffusers 0.35 requires the parameter
    iterable as the first argument instead. ioailab keeps the newer
    Diffusers version because LeRobot requires ``diffusers>=0.27.2``. This
    backend-local adapter keeps robomimic training and checkpoint loading
    working without Docker site-package edits.
    """

    import robomimic.algo.diffusion_policy as diffusion_policy

    original_ema_model = diffusion_policy.EMAModel
    if getattr(original_ema_model, "_ioailab_patched", False):
        return

    class _CompatEMAModel(original_ema_model):
        def __init__(self, parameters=None, *args, model=None, **kwargs):
            import copy

            averaged_model = None
            if parameters is None and model is not None:
                averaged_model = copy.deepcopy(model)
                if hasattr(averaged_model, "eval"):
                    averaged_model.eval()
                if hasattr(averaged_model, "requires_grad_"):
                    averaged_model.requires_grad_(False)
                parameters = model.parameters()
            super().__init__(parameters, *args, **kwargs)
            if averaged_model is not None:
                self.averaged_model = averaged_model
                self.copy_to(self.averaged_model.parameters())

        def step(self, parameters):
            if hasattr(parameters, "parameters"):
                parameters = parameters.parameters()
            result = super().step(parameters)
            if hasattr(self, "averaged_model"):
                self.copy_to(self.averaged_model.parameters())
            return result

    _CompatEMAModel._ioailab_patched = True
    diffusion_policy.EMAModel = _CompatEMAModel


@dataclass(slots=True)
class RobomimicDiffusionPolicyTrainer:
    """Robomimic Diffusion Policy trainer using the robomimic API directly."""

    def train(
        self,
        train_config: RobomimicDiffusionTrainCfg,
        dataset: DatasetRef,
        *,
        backend: str,
    ) -> PolicyCheckpoint:
        """Train a robomimic Diffusion Policy."""

        import time

        import torch
        from robomimic.config import config_factory
        from robomimic.scripts.train import train as robomimic_train

        _install_robomimic_diffusers_ema_compat()
        if not train_config.normalize_actions:
            raise ValueError(
                "RobomimicDiffusionTrainCfg.normalize_actions=False is not supported "
                "because policy inference requires saved action normalization metadata."
            )

        dataset_path = _resolve_training_data_path(dataset)
        base_dir = (
            Path(train_config.output_dir).resolve()
            if train_config.output_dir
            else Path("outputs/robomimic").resolve()
        )
        output_dir = base_dir / time.strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        normalized_path, action_metadata = _normalize_actions(
            dataset_path,
            output_dir,
            env_name=dataset.task_id,
        )
        obs_modalities = _read_obs_modalities_from_hdf5(normalized_path)
        low_dim_obs_keys = obs_modalities["low_dim"]
        rgb_obs_keys = obs_modalities["rgb"]
        obs_keys = (*low_dim_obs_keys, *rgb_obs_keys)

        config = config_factory(algo_name="diffusion_policy")
        config.experiment.name = dataset.task_id or "ioailab"
        config.train.data = str(normalized_path)
        config.train.output_dir = str(output_dir)
        config.observation.modalities.obs.low_dim = list(low_dim_obs_keys)
        config.observation.modalities.obs.rgb = list(rgb_obs_keys)
        _apply_robomimic_diffusion_train_cfg(config, train_config)
        config.train.seq_length = config.algo.horizon.prediction_horizon
        config.train.pad_seq_length = True
        config.train.frame_stack = config.algo.horizon.observation_horizon
        config.train.pad_frame_stack = True
        _configure_dataset_loading(config, has_rgb_observations=bool(rgb_obs_keys))

        config_path = output_dir / "config.json"
        config.dump(filename=str(config_path))

        device = torch.device(
            train_config.device
            if train_config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        robomimic_train(config, device)

        checkpoint_path = _find_best_checkpoint(output_dir, config)
        return PolicyCheckpoint(
            path=checkpoint_path,
            backend=backend,
            metadata={
                "obs_keys": tuple(obs_keys),
                "low_dim_obs_keys": tuple(low_dim_obs_keys),
                "rgb_obs_keys": tuple(rgb_obs_keys),
                "dataset_path": str(normalized_path),
                "config_path": str(config_path),
                **action_metadata,
            },
        )


def _apply_robomimic_diffusion_train_cfg(
    config: Any, train_config: RobomimicDiffusionTrainCfg
) -> None:
    """Apply ioailab train settings to a robomimic Diffusion Policy config."""

    config.experiment.rollout.enabled = bool(train_config.rollout_enabled)
    if train_config.epochs is not None:
        config.train.num_epochs = int(train_config.epochs)
    if train_config.batch_size is not None:
        config.train.batch_size = int(train_config.batch_size)
    if train_config.num_data_workers is not None:
        config.train.num_data_workers = int(train_config.num_data_workers)
    if train_config.seed is not None:
        config.train.seed = int(train_config.seed)

    horizon = config.algo.horizon
    if train_config.prediction_horizon is not None:
        horizon.prediction_horizon = int(train_config.prediction_horizon)
    if train_config.observation_horizon is not None:
        horizon.observation_horizon = int(train_config.observation_horizon)
    if train_config.action_horizon is not None:
        horizon.action_horizon = int(train_config.action_horizon)

    _apply_robomimic_optimizer_cfg(config, train_config.optimizer)


def _apply_robomimic_optimizer_cfg(config: Any, optimizer: OptimizerCfg) -> None:
    """Apply shared optimizer settings to robomimic policy optimizer config."""

    policy_optim = _cfg_get(config.algo.optim_params, "policy")
    learning_rate = _cfg_get(policy_optim, "learning_rate")
    _cfg_set(learning_rate, "initial", float(optimizer.learning_rate))

    regularization = _cfg_get(policy_optim, "regularization")
    _cfg_set(regularization, "L2", float(optimizer.weight_decay))

    if optimizer.grad_clip_norm is not None:
        config.train.max_grad_norm = float(optimizer.grad_clip_norm)


def _cfg_get(config: Any, key: str) -> Any:
    """Read a key from a robomimic config object or plain dict."""

    if isinstance(config, dict):
        return config[key]
    try:
        return config[key]
    except (KeyError, TypeError, AttributeError):
        pass
    return getattr(config, key)


def _cfg_set(config: Any, key: str, value: Any) -> None:
    """Set a key on a robomimic config object or plain dict."""

    if isinstance(config, dict):
        config[key] = value
        return
    try:
        config[key] = value
        return
    except (KeyError, TypeError, AttributeError):
        pass
    setattr(config, key, value)


_ROBOMIMIC_GYM_ENV_TYPE = 2


def _normalize_actions(
    dataset_path: Path, output_dir: Path, *, env_name: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """Normalize dataset actions to [-1, 1] for Diffusion Policy training.

    Creates a normalized copy of the dataset in the output directory. Saves
    normalization parameters alongside for inference-time denormalization.
    """

    import json
    import shutil

    import h5py
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / f"{dataset_path.stem}_normalized.hdf5"
    shutil.copyfile(dataset_path, normalized_path)
    _ensure_robomimic_env_metadata(normalized_path, env_name=env_name)

    with h5py.File(normalized_path, "r") as f:
        demos = sorted(f["data"].keys())
        all_actions = np.concatenate(
            [np.asarray(f[f"data/{d}/actions"], dtype=np.float64) for d in demos],
            axis=0,
        )

    action_min = all_actions.min(axis=0)
    action_max = all_actions.max(axis=0)
    scale = action_max - action_min
    safe_scale = np.where(np.abs(scale) > 1e-8, scale, 1.0)

    with h5py.File(normalized_path, "r+") as f:
        for demo_name in sorted(f["data"].keys()):
            action_path = f"data/{demo_name}/actions"
            actions = np.asarray(f[action_path], dtype=np.float32)
            normalized = 2.0 * ((actions - action_min) / safe_scale) - 1.0
            normalized = np.where(np.abs(scale) > 1e-8, normalized, 0.0).astype(
                np.float32
            )
            del f[action_path]
            f.create_dataset(action_path, data=normalized)

    norm_params = {
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "action_shape": tuple(int(dim) for dim in all_actions.shape[1:]),
    }
    (output_dir / "action_norm_params.json").write_text(
        json.dumps(norm_params, indent=2)
    )

    return normalized_path, norm_params


def _ensure_robomimic_env_metadata(
    dataset_path: Path, *, env_name: str | None = None
) -> None:
    """Ensure robomimic can read ioailab dataset environment metadata."""

    import json

    import h5py

    with h5py.File(dataset_path, "r+") as f:
        data = f.get("data")
        if data is None:
            return

        raw_env_args = data.attrs.get("env_args")
        if isinstance(raw_env_args, bytes):
            raw_env_args = raw_env_args.decode("utf-8")
        env_args = json.loads(raw_env_args) if raw_env_args else {}
        if not isinstance(env_args, dict):
            env_args = {}
        if env_name and not env_args.get("env_name"):
            env_args["env_name"] = str(env_name)
        else:
            env_args.setdefault("env_name", "")
        env_args.setdefault("type", _ROBOMIMIC_GYM_ENV_TYPE)
        if not isinstance(env_args.get("env_kwargs"), dict):
            env_args["env_kwargs"] = {}
        data.attrs["env_args"] = json.dumps(env_args)


def _configure_dataset_loading(
    config: Any, *, has_rgb_observations: bool = False
) -> None:
    """Configure robomimic dataset loading for ioailab HDF5 files."""

    # ioailab expert datasets store current observations under data/<demo>/obs
    # but do not materialize data/<demo>/next_obs. Diffusion Policy training here
    # is behavior cloning and does not need next_obs.
    config.train.hdf5_load_next_obs = False

    # Robomimic's default diffusion-policy config caches the full SequenceDataset
    # when hdf5_cache_mode == "all". For vision datasets this expands every RGB
    # frame-stack / sequence into RAM; 256 demos at 298x224 can be SIGKILLed by
    # the OS before training starts. Keep low-dimensional observations cached,
    # and stream RGB from HDF5.
    if has_rgb_observations:
        config.train.hdf5_cache_mode = "low_dim"


GROUND_TRUTH_OBS_KEYS = frozenset(
    {
        "cube_pos",
        "cube_quat",
        "blue_block_pos",
        "blue_block_quat",
    }
)


def _read_obs_keys_from_hdf5(dataset_path: Path) -> list[str]:
    """Read non-action, non-ground-truth observation key names from an HDF5 dataset."""

    modalities = _read_obs_modalities_from_hdf5(dataset_path)
    return [*modalities["low_dim"], *modalities["rgb"]]


def _read_obs_modalities_from_hdf5(dataset_path: Path) -> dict[str, list[str]]:
    """Classify robomimic observation keys into low-dimensional and RGB modalities.

    Cube/object pose ground-truth observations are intentionally excluded so
    vision-based policies train from camera observations plus robot proprioception.
    """

    import h5py

    with h5py.File(dataset_path, "r") as f:
        demos = sorted(f["data"].keys())
        if not demos:
            raise ValueError(f"No demos found in {dataset_path}")
        obs_node = f[f"data/{demos[0]}/obs"]
        if not hasattr(obs_node, "keys"):
            raise ValueError(
                f"Expected grouped observations under data/{demos[0]}/obs in {dataset_path}"
            )

        low_dim_keys: list[str] = []
        rgb_keys: list[str] = []
        for key in sorted(obs_node.keys()):
            if key == "actions" or key in GROUND_TRUTH_OBS_KEYS:
                continue
            value = obs_node[key]
            shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
            if _is_rgb_observation_key(key, shape):
                rgb_keys.append(key)
            else:
                low_dim_keys.append(key)
        return {"low_dim": low_dim_keys, "rgb": rgb_keys}


def _is_rgb_observation_key(key: str, shape: tuple[int, ...]) -> bool:
    """Return whether an HDF5 observation dataset should use robomimic's RGB modality."""

    name = key.lower()
    return (
        len(shape) >= 4
        and shape[-1] in {3, 4}
        and ("rgb" in name or "image" in name or "camera" in name)
    )


def _resolve_training_data_path(dataset: DatasetRef) -> Path:
    """Resolve the actual HDF5 file path for training.

    If the dataset path exists, use it directly. If not (e.g. mimic output
    not yet generated), walk provenance to find the most recent existing
    source file.
    """

    path = Path(dataset.path)
    if path.is_file():
        return path
    for prov in reversed(dataset.provenance):
        if prov.source_path is not None and Path(prov.source_path).is_file():
            return Path(prov.source_path)
    raise FileNotFoundError(
        f"Training dataset not found at {path}. "
        f"If using mimic(), run the augmentation generation first "
        f"(e.g. via run_mimic_generation(...))."
    )


def _find_best_checkpoint(output_dir: Path, config: Any = None) -> Path:
    """Find the best model checkpoint under the training output directory."""

    candidates = sorted(output_dir.rglob("model_best_training.pth"))
    if candidates:
        return candidates[-1]
    candidates = sorted(output_dir.rglob("*.pth"))
    # Exclude backup files
    candidates = [c for c in candidates if "_bak" not in c.name]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No checkpoint found in {output_dir}")


@dataclass(slots=True)
class _CheckpointActionSource:
    """Lazy robomimic checkpoint inference action source."""

    checkpoint: PolicyCheckpoint
    _policies: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _needs_start_episode: dict[int, bool] = field(
        default_factory=dict, init=False, repr=False
    )
    _obs_history: dict[int, deque] = field(default_factory=dict, init=False, repr=False)

    def reset(self, env: Any = None, env_ids: Any = None) -> None:
        """Mark selected env-row policy states for episode reset."""

        ids = _target_env_ids(env, env_ids, known_ids=self._policies)
        for env_id in ids:
            self._needs_start_episode[env_id] = True
            self._obs_history.pop(env_id, None)

    def __call__(self, env: Any, env_ids: Any = None) -> Any:
        """Return policy actions for all requested env rows."""

        ids = _target_env_ids(env, env_ids)
        rows = []
        for env_id in ids:
            policy = self._policy_for_env(env_id)
            if self._needs_start_episode.get(env_id, True):
                start_episode = getattr(policy, "start_episode", None)
                if callable(start_episode):
                    start_episode()
                self._needs_start_episode[env_id] = False
            obs = _checkpoint_observations(self.checkpoint, env, env_id=env_id)
            action = _call_policy(
                policy, _history_observations(self, env_id=env_id, obs=obs)
            )
            rows.append(_denormalize_and_validate_action(self.checkpoint, action))
        return _as_action_tensor(rows, env)

    def _policy_for_env(self, env_id: int) -> Any:
        env_id = int(env_id)
        if env_id not in self._policies:
            self._policies[env_id] = self._load_policy()
            self._needs_start_episode[env_id] = True
        return self._policies[env_id]

    def _load_policy(self) -> Any:
        if not self.checkpoint.path.is_file():
            raise FileNotFoundError(
                f"Robomimic policy checkpoint not found: {self.checkpoint.path}"
            )

        loader = self.checkpoint.metadata.get("policy_loader")
        if loader is None:
            try:
                _install_robomimic_diffusers_ema_compat()
                from robomimic.utils.file_utils import policy_from_checkpoint
            except ImportError as exc:
                raise RuntimeError(
                    "Robomimic policy inference requires robomimic or checkpoint metadata['policy_loader']."
                ) from exc

            def loader(path: Path) -> Any:
                policy, _ckpt_dict = policy_from_checkpoint(ckpt_path=str(path))
                return policy

        return loader(self.checkpoint.path)


def _target_env_ids(
    env: Any = None,
    env_ids: Any = None,
    *,
    known_ids: dict[int, Any] | None = None,
) -> tuple[int, ...]:
    """Return concrete env-row ids for full or subset policy calls."""

    if env_ids is not None:
        return tuple(int(env_id) for env_id in env_ids)
    if env is not None:
        return tuple(range(int(getattr(env, "num_envs", 1))))
    if known_ids:
        return tuple(sorted(int(env_id) for env_id in known_ids))
    return ()


def _history_observations(
    source: _CheckpointActionSource, *, env_id: int, obs: dict[str, Any]
) -> dict[str, Any]:
    """Stack one env row's recent observations into robomimic's [T, ...] form."""

    import numpy as np

    horizon = _observation_horizon(source.checkpoint)
    history = source._obs_history.setdefault(int(env_id), deque())
    history.append(obs)
    while len(history) < horizon:
        history.appendleft(obs)
    while len(history) > horizon:
        history.popleft()
    if horizon <= 1:
        return obs
    return {key: np.stack([frame[key] for frame in history], axis=0) for key in obs}


def _observation_horizon(checkpoint: PolicyCheckpoint) -> int:
    """Return the robomimic observation horizon needed for checkpoint inference."""

    value = checkpoint.metadata.get("observation_horizon")
    if value is not None:
        return max(1, int(value))
    config_path = checkpoint.metadata.get("config_path")
    if config_path and Path(config_path).is_file():
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        value = config.get("algo", {}).get("horizon", {}).get("observation_horizon")
        if value is not None:
            return max(1, int(value))
    return 1


def _checkpoint_observations(
    checkpoint: PolicyCheckpoint, env: Any, *, env_id: int
) -> dict[str, Any]:
    """Build one env row's observation dict required by checkpoint metadata."""

    obs_keys = _checkpoint_obs_keys(checkpoint)

    if hasattr(env, "_get_obs_dict"):
        available = env._get_obs_dict()
    else:
        obs_manager = getattr(
            getattr(env, "unwrapped", env), "observation_manager", None
        )
        if obs_manager is None:
            raise RuntimeError(
                "Policy inference requires a workflow env observation manager."
            )
        obs_groups = obs_manager.compute()
        available = {}
        for _group, values in obs_groups.items():
            if isinstance(values, dict):
                available.update(values)
            else:
                available["obs"] = values

    missing = [key for key in obs_keys if key not in available]
    if missing:
        raise RuntimeError(
            f"Robomimic checkpoint observation metadata references missing key(s): {missing}"
        )
    num_envs = int(getattr(env, "num_envs", 1))
    return {
        key: _env_row_observation(available[key], env_id=env_id, num_envs=num_envs)
        for key in obs_keys
    }


def _checkpoint_obs_keys(checkpoint: PolicyCheckpoint) -> tuple[str, ...]:
    """Return inference observation keys from checkpoint modality metadata."""

    low_dim_keys = checkpoint.metadata.get("low_dim_obs_keys")
    rgb_keys = checkpoint.metadata.get("rgb_obs_keys")
    if low_dim_keys is not None or rgb_keys is not None:
        obs_keys = [*(low_dim_keys or ()), *(rgb_keys or ())]
    else:
        obs_keys = _checkpoint_config_obs_keys(checkpoint) or checkpoint.metadata.get(
            "obs_keys"
        )

    if obs_keys is None:
        raise RuntimeError(
            "Robomimic checkpoint metadata must include obs_keys for inference."
        )
    obs_keys = tuple(str(key) for key in obs_keys)
    if not obs_keys:
        raise RuntimeError("Robomimic checkpoint metadata obs_keys must not be empty.")
    return obs_keys


def _checkpoint_config_obs_keys(checkpoint: PolicyCheckpoint) -> tuple[str, ...] | None:
    """Read observation keys from a robomimic config next to the checkpoint."""

    config_path = checkpoint.metadata.get("config_path")
    if not config_path or not Path(config_path).is_file():
        return None
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    obs_modalities = config.get("observation", {}).get("modalities", {}).get("obs", {})
    obs_keys = [*obs_modalities.get("low_dim", ()), *obs_modalities.get("rgb", ())]
    return tuple(str(key) for key in obs_keys) if obs_keys else None


def _env_row_observation(value: Any, *, env_id: int, num_envs: int) -> Any:
    """Convert vector-env observations to one env row's robomimic observation."""

    import numpy as np

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required by runtime inference
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if int(num_envs) > 0 and array.shape[:1] == (int(num_envs),):
        array = array[int(env_id)]
    return (
        array.astype(np.float32, copy=False)
        if np.issubdtype(array.dtype, np.number)
        else array
    )


def _call_policy(policy: Any, obs: dict[str, Any]) -> Any:
    """Call a robomimic-like policy object using common inference APIs."""

    if callable(policy):
        return policy(obs)
    get_action = getattr(policy, "get_action", None)
    if callable(get_action):
        return get_action(obs)
    raise TypeError(
        "Loaded robomimic policy must be callable or expose get_action(obs)."
    )


def _denormalize_and_validate_action(checkpoint: PolicyCheckpoint, action: Any) -> Any:
    """Denormalize one policy action row and validate task action metadata."""

    import numpy as np

    if "action_shape" not in checkpoint.metadata:
        raise RuntimeError(
            "Robomimic checkpoint metadata must include action_shape for inference."
        )
    if (
        "action_min" not in checkpoint.metadata
        or "action_max" not in checkpoint.metadata
    ):
        raise RuntimeError(
            "Robomimic checkpoint metadata must include action_min and action_max normalization values."
        )

    expected_tail = tuple(int(dim) for dim in checkpoint.metadata["action_shape"])
    action_array = np.asarray(action, dtype=np.float32)
    action_min = np.asarray(checkpoint.metadata["action_min"], dtype=np.float32)
    action_max = np.asarray(checkpoint.metadata["action_max"], dtype=np.float32)
    denormalized = ((action_array + 1.0) / 2.0) * (action_max - action_min) + action_min

    if denormalized.shape == expected_tail:
        return denormalized.astype(np.float32, copy=False)
    if denormalized.shape == (1, *expected_tail):
        return denormalized.reshape(expected_tail).astype(np.float32, copy=False)
    raise ValueError(
        f"Robomimic policy emitted action shape {denormalized.shape}; "
        f"expected {expected_tail} for one env row."
    )


def _as_action_tensor(rows: list[Any], env: Any) -> Any:
    """Stack per-env action rows and move them to the env device."""

    import numpy as np
    import torch

    denormalized = np.stack(rows, axis=0).astype(np.float32, copy=False)
    device = getattr(getattr(env, "unwrapped", env), "device", None)
    if device is None:
        return torch.as_tensor(denormalized, dtype=torch.float32)
    return torch.as_tensor(denormalized, dtype=torch.float32, device=device)


@dataclass(slots=True)
class RobomimicDiffusionPolicy(Policy):
    """ioailab adapter for robomimic native Diffusion Policy."""

    backend: str = "robomimic_diffusion"
    trainer: RobomimicDiffusionTrainer | None = field(default=None, repr=False)

    def train(
        self,
        dataset: DatasetRef | str | Path,
        train_config: RobomimicDiffusionTrainCfg | None = None,
    ) -> PolicyCheckpoint:
        """Train this policy on a dataset reference.

        Preferred form::

            cfg = RobomimicDiffusionTrainCfg(output_dir="outputs/pick_cube", epochs=1)
            checkpoint = policy.train(dataset, cfg)
        """

        dataset_ref = ensure_dataset_ref(dataset)
        train_config = train_config or RobomimicDiffusionTrainCfg()
        if not isinstance(train_config, RobomimicDiffusionTrainCfg):
            raise TypeError(
                "RobomimicDiffusionPolicy.train expects a RobomimicDiffusionTrainCfg."
            )

        trainer = (
            self.trainer
            if self.trainer is not None
            else RobomimicDiffusionPolicyTrainer()
        )
        return trainer.train(train_config, dataset_ref, backend=self.backend)

    def checkpoint(
        self, path: PolicyCheckpoint | str | Path, **metadata: Any
    ) -> PolicyCheckpoint:
        """Create a checkpoint reference for this policy backend."""

        if isinstance(path, PolicyCheckpoint):
            if metadata:
                checkpoint_metadata = {**path.metadata, **metadata}
                return PolicyCheckpoint(
                    path=path.path, backend=path.backend, metadata=checkpoint_metadata
                )
            return path
        return PolicyCheckpoint(path=path, backend=self.backend, metadata=metadata)

    def load_checkpoint(self, checkpoint: PolicyCheckpoint | str | Path) -> PolicyAgent:
        """Create a ``PolicyAgent`` for replay or evaluation."""

        if isinstance(checkpoint, PolicyCheckpoint):
            checkpoint_ref = checkpoint
            if not checkpoint_ref.metadata:
                metadata = checkpoint_metadata_from_artifacts(checkpoint_ref.path)
                if metadata:
                    checkpoint_ref = PolicyCheckpoint(
                        path=checkpoint_ref.path,
                        backend=checkpoint_ref.backend,
                        metadata=metadata,
                    )
        else:
            checkpoint_path = Path(checkpoint)
            checkpoint_ref = PolicyCheckpoint(
                checkpoint_path,
                self.backend,
                metadata=checkpoint_metadata_from_artifacts(checkpoint_path),
            )
        return PolicyAgent(
            _CheckpointActionSource(checkpoint_ref),
            checkpoint_path=checkpoint_ref.path,
            backend=checkpoint_ref.backend,
            **checkpoint_ref.metadata,
        )
