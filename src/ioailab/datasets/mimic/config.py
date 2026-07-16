"""User-facing Mimic stage configuration helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any

from isaaclab.envs.mimic_env_cfg import SubTaskConfig

DEFAULT_MIMIC_ACTION_NOISE = 0.005
DEFAULT_MIMIC_INTERPOLATION_STEPS = 15
DEFAULT_MIMIC_SELECTION_STRATEGY = "nearest_neighbor_object"
DEFAULT_MIMIC_SELECTION_STRATEGY_KWARGS = {"nn_k": 3}


@dataclass(frozen=True, slots=True)
class _MimicStageCfg:
    """One internal Mimic stage converted to IsaacLab SubTaskConfig."""

    name: str
    object_ref: str
    done_signal: str | None = None
    done: Callable[[Any], Any] | None = None
    offset_range: tuple[int, int] = (0, 0)
    selection_strategy: str = DEFAULT_MIMIC_SELECTION_STRATEGY
    selection_strategy_kwargs: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_MIMIC_SELECTION_STRATEGY_KWARGS)
    )
    action_noise: float = DEFAULT_MIMIC_ACTION_NOISE
    interpolation_steps: int = DEFAULT_MIMIC_INTERPOLATION_STEPS
    fixed_steps: int = 0
    apply_noise_during_interpolation: bool = False
    next_stage: str | None = None

    def __post_init__(self) -> None:
        """Normalize stage fields and validate simple numeric settings."""

        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "object_ref", str(self.object_ref))
        done_signal = self.done_signal
        if done_signal is None and self.done is not None:
            done_signal = self.name
        object.__setattr__(
            self, "done_signal", None if done_signal is None else str(done_signal)
        )
        object.__setattr__(
            self,
            "offset_range",
            (int(self.offset_range[0]), int(self.offset_range[1])),
        )
        object.__setattr__(
            self, "selection_strategy_kwargs", dict(self.selection_strategy_kwargs)
        )
        if self.interpolation_steps < 0:
            raise ValueError("interpolation_steps must be non-negative.")
        if self.fixed_steps < 0:
            raise ValueError("fixed_steps must be non-negative.")

    def to_isaaclab(self) -> SubTaskConfig:
        """Return the IsaacLab Mimic config for this ioailab stage."""

        return SubTaskConfig(
            object_ref=self.object_ref,
            subtask_term_signal=self.done_signal,
            subtask_term_offset_range=self.offset_range,
            selection_strategy=self.selection_strategy,
            selection_strategy_kwargs=dict(self.selection_strategy_kwargs),
            action_noise=float(self.action_noise),
            num_interpolation_steps=int(self.interpolation_steps),
            num_fixed_steps=int(self.fixed_steps),
            apply_noise_during_interpolation=bool(
                self.apply_noise_during_interpolation
            ),
            description=self.name,
            next_subtask_description=self.next_stage,
        )


@dataclass(frozen=True, slots=True)
class MimicCfg:
    """ioailab Mimic settings applied to an IsaacLab Mimic env cfg."""

    eef_name: str
    stages: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]
    datagen_name: str | None = None
    converter: Any | None = None
    object_names: Sequence[str] = ()
    stage_signals: Mapping[str, Callable[[Any], Any]] = field(default_factory=dict)
    stage_cfgs: tuple[_MimicStageCfg, ...] = field(init=False, repr=False)
    generation_num_trials: int = 10
    generation_guarantee: bool = True
    generation_keep_failed: bool = True
    generation_select_src_per_phase: bool = True
    generation_transform_first_robot_pose: bool = False
    generation_interpolate_from_last_target_pose: bool = True
    generation_relative: bool = True
    max_num_failures: int = 25
    seed: int = 1

    def __post_init__(self) -> None:
        """Normalize config values."""

        object.__setattr__(self, "eef_name", str(self.eef_name))
        stage_cfgs = _normalize_stages(self.stages)
        object.__setattr__(self, "stage_cfgs", stage_cfgs)
        datagen_name = None if self.datagen_name is None else str(self.datagen_name)
        object.__setattr__(self, "datagen_name", datagen_name)
        object_names = tuple(str(name) for name in self.object_names)
        if not object_names:
            object_names = tuple(
                dict.fromkeys(stage.object_ref for stage in stage_cfgs)
            )
        object.__setattr__(
            self,
            "object_names",
            object_names,
        )
        stage_signals = dict(self.stage_signals)
        for stage in stage_cfgs:
            if stage.done is not None and stage.done_signal is not None:
                stage_signals.setdefault(str(stage.done_signal), stage.done)
        object.__setattr__(self, "stage_signals", stage_signals)
        if not self.stage_cfgs:
            raise ValueError("MimicCfg.stages must contain at least one stage.")
        missing_signals = [
            str(stage.done_signal)
            for stage in stage_cfgs
            if stage.done_signal is not None
            and stage.done is None
            and str(stage.done_signal) not in stage_signals
        ]
        if missing_signals:
            raise ValueError(
                "Mimic stages require done callables or stage_signals entries for: "
                f"{missing_signals!r}."
            )
        if self.generation_num_trials < 1:
            raise ValueError("generation_num_trials must be greater than zero.")
        if self.max_num_failures < 0:
            raise ValueError("max_num_failures must be non-negative.")

    def apply_to(self, env_cfg: Any) -> None:
        """Apply this ioailab config to an IsaacLab Mimic env cfg."""

        env_cfg.mimic = self
        env_cfg.datagen_config.name = self.datagen_name or _default_datagen_name(
            env_cfg
        )
        env_cfg.datagen_config.generation_guarantee = bool(self.generation_guarantee)
        env_cfg.datagen_config.generation_keep_failed = bool(
            self.generation_keep_failed
        )
        env_cfg.datagen_config.generation_num_trials = int(self.generation_num_trials)
        env_cfg.datagen_config.generation_select_src_per_subtask = bool(
            self.generation_select_src_per_phase
        )
        env_cfg.datagen_config.generation_transform_first_robot_pose = bool(
            self.generation_transform_first_robot_pose
        )
        env_cfg.datagen_config.generation_interpolate_from_last_target_pose = bool(
            self.generation_interpolate_from_last_target_pose
        )
        env_cfg.datagen_config.generation_relative = bool(self.generation_relative)
        env_cfg.datagen_config.max_num_failures = int(self.max_num_failures)
        env_cfg.datagen_config.seed = int(self.seed)
        env_cfg.subtask_configs = {
            self.eef_name: [stage.to_isaaclab() for stage in self.stage_cfgs]
        }
        env_cfg.task_constraint_configs = []


def _default_datagen_name(env_cfg: Any) -> str:
    """Infer a readable Mimic datagen name from the env cfg class name."""

    cfg_name = type(env_cfg).__name__
    cfg_name = re.sub(r"EnvCfg$", "", cfg_name)
    cfg_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cfg_name)
    return cfg_name.lower()


def _normalize_stages(
    stages: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> tuple[_MimicStageCfg, ...]:
    """Return internal stage cfgs from compact user-facing stage definitions."""

    if isinstance(stages, Mapping):
        stage_items = [
            _stage_from_mapping(name=name, values=values)
            for name, values in stages.items()
        ]
    else:
        stage_items = [
            _stage_from_mapping(name=None, values=values) for values in stages
        ]
    return tuple(stage_items)


def _stage_from_mapping(
    *, name: str | None, values: Mapping[str, Any]
) -> _MimicStageCfg:
    """Build one internal stage cfg from a compact mapping."""

    if not isinstance(values, Mapping):
        raise TypeError("Each Mimic stage must be a mapping.")

    stage_name = str(name if name is not None else values["name"])
    object_ref = values.get("object_ref", values.get("object"))
    if object_ref is None:
        raise ValueError(f"Mimic stage {stage_name!r} requires object or object_ref.")

    return _MimicStageCfg(
        name=stage_name,
        object_ref=str(object_ref),
        done_signal=values.get("done_signal"),
        done=values.get("done"),
        offset_range=values.get("offset_range", (0, 0)),
        selection_strategy=values.get(
            "selection_strategy", DEFAULT_MIMIC_SELECTION_STRATEGY
        ),
        selection_strategy_kwargs=values.get(
            "selection_strategy_kwargs", DEFAULT_MIMIC_SELECTION_STRATEGY_KWARGS
        ),
        action_noise=values.get("action_noise", DEFAULT_MIMIC_ACTION_NOISE),
        interpolation_steps=values.get(
            "interpolation_steps", DEFAULT_MIMIC_INTERPOLATION_STEPS
        ),
        fixed_steps=values.get("fixed_steps", 0),
        apply_noise_during_interpolation=values.get(
            "apply_noise_during_interpolation", False
        ),
        next_stage=values.get(
            "next_stage", values.get("next_phase", values.get("next"))
        ),
    )


__all__ = ["MimicCfg"]
