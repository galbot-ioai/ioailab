"""Base reset-time domain randomizer for ioailab tasks.

A ``Randomizer`` is task-agnostic IsaacLab event mechanics. Subclasses implement
``apply(env, env_ids, **params)`` as a ``@staticmethod`` and are referenced as the
``func`` of an IsaacLab ``EventTermCfg`` (``mode="reset"``):

    EventTerm(func=ObjectPoseRandomizer.apply, mode="reset", params={...})

Task-specific object names, ranges, material categories, and event policy stay in
each task's ``mdp/events.py`` (the event ``params``); the randomizer only owns the
reusable sampling/binding mechanics. The base class supplies the shared env-id,
USD stage, and path helpers used across domains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import torch

from ioailab.utils.tensors import as_torch_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Warp arrays are also accepted at runtime and normalized via ``as_torch_tensor``.
EnvIds: TypeAlias = "Sequence[int] | torch.Tensor | None"


class Randomizer(ABC):
    """Reusable reset-time domain randomizer usable as an IsaacLab event func."""

    @staticmethod
    @abstractmethod
    def apply(env: ManagerBasedEnv, env_ids: EnvIds, **params) -> None:
        """Apply the randomization to ``env`` for ``env_ids`` using event params."""

    @staticmethod
    def _resolve_env_ids(
        env: ManagerBasedEnv,
        env_ids: EnvIds,
        *,
        device: str | torch.device,
        num_envs_hint: int = 0,
    ) -> torch.Tensor:
        """Return environment ids as a device-local integer tensor."""

        if env_ids is None:
            num_envs = getattr(env.scene, "num_envs", None)
            if num_envs is None:
                num_envs = getattr(env, "num_envs", num_envs_hint)
            return torch.arange(int(num_envs), device=device, dtype=torch.long)
        return as_torch_tensor(env_ids, device=device, dtype=torch.long).flatten()

    @staticmethod
    def _resolve_event_env_ids(env: ManagerBasedEnv, env_ids: EnvIds) -> torch.Tensor:
        """Return event env ids as a CPU integer tensor."""

        return Randomizer._resolve_env_ids(env, env_ids, device="cpu")

    @staticmethod
    def _stage_from_env(env: ManagerBasedEnv):
        """Return the USD stage owned by the environment."""

        stage = getattr(getattr(env, "sim", None), "stage", None)
        if stage is not None:
            return stage
        stage = getattr(getattr(env, "scene", None), "stage", None)
        if stage is not None:
            return stage

        from isaaclab.sim import utils as sim_utils  # noqa: PLC0415

        return sim_utils.get_current_stage()

    @staticmethod
    def _validated_path_strings(name: str, paths: Sequence[str]) -> tuple[str, ...]:
        """Return non-empty normalized path strings for event asset files."""

        path_strings = tuple(str(Path(path)) for path in paths)
        if not path_strings:
            raise ValueError(f"Expected at least one path in '{name}'.")
        return path_strings


__all__ = ["Randomizer"]
