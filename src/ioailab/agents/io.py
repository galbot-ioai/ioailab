"""Common input/output contracts for ioailab action agents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

EnvIds = Sequence[int] | None
"""Optional subset of vectorized environment row indices."""

ActionSource = Callable[[Any, EnvIds], Any]
"""Callable that returns a full task action, optionally scoped to env rows."""


@dataclass(frozen=True, slots=True)
class AgentIO:
    """Normalized per-step agent input context.

    ``env`` is the caller-owned IsaacLab/Gymnasium environment or workflow env.
    ``env_ids`` is ``None`` for the full vectorized env or a tuple of row ids
    for a masked agent call. Agents still return the action object
    expected by the task MDP; this class does not wrap action tensors.
    """

    env: Any
    env_ids: tuple[int, ...] | None = None

    @classmethod
    def from_env(cls, env: Any, env_ids: EnvIds = None) -> "AgentIO":
        """Build a normalized IO context from an env and optional row ids."""

        return cls(env=env, env_ids=normalize_env_ids(env_ids))

    @property
    def is_full_env(self) -> bool:
        """Return whether this context targets the full vectorized env."""

        return self.env_ids is None


def normalize_env_ids(env_ids: EnvIds = None) -> tuple[int, ...] | None:
    """Return ``env_ids`` as an integer tuple while preserving full-env ``None``."""

    if env_ids is None:
        return None
    return tuple(int(env_id) for env_id in env_ids)


def num_envs(env: Any) -> int:
    """Return the vectorized environment row count for an env-like object."""

    return int(getattr(env, "num_envs"))
