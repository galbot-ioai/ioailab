"""Base classes for env action agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ioailab.agents.io import ActionSource, AgentIO, EnvIds


class BaseAgent(ABC):
    """Base class for per-step ioailab action sources.

    Agents own controller state and produce full task actions for all env rows or
    for the requested ``env_ids`` subset. They do not construct environments,
    call ``env.step(...)``, or retain simulator lifetimes.
    """

    def reset(self, env: Any, env_ids: EnvIds = None) -> None:
        """Reset controller state for a workflow env or env-row subset."""

    @abstractmethod
    def act(self, env: Any, env_ids: EnvIds = None) -> Any:
        """Return a task action for all rows or the requested ``env_ids``."""

    def done(self, env: Any, env_ids: EnvIds = None) -> bool | Sequence[bool]:
        """Return completion state for all rows or the requested ``env_ids``."""

        if env_ids is None:
            return [False] * int(env.num_envs)
        return [False] * len(env_ids)

    def close(self) -> None:
        """Release controller-owned resources."""


class _ActionSourceAgent(BaseAgent):
    """Base implementation for agents backed by a callable action source."""

    def __init__(
        self, action_source: ActionSource | None = None, **metadata: Any
    ) -> None:
        """Initialize an action-source agent with lightweight metadata."""

        self.action_source = action_source
        self.metadata = dict(metadata)

    def reset(self, env: Any, env_ids: EnvIds = None) -> None:
        """Reset the configured action source when it owns controller state."""

        reset = getattr(self.action_source, "reset", None)
        if callable(reset):
            reset(env, env_ids)

    def act(self, env: Any, env_ids: EnvIds = None) -> Any:
        """Return one action from the configured action source."""

        if self.action_source is None:
            raise NotImplementedError(
                f"{type(self).__name__} requires an action source before act(...)."
            )
        io = AgentIO.from_env(env, env_ids)
        return self.action_source(io.env, io.env_ids)
