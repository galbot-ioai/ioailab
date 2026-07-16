"""Task-aware flow agent for coherent multi-phase tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ioailab.agents.base import BaseAgent, EnvIds
from ioailab.agents.flow.sequence import (
    AgentProvider,
    AgentStep,
    SequenceAgent,
    agent_step,
)


@dataclass(frozen=True, slots=True)
class TaskPhaseSpec:
    """Metadata for one phase in a coherent task flow."""

    name: str
    phase_task_id: str
    success: Callable[[Any], Any] | None = None
    default_agent: AgentProvider | None = None
    action_terms: tuple[str, ...] = ()
    fixed_base: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("TaskPhaseSpec.name must be non-empty.")
        task_id = str(self.phase_task_id).strip()
        if not task_id:
            raise ValueError("TaskPhaseSpec.phase_task_id must be non-empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "phase_task_id", task_id)
        object.__setattr__(
            self, "action_terms", tuple(str(term) for term in self.action_terms)
        )
        object.__setattr__(self, "fixed_base", bool(self.fixed_base))

    @property
    def agent(self) -> AgentProvider | None:
        """Alias for the default phase agent provider."""

        return self.default_agent


@dataclass(frozen=True, slots=True)
class TaskFlowSpec:
    """Small task-owned phase metadata for a normal coherent task."""

    phases: tuple[TaskPhaseSpec, ...]
    final_phase: str
    phase_state_getter: str = "current_task_phases"

    def __post_init__(self) -> None:
        phases = tuple(self.phases)
        if not phases:
            raise ValueError("TaskFlowSpec requires at least one phase.")
        names = tuple(phase.name for phase in phases)
        if len(set(names)) != len(names):
            raise ValueError(f"TaskFlowSpec has duplicate phase names: {names!r}.")
        final = str(self.final_phase).strip()
        if final not in names:
            raise ValueError(
                f"TaskFlowSpec.final_phase {final!r} is not one of {names!r}."
            )
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "final_phase", final)
        object.__setattr__(
            self, "phase_state_getter", str(self.phase_state_getter).strip()
        )

    @property
    def initial_phase(self) -> str:
        """Return the first phase name."""

        return self.phases[0].name

    @property
    def phase_names(self) -> tuple[str, ...]:
        """Return phase names in execution order."""

        return tuple(phase.name for phase in self.phases)

    def phase(self, name: str) -> TaskPhaseSpec:
        """Return metadata for ``name``."""

        phase_name = str(name)
        for phase in self.phases:
            if phase.name == phase_name:
                return phase
        raise KeyError(f"Unknown task-flow phase {phase_name!r}.")

    def next_phase_name(self, name: str) -> str | None:
        """Return the next phase name, or ``None`` for the final phase."""

        names = self.phase_names
        index = names.index(str(name))
        if index + 1 >= len(names):
            return None
        return names[index + 1]


def taskspec(**kwargs: Any) -> TaskPhaseSpec:
    """Build a :class:`TaskPhaseSpec` with concise authoring syntax."""

    return TaskPhaseSpec(**kwargs)


def taskflow(
    phases: Sequence[TaskPhaseSpec],
    *,
    final_phase: str | None = None,
    phase_state_getter: str = "current_task_phases",
) -> TaskFlowSpec:
    """Build a :class:`TaskFlowSpec` from ordered phase specs."""

    ordered = tuple(phases)
    return TaskFlowSpec(
        phases=ordered,
        final_phase=final_phase or ordered[-1].name,
        phase_state_getter=phase_state_getter,
    )


class TaskFlowAgent(BaseAgent):
    """Task-aware adapter over :class:`SequenceAgent`."""

    def __init__(
        self,
        flow: TaskFlowSpec,
        *,
        agents: Mapping[str, AgentProvider] | None = None,
        env: Any | None = None,
    ) -> None:
        if not isinstance(flow, TaskFlowSpec):
            raise TypeError("flow must be a TaskFlowSpec.")
        self._flow = flow
        self._providers = _merged_agent_providers(flow, agents)
        self._sequence = SequenceAgent(
            _phase_steps(flow, self._providers),
            env=env,
            state_reader=self._read_env_phases,
            state_writer=self._set_env_phases,
        )

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        agents: Mapping[str, AgentProvider] | None = None,
    ) -> "TaskFlowAgent":
        """Create a task-flow agent from ``env.task_id`` metadata."""

        from ioailab import tasks

        task_id = getattr(env, "task_id", None)
        if not task_id:
            raise ValueError("TaskFlowAgent.from_env(...) requires env.task_id.")
        flow = _flow_from_env(env) or tasks.task_flow_for_task(str(task_id))
        return cls(flow, agents=agents, env=env)

    @property
    def flow(self) -> TaskFlowSpec:
        """Return the task-flow metadata."""

        return self._flow

    @property
    def active_phases(self) -> tuple[str, ...]:
        """Return the last phases observed by this agent, one per env row."""

        return self._sequence.active_steps

    def phase_agent(self, name: str) -> BaseAgent:
        """Return the live agent for a phase."""

        return self._sequence.step_agent(str(name))

    def reset(self, env: Any, env_ids: EnvIds = None) -> None:
        """Reset phase state and the initial phase agent."""

        self._sequence.reset(env, env_ids=env_ids)

    def act(self, env: Any, env_ids: EnvIds = None) -> Any:
        """Return a full action tensor assembled from active phase agents."""

        return self._sequence.act(env, env_ids=env_ids)

    def done(self, env: Any, env_ids: EnvIds = None) -> bool | Sequence[bool]:
        """Return whether requested rows are successful in the final phase."""

        return self._sequence.done(env, env_ids=env_ids)

    def close(self) -> None:
        """Close all phase agents."""

        self._sequence.close()

    def _read_env_phases(self, env: Any, env_ids: tuple[int, ...]) -> tuple[str, ...]:
        return _read_env_phases(env, self._flow, env_ids)

    def _set_env_phases(
        self, env: Any, env_ids: tuple[int, ...], phase_name: str
    ) -> None:
        _set_env_phases(env, self._flow, env_ids, phase_name)


def _phase_steps(
    flow: TaskFlowSpec, providers: Mapping[str, AgentProvider]
) -> tuple[AgentStep, ...]:
    return tuple(
        agent_step(
            phase.name,
            providers[phase.name],
            action_terms=phase.action_terms,
            done=phase.success,
            fixed_base=phase.fixed_base,
        )
        for phase in flow.phases
    )


def _merged_agent_providers(
    flow: TaskFlowSpec, overrides: Mapping[str, AgentProvider] | None
) -> dict[str, AgentProvider]:
    providers: dict[str, AgentProvider] = {}
    overrides = {} if overrides is None else dict(overrides)
    unknown = tuple(sorted(set(overrides) - set(flow.phase_names)))
    if unknown:
        raise ValueError(f"Unknown task-flow phase agent override(s): {unknown}.")
    for phase in flow.phases:
        provider = overrides.get(phase.name, phase.default_agent)
        if provider is None:
            raise ValueError(
                f"Task-flow phase {phase.name!r} does not define an agent."
            )
        providers[phase.name] = provider
    return providers


def _flow_from_env(env: Any) -> TaskFlowSpec | None:
    raw = getattr(env, "raw_env", env)
    unwrapped = getattr(raw, "unwrapped", getattr(env, "unwrapped", raw))
    cfg = getattr(unwrapped, "cfg", None)
    flow = getattr(cfg, "task_flow", None)
    if isinstance(flow, TaskFlowSpec):
        return flow
    flow = getattr(unwrapped, "task_flow", None)
    if isinstance(flow, TaskFlowSpec):
        return flow
    return None


def _unwrapped_env(env: Any) -> Any:
    raw = getattr(env, "raw_env", env)
    return getattr(raw, "unwrapped", getattr(env, "unwrapped", raw))


def _num_envs(env: Any) -> int:
    unwrapped_count = getattr(_unwrapped_env(env), "num_envs", None)
    if unwrapped_count is not None:
        return int(unwrapped_count)
    return int(getattr(env, "num_envs"))


def _read_env_phases(
    env: Any, flow: TaskFlowSpec, env_ids: tuple[int, ...]
) -> tuple[str, ...]:
    getter = getattr(env, flow.phase_state_getter, None)
    if not callable(getter):
        getter = getattr(_unwrapped_env(env), flow.phase_state_getter, None)
    if callable(getter):
        phases = tuple(str(phase) for phase in getter(env_ids=env_ids))
    else:
        phases = _read_default_phase_state(env, flow, env_ids)
    unknown = tuple(phase for phase in phases if phase not in flow.phase_names)
    if unknown:
        raise ValueError(f"Task-flow env returned unknown phase(s): {unknown}.")
    return phases


def _read_default_phase_state(
    env: Any, flow: TaskFlowSpec, env_ids: tuple[int, ...]
) -> tuple[str, ...]:
    unwrapped = _unwrapped_env(env)
    state = getattr(unwrapped, "_ioailab_task_phase_names", None)
    if state is None or len(state) != _num_envs(env):
        state = [flow.initial_phase for _ in range(_num_envs(env))]
        setattr(unwrapped, "_ioailab_task_phase_names", state)
    return tuple(str(state[env_id]) for env_id in env_ids)


def _set_env_phases(
    env: Any, flow: TaskFlowSpec, env_ids: Sequence[int], phase_name: str
) -> None:
    setter = getattr(env, "set_task_phases", None)
    if not callable(setter):
        setter = getattr(_unwrapped_env(env), "set_task_phases", None)
    if callable(setter):
        setter(env_ids=tuple(env_ids), phase=phase_name)
        return
    unwrapped = _unwrapped_env(env)
    state = getattr(unwrapped, "_ioailab_task_phase_names", None)
    if state is None or len(state) != _num_envs(env):
        state = [flow.initial_phase for _ in range(_num_envs(env))]
        setattr(unwrapped, "_ioailab_task_phase_names", state)
    for env_id in env_ids:
        state[int(env_id)] = str(phase_name)


__all__ = [
    "AgentProvider",
    "TaskFlowAgent",
    "TaskFlowSpec",
    "TaskPhaseSpec",
    "taskflow",
    "taskspec",
]
