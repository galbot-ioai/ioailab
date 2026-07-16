"""Runtime helpers for coherent multi-phase ioailab envs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from ioailab.agents.flow import TaskFlowSpec


def current_task_phases(
    env: Any, env_ids: Sequence[int] | None = None
) -> tuple[str, ...]:
    """Return coherent-task phase names for requested env rows."""

    unwrapped = _unwrapped_env(env)
    flow = _task_flow_from_env(unwrapped)
    state = _ensure_phase_state(unwrapped, flow)
    ids = _resolve_env_ids(unwrapped, env_ids)
    return tuple(str(state[env_id]) for env_id in ids)


def set_task_phases(
    env: Any,
    *,
    env_ids: Sequence[int] | None = None,
    phase: str | None = None,
) -> None:
    """Set requested env rows to ``phase``."""

    unwrapped = _unwrapped_env(env)
    flow = _task_flow_from_env(unwrapped)
    phase_name = flow.initial_phase if phase is None else str(phase)
    if phase_name not in flow.phase_names:
        raise ValueError(
            f"Unknown task-flow phase {phase_name!r}; "
            f"valid phases: {flow.phase_names!r}."
        )
    state = _ensure_phase_state(unwrapped, flow)
    for env_id in _resolve_env_ids(unwrapped, env_ids):
        state[env_id] = phase_name


def reset_task_phases(env: Any, env_ids: Sequence[int] | None = None) -> None:
    """Reset requested env rows to the coherent task's initial phase."""

    set_task_phases(env, env_ids=env_ids, phase=None)


def phase_gated_success(phase_name: str, term: Any) -> Callable[[Any], torch.Tensor]:
    """Return a termination predicate gated to rows currently in ``phase_name``."""

    def _predicate(env: Any) -> torch.Tensor:
        phases = _current_phases(env)
        success = torch.as_tensor(_call_term(term, env), dtype=torch.bool)
        if success.ndim == 0:
            success = success.reshape(1)
        phase_mask = torch.tensor(
            [phase == phase_name for phase in phases],
            device=success.device,
            dtype=torch.bool,
        )
        return torch.logical_and(phase_mask, success.reshape(-1))

    _predicate.__name__ = "final_phase_success"
    return _predicate


def _current_phases(env: Any) -> tuple[str, ...]:
    getter = getattr(env, "current_task_phases", None)
    if not callable(getter):
        getter = getattr(getattr(env, "unwrapped", env), "current_task_phases", None)
    if not callable(getter):
        return current_task_phases(env)
    return tuple(str(phase_name) for phase_name in getter())


def _call_term(term: Any, env: Any) -> Any:
    return term.func(env, **dict(getattr(term, "params", {}) or {}))


def _task_flow_from_env(env: Any) -> TaskFlowSpec:
    unwrapped = _unwrapped_env(env)
    cfg = getattr(unwrapped, "cfg", None)
    flow = getattr(cfg, "task_flow", None)
    if flow is None:
        flow = getattr(unwrapped, "task_flow", None)
    if not isinstance(flow, TaskFlowSpec):
        raise ValueError("Combined task env cfg must expose TaskFlowSpec task_flow.")
    return flow


def _ensure_phase_state(env: Any, flow: TaskFlowSpec) -> list[str]:
    unwrapped = _unwrapped_env(env)
    num_envs = int(getattr(unwrapped, "num_envs"))
    state_attr = "_ioailab_task_flow_phase_names"
    state = getattr(unwrapped, state_attr, None)
    if state is None or len(state) != num_envs:
        state = [flow.initial_phase for _ in range(num_envs)]
        setattr(unwrapped, state_attr, state)
        setattr(
            unwrapped,
            flow.phase_state_getter,
            current_task_phases.__get__(unwrapped),
        )
        setattr(unwrapped, "set_task_phases", set_task_phases.__get__(unwrapped))
    return state


def _resolve_env_ids(env: Any, env_ids: Sequence[int] | None) -> tuple[int, ...]:
    if env_ids is None:
        return tuple(range(int(getattr(_unwrapped_env(env), "num_envs"))))
    ids = tuple(int(env_id) for env_id in env_ids)
    if not ids:
        raise ValueError("env_ids must not be empty.")
    return ids


def _unwrapped_env(env: Any) -> Any:
    raw = getattr(env, "raw_env", env)
    return getattr(raw, "unwrapped", getattr(env, "unwrapped", raw))
