"""Generic row-wise sequence agent for ordered agent execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ioailab.agents.base import BaseAgent, EnvIds

if TYPE_CHECKING:
    import torch

    from ioailab.agents.motion_plan.contracts.g1 import G1ActionLayout

AgentProvider = BaseAgent | Callable[[Any], BaseAgent]
"""An agent object or env-aware factory returning one."""

DonePredicate = Callable[[Any], Any]
"""A row-vector completion predicate evaluated on the unwrapped env."""

StepStateReader = Callable[[Any, tuple[int, ...]], tuple[str, ...]]
StepStateWriter = Callable[[Any, tuple[int, ...], str], None]


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One ordered step in a :class:`SequenceAgent`."""

    name: str
    agent: AgentProvider
    action_terms: tuple[str, ...] = ()
    done: DonePredicate | None = None
    fixed_base: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("AgentStep.name must be non-empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "action_terms", tuple(str(term) for term in self.action_terms)
        )
        object.__setattr__(self, "fixed_base", bool(self.fixed_base))


def agent_step(
    name: str,
    agent: AgentProvider,
    *,
    action_terms: Sequence[str] = (),
    done: DonePredicate | None = None,
    fixed_base: bool = False,
) -> AgentStep:
    """Build an :class:`AgentStep` with concise authoring syntax."""

    return AgentStep(
        name=name,
        agent=agent,
        action_terms=tuple(action_terms),
        done=done,
        fixed_base=fixed_base,
    )


def agent_sequence(*steps: AgentStep) -> "SequenceAgent":
    """Build a :class:`SequenceAgent` from ordered agent steps."""

    return SequenceAgent(steps)


class SequenceAgent(BaseAgent):
    """Run ordered agents row-wise, preserving async vector-env progress."""

    def __init__(
        self,
        steps: Sequence[AgentStep],
        *,
        env: Any | None = None,
        state_reader: StepStateReader | None = None,
        state_writer: StepStateWriter | None = None,
    ) -> None:
        ordered = tuple(steps)
        if not ordered:
            raise ValueError("SequenceAgent requires at least one AgentStep.")
        names = tuple(step.name for step in ordered)
        if len(set(names)) != len(names):
            raise ValueError(f"SequenceAgent has duplicate step names: {names!r}.")
        self._steps = ordered
        self._step_by_name = {step.name: step for step in ordered}
        self._providers = {step.name: step.agent for step in ordered}
        self._agents: dict[str, BaseAgent] = {}
        self._known_step_by_env: list[str] = []
        self._inactive_targets_by_env: dict[tuple[int, str], Any] = {}
        self._fixed_base_root_pose_by_env: dict[int, Any] = {}
        self._state_reader = state_reader
        self._state_writer = state_writer
        if env is not None:
            self._ensure_agents(env)

    @property
    def steps(self) -> tuple[AgentStep, ...]:
        """Return ordered sequence metadata."""

        return self._steps

    @property
    def step_names(self) -> tuple[str, ...]:
        """Return ordered step names."""

        return tuple(step.name for step in self._steps)

    @property
    def initial_step(self) -> str:
        """Return the first step name."""

        return self._steps[0].name

    @property
    def final_step(self) -> str:
        """Return the final step name."""

        return self._steps[-1].name

    @property
    def active_steps(self) -> tuple[str, ...]:
        """Return the last observed active step for every env row."""

        return tuple(self._known_step_by_env)

    def step(self, name: str) -> AgentStep:
        """Return step metadata by name."""

        try:
            return self._step_by_name[str(name)]
        except KeyError as exc:
            raise KeyError(f"Unknown sequence step {name!r}.") from exc

    def step_agent(self, name: str) -> BaseAgent:
        """Return the live agent for a step."""

        return self._agents[str(name)]

    def next_step_name(self, name: str) -> str | None:
        """Return the next step name, or ``None`` for the final step."""

        names = self.step_names
        index = names.index(str(name))
        if index + 1 >= len(names):
            return None
        return names[index + 1]

    def reset(self, env: Any, env_ids: EnvIds = None) -> None:
        """Reset sequence state and the first step agent for requested rows."""

        self._ensure_agents(env)
        ids = _resolve_env_ids(env, env_ids)
        self._set_steps(env, ids, self.initial_step)
        self._ensure_known_steps(env)
        self._clear_inactive_targets(ids)
        self._clear_fixed_base_targets(ids)
        for env_id in ids:
            self._known_step_by_env[env_id] = self.initial_step
        self._agents[self.initial_step].reset(env, env_ids=ids)
        self._sync_inactive_targets(
            env,
            tuple((env_id, self.initial_step) for env_id in ids),
            overwrite=True,
        )
        self._capture_fixed_base_targets(env, ids)

    def act(self, env: Any, env_ids: EnvIds = None) -> Any:
        """Return a full action tensor assembled from active step agents."""

        self._ensure_agents(env)
        ids = _resolve_env_ids(env, env_ids)
        self._ensure_known_steps(env)
        changed = self._advance_completed_steps(env, ids)
        if changed:
            self._sync_inactive_targets(env, changed, overwrite=True)
        for step_name, rows in _group_rows_by_step(changed).items():
            self._agents[step_name].reset(env, env_ids=rows)
        if changed:
            self._capture_fixed_base_targets(
                env, tuple(env_id for env_id, _ in changed)
            )
        self._restore_fixed_base_targets(env, ids)

        step_names = self._read_steps(env, ids)
        for env_id, step_name in zip(ids, step_names, strict=True):
            self._known_step_by_env[env_id] = step_name
        self._sync_inactive_targets(
            env,
            tuple(zip(ids, step_names, strict=True)),
            overwrite=False,
        )
        grouped_actions: dict[str, Any] = {}
        grouped_rows: dict[str, tuple[int, ...]] = {}
        for step_name, rows in _group_rows_by_step(zip(ids, step_names)).items():
            grouped_rows[step_name] = rows
            grouped_actions[step_name] = self._agents[step_name].act(env, env_ids=rows)
        self._restore_fixed_base_targets(env, ids)
        return _compose_sequence_actions(
            env,
            self,
            grouped_actions,
            grouped_rows=grouped_rows,
            env_ids=ids,
            inactive_targets_by_env=self._inactive_targets_by_env,
        )

    def done(self, env: Any, env_ids: EnvIds = None) -> bool | Sequence[bool]:
        """Return whether requested rows completed the final sequence step."""

        self._ensure_agents(env)
        ids = _resolve_env_ids(env, env_ids)
        live_env = _unwrapped_env(env)
        final = self.step(self.final_step)
        mask = self._step_done_mask(final, live_env)
        steps = self._read_steps(env, ids)
        result = [
            step_name == self.final_step and bool(mask[env_id])
            for env_id, step_name in zip(ids, steps, strict=True)
        ]
        if env_ids is None and all(result):
            return True
        return tuple(result)

    def close(self) -> None:
        """Close all resolved step agents."""

        for agent in self._agents.values():
            agent.close()
        self._agents.clear()
        self._fixed_base_root_pose_by_env.clear()

    def _ensure_agents(self, env: Any) -> None:
        missing = tuple(name for name in self._providers if name not in self._agents)
        for name in missing:
            self._agents[name] = _resolve_agent_provider(self._providers[name], env)

    def _read_steps(self, env: Any, env_ids: tuple[int, ...]) -> tuple[str, ...]:
        if self._state_reader is not None:
            steps = tuple(str(step) for step in self._state_reader(env, env_ids))
        else:
            steps = self._read_internal_steps(env, env_ids)
        unknown = tuple(step for step in steps if step not in self._step_by_name)
        if unknown:
            raise ValueError(f"SequenceAgent returned unknown step(s): {unknown}.")
        return steps

    def _read_internal_steps(
        self, env: Any, env_ids: tuple[int, ...]
    ) -> tuple[str, ...]:
        num_envs = _num_envs(env)
        if len(self._known_step_by_env) != num_envs:
            self._known_step_by_env = [self.initial_step for _ in range(num_envs)]
        return tuple(self._known_step_by_env[env_id] for env_id in env_ids)

    def _set_steps(self, env: Any, env_ids: Sequence[int], step_name: str) -> None:
        if self._state_writer is not None:
            self._state_writer(env, tuple(int(env_id) for env_id in env_ids), step_name)
        num_envs = _num_envs(env)
        if len(self._known_step_by_env) != num_envs:
            self._known_step_by_env = [self.initial_step for _ in range(num_envs)]
        for env_id in env_ids:
            self._known_step_by_env[int(env_id)] = str(step_name)

    def _ensure_known_steps(self, env: Any) -> None:
        num_envs = _num_envs(env)
        if len(self._known_step_by_env) != num_envs:
            self._known_step_by_env = list(
                self._read_steps(env, tuple(range(num_envs)))
            )

    def _advance_completed_steps(
        self, env: Any, env_ids: tuple[int, ...]
    ) -> tuple[tuple[int, str], ...]:
        """Advance rows whose current step done predicate is true."""

        live_env = _unwrapped_env(env)
        current = self._read_steps(env, env_ids)
        changed: list[tuple[int, str]] = []
        for step_name in self.step_names:
            step = self.step(step_name)
            next_step = self.next_step_name(step_name)
            if next_step is None:
                continue
            rows = tuple(
                env_id
                for env_id, row_step in zip(env_ids, current, strict=True)
                if row_step == step_name
            )
            if not rows:
                continue
            mask = self._step_done_mask(step, live_env)
            advanced = tuple(env_id for env_id in rows if bool(mask[env_id]))
            if not advanced:
                continue
            self._set_steps(env, advanced, next_step)
            for env_id in advanced:
                changed.append((env_id, next_step))
        return tuple(changed)

    def _step_done_mask(self, step: AgentStep, env: Any) -> torch.Tensor:
        if step.done is not None:
            return _as_bool_mask(step.done(env), env=env)
        return _as_bool_mask(self._agents[step.name].done(env), env=env)

    def _clear_inactive_targets(self, env_ids: tuple[int, ...]) -> None:
        ids = set(env_ids)
        for key in tuple(self._inactive_targets_by_env):
            if key[0] in ids:
                del self._inactive_targets_by_env[key]

    def _clear_fixed_base_targets(self, env_ids: tuple[int, ...]) -> None:
        for env_id in env_ids:
            self._fixed_base_root_pose_by_env.pop(int(env_id), None)

    def _capture_fixed_base_targets(self, env: Any, env_ids: tuple[int, ...]) -> None:
        if not env_ids:
            return
        step_names = self._read_steps(env, env_ids)
        fixed_pairs = tuple(
            (env_id, step_name)
            for env_id, step_name in zip(env_ids, step_names, strict=True)
            if self.step(step_name).fixed_base
        )
        if not fixed_pairs:
            for env_id in env_ids:
                self._fixed_base_root_pose_by_env.pop(env_id, None)
            return
        import torch

        robot = _unwrapped_env(env).scene["robot"]
        for env_id, _step_name in fixed_pairs:
            root_pose = torch.cat(
                (robot.data.root_pos_w[env_id], robot.data.root_quat_w[env_id]),
                dim=-1,
            )
            self._fixed_base_root_pose_by_env[env_id] = root_pose.detach().clone()

    def _restore_fixed_base_targets(self, env: Any, env_ids: tuple[int, ...]) -> None:
        import torch

        step_names = self._read_steps(env, env_ids)
        fixed_env_ids = tuple(
            env_id
            for env_id, step_name in zip(env_ids, step_names, strict=True)
            if self.step(step_name).fixed_base
        )
        if not fixed_env_ids:
            return
        missing = tuple(
            env_id
            for env_id in fixed_env_ids
            if env_id not in self._fixed_base_root_pose_by_env
        )
        if missing:
            raise RuntimeError(
                f"Missing fixed-base root pose target for env rows {missing}."
            )

        robot = _unwrapped_env(env).scene["robot"]
        root_pose = torch.stack(
            [
                self._fixed_base_root_pose_by_env[env_id].to(
                    robot.data.root_pos_w[env_id]
                )
                for env_id in fixed_env_ids
            ],
            dim=0,
        )
        env_ids_tensor = torch.tensor(
            fixed_env_ids,
            dtype=torch.long,
            device=root_pose.device,
        )
        zero_velocity = torch.zeros(
            (len(fixed_env_ids), 6),
            dtype=root_pose.dtype,
            device=root_pose.device,
        )
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids_tensor)
        robot.write_root_velocity_to_sim_index(
            root_velocity=zero_velocity, env_ids=env_ids_tensor
        )

    def _sync_inactive_targets(
        self,
        env: Any,
        env_step_pairs: tuple[tuple[int, str], ...],
        *,
        overwrite: bool,
    ) -> None:
        if not env_step_pairs:
            return
        unwrapped = _unwrapped_env(env)
        try:
            from ioailab.agents.motion_plan.contracts.g1 import (
                current_g1_group_joint_positions,
                make_g1_action_layout_from_env,
            )

            layout = make_g1_action_layout_from_env(unwrapped)
        except Exception:
            return

        group_names = tuple(
            term.group_name for term in layout.terms if term.group_name != "base"
        )
        positions_by_group: dict[str, Any] = {}

        def _positions_for_group(group_name: str) -> Any:
            if group_name not in positions_by_group:
                positions_by_group[group_name] = current_g1_group_joint_positions(
                    unwrapped,
                    robot_asset_name="robot",
                    group_name=group_name,
                )
            return positions_by_group[group_name]

        for env_id, step_name in env_step_pairs:
            step = self.step(step_name)
            owns_full_action = not step.action_terms
            owned_groups = set(step.action_terms)
            for group_name in group_names:
                key = (int(env_id), group_name)
                if owns_full_action or group_name in owned_groups:
                    self._inactive_targets_by_env.pop(key, None)
                    continue
                if overwrite or key not in self._inactive_targets_by_env:
                    positions = _positions_for_group(group_name)
                    self._inactive_targets_by_env[key] = (
                        positions[int(env_id)].detach().clone()
                    )


def _resolve_agent_provider(provider: AgentProvider, env: Any) -> BaseAgent:
    if isinstance(provider, BaseAgent):
        return provider
    if not callable(provider):
        raise TypeError(
            f"Agent provider must be a BaseAgent or callable: {provider!r}."
        )
    try:
        agent = provider(env)
    except TypeError:
        agent = provider()
    if not isinstance(agent, BaseAgent):
        raise TypeError(
            f"Agent factory returned {type(agent).__name__}, expected BaseAgent."
        )
    return agent


def _resolve_env_ids(env: Any, env_ids: EnvIds = None) -> tuple[int, ...]:
    if env_ids is None:
        return tuple(range(_num_envs(env)))
    ids = tuple(int(env_id) for env_id in env_ids)
    if not ids:
        raise ValueError("env_ids must not be empty.")
    if len(set(ids)) != len(ids):
        raise ValueError("env_ids must be unique.")
    num_envs = _num_envs(env)
    if any(env_id < 0 or env_id >= num_envs for env_id in ids):
        raise ValueError(f"env_ids out of range for num_envs={num_envs}.")
    return ids


def _num_envs(env: Any) -> int:
    unwrapped_count = getattr(_unwrapped_env(env), "num_envs", None)
    if unwrapped_count is not None:
        return int(unwrapped_count)
    return int(getattr(env, "num_envs"))


def _unwrapped_env(env: Any) -> Any:
    raw = getattr(env, "raw_env", env)
    return getattr(raw, "unwrapped", getattr(env, "unwrapped", raw))


def _group_rows_by_step(
    row_step_pairs: Sequence[tuple[int, str]],
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for env_id, step_name in row_step_pairs:
        grouped.setdefault(str(step_name), []).append(int(env_id))
    return {step_name: tuple(rows) for step_name, rows in grouped.items()}


def _as_bool_mask(value: Any, *, env: Any) -> torch.Tensor:
    import torch

    tensor = torch.as_tensor(value, device=getattr(_unwrapped_env(env), "device", None))
    if tensor.ndim == 0:
        tensor = tensor.reshape(1).repeat(_num_envs(env))
    tensor = tensor.reshape(-1).to(dtype=torch.bool)
    if tensor.numel() != _num_envs(env):
        raise ValueError(
            f"SequenceAgent done predicate returned {tensor.numel()} rows for num_envs={_num_envs(env)}."
        )
    return tensor


def _compose_sequence_actions(
    env: Any,
    sequence: SequenceAgent,
    grouped_actions: Mapping[str, Any],
    *,
    grouped_rows: Mapping[str, tuple[int, ...]],
    env_ids: tuple[int, ...],
    inactive_targets_by_env: Mapping[tuple[int, str], Any] | None = None,
) -> torch.Tensor:
    """Merge step-agent actions into the env's union action tensor."""

    import torch
    from ioailab.agents.motion_plan.contracts.g1 import (
        make_g1_action_layout_from_env,
    )

    unwrapped = _unwrapped_env(env)
    first_action = torch.as_tensor(next(iter(grouped_actions.values())))
    try:
        layout = make_g1_action_layout_from_env(unwrapped)
    except Exception as exc:
        return _compose_same_width_actions(
            grouped_actions,
            grouped_rows=grouped_rows,
            env_ids=env_ids,
            num_envs=_num_envs(env),
            original_error=exc,
        )

    action = _make_stable_hold_action(
        unwrapped,
        layout=layout,
        env_ids=env_ids,
        dtype=first_action.dtype
        if first_action.dtype.is_floating_point
        else torch.float32,
        device=first_action.device,
        inactive_targets_by_env=inactive_targets_by_env,
    )
    local_row = {env_id: index for index, env_id in enumerate(env_ids)}
    for step_name, raw_action in grouped_actions.items():
        rows = grouped_rows[step_name]
        source = _select_action_rows(
            torch.as_tensor(raw_action, device=action.device, dtype=action.dtype),
            rows=rows,
            num_envs=_num_envs(env),
            step_name=step_name,
        )
        _write_step_action(
            action,
            source,
            layout=layout,
            step=sequence.step(step_name),
            local_rows=[local_row[env_id] for env_id in rows],
        )
    return action


def _make_stable_hold_action(
    env: Any,
    *,
    layout: Any,
    env_ids: tuple[int, ...],
    dtype: Any,
    device: Any,
    inactive_targets_by_env: Mapping[tuple[int, str], Any] | None = None,
) -> torch.Tensor:
    import torch
    from ioailab.agents.motion_plan.contracts.g1 import (
        current_g1_group_joint_positions,
    )

    manager = getattr(env, "action_manager", None)
    total_dim = max(
        int(getattr(manager, "total_action_dim", 0) or 0),
        int(layout.action_dim),
    )
    action = torch.zeros((len(env_ids), total_dim), device=device, dtype=dtype)
    inactive_targets = (
        {} if inactive_targets_by_env is None else inactive_targets_by_env
    )
    for term in layout.terms:
        if term.group_name == "base":
            continue
        positions = None
        for local_row, env_id in enumerate(env_ids):
            target = inactive_targets.get((env_id, term.group_name))
            if target is None:
                if positions is None:
                    positions = current_g1_group_joint_positions(
                        env, robot_asset_name="robot", group_name=term.group_name
                    ).to(device=device, dtype=dtype)
                action[local_row, term.action_slice] = positions[env_id, :]
            else:
                action[local_row, term.action_slice] = torch.as_tensor(
                    target, device=device, dtype=dtype
                )
    return action


def _write_step_action(
    action: torch.Tensor,
    source: torch.Tensor,
    *,
    layout: G1ActionLayout,
    step: AgentStep,
    local_rows: Sequence[int],
) -> None:
    if source.ndim == 1:
        source = source.reshape(1, -1)
    full_width = int(action.shape[1])
    if not step.action_terms:
        if int(source.shape[1]) != full_width:
            raise ValueError(
                f"Step {step.name!r} emitted compact actions but declares no action_terms."
            )
        action[list(local_rows), :] = source[:, :full_width]
        return

    if int(source.shape[1]) >= full_width:
        for group_name in step.action_terms:
            term = layout.term_for_group(group_name)
            action[list(local_rows), term.action_slice] = source[:, term.action_slice]
        return

    cursor = 0
    for group_name in step.action_terms:
        term = layout.term_for_group(group_name)
        next_cursor = cursor + term.width
        if next_cursor > int(source.shape[1]):
            raise ValueError(
                f"Step {step.name!r} emitted action width {source.shape[1]}, "
                f"too small for action_terms={step.action_terms!r}."
            )
        action[list(local_rows), term.action_slice] = source[:, cursor:next_cursor]
        cursor = next_cursor
    if cursor != int(source.shape[1]):
        raise ValueError(
            f"Step {step.name!r} emitted action width {source.shape[1]}, "
            f"but declared action_terms width {cursor}."
        )


def _select_action_rows(
    value: torch.Tensor,
    *,
    rows: tuple[int, ...],
    num_envs: int,
    step_name: str,
) -> torch.Tensor:
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if int(value.shape[0]) == len(rows):
        return value
    if int(value.shape[0]) == int(num_envs):
        return value[list(rows)]
    raise ValueError(
        f"Step {step_name!r} emitted {tuple(value.shape)} for env rows {rows}; "
        f"expected {len(rows)} rows or full vectorized row count {num_envs}."
    )


def _compose_same_width_actions(
    grouped_actions: Mapping[str, Any],
    *,
    grouped_rows: Mapping[str, tuple[int, ...]],
    env_ids: tuple[int, ...],
    num_envs: int,
    original_error: Exception,
) -> torch.Tensor:
    import torch

    first = torch.as_tensor(next(iter(grouped_actions.values())))
    if first.ndim == 1:
        first = first.reshape(1, -1)
    action = torch.zeros(
        (len(env_ids), int(first.shape[1])), device=first.device, dtype=first.dtype
    )
    local_row = {env_id: index for index, env_id in enumerate(env_ids)}
    for step_name, raw_action in grouped_actions.items():
        value = _select_action_rows(
            torch.as_tensor(raw_action, device=action.device, dtype=action.dtype),
            rows=grouped_rows[step_name],
            num_envs=num_envs,
            step_name=step_name,
        )
        if int(value.shape[1]) != int(action.shape[1]):
            raise ValueError(
                "SequenceAgent could not resolve a G1 action layout and step "
                "agents emitted incompatible action widths."
            ) from original_error
        for offset, env_id in enumerate(grouped_rows[step_name]):
            action[local_row[env_id], :] = value[offset]
    return action


__all__ = [
    "AgentProvider",
    "AgentStep",
    "DonePredicate",
    "SequenceAgent",
    "agent_sequence",
    "agent_step",
]
