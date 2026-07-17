"""ioailab workflow environment: one transparent class over the IsaacLab env.

``ioailabEnv`` wraps the Gymnasium/IsaacLab env that ``make_env`` builds and
adds the ioailab workflow surface (``collect`` / ``evaluate``)
plus app-lifecycle helpers. It stays transparent: ``__getattr__`` delegates to
the live ``raw_env`` and ``scene`` / ``unwrapped`` / managers / ``step(action)``
remain directly accessible, so IsaacLab is never hidden.

``ioailabEnv`` and ``make_env`` are the only public surfaces in this module.
Factory, recorder, and mask/result helpers live in neighboring private modules
so workflow internals can evolve without widening the user-facing API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ioailab.envs._factory as _factory
import ioailab.envs._masks as _masks
import ioailab.envs._recorder as _recorder
from ioailab.agents.base import BaseAgent

if TYPE_CHECKING:
    from ioailab.datasets import DatasetRef


@dataclass(slots=True)
class ioailabEnv:
    """Transparent ioailab workflow env returned by :func:`make_env`."""

    task_id: str
    raw_env: Any
    app: Any
    num_envs: int
    env_cfg: Any | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def scene(self) -> Any:
        """Return the live IsaacLab scene."""

        return self.raw_env.unwrapped.scene

    @property
    def action_space(self) -> Any:
        """Return the underlying env action space."""

        return self.raw_env.action_space

    @property
    def unwrapped(self) -> Any:
        """Return the underlying unwrapped IsaacLab env."""

        return self.raw_env.unwrapped

    @property
    def device(self) -> Any:
        """Return the underlying IsaacLab device."""

        return self.raw_env.unwrapped.device

    @property
    def task_options(self) -> dict[str, Any]:
        """Return a copy of task-local options used to create this env."""

        options = self.options.get("task_options")
        if isinstance(options, Mapping):
            return dict(options)
        return {}

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying Gym/IsaacLab env."""

        if name == "env_id":
            raise AttributeError
        return getattr(self.raw_env, name)

    def is_running(self) -> bool:
        """Return whether the Isaac application is still running."""

        if self.app is None or not hasattr(self.app, "is_running"):
            return True
        return bool(self.app.is_running())

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset the live IsaacLab env."""

        env_ids = kwargs.pop("env_ids", None)
        if env_ids is not None:
            if args:
                raise TypeError("reset(env_ids=...) does not accept positional args.")
            isaac_env = self.raw_env.unwrapped
            return isaac_env.reset(env_ids=_torch_env_ids(isaac_env, env_ids))
        return self.raw_env.reset(*args, **kwargs)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Render the underlying env and advance the Isaac app if present."""

        render = getattr(self.raw_env, "render", None)
        result = render(*args, **kwargs) if callable(render) else None
        if self.app is not None and hasattr(self.app, "update"):
            self.app.update()
        return result

    def close(self) -> None:
        """Close env and Isaac app."""

        recorder_manager = getattr(self.raw_env.unwrapped, "recorder_manager", None)
        if recorder_manager is not None:
            _recorder.close_recorder_file_handlers(recorder_manager)
        if hasattr(self.raw_env, "close"):
            self.raw_env.close()
        if self.app is not None and hasattr(self.app, "close"):
            self.app.close()

    def save_demo(
        self,
        path: str | Path,
        *,
        env_ids: Sequence[int] | None = None,
        demo_ids: Sequence[int] | None = None,
        format: str = "robomimic_hdf5",
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetRef:
        """Export the currently recorded demo rows to ``path``.

        This is the explicit save hook for caller-owned loops. Users can freely
        run ``reset -> agent.act -> env.step`` themselves and call this method
        when the current candidate episode should become data. The underlying
        data capture and file export still use IsaacLab's ``RecorderManager``.
        """

        if format not in {"hdf5", "robomimic_hdf5"}:
            raise ValueError(f"Unsupported dataset format: {format}")
        dataset_path = Path(path)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        unwrapped_env = self.raw_env.unwrapped
        recorder_manager = _recorder.require_recorder_manager(unwrapped_env)
        selected_env_ids = _recorder.selected_env_ids(self, env_ids)
        selected_demo_ids = tuple(demo_ids) if demo_ids is not None else None
        if selected_demo_ids is not None and len(selected_demo_ids) != len(
            selected_env_ids
        ):
            raise ValueError("demo_ids must have the same length as env_ids.")

        _recorder.retarget_recorder_export_path(
            unwrapped_env,
            recorder_manager=recorder_manager,
            dataset_path=dataset_path,
        )
        export_metadata = _recorder.export_recorded_episodes(
            recorder_manager,
            env_ids=selected_env_ids,
            demo_ids=selected_demo_ids,
        )

        from ioailab.datasets import DatasetRef

        return DatasetRef(
            path=dataset_path,
            format=format,
            task_id=self.task_id,
            metadata={
                "num_envs": int(self.num_envs),
                "saved_env_ids": selected_env_ids,
                "saved_demo_ids": tuple(int(demo_id) for demo_id in selected_demo_ids)
                if selected_demo_ids is not None
                else (),
                "saved_demos": len(selected_env_ids),
                "recorder_manager": type(recorder_manager).__name__,
                **dict(export_metadata),
                **dict(metadata or {}),
            },
        )

    def drop_demo(self, *, env_ids: Sequence[int] | None = None) -> None:
        """Discard the currently recorded demo rows without exporting them."""

        recorder_manager = _recorder.require_recorder_manager(self.raw_env.unwrapped)
        _recorder.drop_recorded_episodes(
            recorder_manager,
            env_ids=_recorder.selected_env_ids(self, env_ids),
        )

    def collect(
        self,
        *,
        agent: BaseAgent,
        path: str | Path,
        episodes: int,
        max_steps: int = 1000,
        format: str = "robomimic_hdf5",
        save_end_scenario: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetRef:
        """Collect a total number of agent episodes across all vector-env rows.

        ioailab owns the agent stepping loop. A row is exported when IsaacLab
        reports ``terminated``/``truncated``, task success is observed,
        ``max_steps`` is reached, or the agent explicitly requests operator exit;
        ``agent.done(env)`` is agent command state, not a collection boundary.
        Data capture/export remain exclusively with IsaacLab recorder
        hooks/file handlers configured on the task env. No handwritten HDF5
        fallback is provided.
        """

        if int(episodes) < 1:
            raise ValueError("episodes must be greater than zero.")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be greater than zero.")
        target_demos = int(episodes)
        max_steps = int(max_steps)
        if save_end_scenario is not None and target_demos != 1:
            raise ValueError("save_end_scenario requires episodes=1.")
        if format not in {"hdf5", "robomimic_hdf5"}:
            raise ValueError(f"Unsupported dataset format: {format}")

        dataset_path = Path(path)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        recorder_manager = _recorder.require_recorder_manager(self.raw_env.unwrapped)

        if not isinstance(agent, BaseAgent):
            raise TypeError("agent must inherit from BaseAgent.")
        export_metadata: dict[str, Any] = {}
        saved_demo_ids: list[int] = []
        auto_export_on_env_done = _recorder.recorder_auto_exports_on_reset(
            recorder_manager
        )
        _recorder.retarget_recorder_export_path(
            self.raw_env.unwrapped,
            recorder_manager=recorder_manager,
            dataset_path=dataset_path,
        )
        start_demo_id = _recorder.next_dataset_demo_id(dataset_path)

        planned_collection_rounds = (target_demos + int(self.num_envs) - 1) // int(
            self.num_envs
        )
        num_envs = int(self.num_envs)
        exported_demos = 0
        attempted_collection_rounds = 0
        exported_collection_rounds = 0
        vector_steps = 0
        current_lengths = [0] * num_envs
        current_rewards = [0.0] * num_envs
        episode_lengths: list[int] = []
        reward_totals: list[float] = []
        termination_reasons: list[str] = []
        lengths_by_env = [[] for _ in range(num_envs)]
        rewards_by_env = [[] for _ in range(num_envs)]
        reasons_by_env = [[] for _ in range(num_envs)]
        terminal_scenarios: dict[int, Any] = {}
        selected_end_scenario: Any | None = None
        saved_end_scenario_path: Path | None = None
        scenario_name = (
            _default_end_scenario_name(self.task_id)
            if save_end_scenario is not None
            else None
        )
        original_record_pre_reset = None
        try:
            self.reset()
            agent.reset(self)
            if save_end_scenario is not None:
                original_record_pre_reset = recorder_manager.record_pre_reset
                recorder_manager.record_pre_reset = (
                    _wrap_pre_reset_terminal_scenario_capture(
                        self,
                        original_record_pre_reset,
                        terminal_scenarios,
                        name=scenario_name,
                    )
                )
            while exported_demos < target_demos:
                action = agent.act(self)
                result = self.raw_env.step(action)
                _obs, reward, terminated, truncated, extras = _masks.unpack_step_result(
                    result
                )
                vector_steps += 1

                reward_values = _masks.numeric_vector(reward, num_envs)
                for env_id, reward_value in enumerate(reward_values):
                    current_lengths[env_id] += 1
                    current_rewards[env_id] += reward_value

                masks = _masks.completion_masks(
                    env_cfg=self.env_cfg,
                    raw_env=self.raw_env,
                    terminated=terminated,
                    truncated=truncated,
                    extras=extras,
                    current_lengths=current_lengths,
                    max_steps=max_steps,
                    num_envs=num_envs,
                )
                user_exit_mask = _recorder.agent_exit_requested_mask(agent, num_envs)
                pending_completed_ids = list(
                    _recorder.collection_completed_env_ids(
                        env_done=masks.env_done,
                        success=masks.success,
                        max_step=masks.max_step,
                        user_exit=user_exit_mask,
                    )
                )
                while pending_completed_ids and exported_demos < target_demos:
                    remaining_demos = target_demos - exported_demos
                    env_ids = tuple(pending_completed_ids[:remaining_demos])
                    del pending_completed_ids[: len(env_ids)]
                    demo_ids = tuple(
                        range(
                            start_demo_id + exported_demos,
                            start_demo_id + exported_demos + len(env_ids),
                        )
                    )
                    round_lengths = tuple(
                        int(current_lengths[env_id]) for env_id in env_ids
                    )
                    round_rewards = tuple(
                        float(current_rewards[env_id]) for env_id in env_ids
                    )
                    round_reasons = tuple(
                        _masks.collection_termination_reason(
                            env_id,
                            terminated=masks.terminated,
                            truncated=masks.truncated,
                            success=masks.success,
                            max_step=masks.max_step,
                            user_exit=user_exit_mask,
                            extras=extras,
                        )
                        for env_id in env_ids
                    )
                    attempted_collection_rounds += 1
                    episode_lengths.extend(round_lengths)
                    reward_totals.extend(round_rewards)
                    termination_reasons.extend(round_reasons)
                    for index, env_id in enumerate(env_ids):
                        lengths_by_env[env_id].append(round_lengths[index])
                        rewards_by_env[env_id].append(round_rewards[index])
                        reasons_by_env[env_id].append(round_reasons[index])

                    if save_end_scenario is not None:
                        env_id = int(env_ids[0])
                        scenario = terminal_scenarios.get(env_id)
                        if scenario is None:
                            if masks.env_done[env_id]:
                                raise RuntimeError(
                                    "IsaacLab reset completed before ioailab captured "
                                    f"the terminal scenario for env {env_id}."
                                )
                            scenario = self.get_scenario(
                                env_id=env_id,
                                name=scenario_name,
                            )
                        selected_end_scenario = _scenario_with_metadata(
                            scenario,
                            name=scenario_name,
                            metadata={
                                "kind": "end",
                                "task_id": self.task_id,
                                "dataset_path": str(dataset_path),
                                "demo_id": int(demo_ids[0]),
                                "episode_length": int(round_lengths[0]),
                                "reward_total": float(round_rewards[0]),
                                "termination_reason": str(round_reasons[0]),
                            },
                        )

                    manual_env_ids = _recorder.manual_collection_env_ids(
                        env_ids,
                        env_done=masks.env_done,
                        auto_export_on_env_done=auto_export_on_env_done,
                    )
                    if manual_env_ids:
                        saved_dataset = self.save_demo(
                            dataset_path,
                            env_ids=manual_env_ids,
                            format=format,
                        )
                        export_metadata.update(
                            _recorder.collect_export_metadata(saved_dataset.metadata)
                        )
                    saved_demo_ids.extend(int(demo_id) for demo_id in demo_ids)
                    exported_demos += len(env_ids)
                    exported_collection_rounds += 1

                    if exported_demos < target_demos:
                        if manual_env_ids:
                            _reset_env_rows(self, manual_env_ids)
                        _reset_agent_rows(agent, self, env_ids)
                        for env_id in env_ids:
                            current_lengths[env_id] = 0
                            current_rewards[env_id] = 0.0
            if save_end_scenario is not None:
                if selected_end_scenario is None:
                    raise RuntimeError("No completed episode was available to save.")
                from ioailab.tasks.common.scenario import save_scenario

                saved_end_scenario_path = save_scenario(
                    save_end_scenario, selected_end_scenario
                )
        finally:
            if original_record_pre_reset is not None:
                recorder_manager.record_pre_reset = original_record_pre_reset
            _recorder.close_recorder_file_handlers(recorder_manager)
        dataset_metadata = {
            "episodes": target_demos,
            "num_envs": num_envs,
            "total_demos": exported_demos,
            "saved_demo_ids": tuple(saved_demo_ids),
            "attempted_demos": len(episode_lengths),
            "collection_rounds": attempted_collection_rounds,
            "planned_collection_rounds": planned_collection_rounds,
            "exported_collection_rounds": exported_collection_rounds,
            "steps": vector_steps,
            "vector_steps": vector_steps,
            "row_steps": int(sum(episode_lengths)),
            "episode_lengths": tuple(episode_lengths),
            "episode_lengths_by_env": tuple(tuple(values) for values in lengths_by_env),
            "reward_totals": tuple(reward_totals),
            "reward_totals_by_env": tuple(tuple(values) for values in rewards_by_env),
            "termination_reasons": tuple(termination_reasons),
            "termination_reasons_by_env": tuple(
                tuple(values) for values in reasons_by_env
            ),
            "recorder_manager": type(recorder_manager).__name__,
            **dict(export_metadata),
            **dict(metadata or {}),
        }
        if saved_end_scenario_path is not None:
            dataset_metadata["saved_end_scenario_path"] = str(saved_end_scenario_path)

        from ioailab.datasets import DatasetRef

        return DatasetRef(
            path=dataset_path,
            format=format,
            task_id=self.task_id,
            metadata=dataset_metadata,
        )

    def evaluate(
        self,
        *,
        agent: BaseAgent,
        episodes: int = 1,
        max_steps: int = 1000,
    ) -> dict[str, Any]:
        """Evaluate a total number of episodes across all vector-env rows.

        Rows finish independently on env termination, task success, or max steps.
        """

        if not isinstance(agent, BaseAgent):
            raise TypeError("agent must inherit from BaseAgent.")
        if int(episodes) < 1:
            raise ValueError("episodes must be greater than zero.")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be greater than zero.")
        action_validation_failures = 0
        try:
            stats = self._run_evaluation(
                agent,
                episodes=int(episodes),
                max_steps=int(max_steps),
            )
        except (TypeError, ValueError):
            action_validation_failures += 1
            raise

        return {
            "task_id": self.task_id,
            "num_envs": int(self.num_envs),
            "episodes": int(episodes),
            "total_episodes": int(stats["total_episodes"]),
            "steps": int(stats["vector_steps"]),
            "vector_steps": int(stats["vector_steps"]),
            "row_steps": int(stats["row_steps"]),
            "success_count": int(stats["success_count"]),
            "success_rate": float(stats["success_rate"]),
            "success_masks": tuple(stats["success_masks"]),
            "episode_lengths": tuple(stats["episode_lengths"]),
            "episode_lengths_by_env": tuple(stats["episode_lengths_by_env"]),
            "average_length": float(stats["average_length"]),
            "reward_totals": tuple(stats["reward_totals"]),
            "reward_totals_by_env": tuple(stats["reward_totals_by_env"]),
            "reward_total_mean": float(stats["reward_total_mean"]),
            "termination_reasons": tuple(stats["termination_reasons"]),
            "termination_reasons_by_env": tuple(stats["termination_reasons_by_env"]),
            "action_validation_failures": action_validation_failures,
        }

    def get_scenario(
        self,
        *,
        env_id: int = 0,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Capture one env row as a human-readable reset scenario."""

        from ioailab.tasks.common.scenario import get_scenario

        scenario_metadata = {"task_id": self.task_id}
        scenario_metadata.update(dict(metadata or {}))
        return get_scenario(
            self.raw_env.unwrapped,
            env_id=env_id,
            name=name,
            metadata=scenario_metadata,
        )

    def _ensure_recorder_export_path(self, dataset_path: Path) -> None:
        """Ensure RecorderManager was constructed with the requested export path."""

        if self.env_cfg is None:
            if _recorder.recorder_export_path_matches(
                self.raw_env.unwrapped.cfg, dataset_path
            ):
                return
            raise RuntimeError(
                "ioailab collect(path=...) requires a make_env-created ioailabEnv so RecorderManager "
                "can be constructed with the requested dataset path before recording."
            )
        if _recorder.recorder_export_path_matches(self.env_cfg, dataset_path):
            return

        _recorder.configure_recorder_export_path(self.env_cfg, dataset_path)
        if hasattr(self.raw_env, "close"):
            self.raw_env.close()
        self.raw_env = _factory.make_gym_env(self.task_id, self.env_cfg)

    def _get_obs_dict(self) -> dict[str, Any]:
        """Extract current observations as a flat dict of tensors."""

        obs_manager = getattr(self.raw_env.unwrapped, "observation_manager", None)
        if obs_manager is None:
            return {}
        obs_groups = obs_manager.compute()
        result: dict[str, Any] = {}
        for _group_name, group_data in obs_groups.items():
            if isinstance(group_data, dict):
                for key, value in group_data.items():
                    result[key] = value
            else:
                result["obs"] = group_data
        return result

    def _run_evaluation(
        self,
        agent: BaseAgent,
        *,
        episodes: int,
        max_steps: int,
    ) -> dict[str, Any]:
        """Run total-episode evaluation using IsaacLab vector-env row resets."""

        num_envs = int(self.num_envs)
        target_episodes = int(episodes)
        self.reset()
        agent.reset(self)

        current_lengths = [0] * num_envs
        current_rewards = [0.0] * num_envs
        episode_lengths: list[int] = []
        reward_totals: list[float] = []
        successes: list[bool] = []
        termination_reasons: list[str] = []
        lengths_by_env = [[] for _ in range(num_envs)]
        rewards_by_env = [[] for _ in range(num_envs)]
        successes_by_env = [[] for _ in range(num_envs)]
        current_successes = [False] * num_envs
        reasons_by_env = [[] for _ in range(num_envs)]
        vector_steps = 0
        completed_total = 0

        while completed_total < target_episodes:
            action = agent.act(self)
            result = self.raw_env.step(action)
            _obs, reward, terminated, truncated, extras = _masks.unpack_step_result(
                result
            )
            vector_steps += 1

            reward_values = _masks.numeric_vector(reward, num_envs)
            for env_id, reward_value in enumerate(reward_values):
                current_lengths[env_id] += 1
                current_rewards[env_id] += reward_value

            masks = _masks.completion_masks(
                env_cfg=self.env_cfg,
                raw_env=self.raw_env,
                terminated=terminated,
                truncated=truncated,
                extras=extras,
                current_lengths=current_lengths,
                max_steps=max_steps,
                num_envs=num_envs,
            )
            for env_id, success in enumerate(masks.success):
                current_successes[env_id] = current_successes[env_id] or bool(success)

            completed_ids = _masks.completed_env_ids(masks)
            if not completed_ids:
                continue

            recorded_completed_ids: list[int] = []
            for env_id in completed_ids:
                if completed_total >= target_episodes:
                    break
                reason = _masks.row_termination_reason(
                    env_id,
                    masks=masks,
                    extras=extras,
                )
                length = int(current_lengths[env_id])
                reward_total = float(current_rewards[env_id])
                success = bool(current_successes[env_id])

                completed_total += 1
                recorded_completed_ids.append(env_id)
                episode_lengths.append(length)
                reward_totals.append(reward_total)
                successes.append(success)
                termination_reasons.append(reason)
                lengths_by_env[env_id].append(length)
                rewards_by_env[env_id].append(reward_total)
                successes_by_env[env_id].append(success)
                reasons_by_env[env_id].append(reason)

                current_lengths[env_id] = 0
                current_rewards[env_id] = 0.0
                current_successes[env_id] = False

            if completed_total >= target_episodes:
                break

            manual_reset_ids = tuple(
                env_id
                for env_id in recorded_completed_ids
                if not masks.env_done[env_id]
            )
            if manual_reset_ids:
                _reset_env_rows(self, manual_reset_ids)
            _reset_agent_rows(agent, self, recorded_completed_ids)

        total_episodes = target_episodes
        row_steps = int(sum(episode_lengths))
        return {
            "total_episodes": total_episodes,
            "vector_steps": vector_steps,
            "row_steps": row_steps,
            "success_count": sum(1 for success in successes if success),
            "success_rate": sum(1 for success in successes if success)
            / float(total_episodes),
            "success_masks": tuple(tuple(values) for values in successes_by_env),
            "episode_lengths": tuple(episode_lengths),
            "episode_lengths_by_env": tuple(tuple(values) for values in lengths_by_env),
            "average_length": row_steps / float(total_episodes),
            "reward_totals": tuple(reward_totals),
            "reward_totals_by_env": tuple(tuple(values) for values in rewards_by_env),
            "reward_total_mean": sum(reward_totals) / float(total_episodes),
            "termination_reasons": tuple(termination_reasons),
            "termination_reasons_by_env": tuple(
                tuple(values) for values in reasons_by_env
            ),
        }


def make_env(
    task_id: str,
    *,
    num_envs: int = 1,
    **options: Any,
) -> ioailabEnv:
    """Create a live ioailab workflow env through IsaacLab/Gymnasium."""

    if not str(task_id).strip():
        raise ValueError("task_id must be a non-empty string.")
    if int(num_envs) < 1:
        raise ValueError("num_envs must be greater than zero.")
    _factory.validate_make_env_options(options)

    app, env_cfg = _factory.build_isaaclab_app_and_cfg(
        str(task_id), int(num_envs), options
    )
    raw_env = _factory.make_gym_env(str(task_id), env_cfg)
    return ioailabEnv(
        task_id=str(task_id),
        raw_env=raw_env,
        app=app,
        num_envs=int(num_envs),
        env_cfg=env_cfg,
        options=dict(options),
    )


def _reset_env_rows(env: ioailabEnv, env_ids: Sequence[int]) -> None:
    """Reset env rows through IsaacLab's manager-based ``reset(env_ids=...)`` API."""

    ids = tuple(int(env_id) for env_id in env_ids)
    if not ids:
        return
    env.reset(env_ids=ids)


def _default_end_scenario_name(task_id: str) -> str:
    """Return the derived scenario name used for terminal collection state."""

    return f"{task_id.removesuffix('-v0')}_end"


def _wrap_pre_reset_terminal_scenario_capture(
    env: ioailabEnv,
    record_pre_reset: Any,
    terminal_scenarios: dict[int, Any],
    *,
    name: str | None,
) -> Any:
    """Capture terminal scene state before IsaacLab auto-resets completed rows."""

    def wrapped_record_pre_reset(
        env_ids: Sequence[int] | Any | None, force_export_or_skip: Any = None
    ) -> Any:
        from ioailab.tasks.common.scenario import get_scenario

        for env_id in _env_id_tuple(env_ids, num_envs=int(env.num_envs)):
            terminal_scenarios[int(env_id)] = get_scenario(
                env.raw_env.unwrapped,
                env_id=int(env_id),
                name=name,
                metadata={"task_id": env.task_id},
            )
        if force_export_or_skip is None:
            return record_pre_reset(env_ids)
        return record_pre_reset(env_ids, force_export_or_skip=force_export_or_skip)

    return wrapped_record_pre_reset


def _env_id_tuple(
    env_ids: Sequence[int] | Any | None, *, num_envs: int
) -> tuple[int, ...]:
    """Normalize IsaacLab env-id inputs to Python row ids."""

    if env_ids is None:
        return tuple(range(int(num_envs)))
    if hasattr(env_ids, "detach"):
        env_ids = env_ids.detach().cpu().tolist()
    return tuple(int(env_id) for env_id in env_ids)


def _scenario_with_metadata(
    scenario: Any,
    *,
    name: str | None,
    metadata: Mapping[str, Any],
) -> Any:
    """Return a scenario copy with collection-end metadata attached."""

    from ioailab.tasks.common.scenario import Scenario

    scenario_metadata = dict(getattr(scenario, "metadata", {}) or {})
    scenario_metadata.update(dict(metadata))
    return Scenario(
        name=name if name is not None else getattr(scenario, "name", None),
        frame=str(getattr(scenario, "frame", "env")),
        assets=getattr(scenario, "assets", {}),
        metadata=scenario_metadata,
        schema=str(getattr(scenario, "schema", "ioailabScenario-v0")),
    )


def _torch_env_ids(target: Any, env_ids: Sequence[int]) -> Any:
    """Return a torch int64 env-id tensor on the target env device."""

    import torch

    unwrapped = getattr(target, "unwrapped", target)
    return torch.tensor(
        tuple(int(env_id) for env_id in env_ids),
        dtype=torch.int64,
        device=getattr(unwrapped, "device", None),
    )


def _reset_agent_rows(
    agent: BaseAgent, env: ioailabEnv, env_ids: Sequence[int]
) -> None:
    """Reset controller state for completed env rows."""

    ids = tuple(int(env_id) for env_id in env_ids)
    if not ids:
        return
    try:
        agent.reset(env, env_ids=ids)
    except NotImplementedError as exc:
        raise RuntimeError(
            f"{type(agent).__name__} does not support per-env reset. "
            "Use num_envs=1 or an agent that honors reset(env, env_ids=...)."
        ) from exc
