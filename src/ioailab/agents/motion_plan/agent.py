"""Motion-planning agent facades."""

from __future__ import annotations

from typing import Any

from ioailab.agents.base import _ActionSourceAgent
from ioailab.agents.io import EnvIds


class PlannerAgent(_ActionSourceAgent):
    """Motion-planning agent facade that emits task actions."""


class CuroboPlannerAgent(PlannerAgent):
    """cuRobo v2 planner agent that returns full IsaacLab action tensors.

    The agent owns only planner/action-source state. It lazily builds the
    action source from task metadata or uses an injected source for tests.
    Caller code owns simulator construction and stepping.
    """

    def __init__(
        self,
        *,
        motion_plan: Any | None = None,
        motion_cfg: Any | None = None,
        action_source: Any | None = None,
        robot_asset_name: str = "robot",
        **action_source_kwargs: Any,
    ) -> None:
        if motion_cfg is None:
            motion_cfg = getattr(motion_plan, "config", None)
        super().__init__(
            None,
            motion_plan=_optional_qualified_name(motion_plan),
            motion_cfg=motion_cfg,
            robot_asset_name=robot_asset_name,
            **action_source_kwargs,
        )
        self.motion_plan = motion_plan
        self.motion_cfg = motion_cfg
        self.robot_asset_name = str(robot_asset_name)
        self.action_source_kwargs = dict(action_source_kwargs)
        self._action_source: Any | None = action_source

    @classmethod
    def from_task(cls, task_id: str, **overrides: Any) -> "CuroboPlannerAgent":
        """Create a cuRobo planner agent from a task-local motion plan.

        ``motion_cfg`` overrides the plan's bundled config.
        """

        from ioailab import tasks

        if "subtask" in overrides:
            raise TypeError(
                "CuroboPlannerAgent.from_task(...) no longer accepts subtask=; "
                "use the phase task ID directly."
            )
        task_options = overrides.pop("task_options", None)
        plan = tasks.motion_plan_for_task(
            str(task_id), config=overrides.pop("motion_cfg", None)
        )
        if task_options:
            apply_task_options = getattr(plan.config, "apply_task_options", None)
            if not callable(apply_task_options):
                raise ValueError(
                    f"Motion-planning cfg {type(plan.config).__name__} does not "
                    "support task_options."
                )
            apply_task_options(dict(task_options))
        robot_asset_name = overrides.pop(
            "robot_asset_name",
            getattr(plan.config, "robot_asset_name", "robot"),
        )
        return cls(
            motion_plan=plan,
            motion_cfg=plan.config,
            robot_asset_name=robot_asset_name,
            **overrides,
        )

    @classmethod
    def from_env(cls, env: Any, **overrides: Any) -> "CuroboPlannerAgent":
        """Create a cuRobo planner agent from a ioailab env's task state."""

        if "subtask" in overrides:
            raise TypeError(
                "CuroboPlannerAgent.from_env(...) no longer accepts subtask=; "
                "use the phase task ID directly."
            )
        if "task_options" in overrides:
            raise ValueError(
                "CuroboPlannerAgent.from_env(...) reads task_options from env; "
                "pass task options when constructing the env, not to the agent."
            )

        task_id = getattr(env, "task_id", None)
        if not task_id:
            raise ValueError("CuroboPlannerAgent.from_env(...) requires env.task_id.")

        task_options = getattr(env, "task_options", {})
        if task_options:
            overrides["task_options"] = task_options

        return cls.from_task(str(task_id), **overrides)

    @property
    def action_source(self) -> Any | None:  # type: ignore[override]
        """Return the lazily constructed underlying action source."""

        return self._action_source

    @action_source.setter
    def action_source(self, value: Any | None) -> None:
        self._action_source = value

    @property
    def is_complete(self) -> bool:
        """Return whether the underlying planner has emitted all planned actions."""

        return bool(getattr(self._require_action_source(), "is_complete", False))

    @property
    def final_action_tensor(self) -> Any | None:
        """Return the final full action tensor emitted by the planner, if any."""

        return getattr(self._require_action_source(), "final_action_tensor", None)

    @property
    def current_target_name(self) -> str:
        """Return the current planner target/stage name for diagnostics."""

        return str(getattr(self._require_action_source(), "current_target_name", ""))

    def reset(self, env: Any, env_ids: EnvIds = None) -> None:
        """Reset/replan against the caller-owned simulator state."""

        planner_env = self._planner_env(env)
        if self._action_source is None:
            if self.motion_plan is None:
                raise ValueError(
                    "CuroboPlannerAgent requires a motion plan from from_task(...) "
                    "or an injected action source."
                )
            self._action_source = self._make_action_source()
        self._action_source.reset(planner_env, env_ids=env_ids)

    def act(self, env: Any, env_ids: EnvIds = None) -> Any:
        """Return the next full action tensor for the caller-owned loop."""

        planner_env = self._planner_env(env)
        if self._action_source is None:
            if self.motion_plan is None:
                raise RuntimeError(
                    "CuroboPlannerAgent must be reset with a motion plan before act(...), "
                    "or be given an action source."
                )
            self.reset(env)
        return self._require_action_source().act(planner_env, env_ids=env_ids)

    def done(self, env: Any | None = None, env_ids: EnvIds = None) -> Any:
        """Return whether the planner has emitted all planned actions."""

        source = self._require_action_source()
        done = getattr(source, "done", None)
        if callable(done):
            planner_env = self._planner_env(env) if env is not None else env
            return done(planner_env, env_ids=env_ids)
        return self.is_complete

    @staticmethod
    def _planner_env(env: Any) -> Any:
        """Return the IsaacLab env behind optional Gymnasium wrappers."""

        return getattr(env, "unwrapped", env)

    def diagnostics(self) -> dict[str, Any]:
        """Return lightweight planner diagnostics without importing planner backends."""

        source = self._action_source
        return {
            "planner": "curobov2",
            "motion_plan": _optional_qualified_name(self.motion_plan),
            "current_target_name": getattr(source, "current_target_name", ""),
            "is_complete": bool(getattr(source, "is_complete", False)),
            "has_final_action_tensor": getattr(source, "final_action_tensor", None)
            is not None,
        }

    def _make_action_source(self) -> Any:
        from ioailab.agents.motion_plan.action_source import (
            make_g1_curobo_motion_plan_action_source,
        )

        return make_g1_curobo_motion_plan_action_source(
            motion_plan=self.motion_plan,
            motion_cfg=self.motion_cfg,
            robot_asset_name=self.robot_asset_name,
            **self.action_source_kwargs,
        )

    def _require_action_source(self) -> Any:
        if self._action_source is None:
            raise RuntimeError(
                "CuroboPlannerAgent must be reset before planner state is available."
            )
        return self._action_source


def _optional_qualified_name(obj: Any | None) -> str | None:
    if obj is None:
        return None
    source = obj if isinstance(obj, type) else type(obj)
    module = getattr(source, "__module__", "")
    name = getattr(source, "__qualname__", getattr(source, "__name__", repr(source)))
    return f"{module}:{name}" if module else str(name)
