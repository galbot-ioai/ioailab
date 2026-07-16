"""Built-in ioailab task registry.

Importing this package is intentionally lightweight. Task modules are loaded
lazily when their exports are accessed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Minimal task metadata that is not part of Gymnasium registration."""

    task_id: str
    entry_point: str
    isaaclab_kwargs: Mapping[str, str]
    motion_plan_entry_point: str | None = None
    nav_agent_entry_point: str | None = None
    task_flow_entry_point: str | None = None
    requires_cameras: bool = False
    reset_randomization_events: Sequence[str] = ()

    def __post_init__(self) -> None:
        kwargs = dict(self.isaaclab_kwargs)
        if "env_cfg_entry_point" not in kwargs:
            raise ValueError("TaskSpec requires an IsaacLab env_cfg_entry_point kwarg.")
        for key in kwargs:
            if key != "env_cfg_entry_point" and not key.endswith("_cfg_entry_point"):
                raise ValueError(
                    "IsaacLab task kwargs must use '*_cfg_entry_point' keys."
                )
        object.__setattr__(self, "isaaclab_kwargs", kwargs)
        object.__setattr__(
            self, "reset_randomization_events", tuple(self.reset_randomization_events)
        )

    @property
    def env_cfg_entry_point(self) -> str:
        """Return the IsaacLab env cfg entry point."""

        return self.isaaclab_kwargs["env_cfg_entry_point"]

    @property
    def motion_plan_factory(self) -> Any | None:
        """Return the optional ``(config=None) -> TaskMotionPlan`` factory."""

        return _load_entry_point(self.motion_plan_entry_point)

    @property
    def nav_agent_factory(self) -> Any | None:
        """Return the optional direct navigation-agent factory."""

        return _load_entry_point(self.nav_agent_entry_point)

    @property
    def task_flow(self) -> Any | None:
        """Return the optional coherent task-flow metadata."""

        return _load_entry_point(self.task_flow_entry_point)

    def gym_kwargs(self) -> dict[str, str]:
        """Return a copy of the IsaacLab Gymnasium kwargs."""

        return dict(self.isaaclab_kwargs)


_TASK_MODULE_NAMES = (
    "reach",
    "pick_cube",
    "stack_cube",
    "base_nav",
    "pick_to_shelf",
    "pick_to_shelf_pick",
    "pick_to_shelf_nav",
    "pick_to_shelf_place",
    "sort_to_shelf",
    "sort_to_shelf_pick",
    "sort_to_shelf_nav",
    "sort_to_shelf_place",
)

_TASK_EXPORTS = {
    "GALBOT_G1_BASE_NAV_TASK_ID",
    "GALBOT_G1_BASE_NAV_TASK_IDS",
    "GALBOT_G1_BASE_NAV_TASK",
    "GALBOT_G1_BASE_NAV_TASKS",
    "GALBOT_G1_PICK_TO_SHELF_TASK_ID",
    "GALBOT_G1_PICK_TO_SHELF_TASK_IDS",
    "GALBOT_G1_PICK_TO_SHELF_TASK",
    "GALBOT_G1_PICK_TO_SHELF_TASKS",
    "GALBOT_G1_PICK_TO_SHELF_PICK_TASK_ID",
    "GALBOT_G1_PICK_TO_SHELF_PICK_TASK_IDS",
    "GALBOT_G1_PICK_TO_SHELF_PICK_TASK",
    "GALBOT_G1_PICK_TO_SHELF_PICK_TASKS",
    "GALBOT_G1_PICK_TO_SHELF_NAV_TASK_ID",
    "GALBOT_G1_PICK_TO_SHELF_NAV_TASK_IDS",
    "GALBOT_G1_PICK_TO_SHELF_NAV_TASK",
    "GALBOT_G1_PICK_TO_SHELF_NAV_TASKS",
    "GALBOT_G1_PICK_TO_SHELF_PLACE_TASK_ID",
    "GALBOT_G1_PICK_TO_SHELF_PLACE_TASK_IDS",
    "GALBOT_G1_PICK_TO_SHELF_PLACE_TASK",
    "GALBOT_G1_PICK_TO_SHELF_PLACE_TASKS",
    "GALBOT_G1_SORT_TO_SHELF_TASK_ID",
    "GALBOT_G1_SORT_TO_SHELF_TASK_IDS",
    "GALBOT_G1_SORT_TO_SHELF_TASK",
    "GALBOT_G1_SORT_TO_SHELF_TASKS",
    "GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID",
    "GALBOT_G1_SORT_TO_SHELF_PICK_TASK_IDS",
    "GALBOT_G1_SORT_TO_SHELF_PICK_TASK",
    "GALBOT_G1_SORT_TO_SHELF_PICK_TASKS",
    "GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID",
    "GALBOT_G1_SORT_TO_SHELF_NAV_TASK_IDS",
    "GALBOT_G1_SORT_TO_SHELF_NAV_TASK",
    "GALBOT_G1_SORT_TO_SHELF_NAV_TASKS",
    "GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID",
    "GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_IDS",
    "GALBOT_G1_SORT_TO_SHELF_PLACE_TASK",
    "GALBOT_G1_SORT_TO_SHELF_PLACE_TASKS",
    "GALBOT_G1_PICK_CUBE_TASK_ID",
    "GALBOT_G1_PICK_CUBE_TASK_IDS",
    "GALBOT_G1_PICK_CUBE_MIMIC_TASK_ID",
    "GALBOT_G1_PICK_CUBE_MIMIC_TASK",
    "GALBOT_G1_PICK_CUBE_TELEOP_TASK_ID",
    "GALBOT_G1_PICK_CUBE_TELEOP_TASK",
    "GALBOT_G1_PICK_CUBE_TASK",
    "GALBOT_G1_PICK_CUBE_TASKS",
    "GALBOT_G1_REACH_TASK_ID",
    "GALBOT_G1_REACH_TASK_IDS",
    "GALBOT_G1_REACH_TASK",
    "GALBOT_G1_REACH_TASKS",
    "GALBOT_G1_STACK_CUBE_TASK_ID",
    "GALBOT_G1_STACK_CUBE_TASK_IDS",
    "GALBOT_G1_STACK_CUBE_TASK",
    "GALBOT_G1_STACK_CUBE_TASKS",
}

__all__ = sorted(
    _TASK_EXPORTS
    | {
        "BUILTIN_TASK_IDS",
        "BUILTIN_TASKS",
        "DEFAULT_TASK_ID",
        "TaskSpec",
        "motion_plan_for_task",
        "nav_agent_for_task",
        "register_tasks",
        "task_flow_for_task",
        "task_entry_for_task_id",
    }
)


def _load_entry_point(entry_point: str | None) -> Any | None:
    """Load a ``module:object`` task metadata entry point lazily."""

    if entry_point is None:
        return None
    module_name, _, object_name = entry_point.partition(":")
    if not module_name or not object_name:
        raise ValueError(
            f"Task entry point must use 'module:object' format, got {entry_point!r}."
        )
    obj: Any = import_module(module_name)
    for attr in object_name.split("."):
        obj = getattr(obj, attr)
    return obj


def _load_task_module(name: str) -> ModuleType:
    return import_module(f"{__name__}.{name}")


def _load_task_modules() -> tuple[ModuleType, ...]:
    return tuple(_load_task_module(name) for name in _TASK_MODULE_NAMES)


def _builtin_tasks() -> tuple[Any, ...]:
    entries: list[Any] = []
    for module in _load_task_modules():
        for attr in dir(module):
            if attr.endswith("_TASKS") and not attr.startswith("_"):
                value = getattr(module, attr)
                if isinstance(value, tuple):
                    entries.extend(value)
    return tuple(entries)


def _builtin_task_ids() -> tuple[str, ...]:
    return tuple(entry.task_id for entry in _builtin_tasks())


def _register_task_spec(task: TaskSpec) -> str | None:
    """Register one task spec with Gymnasium if it is not already registered."""

    import gymnasium as gym
    from gymnasium.envs.registration import registry

    if task.task_id in registry:
        return None
    gym.register(
        id=task.task_id,
        entry_point=task.entry_point,
        kwargs=task.gym_kwargs(),
        disable_env_checker=True,
    )
    return task.task_id


def task_entry_for_task_id(task_id: str) -> Any:
    """Return the task metadata for ``task_id``."""

    for entry in _builtin_tasks():
        if entry.task_id == str(task_id):
            return entry
    raise ValueError(
        f"Unknown ioailab task ID {task_id!r}. Available: {_builtin_task_ids()}."
    )


def motion_plan_for_task(task_id: str, *, config: Any | None = None) -> Any:
    """Return a task-local motion-plan instance, bundling its planning config.

    ``config`` overrides the plan's default config.
    """

    entry = task_entry_for_task_id(task_id)
    factory = entry.motion_plan_factory
    if factory is None:
        raise ValueError(f"Task {task_id!r} does not define a motion plan.")
    return factory(config=config)


def nav_agent_for_task(task_id: str, **overrides: Any) -> Any:
    """Return a task-local navigation agent for a task."""

    entry = task_entry_for_task_id(task_id)
    factory = entry.nav_agent_factory
    if factory is None:
        raise ValueError(f"Task {task_id!r} does not define a navigation agent.")
    return factory(**overrides)


def task_flow_for_task(task_id: str) -> Any:
    """Return coherent task-flow metadata for ``task_id``."""

    entry = task_entry_for_task_id(task_id)
    flow = entry.task_flow
    if flow is None:
        raise ValueError(f"Task {task_id!r} does not define a task flow.")
    return flow


def register_tasks() -> tuple[str, ...]:
    """Register all built-in ioailab tasks with Gymnasium."""

    registered: list[str] = []
    for task in _builtin_tasks():
        task_id = _register_task_spec(task)
        if task_id is not None:
            registered.append(task_id)
    return tuple(registered)


def __getattr__(name: str) -> Any:
    if name == "DEFAULT_TASK_ID":
        return "GalbotG1-PickCube-v0"
    if name == "BUILTIN_TASK_IDS":
        value = _builtin_task_ids()
    elif name == "BUILTIN_TASKS":
        value = _builtin_tasks()
    elif name in _TASK_EXPORTS:
        for module in _load_task_modules():
            if hasattr(module, name):
                value = getattr(module, name)
                break
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
