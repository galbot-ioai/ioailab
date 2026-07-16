"""Private make_env factory helpers for :mod:`ioailab.envs.env`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_APP_OPTION_KEYS = {"headless", "visualizer", "enable_cameras"}
_ENV_CFG_OPTION_KEYS = {"device", "randomize", "randomization_events", "use_fabric"}
_TASK_OPTION_KEYS = {"task_options"}
_ALLOWED_MAKE_ENV_OPTION_KEYS = (
    _APP_OPTION_KEYS | _ENV_CFG_OPTION_KEYS | _TASK_OPTION_KEYS
)


def validate_make_env_options(options: Mapping[str, Any]) -> None:
    """Reject unknown make_env options instead of silently ignoring them."""

    unknown = tuple(sorted(set(options) - _ALLOWED_MAKE_ENV_OPTION_KEYS))
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED_MAKE_ENV_OPTION_KEYS))
        raise ValueError(
            f"Unknown make_env option(s): {unknown}. Allowed options: {allowed}."
        )


def build_isaaclab_app_and_cfg(
    task_id: str,
    num_envs: int,
    options: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Launch IsaacLab and build the env cfg."""

    import ioailab.tasks

    ioailab.tasks.register_tasks()
    task_entry = ioailab.tasks.task_entry_for_task_id(task_id)

    from isaaclab.app import AppLauncher

    app_kwargs = make_app_kwargs(options, requires_cameras=task_entry.requires_cameras)
    if app_kwargs.get("visualizer") is None and not bool(
        app_kwargs.get("headless", False)
    ):
        app_kwargs["visualizer"] = "kit"
    launcher = AppLauncher(**app_kwargs)
    app = launcher.app

    from isaaclab_tasks.utils import parse_env_cfg

    parse_kwargs: dict[str, Any] = {"num_envs": num_envs}
    for key in ("device", "use_fabric"):
        if key in options:
            parse_kwargs[key] = options[key]
    env_cfg = parse_env_cfg(task_id, **parse_kwargs)
    configure_reset_randomization(
        env_cfg, task_entry.reset_randomization_events, options
    )
    configure_task_options(env_cfg, options)
    return app, env_cfg


def configure_task_options(env_cfg: Any, options: Mapping[str, Any]) -> None:
    """Apply ioailab task-specific options after IsaacLab cfg parsing.

    ``task_options`` is a task-local mapping (e.g. the sorting object to collect)
    delegated to the env cfg's ``apply_task_options`` hook, so per-task selection
    stays out of the generic make_env option set.
    """

    task_options = options.get("task_options")
    if not task_options:
        return
    if not isinstance(task_options, Mapping):
        raise ValueError("make_env task_options must be a mapping.")
    apply_options = getattr(env_cfg, "apply_task_options", None)
    if not callable(apply_options):
        raise ValueError(
            f"Task env cfg {type(env_cfg).__name__} does not accept task_options."
        )
    apply_options(dict(task_options))


def apply_env_cfg_runtime(
    env_cfg: Any, num_envs: int, options: Mapping[str, Any]
) -> None:
    """Apply runtime wiring (num_envs, device, fabric) to a resolved env cfg."""

    env_cfg.scene.num_envs = int(num_envs)
    if "device" in options:
        env_cfg.sim.device = options["device"]
    if "use_fabric" in options:
        env_cfg.sim.use_fabric = bool(options["use_fabric"])


def make_app_kwargs(
    options: Mapping[str, Any], *, requires_cameras: bool
) -> dict[str, Any]:
    """Return IsaacLab ``AppLauncher`` kwargs from workflow options and task metadata."""

    app_kwargs = {key: options[key] for key in _APP_OPTION_KEYS if key in options}
    if requires_cameras:
        if "enable_cameras" in app_kwargs and not bool(app_kwargs["enable_cameras"]):
            raise ValueError("Task requires cameras; enable_cameras cannot be False.")
        app_kwargs["enable_cameras"] = True
    return app_kwargs


def make_gym_env(task_id: str, env_cfg: Any) -> Any:
    """Construct the Gymnasium env after IsaacLab app launch."""

    import gymnasium as gym

    return gym.make(task_id, cfg=env_cfg)


def configure_reset_randomization(
    env_cfg: Any, event_names: Sequence[str], options: Mapping[str, Any]
) -> None:
    """Apply reset-randomization options to an IsaacLab env cfg."""

    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    if not bool(options.get("randomize", False)):
        enabled_events: tuple[str, ...] = ()
    else:
        requested = options.get("randomization_events")
        if requested is None:
            enabled_events = tuple(event_names)
        elif isinstance(requested, str):
            enabled_events = (requested,)
        else:
            enabled_events = tuple(str(event_name) for event_name in requested)
    for event_name in event_names:
        if event_name not in enabled_events and hasattr(events, event_name):
            setattr(events, event_name, None)
