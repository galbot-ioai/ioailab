"""Private IsaacLab RecorderManager helpers for :mod:`ioailab.envs.env`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ioailab.agents.base import BaseAgent
from ioailab.envs._masks import bool_mask as _bool_mask

_RECORDER_EXPORT_DIR_ATTR = "dataset_export_dir_path"
_RECORDER_EXPORT_FILE_ATTR = "dataset_filename"


def close_recorder_file_handlers(recorder_manager: Any, *, clear: bool = False) -> None:
    """Close IsaacLab recorder HDF5 handlers so exported datasets are readable immediately.

    Pass ``clear=True`` at the retarget seam to also drop the manager's handler
    references after closing, so new handlers can be created for the new path.
    """

    for attr_name in ("_dataset_file_handler", "_failed_episode_dataset_file_handler"):
        handler = getattr(recorder_manager, attr_name, None)
        close = getattr(handler, "close", None)
        if callable(close):
            close()
        if clear and hasattr(recorder_manager, attr_name):
            setattr(recorder_manager, attr_name, None)


def require_recorder_manager(unwrapped_env: Any) -> Any:
    """Return the live IsaacLab recorder manager or raise a clear config error."""

    recorder_manager = getattr(unwrapped_env, "recorder_manager", None)
    if recorder_manager is None:
        raise RuntimeError(
            "ioailab collect requires env.unwrapped.recorder_manager. "
            "Configure IsaacLab recorders for this task before collecting demos."
        )
    return recorder_manager


def selected_env_ids(env: Any, env_ids: Sequence[int] | None) -> tuple[int, ...]:
    """Return validated env ids for save/drop operations."""

    if env_ids is None:
        return tuple(range(int(env.num_envs)))
    ids = tuple(int(env_id) for env_id in env_ids)
    if not ids:
        raise ValueError("env_ids must not be empty.")
    if len(set(ids)) != len(ids):
        raise ValueError("env_ids must be unique.")
    if any(env_id < 0 or env_id >= int(env.num_envs) for env_id in ids):
        raise ValueError(f"env_ids out of range for num_envs={env.num_envs}.")
    return ids


def require_recorder_cfg(env_or_cfg: Any) -> Any:
    """Return the task/env recorder cfg or raise a clear configuration error."""

    cfg = (
        env_or_cfg
        if hasattr(env_or_cfg, "recorders")
        else getattr(env_or_cfg, "cfg", None)
    )
    if isinstance(cfg, Mapping):
        recorders = cfg.get("recorders")
    else:
        recorders = getattr(cfg, "recorders", None) if cfg is not None else None
    if recorders is None:
        raise RuntimeError(
            "ioailab collect requires cfg.recorders to be configured; "
            "handwritten HDF5 collection fallback has been removed."
        )
    if isinstance(recorders, Mapping) and not recorders:
        raise RuntimeError(
            "ioailab collect requires at least one configured recorder term."
        )
    return recorders


def retarget_recorder_export_path(
    unwrapped_env: Any,
    *,
    recorder_manager: Any,
    dataset_path: Path,
) -> None:
    """Point the live IsaacLab recorder manager at ``dataset_path``."""

    if recorder_export_path_matches(unwrapped_env.cfg, dataset_path):
        return

    close_recorder_file_handlers(recorder_manager, clear=True)
    configure_recorder_export_path(unwrapped_env.cfg, dataset_path)
    recorder_cfg = require_recorder_cfg(unwrapped_env.cfg)
    create_recorder_file_handlers(
        recorder_manager,
        recorder_cfg=recorder_cfg,
        env_name=getattr(unwrapped_env.cfg, "env_name", None),
    )


def create_recorder_file_handlers(
    recorder_manager: Any,
    *,
    recorder_cfg: Any,
    env_name: str | None,
) -> None:
    """Recreate IsaacLab recorder file handlers after changing the target path."""

    handler_type = getattr(recorder_cfg, "dataset_file_handler_class_type", None)
    if handler_type is None:
        return

    from isaaclab.managers.recorder_manager import DatasetExportMode

    export_mode = getattr(recorder_cfg, "dataset_export_mode", None)
    dataset_path = Path(getattr(recorder_cfg, _RECORDER_EXPORT_DIR_ATTR)) / str(
        getattr(recorder_cfg, _RECORDER_EXPORT_FILE_ATTR)
    )
    if export_mode != DatasetExportMode.EXPORT_NONE:
        handler = handler_type()
        open_or_create_hdf5_handler(handler, dataset_path, env_name=env_name)
        setattr(recorder_manager, "_dataset_file_handler", handler)

    if export_mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
        failed_handler = handler_type()
        open_or_create_hdf5_handler(
            failed_handler, Path(str(dataset_path) + "_failed"), env_name=env_name
        )
        setattr(
            recorder_manager, "_failed_episode_dataset_file_handler", failed_handler
        )


def open_or_create_hdf5_handler(
    handler: Any, dataset_path: Path, *, env_name: str | None
) -> None:
    """Open an existing IsaacLab HDF5 dataset or create a new one."""

    hdf5_path = hdf5_file_path(dataset_path)
    if hdf5_path.is_file():
        open_handler = getattr(handler, "open", None)
        if callable(open_handler):
            open_handler(str(hdf5_path), mode="r+")
            if env_name is not None:
                set_env_name = getattr(handler, "set_env_name", None)
                if callable(set_env_name):
                    set_env_name(env_name)
            sync_handler_demo_count(handler, hdf5_path)
            return

    create_handler = getattr(handler, "create", None)
    if not callable(create_handler):
        raise RuntimeError(
            "IsaacLab recorder file handler does not expose create(...)."
        )
    create_handler(str(hdf5_path), env_name=env_name)


def hdf5_file_path(dataset_path: Path) -> Path:
    """Return the concrete .hdf5 path used by IsaacLab's HDF5 handler."""

    path_text = str(dataset_path)
    if path_text.endswith(".hdf5"):
        return dataset_path
    return Path(path_text + ".hdf5")


def sync_handler_demo_count(handler: Any, dataset_path: Path) -> None:
    """Keep IsaacLab's default demo id after the highest existing demo group."""

    if hasattr(handler, "_demo_count"):
        setattr(handler, "_demo_count", next_dataset_demo_id(dataset_path))


def drop_recorded_episodes(
    recorder_manager: Any,
    *,
    env_ids: Sequence[int],
) -> None:
    """Discard buffered IsaacLab recorder episodes for ``env_ids``."""

    export_episodes = getattr(recorder_manager, "export_episodes", None)
    cfg = getattr(recorder_manager, "cfg", None)
    if (
        callable(export_episodes)
        and cfg is not None
        and hasattr(cfg, "dataset_export_mode")
    ):
        from isaaclab.managers.recorder_manager import DatasetExportMode

        original_mode = cfg.dataset_export_mode
        cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        try:
            export_episodes(env_ids=tuple(env_ids), demo_ids=None)
        finally:
            cfg.dataset_export_mode = original_mode
        return

    episodes = getattr(recorder_manager, "_episodes", None)
    if isinstance(episodes, dict):
        from isaaclab.utils.datasets import EpisodeData

        for env_id in env_ids:
            episodes[int(env_id)] = EpisodeData()


def configure_recorder_export_path(env_cfg: Any, dataset_path: Path) -> None:
    """Point IsaacLab's recorder config at the requested dataset path."""

    recorder_cfg = require_recorder_cfg(env_cfg)
    configured = False
    for cfg in iter_recorder_cfg_objects(recorder_cfg):
        if hasattr(cfg, _RECORDER_EXPORT_DIR_ATTR) and hasattr(
            cfg, _RECORDER_EXPORT_FILE_ATTR
        ):
            setattr(cfg, _RECORDER_EXPORT_DIR_ATTR, str(dataset_path.parent))
            setattr(cfg, _RECORDER_EXPORT_FILE_ATTR, dataset_path.stem)
            configured = True
    if not configured:
        raise RuntimeError(
            "ioailab collect requires cfg.recorders to inherit IsaacLab RecorderManagerBaseCfg "
            f"and expose {_RECORDER_EXPORT_DIR_ATTR!r} plus {_RECORDER_EXPORT_FILE_ATTR!r}."
        )


def recorder_export_path_matches(env_cfg: Any, dataset_path: Path) -> bool:
    """Return whether every recorder cfg object already targets ``dataset_path``."""

    recorder_cfg = require_recorder_cfg(env_cfg)
    cfg_objects = iter_recorder_cfg_objects(recorder_cfg)
    if not cfg_objects:
        return False
    for cfg in cfg_objects:
        if not hasattr(cfg, _RECORDER_EXPORT_DIR_ATTR) or not hasattr(
            cfg, _RECORDER_EXPORT_FILE_ATTR
        ):
            return False
        if Path(getattr(cfg, _RECORDER_EXPORT_DIR_ATTR)) != dataset_path.parent:
            return False
        if str(getattr(cfg, _RECORDER_EXPORT_FILE_ATTR)) != dataset_path.stem:
            return False
    return True


def iter_recorder_cfg_objects(recorders: Any) -> tuple[Any, ...]:
    """Return recorder manager cfg candidates without accepting signature fallbacks."""

    if isinstance(recorders, Mapping):
        return tuple(recorders.values())
    if isinstance(recorders, Sequence) and not isinstance(recorders, (str, bytes)):
        return tuple(recorders)
    return (recorders,)


def next_dataset_demo_id(dataset_path: Path) -> int:
    """Return the next Robomimic demo id for appending to ``dataset_path``."""

    if not dataset_path.is_file():
        return 0
    try:
        import h5py
    except ModuleNotFoundError:
        return 0
    try:
        with h5py.File(dataset_path, "r") as file:
            data_group = file.get("data")
            if data_group is None:
                return 0
            demo_ids = [
                int(name.removeprefix("demo_"))
                for name in data_group.keys()
                if name.startswith("demo_") and name.removeprefix("demo_").isdigit()
            ]
    except OSError:
        return 0
    return max(demo_ids, default=-1) + 1


def collection_completed_env_ids(
    *,
    env_done: Sequence[bool],
    success: Sequence[bool],
    max_step: Sequence[bool],
    user_exit: Sequence[bool],
) -> tuple[int, ...]:
    """Rows whose collection episode must end. No generic agent.done here."""

    return tuple(
        env_id
        for env_id in range(len(env_done))
        if bool(env_done[env_id])
        or bool(success[env_id])
        or bool(max_step[env_id])
        or bool(user_exit[env_id])
    )


def manual_collection_env_ids(
    env_ids: Sequence[int],
    *,
    env_done: Sequence[bool],
    auto_export_on_env_done: bool,
) -> tuple[int, ...]:
    """Rows ioailab must explicitly export/reset outside IsaacLab auto-reset."""

    ids = tuple(int(env_id) for env_id in env_ids)
    if not auto_export_on_env_done:
        return ids
    return tuple(env_id for env_id in ids if not bool(env_done[env_id]))


def agent_exit_requested_mask(agent: BaseAgent, num_envs: int) -> tuple[bool, ...]:
    """Return explicit operator-exit requests without using generic agent.done()."""

    exit_requested = getattr(agent, "exit_requested", None)
    if not callable(exit_requested):
        return (False,) * int(num_envs)
    return _bool_mask(exit_requested(), num_envs)


def recorder_auto_exports_on_reset(recorder_manager: Any) -> bool:
    """Return whether IsaacLab exports demos automatically during env reset."""

    cfg = getattr(recorder_manager, "cfg", None)
    return bool(getattr(cfg, "export_in_record_pre_reset", False))


def collect_export_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return recorder-export metadata, excluding per-save bookkeeping keys."""

    excluded = {
        "num_envs",
        "saved_env_ids",
        "saved_demo_ids",
        "saved_demos",
        "recorder_manager",
    }
    return {key: value for key, value in metadata.items() if key not in excluded}


def export_recorded_episodes(
    recorder_manager: Any,
    *,
    env_ids: Sequence[int],
    demo_ids: Sequence[int] | None,
) -> dict[str, Any]:
    """Export recorded episodes through IsaacLab's RecorderManager API."""

    export_episodes = getattr(recorder_manager, "export_episodes", None)
    if not callable(export_episodes):
        raise RuntimeError(
            "ioailab demo saving requires an IsaacLab RecorderManager with export_episodes(...); "
            "configure a task recorder manager before saving demos."
        )
    result = export_episodes(env_ids=env_ids, demo_ids=demo_ids)

    metadata: dict[str, Any] = {}
    if isinstance(result, Mapping):
        metadata.update(result)
    elif result is not None:
        metadata["export_result"] = result
    return metadata
