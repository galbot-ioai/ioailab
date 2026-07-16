"""Record motion-planning episodes and export them as LeRobot v3 datasets."""

from __future__ import annotations

import importlib
import inspect
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_HDF5_FILENAME = "motion_plan_staging.hdf5"
DEFAULT_ROBOT_TYPE = "galbot_g1"
CAMERA_FEATURE_PREFIX = "observation.images."


def default_motion_plan_hdf5_path(lerobot_root: str | Path) -> Path:
    """Return the default HDF5 staging path for a LeRobot output root.

    LeRobotDataset.create() expects its root directory not to exist. Keep the
    staging file next to the LeRobot root so recording does not create that
    root before export.
    """

    root = Path(lerobot_root)
    stem = root.name or "lerobot"
    return root.parent / f"{stem}_{DEFAULT_HDF5_FILENAME}"


def validate_motion_plan_record_paths(
    *, lerobot_root: str | Path, hdf5_path: str | Path
) -> None:
    """Validate that HDF5 staging cannot pre-create the LeRobot dataset root."""

    root = Path(lerobot_root).resolve(strict=False)
    hdf5 = Path(hdf5_path).resolve(strict=False)
    if hdf5 == root or root in hdf5.parents:
        raise ValueError(
            "motion-plan HDF5 staging path must not be inside --record-lerobot-root. "
            "LeRobot creates that root itself during export; use the default sibling "
            "staging file or pass --record-hdf5-path outside the LeRobot root."
        )


def validate_lerobot_root_available(lerobot_root: str | Path) -> None:
    """Validate that LeRobot can create a new dataset root."""

    root = Path(lerobot_root)
    if root.exists():
        raise FileExistsError(
            f"LeRobot dataset root already exists: {root}. "
            "LeRobotDataset.create() requires a new root; remove the stale/failed "
            "dataset directory or choose a fresh --record-lerobot-root."
        )


def default_motion_plan_repo_id(task_id: str) -> str:
    """Return a stable local LeRobot repo id for a ioailab task id."""

    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task_id).strip("-").lower()
    return f"ioailab/{normalized or 'motion-plan'}"


@dataclass(slots=True)
class MotionPlanCameraSpec:
    """Configuration for one recorded IsaacLab camera output."""

    feature_key: str
    sensor_name: str
    shape: tuple[int, int, int]
    output_key: str = "rgb"


@dataclass(slots=True)
class MotionPlanRecordConfig:
    """Configuration for motion-plan HDF5 staging and LeRobot export."""

    lerobot_root: Path
    hdf5_path: Path
    repo_id: str
    task: str
    fps: float
    robot_type: str = DEFAULT_ROBOT_TYPE
    include_failed: bool = False
    export_on_close: bool = True
    camera_specs: tuple[MotionPlanCameraSpec, ...] = ()


class MotionPlanHdf5Writer:
    """Append vectorized motion-planning frames into an HDF5 staging database."""

    def __init__(
        self, file_path: str | Path, *, compression: str | None = "gzip"
    ) -> None:
        self.file_path = Path(file_path)
        self.compression = compression
        self._h5py = _require_h5py()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._h5py.File(self.file_path, mode="a")
        self._episodes_group = self._file.require_group("episodes")
        self._active_episode_groups: list[Any] = []
        self._active_frame_counts: list[int] = []
        self._next_episode_index = self._read_next_episode_index()

    @property
    def has_active_episode(self) -> bool:
        """Return whether an episode has been opened but not finished."""

        return bool(self._active_episode_groups)

    def start_episode(
        self,
        *,
        task_id: str,
        planner: str,
        fps: float,
        num_envs: int,
        record_task: str,
        workflow_name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...]:
        """Create one active HDF5 episode group per vectorized env row."""

        if self.has_active_episode:
            raise RuntimeError(
                "finish the active motion-plan episode before starting another one."
            )
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")

        created_names: list[str] = []
        shared_metadata = dict(metadata or {})
        shared_metadata.update(
            {
                "task_id": task_id,
                "planner": planner,
                "fps": float(fps),
                "num_envs": int(num_envs),
                "record_task": record_task,
                "workflow_name": workflow_name,
            }
        )
        for env_index in range(int(num_envs)):
            name = self._allocate_episode_name()
            group = self._episodes_group.create_group(name)
            group.attrs["task_id"] = task_id
            group.attrs["planner"] = planner
            group.attrs["fps"] = float(fps)
            group.attrs["num_envs"] = int(num_envs)
            group.attrs["env_index"] = int(env_index)
            group.attrs["record_task"] = record_task
            group.attrs["workflow_name"] = workflow_name
            group.attrs["success"] = False
            group.attrs["failure_reason"] = ""
            group.attrs["frame_count"] = 0
            group.attrs["exported"] = False
            self._write_json_dataset(
                group,
                "metadata_json",
                {
                    **shared_metadata,
                    "env_index": int(env_index),
                    "episode_name": name,
                },
            )
            self._active_episode_groups.append(group)
            self._active_frame_counts.append(0)
            created_names.append(name)

        self._file.attrs["next_episode_index"] = int(self._next_episode_index)
        self._file.flush()
        return tuple(created_names)

    def append_frame_batch(self, frame_batch: Mapping[str, Any]) -> None:
        """Append one vectorized frame to the active per-env episode groups."""

        if not self.has_active_episode:
            raise RuntimeError("start_episode must be called before appending frames.")
        action = _as_2d_array(frame_batch["action"], dtype=np.float32)
        num_envs = action.shape[0]
        if num_envs != len(self._active_episode_groups):
            raise ValueError(
                f"frame batch has {num_envs} env rows but active episode has "
                f"{len(self._active_episode_groups)} rows."
            )

        numeric_fields = {
            "action": action,
            "observation_state": _as_2d_array(
                frame_batch["observation_state"], dtype=np.float32
            ),
            "joint_pos": _as_2d_array(frame_batch["joint_pos"], dtype=np.float32),
            "joint_vel": _as_2d_array(frame_batch["joint_vel"], dtype=np.float32),
            "root_pose": _as_2d_array(frame_batch["root_pose"], dtype=np.float32),
            "timestamp": _as_1d_array(frame_batch["timestamp"], dtype=np.float64),
            "frame_index": _as_1d_array(frame_batch["frame_index"], dtype=np.int64),
            "terminated": _as_1d_array(
                frame_batch.get("terminated", np.zeros(num_envs)), dtype=np.bool_
            ),
            "truncated": _as_1d_array(
                frame_batch.get("truncated", np.zeros(num_envs)), dtype=np.bool_
            ),
        }
        phase_values = tuple(
            str(value) for value in frame_batch.get("phase", ("",) * num_envs)
        )
        if len(phase_values) != num_envs:
            raise ValueError(
                f"phase must have {num_envs} entries, got {len(phase_values)}."
            )
        camera_fields = _camera_frame_fields(frame_batch, num_envs)

        for env_index, group in enumerate(self._active_episode_groups):
            for name, values in numeric_fields.items():
                self._append_dataset_row(group, name, values[env_index])
            for name, values in camera_fields.items():
                self._append_dataset_row(group, name, values[env_index])
            self._append_string_row(group, "phase", phase_values[env_index])
            self._active_frame_counts[env_index] += 1
            group.attrs["frame_count"] = int(self._active_frame_counts[env_index])
        self._file.flush()

    def finish_episode(self, *, success: bool, failure_reason: str = "") -> None:
        """Mark active episode groups as finished."""

        if not self.has_active_episode:
            return
        for group, frame_count in zip(
            self._active_episode_groups, self._active_frame_counts, strict=True
        ):
            group.attrs["success"] = bool(success)
            group.attrs["failure_reason"] = "" if success else str(failure_reason)
            group.attrs["frame_count"] = int(frame_count)
        self._active_episode_groups = []
        self._active_frame_counts = []
        self._file.flush()

    def close(self) -> None:
        """Flush and close the HDF5 staging file."""

        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None

    def __enter__(self) -> "MotionPlanHdf5Writer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _read_next_episode_index(self) -> int:
        attr_value = self._file.attrs.get("next_episode_index")
        if attr_value is not None:
            return int(attr_value)
        max_index = -1
        for name in self._episodes_group.keys():
            if name.startswith("episode_"):
                try:
                    max_index = max(max_index, int(name.rsplit("_", maxsplit=1)[1]))
                except ValueError:
                    continue
        return max_index + 1

    def _allocate_episode_name(self) -> str:
        while True:
            name = f"episode_{self._next_episode_index:06d}"
            self._next_episode_index += 1
            if name not in self._episodes_group:
                return name

    def _write_json_dataset(
        self, group: Any, name: str, data: Mapping[str, Any]
    ) -> None:
        dtype = self._h5py.string_dtype(encoding="utf-8")
        dataset = group.create_dataset(name, shape=(1,), dtype=dtype)
        dataset[0] = json.dumps(data, ensure_ascii=False, sort_keys=True)

    def _append_dataset_row(self, group: Any, name: str, row: Any) -> None:
        values = np.asarray(row)
        if name not in group:
            shape = (0, *values.shape)
            maxshape = (None, *values.shape)
            kwargs = {}
            if values.shape and self.compression is not None:
                kwargs["compression"] = self.compression
            group.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                chunks=True,
                dtype=values.dtype,
                **kwargs,
            )
        dataset = group[name]
        new_length = dataset.shape[0] + 1
        dataset.resize((new_length, *dataset.shape[1:]))
        dataset[new_length - 1] = values

    def _append_string_row(self, group: Any, name: str, value: str) -> None:
        if name not in group:
            dtype = self._h5py.string_dtype(encoding="utf-8")
            group.create_dataset(
                name, shape=(0,), maxshape=(None,), chunks=True, dtype=dtype
            )
        dataset = group[name]
        new_length = dataset.shape[0] + 1
        dataset.resize((new_length,))
        dataset[new_length - 1] = value


class MotionPlanLeRobotExporter:
    """Export successful HDF5 motion-plan episodes to a LeRobot v3 dataset."""

    def __init__(
        self,
        *,
        hdf5_path: str | Path,
        lerobot_root: str | Path,
        repo_id: str,
        task: str,
        fps: float,
        robot_type: str = DEFAULT_ROBOT_TYPE,
        include_failed: bool = False,
        dataset_cls: Any | None = None,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.lerobot_root = Path(lerobot_root)
        validate_motion_plan_record_paths(
            lerobot_root=lerobot_root, hdf5_path=hdf5_path
        )
        validate_lerobot_root_available(lerobot_root)
        self.repo_id = repo_id
        self.task = task
        self.fps = float(fps)
        self.robot_type = robot_type
        self.include_failed = include_failed
        self.dataset_cls = dataset_cls

    def export(self) -> int:
        """Export eligible HDF5 episodes and return the number written."""

        h5py = _require_h5py()
        dataset = None
        exported_group_names: list[str] = []
        with h5py.File(self.hdf5_path, mode="r") as hdf5_file:
            episode_groups = self._eligible_episode_groups(hdf5_file)
            if not episode_groups:
                return 0
            features = self._features_from_episode(episode_groups[0])
            dataset = self._create_lerobot_dataset(features)
            for group in episode_groups:
                task = str(group.attrs.get("record_task", self.task))
                action = np.asarray(group["action"], dtype=np.float32)
                observation_state = np.asarray(
                    group["observation_state"], dtype=np.float32
                )
                camera_values = {
                    key: np.asarray(group[key], dtype=np.uint8)
                    for key in _camera_feature_keys(features)
                }
                for frame_index in range(action.shape[0]):
                    frame = {
                        "observation.state": _lerobot_frame_value(
                            observation_state[frame_index],
                            features["observation.state"]["shape"],
                        ),
                        "action": _lerobot_frame_value(
                            action[frame_index], features["action"]["shape"]
                        ),
                    }
                    for key, values in camera_values.items():
                        frame[key] = values[frame_index]
                    self._add_lerobot_frame(dataset, frame, task=task)
                dataset.save_episode()
                exported_group_names.append(group.name.rsplit("/", maxsplit=1)[-1])
        if dataset is not None:
            if hasattr(dataset, "finalize"):
                dataset.finalize()
            _make_lerobot_dataset_readable(self.lerobot_root)
        with h5py.File(self.hdf5_path, mode="a") as hdf5_file:
            for group_name in exported_group_names:
                hdf5_file["episodes"][group_name].attrs["exported"] = True
            hdf5_file.flush()
        return len(exported_group_names)

    def _eligible_episode_groups(self, hdf5_file: Any) -> list[Any]:
        if "episodes" not in hdf5_file:
            return []
        groups = []
        for name in sorted(hdf5_file["episodes"].keys()):
            group = hdf5_file["episodes"][name]
            if bool(group.attrs.get("exported", False)):
                continue
            if "action" not in group or int(group.attrs.get("frame_count", 0)) <= 0:
                continue
            if bool(group.attrs.get("success", False)) or self.include_failed:
                groups.append(group)
        return groups

    def _features_from_episode(self, group: Any) -> dict[str, dict[str, Any]]:
        action_shape = _lerobot_feature_shape(
            tuple(np.asarray(group["action"][0]).shape)
        )
        state_shape = _lerobot_feature_shape(
            tuple(np.asarray(group["observation_state"][0]).shape)
        )
        metadata = _read_json_dataset(group, "metadata_json")
        action_names = _metadata_feature_names(
            metadata.get("action_names"), "action", action_shape
        )
        state_names = _metadata_feature_names(
            metadata.get("state_names"), "state", state_shape
        )
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": action_shape,
                "names": action_names,
            },
        }
        for feature_key in _camera_feature_keys_from_episode(group, metadata):
            camera_shape = tuple(
                int(value) for value in np.asarray(group[feature_key][0]).shape
            )
            features[feature_key] = {
                "dtype": "video",
                "shape": camera_shape,
                "names": ("height", "width", "channels"),
            }
        return features

    def _create_lerobot_dataset(self, features: Mapping[str, Any]) -> Any:
        dataset_cls = self.dataset_cls or _import_lerobot_dataset_cls()
        use_videos = _features_include_video(features)
        kwargs = {
            "repo_id": self.repo_id,
            "root": self.lerobot_root,
            "fps": _lerobot_video_fps(self.fps) if use_videos else self.fps,
            "robot_type": self.robot_type,
            "features": dict(features),
            "use_videos": use_videos,
        }
        create = dataset_cls.create
        try:
            signature = inspect.signature(create)
        except (TypeError, ValueError):
            return create(**kwargs)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return create(**kwargs)
        filtered_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
        return create(**filtered_kwargs)

    @staticmethod
    def _add_lerobot_frame(
        dataset: Any, frame: Mapping[str, Any], *, task: str
    ) -> None:
        add_frame = dataset.add_frame
        frame_values = dict(frame)
        try:
            signature = inspect.signature(add_frame)
        except (TypeError, ValueError):
            try:
                add_frame(frame_values, task=task)
                return
            except TypeError:
                frame_values["task"] = task
                add_frame(frame_values)
                return
        if _call_accepts_keyword(signature, "task"):
            add_frame(frame_values, task=task)
            return
        frame_values["task"] = task
        add_frame(frame_values)


def _make_lerobot_dataset_readable(root: str | Path) -> None:
    """Make exported LeRobot files readable from host bind mounts."""

    root_path = Path(root)
    if not root_path.exists():
        return
    for path in (root_path, *root_path.rglob("*")):
        if path.is_symlink():
            continue
        try:
            mode = path.stat().st_mode
        except OSError:
            continue
        if path.is_dir():
            readable_mode = (
                mode
                | stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
        elif path.is_file():
            readable_mode = mode | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        else:
            continue
        if readable_mode != mode:
            path.chmod(readable_mode)


class MotionPlanDatasetRecorder:
    """Coordinate HDF5 staging and optional LeRobot export for one run."""

    def __init__(
        self,
        config: MotionPlanRecordConfig,
        *,
        dataset_cls: Any | None = None,
    ) -> None:
        validate_motion_plan_record_paths(
            lerobot_root=config.lerobot_root,
            hdf5_path=config.hdf5_path,
        )
        validate_lerobot_root_available(config.lerobot_root)
        self.config = config
        self.dataset_cls = dataset_cls
        self.writer = MotionPlanHdf5Writer(config.hdf5_path)
        self._frame_index = 0
        self._closed = False

    def start_episode(
        self, env: Any, *, task_id: str, planner: str, workflow_name: str
    ) -> None:
        """Open one HDF5 episode per env row."""

        self._frame_index = 0
        unwrapped = env.unwrapped
        metadata = collect_motion_plan_metadata(
            env, camera_specs=self.config.camera_specs
        )
        self.writer.start_episode(
            task_id=task_id,
            planner=planner,
            fps=self.config.fps,
            num_envs=int(unwrapped.num_envs),
            record_task=self.config.task,
            workflow_name=workflow_name,
            metadata=metadata,
        )

    def record_step(
        self,
        env: Any,
        action_tensor: Any,
        *,
        phase: str = "",
        terminated: Any | None = None,
        truncated: Any | None = None,
    ) -> None:
        """Append one post-step frame batch to the active HDF5 episode."""

        timestamp = float(self._frame_index) / float(self.config.fps)
        frame_batch = collect_motion_plan_frame_batch(
            env,
            action_tensor,
            frame_index=self._frame_index,
            timestamp=timestamp,
            phase=phase,
            terminated=terminated,
            truncated=truncated,
            camera_specs=self.config.camera_specs,
        )
        self.writer.append_frame_batch(frame_batch)
        self._frame_index += 1

    def finish_episode(self, *, success: bool, failure_reason: str = "") -> None:
        """Finish the active episode if one is open."""

        self.writer.finish_episode(success=success, failure_reason=failure_reason)

    def close(self) -> int:
        """Close HDF5 and export to LeRobot v3 if configured."""

        if self._closed:
            return 0
        if self.writer.has_active_episode:
            self.writer.finish_episode(
                success=False, failure_reason="recorder closed before episode finished"
            )
        self.writer.close()
        self._closed = True
        if not self.config.export_on_close:
            return 0
        exporter = MotionPlanLeRobotExporter(
            hdf5_path=self.config.hdf5_path,
            lerobot_root=self.config.lerobot_root,
            repo_id=self.config.repo_id,
            task=self.config.task,
            fps=self.config.fps,
            robot_type=self.config.robot_type,
            include_failed=self.config.include_failed,
            dataset_cls=self.dataset_cls,
        )
        return exporter.export()


def collect_motion_plan_metadata(
    env: Any,
    *,
    camera_specs: Sequence[MotionPlanCameraSpec] = (),
) -> dict[str, Any]:
    """Collect stable feature metadata from an IsaacLab env."""

    robot = env.unwrapped.scene["robot"]
    joint_names = tuple(str(name) for name in getattr(robot, "joint_names", ()))
    action_dim = int(getattr(env.unwrapped.action_manager, "total_action_dim", 0))
    return {
        "joint_names": joint_names,
        "state_names": _state_feature_names(joint_names),
        "action_names": _default_feature_names("action", (action_dim,)),
        "camera_features": _camera_specs_metadata(camera_specs),
    }


def collect_motion_plan_frame_batch(
    env: Any,
    action_tensor: Any,
    *,
    frame_index: int,
    timestamp: float,
    phase: str = "",
    terminated: Any | None = None,
    truncated: Any | None = None,
    camera_specs: Sequence[MotionPlanCameraSpec] = (),
) -> dict[str, Any]:
    """Collect one vectorized action/state frame batch from an IsaacLab env."""

    robot = env.unwrapped.scene["robot"]
    joint_pos = _as_2d_array(robot.data.joint_pos, dtype=np.float32)
    joint_vel = _as_2d_array(robot.data.joint_vel, dtype=np.float32)
    root_pose = _robot_root_pose(robot)
    action = _as_2d_array(action_tensor, dtype=np.float32)
    if action.shape[0] != joint_pos.shape[0]:
        raise ValueError(
            f"action rows {action.shape[0]} do not match env rows {joint_pos.shape[0]}."
        )
    observation_state = np.concatenate(
        [joint_pos, joint_vel, root_pose], axis=1
    ).astype(np.float32, copy=False)
    num_envs = action.shape[0]
    frame_batch = {
        "action": action,
        "observation_state": observation_state,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pose": root_pose,
        "timestamp": np.full((num_envs,), float(timestamp), dtype=np.float64),
        "frame_index": np.full((num_envs,), int(frame_index), dtype=np.int64),
        "phase": tuple(str(phase) for _ in range(num_envs)),
        "terminated": _done_array(terminated, num_envs),
        "truncated": _done_array(truncated, num_envs),
    }
    for camera_spec in camera_specs:
        frame_batch[camera_spec.feature_key] = _camera_rgb_frame_batch(
            env, camera_spec, num_envs
        )
    return frame_batch


def _camera_specs_metadata(
    camera_specs: Sequence[MotionPlanCameraSpec],
) -> tuple[dict[str, Any], ...]:
    """Return JSON-serializable camera metadata for HDF5 staging."""

    return tuple(
        {
            "feature_key": spec.feature_key,
            "sensor_name": spec.sensor_name,
            "output_key": spec.output_key,
            "shape": tuple(int(value) for value in spec.shape),
        }
        for spec in camera_specs
    )


def _camera_rgb_frame_batch(
    env: Any, spec: MotionPlanCameraSpec, num_envs: int
) -> np.ndarray:
    """Collect one RGB camera output as a uint8 NHWC batch."""

    camera = env.unwrapped.scene[spec.sensor_name]
    output = camera.data.output[spec.output_key]
    images = _as_rgb_uint8_images(output)
    if images.ndim != 4:
        raise ValueError(
            f"camera '{spec.sensor_name}' output '{spec.output_key}' must have shape "
            f"(num_envs, height, width, channels), got {images.shape}."
        )
    if images.shape[0] != num_envs:
        raise ValueError(
            f"camera '{spec.sensor_name}' rows {images.shape[0]} do not match env rows {num_envs}."
        )
    expected_shape = tuple(int(value) for value in spec.shape)
    if images.shape[1:] != expected_shape:
        raise ValueError(
            f"camera '{spec.sensor_name}' output shape {images.shape[1:]} does not match "
            f"configured LeRobot image shape {expected_shape}."
        )
    return images


def _as_rgb_uint8_images(value: Any) -> np.ndarray:
    """Convert an RGB or RGBA tensor-like batch to uint8 NHWC RGB images."""

    images = _as_numpy_array(value)
    if images.ndim < 1 or images.shape[-1] < 3:
        raise ValueError(
            f"camera RGB output must have at least three channels, got shape {images.shape}."
        )
    images = images[..., :3]
    if np.issubdtype(images.dtype, np.floating):
        finite = np.isfinite(images)
        max_value = float(np.max(images[finite])) if bool(finite.any()) else 0.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        upper = 1.0 if scale == 255.0 else 255.0
        images = np.nan_to_num(images, nan=0.0, posinf=upper, neginf=0.0)
        images = np.clip(images, 0.0, upper) * scale
    else:
        images = np.clip(images, 0, 255)
    return images.astype(np.uint8, copy=False)


def _camera_frame_fields(
    frame_batch: Mapping[str, Any], num_envs: int
) -> dict[str, np.ndarray]:
    """Return camera image fields from a vectorized frame batch."""

    fields: dict[str, np.ndarray] = {}
    for name, value in frame_batch.items():
        if not name.startswith(CAMERA_FEATURE_PREFIX):
            continue
        images = _as_rgb_uint8_images(value)
        if images.ndim != 4:
            raise ValueError(
                f"camera feature '{name}' must have shape (num_envs, height, width, channels)."
            )
        if images.shape[0] != num_envs:
            raise ValueError(
                f"camera feature '{name}' rows {images.shape[0]} do not match env rows {num_envs}."
            )
        fields[name] = images
    return fields


def _camera_feature_keys(features: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Return LeRobot visual feature keys in stable insertion order."""

    return tuple(
        key
        for key, feature in features.items()
        if key.startswith(CAMERA_FEATURE_PREFIX)
        and feature.get("dtype") in {"image", "video"}
    )


def _camera_feature_keys_from_episode(
    group: Any, metadata: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return camera feature keys available in an HDF5 episode group."""

    keys: list[str] = []
    for entry in metadata.get("camera_features", ()):
        if not isinstance(entry, Mapping):
            continue
        feature_key = str(entry.get("feature_key", ""))
        if (
            feature_key.startswith(CAMERA_FEATURE_PREFIX)
            and feature_key in group
            and feature_key not in keys
        ):
            keys.append(feature_key)
    for name in group.keys():
        if name.startswith(CAMERA_FEATURE_PREFIX) and name not in keys:
            keys.append(str(name))
    return tuple(keys)


def _features_include_video(features: Mapping[str, Mapping[str, Any]]) -> bool:
    """Return whether the LeRobot feature schema needs video storage."""

    return any(feature.get("dtype") == "video" for feature in features.values())


def _lerobot_video_fps(fps: float) -> int:
    """Return a PyAV- and LeRobot-compatible integer fps for video encoding."""

    value = float(fps)
    if not value.is_integer():
        raise ValueError(
            f"LeRobot video export requires integer-valued fps, got {fps!r}."
        )
    return int(value)


def _robot_root_pose(robot: Any) -> np.ndarray:
    data = robot.data
    if hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
        root_pos = _as_2d_array(data.root_pos_w, dtype=np.float32)
        root_quat = _as_2d_array(data.root_quat_w, dtype=np.float32)
        return np.concatenate([root_pos, root_quat], axis=1).astype(
            np.float32, copy=False
        )
    if hasattr(data, "root_state_w"):
        root_state = _as_2d_array(data.root_state_w, dtype=np.float32)
        if root_state.shape[1] < 7:
            raise ValueError(
                f"root_state_w must expose at least 7 columns, got {root_state.shape[1]}."
            )
        return root_state[:, :7].astype(np.float32, copy=True)
    raise AttributeError(
        "robot data must expose root_pos_w/root_quat_w or root_state_w."
    )


def _state_feature_names(joint_names: Sequence[str]) -> tuple[str, ...]:
    return (
        *(f"joint_pos.{name}" for name in joint_names),
        *(f"joint_vel.{name}" for name in joint_names),
        "root_pos.x",
        "root_pos.y",
        "root_pos.z",
        "root_quat.w",
        "root_quat.x",
        "root_quat.y",
        "root_quat.z",
    )


def _default_feature_names(prefix: str, shape: Sequence[int]) -> tuple[str, ...]:
    size = int(np.prod(tuple(shape), dtype=np.int64)) if shape else 1
    return tuple(f"{prefix}.{index:04d}" for index in range(size))


def _metadata_feature_names(
    names: Any, prefix: str, shape: Sequence[int]
) -> tuple[str, ...]:
    expected_size = int(np.prod(tuple(shape), dtype=np.int64)) if shape else 1
    if names is None:
        return _default_feature_names(prefix, shape)
    values = tuple(str(name) for name in names)
    if len(values) != expected_size:
        return _default_feature_names(prefix, shape)
    return values


def _lerobot_feature_shape(shape: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in shape)
    if values == (1,):
        return (1, 1)
    return values


def _lerobot_frame_value(value: Any, feature_shape: Sequence[int]) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    target_shape = tuple(int(dim) for dim in feature_shape)
    if values.shape != target_shape:
        values = values.reshape(target_shape)
    return values.astype(np.float32, copy=False)


def _call_accepts_keyword(signature: inspect.Signature, keyword: str) -> bool:
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    parameter = signature.parameters.get(keyword)
    if parameter is None:
        return False
    return parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _done_array(done: Any | None, num_envs: int) -> np.ndarray:
    if done is None:
        return np.zeros((num_envs,), dtype=np.bool_)
    values = _as_numpy(done, dtype=np.bool_)
    if values.ndim == 0:
        values = np.full((num_envs,), bool(values), dtype=np.bool_)
    values = values.reshape(-1)
    if values.shape != (num_envs,):
        raise ValueError(
            f"done array must have shape ({num_envs},), got {values.shape}."
        )
    return values.astype(np.bool_, copy=True)


def _as_1d_array(value: Any, *, dtype: Any) -> np.ndarray:
    values = _as_numpy(value, dtype=dtype).reshape(-1)
    return values.astype(dtype, copy=False)


def _as_2d_array(value: Any, *, dtype: Any) -> np.ndarray:
    values = _as_numpy(value, dtype=dtype)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {values.shape}.")
    return values.astype(dtype, copy=False)


def _as_numpy(value: Any, *, dtype: Any) -> np.ndarray:
    return np.asarray(_as_numpy_array(value), dtype=dtype)


def _as_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    if hasattr(value, "torch") and hasattr(value, "warp"):
        # IsaacLab 3.0 ProxyArray: convert via the cached, warning-free torch view.
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass

    try:
        import warp as wp

        torch_value = wp.to_torch(value)
    except Exception:
        return np.asarray(value)

    if hasattr(torch_value, "detach"):
        torch_value = torch_value.detach()
    if hasattr(torch_value, "cpu"):
        torch_value = torch_value.cpu()
    if hasattr(torch_value, "numpy"):
        return np.asarray(torch_value.numpy())
    return np.asarray(torch_value)


def _read_json_dataset(group: Any, name: str) -> dict[str, Any]:
    if name not in group:
        return {}
    value = group[name][0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _require_h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ImportError as exc:
        raise ImportError(
            "Motion-plan HDF5 recording requires h5py. Install h5py in the Isaac Sim Python environment "
            "or disable --record-lerobot-root."
        ) from exc


def _import_lerobot_dataset_cls() -> Any:
    import_errors: list[ImportError] = []
    for module_name in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            import_errors.append(exc)
            continue
        return module.LeRobotDataset

    cause = import_errors[-1] if import_errors else None
    raise ImportError(
        "LeRobot export requires the lerobot dataset package. The HDF5 staging file was written; "
        "rebuild the ioailab Docker image or install the image-compatible LeRobot dataset dependencies."
    ) from cause
