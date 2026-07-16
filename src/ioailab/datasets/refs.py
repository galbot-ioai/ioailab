"""Dataset references and augmentation provenance metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

DEFAULT_DATASET_FORMAT = "robomimic_hdf5"


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """One provenance event for a derived dataset reference."""

    operation: str
    source_path: Path | None
    source_format: str
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize provenance fields."""

        operation = self.operation.strip()
        if not operation:
            raise ValueError("DatasetProvenance.operation must be a non-empty string.")
        source_path = Path(self.source_path) if self.source_path is not None else None

        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "source_format", str(self.source_format))
        object.__setattr__(self, "config", dict(self.config))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Reference to a dataset artifact plus lightweight metadata."""

    path: Path | str
    format: str = DEFAULT_DATASET_FORMAT
    task_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[DatasetProvenance, ...] = ()

    def __post_init__(self) -> None:
        """Normalize dataset reference fields."""

        dataset_format = str(self.format).strip()
        if not dataset_format:
            raise ValueError("DatasetRef.format must be a non-empty string.")

        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "format", dataset_format)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "provenance", tuple(self.provenance))

    def drop(self) -> "DatasetRef":
        """Remove the demos saved by this dataset reference from its HDF5 file."""

        demo_ids = _metadata_demo_ids(self.metadata)
        if not demo_ids:
            raise ValueError("DatasetRef.drop() requires metadata['saved_demo_ids'].")
        if self.format not in {"hdf5", "robomimic_hdf5"}:
            raise ValueError(f"Unsupported droppable dataset format: {self.format}")
        dropped = _drop_hdf5_demos(self.path, demo_ids)
        metadata = dict(self.metadata)
        metadata["dropped_demo_ids"] = tuple(dropped)
        return DatasetRef(
            path=self.path,
            format=self.format,
            task_id=self.task_id,
            metadata=metadata,
            provenance=self.provenance,
        )


def ensure_dataset_ref(
    data: DatasetRef | str | Path, *, format: str = DEFAULT_DATASET_FORMAT
) -> DatasetRef:
    """Return ``data`` as a ``DatasetRef``."""

    if isinstance(data, DatasetRef):
        return data
    return DatasetRef(path=data, format=format)


def _metadata_demo_ids(metadata: Mapping[str, Any]) -> tuple[int, ...]:
    """Return saved demo ids from dataset metadata."""

    value = metadata.get("saved_demo_ids")
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        return (int(value),)
    return tuple(int(item) for item in value)


def _drop_hdf5_demos(path: Path, demo_ids: Sequence[int]) -> tuple[int, ...]:
    """Delete Robomimic-style ``data/demo_<id>`` groups from an HDF5 file."""

    import h5py

    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")

    demo_names = tuple(f"demo_{int(demo_id)}" for demo_id in demo_ids)
    dropped: list[int] = []
    with h5py.File(path, "r+") as file:
        data_group = file.get("data")
        if data_group is None:
            raise ValueError(f"dataset has no 'data' group: {path}")
        for demo_id, demo_name in zip(demo_ids, demo_names, strict=True):
            if demo_name in data_group:
                del data_group[demo_name]
                dropped.append(int(demo_id))
        _drop_demo_names_from_masks(file, demo_names)
        _refresh_data_group_attrs(data_group)
    return tuple(dropped)


def _drop_demo_names_from_masks(file: Any, demo_names: Sequence[str]) -> None:
    """Remove dropped demo names from Robomimic mask datasets when present."""

    mask_group = file.get("mask")
    if mask_group is None:
        return
    dropped_names = set(demo_names)
    for mask_name in tuple(mask_group.keys()):
        dataset = mask_group[mask_name]
        kept = [
            item
            for item in dataset[()]
            if _decode_hdf5_string(item) not in dropped_names
        ]
        del mask_group[mask_name]
        mask_group.create_dataset(mask_name, data=kept)


def _refresh_data_group_attrs(data_group: Any) -> None:
    """Refresh common Robomimic data-group attributes after deleting demos."""

    demo_names = tuple(name for name in data_group.keys() if name.startswith("demo_"))
    if "num_demos" in data_group.attrs:
        data_group.attrs["num_demos"] = len(demo_names)
    if "total" in data_group.attrs:
        data_group.attrs["total"] = sum(
            _demo_num_samples(data_group[name]) for name in demo_names
        )


def _demo_num_samples(demo_group: Any) -> int:
    """Return the sample count for one Robomimic demo group."""

    if "num_samples" in demo_group.attrs:
        return int(demo_group.attrs["num_samples"])
    actions = demo_group.get("actions")
    if actions is not None and getattr(actions, "shape", ()):
        return int(actions.shape[0])
    return 0


def _decode_hdf5_string(value: Any) -> str:
    """Decode bytes from HDF5 variable-length string datasets."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _default_mimic_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_mimic{path.suffix or '.hdf5'}")


def _default_annotated_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_annotated{path.suffix or '.hdf5'}")


def _require_existing_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _generated_file_error(
    path: Path, *, operation: str, task: str, reason: str
) -> FileNotFoundError:
    return FileNotFoundError(
        f"{operation} did not produce a usable dataset: {path} ({reason}). "
        f"Check IsaacLab Mimic support for task {task!r}; use --skip-mimic to train "
        "directly on expert demonstrations."
    )


def _require_generated_file(path: Path, *, operation: str, task: str) -> None:
    if not path.is_file():
        raise _generated_file_error(
            path, operation=operation, task=task, reason="missing"
        )
    if path.suffix.lower() not in {".hdf5", ".h5"}:
        return

    import h5py

    try:
        with h5py.File(path, "r"):
            return
    except OSError as exc:
        raise _generated_file_error(
            path, operation=operation, task=task, reason="invalid HDF5"
        ) from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(repo_root / "src"), str(repo_root)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def _run_annotate_script(argv: Sequence[str]) -> None:
    repo_root = _repo_root()
    code = (
        "import sys; "
        "from ioailab.datasets.mimic.bridge import run_isaaclab_imitation; "
        "run_isaaclab_imitation('annotate_demos', sys.argv[1:])"
    )
    subprocess.run(
        [sys.executable, "-c", code, *argv],
        check=True,
        cwd=repo_root,
        env=_subprocess_env(repo_root),
    )


def _run_mimic_generate_script(argv: Sequence[str]) -> None:
    repo_root = _repo_root()
    code = (
        "import sys; "
        "from ioailab.datasets.mimic.generation import main; "
        "main(sys.argv[1:])"
    )
    subprocess.run(
        [sys.executable, "-c", code, *argv],
        check=True,
        cwd=repo_root,
        env=_subprocess_env(repo_root),
    )


def run_annotate_demos(
    data: DatasetRef | str | Path,
    *,
    task: str,
    output_path: str | Path | None = None,
    auto: bool = True,
    extra_args: Sequence[str] = (),
    runner: Callable[[Sequence[str]], None] | None = None,
) -> DatasetRef:
    """Run IsaacLab annotation for expert demonstrations and return its output ref.

    IsaacLab Mimic defaults to manual keyboard annotation unless ``--auto`` is
    passed. ioailab pipelines use automatic subtask signals by default so a
    full collect/annotate/generate/train run does not block on key presses.
    """

    task_id = task.strip()
    if not task_id:
        raise ValueError("task must be a non-empty string.")

    source = ensure_dataset_ref(data)
    _require_existing_file(source.path, label="annotation input dataset")
    destination = (
        Path(output_path)
        if output_path is not None
        else _default_annotated_output_path(source.path)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    annotation_args = list(extra_args)
    if auto and "--auto" not in annotation_args:
        annotation_args.insert(0, "--auto")

    argv = [
        "--task",
        task_id,
        "--input_file",
        str(source.path),
        "--output_file",
        str(destination),
        *annotation_args,
    ]
    (runner or _run_annotate_script)(argv)
    _require_generated_file(destination, operation="annotation", task=task_id)

    annotation_metadata = {"status": "complete"}
    provenance = DatasetProvenance(
        operation="annotate",
        source_path=source.path,
        source_format=source.format,
        config={"task": task_id, "auto": auto, "extra_args": annotation_args},
        metadata=annotation_metadata,
    )
    metadata = dict(source.metadata)
    metadata["annotate"] = dict(annotation_metadata)
    return DatasetRef(
        path=destination,
        format=source.format,
        task_id=task_id,
        metadata=metadata,
        provenance=(*source.provenance, provenance),
    )


def run_mimic_generation(
    data: DatasetRef | str | Path,
    *,
    task: str,
    episodes: int,
    output_path: str | Path | None = None,
    extra_args: Sequence[str] = (),
    runner: Callable[[Sequence[str]], None] | None = None,
) -> DatasetRef:
    """Run IsaacLab Mimic generation and return the materialized dataset ref."""

    task_id = task.strip()
    if not task_id:
        raise ValueError("task must be a non-empty string.")
    if episodes <= 0:
        raise ValueError("episodes must be a positive integer.")

    source = ensure_dataset_ref(data)
    _require_existing_file(source.path, label="mimic input dataset")
    destination = (
        Path(output_path)
        if output_path is not None
        else _default_mimic_output_path(source.path)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        "--task",
        task_id,
        "--input_file",
        str(source.path),
        "--output_file",
        str(destination),
        "--generation_num_trials",
        str(episodes),
        *extra_args,
    ]
    (runner or _run_mimic_generate_script)(argv)
    _require_generated_file(destination, operation="mimic generation", task=task_id)

    mimic_metadata = {
        "status": "complete",
        "successful_episodes": episodes,
        "failed_episodes": 0,
    }
    provenance = DatasetProvenance(
        operation="mimic_generation",
        source_path=source.path,
        source_format=source.format,
        config={"task": task_id, "episodes": episodes, "extra_args": list(extra_args)},
        metadata=mimic_metadata,
    )
    metadata = dict(source.metadata)
    metadata["mimic"] = dict(mimic_metadata)
    return DatasetRef(
        path=destination,
        format=source.format,
        task_id=task_id,
        metadata=metadata,
        provenance=(*source.provenance, provenance),
    )


def mimic(
    mimic_config_or_data: Mapping[str, Any] | DatasetRef | str | Path,
    data: DatasetRef | str | Path | None = None,
    *,
    episodes: int | None = None,
    output_path: str | Path | None = None,
    num_envs: int = 1,
    headless: bool = False,
    task: str | None = None,
) -> DatasetRef:
    """Expand a dataset with IsaacLab Mimic or record planned Mimic provenance.

    Preferred runtime form::

        dataset = mimic(dataset, episodes=20)

    The task is read from ``DatasetRef.task_id``. The older metadata-only form
    ``mimic(mimic_config, data)`` is still accepted for lightweight provenance
    construction.
    """

    if isinstance(mimic_config_or_data, Mapping):
        if data is None:
            raise TypeError("mimic(config, data) requires a source dataset.")
        return _planned_mimic(mimic_config_or_data, data, output_path=output_path)

    if data is not None:
        raise TypeError("mimic(dataset, ...) accepts only one dataset argument.")
    if episodes is None:
        raise TypeError("mimic(dataset, ...) requires episodes=... .")

    source = ensure_dataset_ref(mimic_config_or_data)
    source_task_id = _resolve_dataset_task_id(source, task=task)
    mimic_task_id = _mimic_runtime_task_id(source_task_id)
    destination = (
        Path(output_path)
        if output_path is not None
        else _default_mimic_output_path(source.path)
    )
    annotated_path = destination.with_name(
        f"{destination.stem}_annotated{destination.suffix or '.hdf5'}"
    )
    app_args = (
        ("--headless", "--enable_cameras")
        if headless
        else ("--enable_cameras", "--viz", "kit")
    )

    annotated = run_annotate_demos(
        DatasetRef(
            path=source.path,
            format=source.format,
            task_id=source_task_id,
            metadata=dict(source.metadata),
            provenance=source.provenance,
        ),
        task=mimic_task_id,
        output_path=annotated_path,
        extra_args=app_args,
    )
    generated = run_mimic_generation(
        annotated,
        task=mimic_task_id,
        episodes=int(episodes),
        output_path=destination,
        extra_args=("--num_envs", str(int(num_envs)), *app_args),
    )

    metadata = dict(generated.metadata)
    metadata["source_task_id"] = source_task_id
    metadata["mimic_task_id"] = mimic_task_id
    return DatasetRef(
        path=generated.path,
        format=generated.format,
        task_id=source_task_id,
        metadata=metadata,
        provenance=generated.provenance,
    )


def _planned_mimic(
    mimic_config: Mapping[str, Any],
    data: DatasetRef | str | Path,
    *,
    output_path: str | Path | None = None,
) -> DatasetRef:
    """Create a metadata-only Mimic dataset reference."""

    source = ensure_dataset_ref(data)
    destination = (
        Path(output_path)
        if output_path is not None
        else _default_mimic_output_path(source.path)
    )
    mimic_metadata = {
        "status": "planned",
        "successful_episodes": None,
        "failed_episodes": None,
    }
    provenance = DatasetProvenance(
        operation="mimic",
        source_path=source.path,
        source_format=source.format,
        config=dict(mimic_config),
        metadata=mimic_metadata,
    )
    metadata = dict(source.metadata)
    metadata["mimic"] = dict(mimic_metadata)

    return DatasetRef(
        path=destination,
        format=source.format,
        task_id=source.task_id,
        metadata=metadata,
        provenance=(*source.provenance, provenance),
    )


def _resolve_dataset_task_id(source: DatasetRef, *, task: str | None = None) -> str:
    task_id = (task or source.task_id or "").strip()
    if not task_id:
        raise ValueError(
            "mimic(dataset, ...) requires DatasetRef.task_id or task=... ."
        )
    return task_id


def _mimic_runtime_task_id(task_id: str) -> str:
    if task_id.endswith("-Mimic-v0"):
        return task_id
    if task_id.endswith("-v0"):
        return f"{task_id.removesuffix('-v0')}-Mimic-v0"
    return f"{task_id}-Mimic-v0"


__all__ = [
    "DEFAULT_DATASET_FORMAT",
    "DatasetProvenance",
    "DatasetRef",
    "ensure_dataset_ref",
    "mimic",
    "run_annotate_demos",
    "run_mimic_generation",
]
