"""Dispatch IsaacLab Mimic dataset generation for ioailab datasets."""

from __future__ import annotations

from collections.abc import Sequence
import shutil
import sys
from pathlib import Path


_OUTPUT_FILE_FLAGS = ("--output_file", "--output-file")


def _parse_output_file_arg(argv: Sequence[str]) -> Path | None:
    """Return the IsaacLab Mimic output path from forwarded CLI args."""

    for index, arg in enumerate(argv):
        if arg in _OUTPUT_FILE_FLAGS:
            if index + 1 >= len(argv):
                return None
            return Path(argv[index + 1])
        for flag in _OUTPUT_FILE_FLAGS:
            prefix = f"{flag}="
            if arg.startswith(prefix):
                return Path(arg[len(prefix) :])
    return None


def _failed_mimic_hdf5_path(output_path: Path) -> Path:
    """Return the failed HDF5 path IsaacLab creates for an output path."""

    return output_path.parent / f"{output_path.stem}_failed.hdf5"


def _failed_archive_dir(output_path: Path) -> Path:
    """Return the archive directory for a failed Mimic HDF5 file."""

    for parent in output_path.parents:
        if parent.name == "datasets":
            return parent / "failed"
    return output_path.parent / "failed"


def _unique_archive_path(path: Path) -> Path:
    """Return a non-existing archive path without overwriting user data."""

    if not path.exists():
        return path

    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _archive_failed_mimic_hdf5(output_path: str | Path | None) -> Path | None:
    """Move IsaacLab's failed Mimic HDF5 next to other failed datasets.

    Args:
        output_path: The successful Mimic output path passed to IsaacLab.

    Returns:
        The archived failed HDF5 path when a file was moved, otherwise ``None``.
    """

    if output_path is None:
        return None

    output = Path(output_path)
    source = _failed_mimic_hdf5_path(output)
    if not source.is_file():
        return None

    archive_dir = _failed_archive_dir(output)
    target = archive_dir / source.name
    if source.resolve(strict=False) == target.resolve(strict=False):
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_archive_path(target)
    shutil.move(str(source), str(target))
    return target


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch to IsaacLab Mimic generation and archive failed HDF5 output."""

    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    output_path = _parse_output_file_arg(forwarded_args)

    from ioailab.datasets.mimic.batched_generation import (
        install_batched_mimic_generation_patch,
    )
    from ioailab.datasets.mimic.bridge import run_isaaclab_imitation

    try:
        install_batched_mimic_generation_patch()
        run_isaaclab_imitation("generate_dataset", forwarded_args)
    finally:
        archived_path = _archive_failed_mimic_hdf5(output_path)
        if archived_path is not None:
            print(f"Archived failed Mimic HDF5 to {archived_path}.", flush=True)


if __name__ == "__main__":
    main()
