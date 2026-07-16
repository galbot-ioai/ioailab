"""Thin dispatch helpers for IsaacLab Mimic scripts used by ioailab."""

from __future__ import annotations

from collections.abc import Sequence
import os
import runpy
import sys
from pathlib import Path

DEFAULT_ISAACLAB_PATH = "/workspace/isaaclab"

SCRIPT_RELATIVE_PATHS = {
    "annotate_demos": "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py",
    "generate_dataset": "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
}


def resolve_script_path(isaaclab_root: Path, script_name: str) -> Path:
    """Resolve a known upstream IsaacLab Mimic script path."""

    try:
        relative_path = SCRIPT_RELATIVE_PATHS[script_name]
    except KeyError as exc:
        choices = ", ".join(sorted(SCRIPT_RELATIVE_PATHS))
        raise ValueError(
            f"Unknown IsaacLab imitation script {script_name!r}. Choices: {choices}."
        ) from exc

    script_path = isaaclab_root / relative_path
    if not script_path.exists():
        raise FileNotFoundError(
            f"IsaacLab script not found: {script_path}. Set ISAACLAB_PATH to the IsaacLab root."
        )
    return script_path


def run_script(script_path: Path, argv: Sequence[str] | None = None) -> None:
    """Run an IsaacLab script after registering ioailab tasks."""

    import ioailab.tasks

    ioailab.tasks.register_tasks()

    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    old_argv = sys.argv
    script_parent = str(script_path.parent)
    had_script_parent = script_parent in sys.path
    try:
        if not had_script_parent:
            sys.path.insert(0, script_parent)
        sys.argv = [str(script_path), *forwarded_args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        if not had_script_parent:
            try:
                sys.path.remove(script_parent)
            except ValueError:
                pass


def run_isaaclab_imitation(script_name: str, argv: Sequence[str] | None = None) -> None:
    """Register ioailab tasks and dispatch to an IsaacLab imitation script."""

    isaaclab_root = Path(os.environ.get("ISAACLAB_PATH", DEFAULT_ISAACLAB_PATH))
    run_script(resolve_script_path(isaaclab_root, script_name), argv)
