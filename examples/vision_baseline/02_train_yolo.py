"""Train an Ultralytics YOLO segmentation model on generated ioailab data.

Prerequisites:
    Install Ultralytics inside the dev container:
        make shell
        pip install ultralytics

Usage:
    python examples/vision_baseline/02_train_yolo.py

    python examples/vision_baseline/02_train_yolo.py \
      --data playground/Datasets/g1_sorttoshelf_pick_v0/front_head_rgb_camera/data.yaml \
      --model yolo26n-seg.pt \
      --epochs 200 \
      --imgsz 320 \
      --batch 16

By default, the only data.yaml under playground/Datasets is used. Training
results are written to playground/Checkpoints/.

See docs/yolo_seg.md for the full workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# Two levels up from examples/vision_baseline/ to the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = "playground/Datasets"
DEFAULT_PROJECT = "playground/Checkpoints"
DEFAULT_PRETRAINED_ROOT = "playground/Pretrained_models"


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""

    args = parse_args(argv)
    data_path = resolve_data_path(data=args.data, data_root=DEFAULT_DATA_ROOT)
    project_path = resolve_repo_path(DEFAULT_PROJECT)
    model_path = resolve_model_path(args.model, pretrained_root=DEFAULT_PRETRAINED_ROOT)

    YOLO = import_yolo()
    model = YOLO(str(model_path))
    train_kwargs: dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": parse_batch(args.batch),
        "workers": 8,
        "patience": 0,
        "project": str(project_path),
        "name": args.name or default_run_name(data_path),
        "exist_ok": args.exist_ok,
        "resume": False,
        "val": True,
        "amp": False,
        "task": "segment",
    }

    print(f"[train] Dataset yaml: {data_path}")
    print(f"[train] Runs directory: {project_path}")
    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        best_path = Path(save_dir) / "weights" / "best.pt"
        print(f"[train] Results saved to: {save_dir}")
        print(f"[train] Best checkpoint: {best_path}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Path to a YOLO data.yaml. Passing a dataset directory containing "
            "data.yaml is also accepted. If omitted, playground/Datasets must "
            "contain exactly one data.yaml."
        ),
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        default="yolo26n-seg.pt",
        help=(
            "Ultralytics segmentation model checkpoint or model yaml. Relative "
            "checkpoint names are loaded from or downloaded to "
            "playground/Pretrained_models."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument(
        "--batch",
        default="-1",
        help="Batch size. Use -1 for Ultralytics auto-batch.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name under playground/Checkpoints.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args(argv)


def import_yolo() -> Any:
    """Import Ultralytics with a clear setup error."""

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is required for YOLO training but is not included in "
            "the default ioailab Docker image. Install it inside the dev "
            "container with `pip install ultralytics`, then rerun this command."
        ) from exc
    return YOLO


def resolve_data_path(*, data: str | None, data_root: str) -> Path:
    """Return the requested or uniquely detected YOLO data.yaml path."""

    if data:
        path = resolve_repo_path(data)
        if path.is_dir():
            path = path / "data.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset yaml does not exist: {path}")
        return path

    root = resolve_repo_path(data_root)
    candidates = sorted(path for path in root.rglob("data.yaml") if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No data.yaml found under {root}. Generate a dataset first with "
            "`python examples/vision_baseline/01_generate_yolo_dataset.py ...` "
            "or pass `--data <camera_dataset_root>/data.yaml`."
        )
    if len(candidates) > 1:
        formatted = "\n".join(f"  - {path}" for path in candidates)
        raise ValueError(
            f"Multiple data.yaml files found under {root}. Pass --data explicitly.\n"
            f"{formatted}"
        )
    return candidates[0]


def resolve_model_path(model: str, *, pretrained_root: str) -> Path | str:
    """Resolve model paths, keeping checkpoint downloads under pretrained_root."""

    candidate = Path(model).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    pretrained_candidate = resolve_repo_path(str(Path(pretrained_root) / candidate))
    if pretrained_candidate.exists():
        return pretrained_candidate

    repo_candidate = resolve_repo_path(str(candidate))
    if repo_candidate.exists():
        return repo_candidate

    if candidate.parent == Path(".") and candidate.suffix == ".pt":
        pretrained_candidate.parent.mkdir(parents=True, exist_ok=True)
        return pretrained_candidate

    return model


def resolve_repo_path(path: str) -> Path:
    """Resolve a path relative to the repository root when needed."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def parse_batch(value: str) -> int | float:
    """Parse Ultralytics batch values."""

    if "." in value:
        return float(value)
    return int(value)


def default_run_name(data_path: Path) -> str:
    """Return a stable run name from the generated dataset layout."""

    camera_name = data_path.parent.name
    dataset_name = data_path.parent.parent.name
    if dataset_name and camera_name:
        return f"{dataset_name}_{camera_name}"
    return data_path.parent.name or "yolo_seg_train"


if __name__ == "__main__":
    main()
