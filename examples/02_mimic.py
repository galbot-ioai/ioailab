"""Expand collected demonstrations with IsaacLab Mimic."""

from __future__ import annotations

import argparse

from ioailab.datasets import DatasetRef, mimic
from ioailab.utils.log_utils import configure, get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="GalbotG1-PickCube-v0", help="Registered source task ID."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/pick_cube_demos.hdf5",
        help="Input HDF5 dataset path.",
    )
    parser.add_argument(
        "--output-path",
        default="data/pick_cube_demos_mimic.hdf5",
        help="Output HDF5 dataset path.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=36,
        help="Total Mimic trials to generate.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=9, help="Parallel Mimic env rows."
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Mimic generation without viewer; pass --headless to disable "
        "the viewer.",
    )
    args = parser.parse_args(argv)

    dataset = DatasetRef(args.dataset_path, task_id=args.task)
    dataset = mimic(
        dataset,
        episodes=args.episodes,
        output_path=args.output_path,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    logger.info("Mimic dataset: %s", dataset.path)


if __name__ == "__main__":
    main()
