"""Train a robomimic Diffusion Policy from a Mimic-expanded dataset."""

from __future__ import annotations

import argparse

from ioailab.datasets import DatasetRef
from ioailab.agents.policy import (
    OptimizerCfg,
    Policy,
    RobomimicDiffusionTrainCfg,
)
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
        default="data/pick_cube_demos_mimic.hdf5",
        help="Training dataset path.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/pick_cube", help="Checkpoint output directory."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size; uses robomimic default when omitted.",
    )
    parser.add_argument(
        "--num-data-workers",
        type=int,
        default=8,
        help="Robomimic dataloader worker count.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-4,
        help="Policy optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Policy optimizer weight decay.",
    )
    args = parser.parse_args(argv)

    dataset = DatasetRef(args.dataset_path, task_id=args.task)
    policy = Policy.from_backend("robomimic_diffusion")
    train_cfg = RobomimicDiffusionTrainCfg(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_data_workers=args.num_data_workers,
        optimizer=OptimizerCfg(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )
    checkpoint = policy.train(dataset, train_cfg)
    logger.info("Policy checkpoint: %s", checkpoint.path)


if __name__ == "__main__":
    main()
