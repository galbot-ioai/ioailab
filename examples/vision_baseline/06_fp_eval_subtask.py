"""Evaluate the sort-to-shelf pick phase task using YOLO + FoundationPose + CuRobo.

Perception-based alternative to evaluating ``GalbotG1-SortToShelf-Pick-v0`` with
``CuroboPlannerAgent`` (see examples/04_eval.py): this script localizes the target
object via YOLO segmentation and FoundationPose 6-DoF pose estimation instead of
reading ground-truth scene state.

Pipeline:
    YOLO mask  →  FoundationPose 6-DoF pose  →  CuRobo motion plan  →  grasp

Prerequisites:
    1. Generate sorting-object meshes (one-time setup):
           python examples/vision_baseline/scripts/export_sorting_meshes.py

    2. In a separate dev-container terminal, start the FoundationPose server:
           micromamba run -n foundationpose \
             python examples/vision_baseline/04_fp_start_server.py \
             --bridge-dir data/foundationpose_bridge/sort_to_shelf \
             --foundationpose-dir /opt/FoundationPose

    3. Inside the dev container, install Ultralytics:
           make shell
           pip install ultralytics

Usage:
    python examples/vision_baseline/06_fp_eval_subtask.py \
      --yolo-model playground/Checkpoints/g1_sorttoshelf_pick_v0_front_head_rgb_camera/weights/best.pt \
      --sorting-object red_cube \
      --headless

Compared with the ground-truth pipeline:
    CuroboPlannerAgent plans from ground-truth scene positions; a trained
    PolicyAgent replays a DiffusionPolicy checkpoint.
    This example uses SortToShelfFPAgent (YOLO + FoundationPose localization).
"""

from __future__ import annotations

import argparse

from ioailab.envs import make_env
from ioailab.tasks.sort_to_shelf_pick import GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID
from ioailab.utils.log_utils import configure, get_logger

# Sibling import: examples/ is not an importable package, so a script run from
# this directory resolves siblings by bare module name (sys.path[0]).
from foundation_pose_sort_to_shelf import (
    DEFAULT_BRIDGE_DIR,
    DEFAULT_CAMERA,
    DEFAULT_TIMEOUT_S,
    SortToShelfFPAgent,
)

logger = get_logger(__name__)

_SORTING_OBJECTS = ("red_cube", "blue_cuboid", "yellow_cylinder", "green_cylinder")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""

    configure()
    args = parse_args(argv)

    agent = SortToShelfFPAgent(
        sorting_object=args.sorting_object,
        yolo_model=args.yolo_model,
        bridge_dir=args.bridge_dir,
        camera_key=args.camera,
        timeout_s=args.timeout,
    )

    env = make_env(
        args.task,
        num_envs=1,
        headless=args.headless,
        randomize=args.randomize,
        task_options={"sorting_object": args.sorting_object},
    )
    try:
        logger.info(
            "Starting evaluation: episodes=%d max_steps=%d",
            args.episodes,
            args.max_steps,
        )
        metrics = env.evaluate(
            agent=agent,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )
    finally:
        env.close()

    logger.info(
        "pick success: %.1f%% (%d/%d) | avg_len=%.1f | agent=%s",
        100.0 * metrics["success_rate"],
        int(metrics.get("success_count", 0)),
        args.episodes,
        metrics.get("average_length", float("nan")),
        "yolo+foundationpose",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default=GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID)
    parser.add_argument(
        "--sorting-object",
        choices=_SORTING_OBJECTS,
        default="red_cube",
        help="Sorting object to pick. Defaults to red_cube.",
    )
    parser.add_argument(
        "--yolo-model",
        required=True,
        help="Path to a trained YOLO-seg checkpoint (.pt). Required.",
    )
    parser.add_argument(
        "--bridge-dir",
        default=DEFAULT_BRIDGE_DIR,
        help=(
            "Shared directory for the sim↔FoundationPose file bridge. "
            f"Defaults to {DEFAULT_BRIDGE_DIR}."
        ),
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"Camera key in the task scene. Defaults to {DEFAULT_CAMERA}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=(
            "Seconds to wait for the FoundationPose server. "
            f"Defaults to {DEFAULT_TIMEOUT_S}."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=36,
        help="Total evaluation episodes across all env rows.",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize object positions. Default: fixed positions for reproducibility.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
