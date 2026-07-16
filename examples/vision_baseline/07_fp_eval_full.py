"""Evaluate the full sort-to-shelf pipeline using YOLO + FoundationPose for pick,
TrajectoryNav for navigation, and CuRobo motion planning for place.

Runs the coherent task env with a TaskFlowAgent whose pick phase is overridden:

    YOLO + FoundationPose pick  →  TrajectoryNav to shelf  →  CuRobo place  →  done

Nav and place use the task-owned default phase agents; only the pick phase is
replaced with the perception-based SortToShelfFPAgent instead of ground-truth
CuRobo (see examples/06_compound_task.py for the default-agent pipeline).

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
    python examples/vision_baseline/07_fp_eval_full.py \
      --yolo-model playground/Checkpoints/g1_sorttoshelf_pick_v0_front_head_rgb_camera/weights/best.pt \
      --sorting-object red_cube

"""

from __future__ import annotations

import argparse

from ioailab.agents import TaskFlowAgent
from ioailab.envs import make_env
from ioailab.tasks.sort_to_shelf import GALBOT_G1_SORT_TO_SHELF_TASK_ID
from ioailab.utils.log_utils import configure, get_logger

# Sibling import: examples/ is not an importable package.
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

    fp_pick_agent = SortToShelfFPAgent(
        sorting_object=args.sorting_object,
        yolo_model=args.yolo_model,
        bridge_dir=args.bridge_dir,
        camera_key=args.camera,
        timeout_s=args.timeout,
    )

    env = make_env(
        args.task,
        num_envs=args.num_envs,
        headless=args.headless,
        randomize=args.randomize,
        task_options={"sorting_object": args.sorting_object},
    )
    try:
        logger.info(
            "Starting full-pipeline evaluation: episodes=%d max_steps=%d object=%s",
            args.episodes,
            args.max_steps,
            args.sorting_object,
        )
        try:
            metrics = env.evaluate(
                agent=TaskFlowAgent.from_env(env, agents={"pick": fp_pick_agent}),
                episodes=args.episodes,
                max_steps=args.max_steps,
            )
        except BaseException:
            logger.exception("Full-pipeline evaluation failed before metrics returned.")
            raise
    finally:
        env.close()

    logger.info(
        "full-pipeline success: %.1f%% (%d/%d) | avg_len=%.1f | object=%s | agent=yolo+fp+nav+curobo",
        100.0 * metrics["success_rate"],
        int(metrics.get("success_count", 0)),
        args.episodes,
        metrics.get("average_length", float("nan")),
        args.sorting_object,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default=GALBOT_G1_SORT_TO_SHELF_TASK_ID)
    parser.add_argument(
        "--sorting-object",
        choices=_SORTING_OBJECTS,
        default="red_cube",
        help="Sorting object to pick and place. Defaults to red_cube.",
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
    parser.add_argument("--episodes", type=int, default=36)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize object positions. Default: fixed positions for reproducibility.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
