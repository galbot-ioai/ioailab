"""Collect PickToShelf or SortToShelf component tasks."""

from __future__ import annotations

import argparse

from ioailab.agents import CuroboPlannerAgent, TrajectoryNavAgent
from ioailab.envs import make_env
from ioailab.utils.log_utils import configure, get_logger

logger = get_logger(__name__)

# Select one component preset for the tutorial.
COMPONENT_PRESET = "pick_to_shelf_pick"
# COMPONENT_PRESET = "pick_to_shelf_nav"
# COMPONENT_PRESET = "pick_to_shelf_place"
# COMPONENT_PRESET = "sort_to_shelf_pick"
# COMPONENT_PRESET = "sort_to_shelf_nav"
# COMPONENT_PRESET = "sort_to_shelf_place"

SORTING_OBJECT_CHOICES = (
    "red_cube",
    "blue_cuboid",
    "yellow_cylinder",
    "green_cylinder",
)


def main(argv: list[str] | None = None) -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Output HDF5 dataset path. Defaults to data/<component-preset>.hdf5.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Total episodes to collect across all env rows.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--init-scenario",
        default=None,
        help="Optional scenario YAML used by tasks that support scenario resets.",
    )
    parser.add_argument(
        "--save-end-scenario",
        default=None,
        help="Optional YAML path for saving the final scene state after collection.",
    )
    parser.add_argument(
        "--sorting-object",
        choices=SORTING_OBJECT_CHOICES,
        default="red_cube",
        help="SortToShelf object to pick, navigate with, or place.",
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)

    if args.save_end_scenario and int(args.num_envs) != 1:
        raise ValueError("--save-end-scenario requires --num-envs 1.")
    if args.save_end_scenario and int(args.episodes) != 1:
        raise ValueError("--save-end-scenario requires --episodes 1.")

    task_options = {"init_scenario": args.init_scenario} if args.init_scenario else {}

    if COMPONENT_PRESET == "pick_to_shelf_pick":
        task_id = "GalbotG1-PickToShelf-Pick-v0"
        agent = CuroboPlannerAgent.from_task(task_id)
    elif COMPONENT_PRESET == "pick_to_shelf_nav":
        task_id = "GalbotG1-PickToShelf-Nav-v0"
        agent = TrajectoryNavAgent.from_task(task_id)
    elif COMPONENT_PRESET == "pick_to_shelf_place":
        task_id = "GalbotG1-PickToShelf-Place-v0"
        agent = CuroboPlannerAgent.from_task(task_id)
    elif COMPONENT_PRESET == "sort_to_shelf_pick":
        task_id = "GalbotG1-SortToShelf-Pick-v0"
        sort_options = {"sorting_object": args.sorting_object}
        task_options = {**task_options, **sort_options}
        agent = CuroboPlannerAgent.from_task(task_id, task_options=sort_options)
    elif COMPONENT_PRESET == "sort_to_shelf_nav":
        from ioailab.tasks.sort_to_shelf_nav.agent import nav_sequence_agent

        task_id = "GalbotG1-SortToShelf-Nav-v0"
        sort_options = {"sorting_object": args.sorting_object}
        task_options = {**task_options, **sort_options}
        agent = nav_sequence_agent(sorting_object=args.sorting_object)
    elif COMPONENT_PRESET == "sort_to_shelf_place":
        task_id = "GalbotG1-SortToShelf-Place-v0"
        sort_options = {"sorting_object": args.sorting_object}
        task_options = {**task_options, **sort_options}
        agent = CuroboPlannerAgent.from_task(task_id, task_options=sort_options)
    else:
        raise ValueError(f"Unknown COMPONENT_PRESET: {COMPONENT_PRESET!r}")

    dataset_path = args.dataset_path or f"data/{COMPONENT_PRESET}.hdf5"
    env = make_env(
        task_id,
        num_envs=args.num_envs,
        headless=args.headless,
        **({"task_options": task_options} if task_options else {}),
    )
    try:
        dataset = env.collect(
            agent=agent,
            path=dataset_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
            save_end_scenario=args.save_end_scenario,
        )
        logger.info("Saved %s episode(s) to %s", args.episodes, dataset.path)
        if args.save_end_scenario:
            logger.info("Saved end scenario to %s", args.save_end_scenario)
    finally:
        env.close()


if __name__ == "__main__":
    main()
