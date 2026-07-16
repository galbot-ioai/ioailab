"""Collect one normal task with one swappable agent."""

from __future__ import annotations

import argparse

from ioailab.agents import CuroboPlannerAgent
from ioailab.envs import make_env
from ioailab.utils.log_utils import configure, get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="GalbotG1-PickCube-v0", help="Registered task ID."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/pick_cube_demos.hdf5",
        help="Output HDF5 dataset path.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Parallel vectorized env rows."
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Max steps per episode."
    )
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
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    args = parser.parse_args(argv)

    task_id = args.task
    task_options = {"init_scenario": args.init_scenario} if args.init_scenario else {}
    if args.save_end_scenario and int(args.num_envs) != 1:
        raise ValueError("--save-end-scenario requires --num-envs 1.")
    if args.save_end_scenario and int(args.episodes) != 1:
        raise ValueError("--save-end-scenario requires --episodes 1.")

    agent = CuroboPlannerAgent.from_task(task_id)

    # GP001 teleop collection:
    # from ioailab.agents import TeleopAgent
    #
    # Type "done" to finish one candidate, then choose keep/drop/exit.
    # task_id = "GalbotG1-PickCube-Teleop-v0"
    # agent = TeleopAgent.from_device("gp001", task=task_id)
    # env = make_env(task_id, num_envs=1, headless=args.headless)
    #
    # accepted = 0
    # while accepted < args.episodes:
    #     dataset = env.collect(
    #         agent=agent,
    #         path=args.dataset_path,
    #         episodes=1,
    #         max_steps=args.max_steps,
    #     )
    #     decision = agent.review_demo()
    #     if decision == "keep":
    #         accepted += 1
    #         logger.info("Saved teleop episode %s to %s", accepted, dataset.path)
    #     elif decision == "drop":
    #         dataset.drop()
    #     elif decision == "exit":
    #         dataset.drop()
    #         break
    # env.close()
    # return

    env = make_env(
        task_id,
        num_envs=args.num_envs,
        headless=args.headless,
        **({"task_options": task_options} if task_options else {}),
    )
    dataset = env.collect(
        agent=agent,
        path=args.dataset_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_end_scenario=args.save_end_scenario,
    )
    logger.info("Saved %s episode(s) to %s", args.episodes, dataset.path)
    if args.save_end_scenario:
        logger.info("Saved end scenario to %s", args.save_end_scenario)
    env.close()


if __name__ == "__main__":
    main()
