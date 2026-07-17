"""Run a coherent compound task and optionally override phase agents."""

from __future__ import annotations

import argparse

from ioailab.agents import TaskFlowAgent
from ioailab.envs import make_env
from ioailab.tasks.pick_to_shelf import GALBOT_G1_PICK_TO_SHELF_TASK_ID
from ioailab.utils.log_utils import configure, get_logger

logger = get_logger(__name__)

# Select one compound-agent preset for the tutorial.
# COMPOUND_AGENT_PRESET = "task_default"
# COMPOUND_AGENT_PRESET = "pick_to_shelf_experts"
COMPOUND_AGENT_PRESET = "pick_to_shelf_policy"


def main(argv: list[str] | None = None) -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=GALBOT_G1_PICK_TO_SHELF_TASK_ID)
    parser.add_argument(
        "--mode",
        choices=("evaluate", "collect"),
        default="evaluate",
        help="Run evaluation metrics or collect an expert dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/compound_task_demos.hdf5",
        help="Output HDF5 path used with --mode collect.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=36,
        help="Total episodes across all env rows.",
    )
    parser.add_argument("--num-envs", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--sorting-object",
        default=None,
        help="SortToShelf object name, for example red_cube.",
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)

    task_options = (
        {"sorting_object": args.sorting_object} if args.sorting_object else {}
    )
    env = make_env(
        args.task,
        num_envs=args.num_envs,
        headless=args.headless,
        **({"task_options": task_options} if task_options else {}),
    )
    try:
        if COMPOUND_AGENT_PRESET == "task_default":
            agent = TaskFlowAgent.from_env(env)
        elif COMPOUND_AGENT_PRESET == "pick_to_shelf_experts":
            from ioailab.agents import CuroboPlannerAgent, TrajectoryNavAgent

            phase_agents = {
                "pick": CuroboPlannerAgent.from_task("GalbotG1-PickToShelf-Pick-v0"),
                "nav": TrajectoryNavAgent.from_task("GalbotG1-PickToShelf-Nav-v0"),
                "place": CuroboPlannerAgent.from_task("GalbotG1-PickToShelf-Place-v0"),
            }
            agent = TaskFlowAgent.from_env(env, agents=phase_agents)
        elif COMPOUND_AGENT_PRESET == "pick_to_shelf_policy":
            from ioailab.agents import G1ManipulationPolicyActionAdapter, PolicyAgent

            phase_agents = {
                "pick": G1ManipulationPolicyActionAdapter(
                    PolicyAgent.from_checkpoint("outputs/pick/model_best_training.pth")
                ),
                "place": G1ManipulationPolicyActionAdapter(
                    PolicyAgent.from_checkpoint("outputs/place/model_best_training.pth")
                ),
            }
            agent = TaskFlowAgent.from_env(env, agents=phase_agents)
        else:
            raise ValueError(
                f"Unknown COMPOUND_AGENT_PRESET: {COMPOUND_AGENT_PRESET!r}"
            )

        if args.mode == "collect":
            dataset = env.collect(
                agent=agent,
                path=args.dataset_path,
                episodes=args.episodes,
                max_steps=args.max_steps,
                metadata={"collection": "compound_task", "task": args.task},
            )
            logger.info("Saved %d demo(s) to %s", args.episodes, dataset.path)
        else:
            metrics = env.evaluate(
                agent=agent,
                episodes=args.episodes,
                max_steps=args.max_steps,
            )
            logger.info("Evaluated %d episode(s)", metrics["total_episodes"])
            logger.info("Success rate: %.1f%%", metrics["success_rate"] * 100)
    finally:
        env.close()


if __name__ == "__main__":
    main()
